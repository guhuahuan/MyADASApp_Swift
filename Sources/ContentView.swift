import SwiftUI
@preconcurrency import Vision
@preconcurrency import AVFoundation
import CoreML

// 1. 新增：专门用于跨线程传递的原始检测数据结构
struct RawDetection: Sendable {
    let box: CGRect
    let label: String
}

struct TrackedObject: Identifiable, Sendable {
    let id: UUID
    let label: String
    let currentBox: CGRect
    let velocity: CGPoint
}

@MainActor
class GlobalTrafficController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var alertLevel: Double = 0.0
    
    let captureSession = AVCaptureSession()
    
    // 依然使用 unsafe 绕过模型本身的 Sendable 检查
    nonisolated(unsafe) private var internalModel: VNCoreMLModel?
    private let videoDataQueue = DispatchQueue(label: "video_data_queue", qos: .userInitiated)

    override init() {
        super.init()
        loadModel()
        setupCapture()
    }

    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "yolov8s", withExtension: "mlmodelc") else { return }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let m = try MLModel(contentsOf: modelURL, configuration: config)
            self.internalModel = try VNCoreMLModel(for: m)
        } catch {
            print("Model load failed")
        }
    }

    func setupCapture() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: videoDataQueue)
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        captureSession.commitConfiguration()
        
        Task {
            self.captureSession.startRunning()
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let model = self.internalModel else { return }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            
            // 2. 修复：映射为 Sendable 的结构体数组，而不是元组数组
            let detections = results.map { obs in
                RawDetection(
                    box: obs.boundingBox,
                    label: obs.labels.first?.identifier ?? "obj"
                )
            }
            
            // 3. 此时 detections 是 [RawDetection]，符合 Sendable，跨边界安全
            Task { @MainActor in
                self?.updateTracking(with: detections)
            }
        }
        
        try? handler.perform([request])
    }

    // 4. 更新参数类型
    private func updateTracking(with newObservations: [RawDetection]) {
        let targetLabels = ["person", "car", "bus", "truck", "motorbike"]
        var totalRisk = 0.0
        
        let updated = newObservations.compactMap { obs -> TrackedObject? in
            let label = obs.label
            let box = obs.box
            
            guard targetLabels.contains(label) else { return nil }
            let center = CGPoint(x: box.midX, y: box.midY)
            
            let match = self.trackedObjects.first(where: { 
                abs($0.currentBox.midX - box.midX) < 0.15 && abs($0.currentBox.midY - box.midY) < 0.15 
            })
            
            let vel = match != nil ? CGPoint(x: center.x - match!.currentBox.midX, y: center.y - match!.currentBox.midY) : .zero
            
            if box.midX > 0.3 && box.midX < 0.7 && box.width * box.height > 0.08 {
                totalRisk += 0.2
            }
            
            return TrackedObject(id: match?.id ?? UUID(), label: label, currentBox: box, velocity: vel)
        }
        
        self.trackedObjects = updated
        self.alertLevel = min(totalRisk, 1.0)
    }
}

// 以下预览和 UI 代码保持一致
struct CameraPreviewHolder: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.frame = view.layer.bounds
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct ContentView: View {
    @StateObject var controller = GlobalTrafficController()
    var body: some View {
        ZStack {
            CameraPreviewHolder(session: controller.captureSession).ignoresSafeArea()
            Canvas { context, size in
                for obj in controller.trackedObjects {
                    let rect = CGRect(
                        x: obj.currentBox.minX * size.width,
                        y: (1 - obj.currentBox.maxY) * size.height,
                        width: obj.currentBox.width * size.width,
                        height: obj.currentBox.height * size.height
                    )
                    let color: Color = controller.alertLevel > 0.5 ? .red : .green
                    context.stroke(Path(rect), with: .color(color), lineWidth: 2)
                    
                    var line = Path()
                    line.move(to: CGPoint(x: rect.midX, y: rect.midY))
                    line.addLine(to: CGPoint(x: rect.midX + obj.velocity.x * size.width * 10, y: rect.midY - obj.velocity.y * size.height * 10))
                    context.stroke(line, with: .color(color.opacity(0.6)), style: StrokeStyle(lineWidth: 1, dash: [5]))
                }
            }
            VStack {
                Text("ADAS ACTIVE - VIETNAM MODE")
                    .font(.system(.caption, design: .monospaced)).bold()
                    .padding(8).background(controller.alertLevel > 0.5 ? .red : .black.opacity(0.7))
                    .foregroundColor(.white).cornerRadius(8).padding(.top, 50)
                Spacer()
            }
        }
    }
}

@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

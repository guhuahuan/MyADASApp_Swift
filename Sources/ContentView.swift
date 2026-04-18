import SwiftUI
@preconcurrency import Vision
@preconcurrency import AVFoundation
import CoreML

// 1. 显式标记为 Sendable，解决跨线程数据竞争的核心结构
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
    
    // 强制关闭模型并发检查：模型是只读的，由我们手动担保安全
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

    // 关键：nonisolated 回调，避免主线程阻塞
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let model = self.internalModel else { return }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        
        let request = VNCoreMLRequest(model: model) { req, _ in
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            
            // 提取数据
            let detections = results.map { obs in
                RawDetection(
                    box: obs.boundingBox,
                    label: obs.labels.first?.identifier ?? "obj"
                )
            }
            
            // 核心修复：通过 Task 捕获 Sendable 数据，并异步切回主线程执行 UI 更新
            Task { [weak self, detections] in
                await self?.updateTracking(with: detections)
            }
        }
        
        try? handler.perform([request])
    }

    // 控制器本身是 @MainActor，所以这个方法必须在主线程执行
    private func updateTracking(with newObservations: [RawDetection]) {
        let targetLabels = ["person", "car", "bus", "truck", "motorbike"]
        var totalRisk = 0.0
        
        let updated = newObservations.compactMap { obs -> TrackedObject? in
            guard targetLabels.contains(obs.label) else { return nil }
            let center = CGPoint(x: obs.box.midX, y: obs.box.midY)
            
            let match = self.trackedObjects.first(where: { 
                abs($0.currentBox.midX - obs.box.midX) < 0.15 && abs($0.currentBox.midY - obs.box.midY) < 0.15 
            })
            
            let vel = match != nil ? CGPoint(x: center.x - match!.currentBox.midX, y: center.y - match!.currentBox.midY) : .zero
            
            // 针对越南高密度路况的简易碰撞模型
            if obs.box.midX > 0.3 && obs.box.midX < 0.7 && obs.box.width * obs.box.height > 0.08 {
                totalRisk += 0.2
            }
            
            return TrackedObject(id: match?.id ?? UUID(), label: obs.label, currentBox: obs.box, velocity: vel)
        }
        
        self.trackedObjects = updated
        self.alertLevel = min(totalRisk, 1.0)
    }
}

// MARK: - UI 视图
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
                    
                    // 速度矢量线
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

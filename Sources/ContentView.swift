import SwiftUI
import Vision
@preconcurrency import AVFoundation
import CoreML

// 1. 定义 Sendable 数据结构，确保跨线程传输安全
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
    private var internalModel: VNCoreMLModel?
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

    // 关键修正：在 nonisolated 环境下安全处理
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 步骤 A: 立即在后台执行 Vision 请求，不跨线程传递 pixelBuffer
        // 我们通过 run 函数在主线程取回模型引用，但这必须极快
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        
        Task {
            // 在主线程安全获取模型引用
            guard let model = await MainActor.run(resultType: VNCoreMLModel?.self, {
                return self.internalModel 
            }) else { return }
            
            // 步骤 B: 在后台异步执行推理
            let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
                guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
                
                // 将结果转为 Sendable 类型 (CGRect, String)
                let detections = results.map { (obs: VNRecognizedObjectObservation) -> (CGRect, String) in
                    return (obs.boundingBox, obs.labels.first?.identifier ?? "obj")
                }
                
                // 步骤 C: 只将纯数据传回主线程进行轨迹关联
                Task { @MainActor in
                    self?.updateTracking(with: detections)
                }
            }
            try? handler.perform([request])
        }
    }

    private func updateTracking(with newObservations: [(CGRect, String)]) {
        let targetLabels = ["person", "car", "bus", "truck", "motorbike"]
        var totalRisk = 0.0
        
        let updated = newObservations.compactMap { (box, label) -> TrackedObject? in
            guard targetLabels.contains(label) else { return nil }
            let center = CGPoint(x: box.midX, y: box.midY)
            
            // 轨迹匹配
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

// UI 部分代码保持稳定
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
                Text("GLOBAL ADAS: \(Int(controller.alertLevel * 100))%")
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

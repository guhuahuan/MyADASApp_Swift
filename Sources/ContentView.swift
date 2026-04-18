import SwiftUI
import Vision
@preconcurrency import AVFoundation // 关键 1: 降低旧框架的检查严苛度
import CoreML

// 轨迹目标模型
struct TrackedObject: Identifiable {
    let id: UUID
    var label: String
    var currentBox: CGRect
    var velocity: CGPoint
}

@MainActor
class GlobalTrafficController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var alertLevel: Double = 0.0
    
    // 移除所有 nonisolated 修饰，回归标准 MainActor 隔离
    let captureSession = AVCaptureSession()
    private var internalModel: VNCoreMLModel?
    
    // 创建一个专门的后台队列用于处理图像，避免堵塞主线程
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
        // 指定在后台队列回调
        videoOutput.setSampleBufferDelegate(self, queue: videoDataQueue)
        
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        captureSession.commitConfiguration()
        
        // 关键 2: 既然在主线程启动受阻，我们就在主线程异步任务中启动
        Task {
            self.captureSession.startRunning()
        }
    }

    // 关键 3: 必须显式标记为 nonisolated 以符合协议要求
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 在后台线程同步等待获取模型（避开 Actor 隔离检查）
        Task { @MainActor in
            guard let model = self.internalModel else { return }
            
            // 为了性能，我们将 Request 放在后台处理
            let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
                guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
                let mapped = results.map { (obs: VNRecognizedObjectObservation) -> (CGRect, String) in
                    return (obs.boundingBox, obs.labels.first?.identifier ?? "obj")
                }
                
                // 再次回到主线程更新数据
                Task { @MainActor in
                    self?.updateTracking(with: mapped)
                }
            }
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
            try? handler.perform([request])
        }
    }

    private func updateTracking(with newObservations: [(CGRect, String)]) {
        let targetLabels = ["person", "car", "bus", "truck", "motorbike"]
        var totalRisk = 0.0
        
        let updated = newObservations.compactMap { (box, label) -> TrackedObject? in
            guard targetLabels.contains(label) else { return nil }
            let center = CGPoint(x: box.midX, y: box.midY)
            
            let match = self.trackedObjects.first(where: { 
                abs($0.currentBox.midX - box.midX) < 0.15 && abs($0.currentBox.midY - box.midY) < 0.15 
            })
            
            let vel = match != nil ? CGPoint(x: center.x - match!.currentBox.midX, y: center.y - match!.currentBox.midY) : .zero
            
            // 危险预判逻辑
            if box.midX > 0.3 && box.midX < 0.7 && box.width * box.height > 0.08 {
                totalRisk += 0.2
            }
            
            return TrackedObject(id: match?.id ?? UUID(), label: label, currentBox: box, velocity: vel)
        }
        
        self.trackedObjects = updated
        self.alertLevel = min(totalRisk, 1.0)
    }
}

// 预览容器保持一致
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
                Text("TRAFFIC SYSTEM: \(Int(controller.alertLevel * 100))%")
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

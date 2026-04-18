import SwiftUI
import Vision
import AVFoundation
import CoreML

// 1. 轨迹目标模型
struct TrackedObject: Identifiable {
    let id: UUID
    var label: String
    var currentBox: CGRect
    var velocity: CGPoint
}

// 2. 核心分析控制器
@MainActor
class GlobalTrafficController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var alertLevel: Double = 0.0
    
    let captureSession = AVCaptureSession()
    private var model: VNCoreMLModel?
    private let videoOutput = AVCaptureVideoDataOutput()
    
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
            self.model = try VNCoreMLModel(for: m)
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
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "traffic_queue"))
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    // 关键修复：简化异步处理逻辑
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let model = self.model else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            
            // 映射结果
            let mapped = results.map { obs -> (CGRect, String) in
                return (obs.boundingBox, obs.labels.first?.identifier ?? "obj")
            }
            
            Task { @MainActor in
                self?.updateTracking(with: mapped)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform([request])
    }

    private func updateTracking(with newObservations: [(CGRect, String)]) {
        let targetLabels = ["person", "car", "bus", "truck", "motorbike"]
        var totalRisk = 0.0
        
        let updated = newObservations.compactMap { (box, label) -> TrackedObject? in
            guard targetLabels.contains(label) else { return nil }
            
            // 轨迹关联逻辑
            let center = CGPoint(x: box.midX, y: box.midY)
            let match = self.trackedObjects.first(where: { 
                abs($0.currentBox.midX - box.midX) < 0.15 && abs($0.currentBox.midY - box.midY) < 0.15 
            })
            
            let vel = match != nil ? CGPoint(x: center.x - match!.currentBox.midX, y: center.y - match!.currentBox.midY) : .zero
            
            // 危险判定：目标在中心区域且面积较大
            if box.midX > 0.3 && box.midX < 0.7 && box.width * box.height > 0.1 {
                totalRisk += 0.2
            }
            
            return TrackedObject(id: match?.id ?? UUID(), label: label, currentBox: box, velocity: vel)
        }
        
        self.trackedObjects = updated
        self.alertLevel = min(totalRisk, 1.0)
    }
}

// 3. 画面渲染层
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
    func updateUIView(_ uiView: UIView, context: Context) {
        if let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            layer.frame = uiView.bounds
        }
    }
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
                    
                    let color = controller.alertLevel > 0.5 ? Color.red : Color.green
                    context.stroke(Path(rect), with: .color(color), lineWidth: 2)
                    
                    // 绘制预判虚线
                    var line = Path()
                    line.move(to: CGPoint(x: rect.midX, y: rect.midY))
                    line.addLine(to: CGPoint(x: rect.midX + obj.velocity.x * size.width * 5, y: rect.midY - obj.velocity.y * size.height * 5))
                    context.stroke(line, with: .color(color.opacity(0.5)), style: StrokeStyle(lineWidth: 1, dash: [5]))
                }
            }
            
            VStack {
                Text("GLOBAL PREDICTION: \(Int(controller.alertLevel * 100))%")
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

import SwiftUI
import Vision
import AVFoundation
import CoreML

// 1. 数据模型
struct Detection: Sendable {
    let box: CGRect
    let label: String
}

// 2. 核心控制器
@MainActor
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private var model: VNCoreMLModel?

    override init() {
        super.init()
        loadModel()
        setupCapture()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            // 匹配打包脚本编译出的文件名
            if let modelURL = Bundle.main.url(forResource: "yolov8s", withExtension: "mlmodelc") {
                let compiledModel = try MLModel(contentsOf: modelURL, configuration: config)
                self.model = try VNCoreMLModel(for: compiledModel)
                print("✅ YOLOv8 Loaded")
            }
        } catch {
            print("❌ Model Error: \(error)")
        }
    }

    func setupCapture() {
        captureSession.beginConfiguration()
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video_queue"))
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let request = VNCoreMLRequest(model: model ?? self.fallbackModel()) { [weak self] req, _ in
            let results = req.results as? [VNRecognizedObjectObservation] ?? []
            let boxes = results.map { Detection(box: $0.boundingBox, label: $0.labels.first?.identifier ?? "Obj") }
            Task { @MainActor [weak self] in self?.detections = boxes }
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform([request])
    }
    
    // 兜底：如果模型没加载好，返回一个空的 Request 以防崩溃
    private func fallbackModel() -> VNCoreMLModel {
        // 这里只是为了编译器通过，实际运行会优先加载 YOLO
        return try! VNCoreMLModel(for: VNDetectFaceRectanglesRequest().model)
    }
}

// 3. UI 视图
struct CameraView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.layer.bounds
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
    @StateObject var controller = ADASController()
    var body: some View {
        ZStack {
            CameraView(session: controller.captureSession).ignoresSafeArea()
            Canvas { context, size in
                for d in controller.detections {
                    let rect = CGRect(
                        x: d.box.minX * size.width,
                        y: (1 - d.box.maxY) * size.height,
                        width: d.box.width * size.width,
                        height: d.box.height * size.height
                    )
                    context.stroke(Path(rect), with: .color(.green), lineWidth: 2)
                }
            }
        }
    }
}

// 4. 程序入口（把这一段加进去，就不需要其他 .swift 文件了）
@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

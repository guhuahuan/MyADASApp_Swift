import SwiftUI
import Vision
import AVFoundation
import CoreML

struct Detection: Sendable {
    let box: CGRect
    let label: String
}

@MainActor
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    
    // 使用 nonisolated 让后台线程可以访问，不触发 Actor 报错
    nonisolated let captureSession: AVCaptureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    nonisolated private(set) var model: VNCoreMLModel?

    override init() {
        super.init()
        // 在初始化时加载，此时还在主线程环境
        loadModel()
        setupCapture()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
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
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    // 处理摄像头每一帧的输出
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 如果 YOLO 模型加载了就用 YOLO，否则保持空结果
        let request: VNImageBasedRequest
        if let visionModel = self.model {
            request = VNCoreMLRequest(model: visionModel) { [weak self] req, _ in
                let results = req.results as? [VNRecognizedObjectObservation] ?? []
                let boxes = results.map { Detection(box: $0.boundingBox, label: $0.labels.first?.identifier ?? "Obj") }
                Task { @MainActor [weak self] in
                    self?.detections = boxes
                }
            }
        } else {
            // 兜底方案：如果模型没加载，使用简单的矩形检测，避免报错
            request = VNDetectRectanglesRequest { [weak self] req, _ in
                let results = req.results as? [VNRectangleObservation] ?? []
                let boxes = results.map { Detection(box: $0.boundingBox, label: "Scanning...") }
                Task { @MainActor [weak self] in
                    self?.detections = boxes
                }
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform([request])
    }
}

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

@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

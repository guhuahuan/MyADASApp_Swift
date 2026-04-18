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
    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private var model: VNCoreMLModel?

    override init() {
        super.init()
        loadModel()
        setupCapture()
    }

    private func loadModel() {
        // 彻底手动加载：不使用自动生成的 yolov8s() 类
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            // 我们的打包脚本会将模型编译成 yolov8s.mlmodelc 文件夹
            if let modelURL = Bundle.main.url(forResource: "yolov8s", withExtension: "mlmodelc") {
                let compiledModel = try MLModel(contentsOf: modelURL, configuration: config)
                self.model = try VNCoreMLModel(for: compiledModel)
                print("✅ YOLOv8 模型手动加载成功")
            } else {
                print("❌ 没找到编译后的 mlmodelc 文件")
            }
        } catch {
            print("❌ 加载模型出错: \(error)")
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
        
        let request: VNImageBasedRequest
        if let model = self.model {
            request = VNCoreMLRequest(model: model) { [weak self] req, _ in
                let results = req.results as? [VNRecognizedObjectObservation] ?? []
                let boxes = results.map { Detection(box: $0.boundingBox, label: $0.labels.first?.identifier ?? "Obj") }
                Task { @MainActor [weak self] in self?.detections = boxes }
            }
        } else {
            request = VNDetectRectanglesRequest { [weak self] req, _ in
                let results = req.results as? [VNRectangleObservation] ?? []
                let boxes = results.map { Detection(box: $0.boundingBox, label: "Scanning...") }
                Task { @MainActor [weak self] in self?.detections = boxes }
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
            CameraView(session: controller.captureSession)
                .ignoresSafeArea()
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
            VStack {
                Text(controller.model != nil ? "YOLOv8 Active" : "Waiting for Model...")
                    .foregroundColor(.white).padding().background(Color.black.opacity(0.5)).cornerRadius(10).padding(.top, 50)
                Spacer()
            }
        }
    }
}

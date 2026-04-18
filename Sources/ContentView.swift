import SwiftUI
import Vision
@preconcurrency import AVFoundation
import CoreML

struct Detection: Sendable {
    let box: CGRect
    let label: String
}

@MainActor
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    
    // 使用 nonisolated(unsafe) 绕过 Swift 6 的 Sendable 检查
    nonisolated(unsafe) let captureSession: AVCaptureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    nonisolated(unsafe) private(set) var model: VNCoreMLModel?

    override init() {
        super.init()
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
        
        // 放在后台线程启动摄像头，避免卡顿
        let session = self.captureSession
        DispatchQueue.global(qos: .userInitiated).async {
            session.startRunning()
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
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

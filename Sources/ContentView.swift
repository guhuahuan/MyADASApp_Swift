import SwiftUI
import Vision
@preconcurrency import AVFoundation // 关键：处理非 Sendable 警告
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
    @Published var modelStatus: String = "正在初始化..."
    
    // 强制声明为 nonisolated，允许跨线程访问
    nonisolated let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    
    // 关键修复：使用 nonisolated(unsafe) 避开严格的并发检查
    // 因为模型加载后是只读的，这样做是安全的
    nonisolated(unsafe) private var visionModel: VNCoreMLModel?

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
                self.visionModel = try VNCoreMLModel(for: compiledModel)
                self.modelStatus = "YOLOv8 已激活"
            } else {
                self.modelStatus = "使用系统内置检测"
            }
        } catch {
            self.modelStatus = "模型加载失败"
        }
    }

    func setupCapture() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera_queue"))
        
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        captureSession.commitConfiguration()
        
        // 既然 session 是 nonisolated，我们可以安全地在后台启动
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    // 摄像头回调：标记为 nonisolated
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request: VNImageBasedRequest
        // 此时访问 visionModel 不再报错，因为标记了 nonisolated(unsafe)
        if let yolo = self.visionModel {
            request = VNCoreMLRequest(model: yolo) { [weak self] req, _ in
                self?.handleResults(req.results)
            }
        } else {
            request = VNDetectRectanglesRequest { [weak self] req, _ in
                self?.handleResults(req.results)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform([request])
    }
    
    nonisolated private func handleResults(_ results: [Any]?) {
        let observations = results as? [VNDetectedObjectObservation] ?? []
        let newDetections = observations.map { obs in
            Detection(
                box: obs.boundingBox,
                label: (obs as? VNRecognizedObjectObservation)?.labels.first?.identifier ?? "Target"
            )
        }
        
        Task { @MainActor in
            self.detections = newDetections
        }
    }
}

// 3. 画面预览
struct CameraPreviewHolder: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.frame = UIScreen.main.bounds
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            if let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
                layer.frame = uiView.bounds
            }
        }
    }
}

// 4. 主视图
struct ContentView: View {
    @StateObject var controller = ADASController()
    
    var body: some View {
        ZStack {
            CameraPreviewHolder(session: controller.captureSession)
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
                Text(controller.modelStatus)
                    .font(.caption.monospaced())
                    .padding(6)
                    .background(.black.opacity(0.6))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                    .padding(.top, 40)
                Spacer()
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

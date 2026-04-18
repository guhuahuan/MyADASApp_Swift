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
    @Published var modelStatus: String = "正在初始化..."
    
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
            // 尝试加载编译后的 YOLO 模型
            if let modelURL = Bundle.main.url(forResource: "yolov8s", withExtension: "mlmodelc") {
                let compiledModel = try MLModel(contentsOf: modelURL, configuration: config)
                self.model = try VNCoreMLModel(for: compiledModel)
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
        // 设置高质量预览
        captureSession.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera_queue"))
        
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        
        // 确保视频方向正确
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }
        
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request: VNImageBasedRequest
        if let yolomodel = self.model {
            request = VNCoreMLRequest(model: yolomodel) { [weak self] req, _ in
                self?.processResults(req.results)
            }
        } else {
            // 兜底：内置矩形检测
            request = VNDetectRectanglesRequest { [weak self] req, _ in
                self?.processResults(req.results)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform([request])
    }
    
    private func processResults(_ results: [Any]?) {
        let observations = results as? [VNDetectedObjectObservation] ?? []
        let newDetections = observations.map { obs in
            Detection(
                box: obs.boundingBox,
                label: (obs as? VNRecognizedObjectObservation)?.labels.first?.identifier ?? "目标"
            )
        }
        Task { @MainActor in
            self.detections = newDetections
        }
    }
}

// 3. 画面预览层（修复黑屏的关键）
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
        if let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            layer.frame = uiView.bounds
        }
    }
}

// 4. 主视图
struct ContentView: View {
    @StateObject var controller = ADASController()
    
    var body: some View {
        ZStack {
            // 背景摄像头画面
            CameraPreviewHolder(session: controller.captureSession)
                .ignoresSafeArea()
            
            // 覆盖检测框
            Canvas { context, size in
                for d in controller.detections {
                    // 坐标转换：Vision 是归一化坐标 (0-1)，需要转为屏幕像素
                    let rect = CGRect(
                        x: d.box.minX * size.width,
                        y: (1 - d.box.maxY) * size.height,
                        width: d.box.width * size.width,
                        height: d.box.height * size.height
                    )
                    context.stroke(Path(rect), with: .color(.green), lineWidth: 2)
                    context.draw(Text(d.label).foregroundColor(.green), at: CGPoint(x: rect.minX, y: rect.minY - 10))
                }
            }
            
            // 顶部状态栏
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

// 5. 入口
@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

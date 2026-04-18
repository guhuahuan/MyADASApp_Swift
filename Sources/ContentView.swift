import SwiftUI
import Vision
import AVFoundation
import CoreML

// 1. 定义检测结果结构
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
        do {
            // 注意：编译后模型后缀会变成 .mlmodelc
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            if let modelURL = Bundle.main.url(forResource: "yolov8s", withExtension: "mlmodelc") {
                let coreMLModel = try MLModel(contentsOf: modelURL, configuration: config)
                self.model = try VNCoreMLModel(for: coreMLModel)
                print("✅ YOLOv8 模型加载成功")
            } else {
                print("❌ 未找到模型文件")
            }
        } catch {
            print("❌ 模型初始化失败: \(error)")
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
        // 如果加载了 YOLO 就用 YOLO，否则用系统自带的矩形检测作为兜底
        if let model = self.model {
            request = VNCoreMLRequest(model: model) { [weak self] req, _ in
                let results = req.results as? [VNRecognizedObjectObservation] ?? []
                let boxes = results.map { Detection(box: $0.boundingBox, label: $0.labels.first?.identifier ?? "Target") }
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

// 2. 摄像头预览层 (修复黑屏关键)
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

// 3. 主界面
struct ContentView: View {
    @StateObject var controller = ADASController()
    
    var body: some View {
        ZStack {
            CameraView(session: controller.captureSession)
                .ignoresSafeArea()
            
            // 实时绘制检测框
            Canvas { context, size in
                for d in controller.detections {
                    let rect = CGRect(
                        x: d.box.minX * size.width,
                        y: (1 - d.box.maxY) * size.height,
                        width: d.box.width * size.width,
                        height: d.box.height * size.height
                    )
                    context.stroke(Path(rect), with: .color(.green), lineWidth: 2)
                    context.draw(Text(d.label).foregroundColor(.green).font(.caption), at: CGPoint(x: rect.minX, y: rect.minY - 10))
                }
            }
            
            // 状态指示器
            VStack {
                HStack {
                    Circle()
                        .fill(controller.model != nil ? Color.green : Color.red)
                        .frame(width: 10, height: 10)
                    Text(controller.model != nil ? "YOLOv8 Active" : "Vision Mode")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.white)
                }
                .padding(8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(20)
                .padding(.top, 50)
                Spacer()
            }
        }
    }
}

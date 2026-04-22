import SwiftUI
import AVFoundation
import Vision
import CoreML

@main
struct MiniDetectorApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - 极简引擎
class DetectionEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [VNRecognizedObjectObservation] = []
    let session = AVCaptureSession()
    private var requests = [VNRequest]()

    func setup() {
        // 1. 相机配置
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        session.addInput(input)
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video_queue"))
        session.addOutput(output)
        
        // 2. 加载 yolo26x 模型
        guard let modelURL = Bundle.main.url(forResource: "yolo26x", withExtension: "mlmodelc"),
              let visionModel = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else { return }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                DispatchQueue.main.async {
                    // 核心逻辑：只过滤人 (person) 和 车 (car)
                    self?.detections = results.filter { 
                        $0.labels.first?.identifier == "person" || $0.labels.first?.identifier == "car"
                    }
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop
        self.requests = [request]
        
        DispatchQueue.global().async { self.session.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(requests)
    }
}

// MARK: - 主视图
struct ContentView: View {
    @StateObject private var engine = DetectionEngine()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                CameraView(session: engine.session)
                
                // 绘制识别框
                ForEach(0..<engine.detections.count, id: \.self) { i in
                    let obs = engine.detections[i]
                    let rect = VNImageRectForNormalizedRect(obs.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // 翻转坐标系（Vision 的 Y 轴向上，SwiftUI 向下）
                    let correctedRect = CGRect(x: rect.minX, y: geo.size.height - rect.maxY, width: rect.width, height: rect.height)
                    
                    Rectangle()
                        .path(in: correctedRect)
                        .stroke(obs.labels.first?.identifier == "person" ? Color.blue : Color.green, lineWidth: 3)
                        .overlay(
                            Text(obs.labels.first?.identifier.uppercased() ?? "")
                                .font(.caption.bold())
                                .foregroundColor(.white)
                                .background(obs.labels.first?.identifier == "person" ? Color.blue : Color.green)
                                .position(x: correctedRect.minX + 25, y: correctedRect.minY - 10)
                        )
                }
            }
        }
        .ignoresSafeArea()
        .onAppear { engine.setup() }
    }
}

// MARK: - 相机预览层
struct CameraView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.frame = view.frame
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

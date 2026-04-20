import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation

@main
struct FSD_V4_App: App {
    var body: some Scene {
        WindowGroup {
            FSDMainView()
        }
    }
}

// MARK: - 主 UI 界面
struct FSDMainView: View {
    @StateObject private var engine = ADASLogicEngine()
    
    var body: some View {
        ZStack {
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // AI 检测框
            Canvas { context, size in
                for detection in engine.detections {
                    let rect = engine.convertRect(detection.boundingBox, to: size)
                    context.stroke(Path(rect), with: .color(.cyan), lineWidth: 2)
                    
                    var title = detection.label
                    if engine.currentSpeed > 0.5 {
                        title += " \(String(format: "%.1f", engine.currentSpeed * 3.6))km/h"
                    }
                    context.draw(Text(title).font(.caption).bold().foregroundColor(.cyan), 
                                 at: CGPoint(x: rect.minX, y: rect.minY - 10))
                }
            }
            
            // 仪表盘
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("FSD V4 ADAS").font(.title3).bold().foregroundColor(.white)
                        Text(engine.isModelLoaded ? "AI 运行中" : "AI 模型加载失败")
                            .font(.caption).foregroundColor(engine.isModelLoaded ? .green : .red)
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 40, weight: .bold)).foregroundColor(.yellow)
                        Text("KM/H").font(.caption).foregroundColor(.yellow)
                    }
                }
                .padding().background(LinearGradient(gradient: Gradient(colors: [.black.opacity(0.7), .clear]), startPoint: .top, endPoint: .bottom))
                
                Spacer()
                
                if let err = engine.errorMessage {
                    Text(err).font(.system(size: 10, design: .monospaced)).padding(8).background(Color.red).foregroundColor(.white).cornerRadius(5)
                }
            }
        }
        .onAppear { engine.checkPermissions() }
    }
}

// MARK: - 逻辑引擎
class ADASLogicEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: CLLocationSpeed = 0
    @Published var isModelLoaded = false
    @Published var errorMessage: String?
    
    let captureSession = AVCaptureSession()
    private let locationManager = CLLocationManager()
    private var requests = [VNRequest]()
    
    struct Detection {
        let label: String
        let boundingBox: CGRect
    }

    override init() {
        super.init()
        setupModel()
    }

    private func setupModel() {
        // 强化路径搜索逻辑
        let modelName = "yolov8l"
        let possibleURLs = [
            Bundle.main.url(forResource: modelName, withExtension: "mlmodelc"),
            Bundle.main.url(forResource: "model", withExtension: "mlmodelc"),
            Bundle.main.bundleURL.appendingPathComponent("\(modelName).mlmodelc"),
            Bundle.main.bundleURL.appendingPathComponent("model.mlmodelc")
        ]
        
        guard let modelURL = possibleURLs.compactMap({ $0 }).first else {
            self.errorMessage = "错误: 根目录未发现 \(modelName).mlmodelc"
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let visionModel = try VNCoreMLModel(for: model)
            
            let request = VNCoreMLRequest(model: visionModel) { (request, error) in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        self.detections = results.map { Detection(label: $0.labels.first?.identifier ?? "?", boundingBox: $0.boundingBox) }
                    }
                }
            }
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            self.isModelLoaded = true
        } catch {
            self.errorMessage = "模型加载失败: \(error.localizedDescription)"
        }
    }

    func checkPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        try? captureSession.addInput(AVCaptureDeviceInput(device: device))
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        captureSession.addOutput(output)
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = locations.last?.speed ?? 0
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.layer.bounds
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

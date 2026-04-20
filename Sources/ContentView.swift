import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation

// MARK: - 1. App 入口
@main
struct FSD_V4_App: App {
    var body: some Scene {
        WindowGroup {
            FSDMainView()
        }
    }
}

// MARK: - 2. 主 UI 界面
struct FSDMainView: View {
    @StateObject private var engine = ADASLogicEngine()
    
    var body: some View {
        ZStack {
            // 底层：实时相机预览
            CameraPreview(session: engine.captureSession)
                .edgesIgnoringSafeArea(.all)
            
            // 中层：AI 识别框渲染
            Canvas { context, size in
                for detection in engine.detections {
                    let rect = engine.convertRect(detection.boundingBox, to: size)
                    context.stroke(Path(rect), with: .color(.green), lineWidth: 2)
                    context.draw(Text(detection.label).foregroundColor(.green), at: CGPoint(x: rect.minX, y: rect.minY - 10))
                }
            }
            
            // 顶层：仪表盘数据
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("FSD V4 ADAS").font(.headline).foregroundColor(.white)
                        Text("Speed: \(Int(engine.currentSpeed * 3.6)) km/h").foregroundColor(.yellow)
                    }
                    Spacer()
                    if engine.isModelLoaded {
                        Image(systemName: "cpu.fill").foregroundColor(.green)
                    } else {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundColor(.red)
                    }
                }
                .padding()
                .background(Color.black.opacity(0.5))
                
                Spacer()
                
                if let error = engine.errorMessage {
                    Text(error).background(Color.red).padding()
                }
            }
        }
        .onAppear { engine.checkPermissions() }
    }
}

// MARK: - 3. 核心逻辑引擎 (AI + Camera + GPS)
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
        let confidence: Float
        let boundingBox: CGRect
    }

    override init() {
        super.init()
        setupModel()
    }

    // A. 安全加载模型：解决闪退的核心
    private func setupModel() {
        // 尝试多个可能的路径（GitHub Actions 编译后的 .mlmodelc 位置）
        let modelName = "yolov8l"
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") ??
                           Bundle.main.url(forResource: "yolov8l.mlpackage/Data/com.apple.CoreML/model", withExtension: "mlmodelc") else {
            self.errorMessage = "Missing Model: \(modelName)"
            return
        }

        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel) { (request, error) in
                self.processDetections(for: request)
            }
            objectRecognition.imageCropAndScaleOption = .scaleFill
            self.requests = [objectRecognition]
            self.isModelLoaded = true
        } catch {
            self.errorMessage = "Model Init Failed"
        }
    }

    // B. 相机初始化
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        default: self.errorMessage = "Camera Access Denied"
        }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
            
            // 异步启动 session，防止 Watchdog 杀掉 App
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        } catch {
            self.errorMessage = "Camera Setup Failed"
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform(self.requests)
    }

    private func processDetections(for request: VNRequest) {
        DispatchQueue.main.async {
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            self.detections = results.map { res in
                Detection(label: res.labels.first?.identifier ?? "Obj", 
                          confidence: res.confidence, 
                          boundingBox: res.boundingBox)
            }
        }
    }
    
    // 坐标转换
    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width,
                      y: (1 - rect.maxY) * size.height,
                      width: rect.width * size.width,
                      height: rect.height * size.height)
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = locations.last?.speed ?? 0
    }
}

// MARK: - 4. 预览组件
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

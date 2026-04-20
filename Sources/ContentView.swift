import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation
import CoreMotion

@main
struct FSD_V4_Final: App {
    var body: some Scene {
        WindowGroup {
            FSDMasterView()
        }
    }
}

// MARK: - 终极 Master 视图
struct FSDMasterView: View {
    @StateObject private var engine = FSDCoreEngine()
    
    var body: some View {
        ZStack {
            // 1. 底层：工业级相机预览
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 2. 中层：智能 AR 渲染层
            GeometryReader { geo in
                Canvas { context, size in
                    // --- 功能 A: 陀螺仪修正的 3D 引导路径 ---
                    // 即使手机装斜了，路径也会尽量保持与地面平行
                    drawStabilizedPath(context: context, size: size, attitude: engine.deviceAttitude)
                    
                    // --- 功能 B: 智能目标分类与 TTC 预警 ---
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazardLevel = engine.calculateHazard(rect: rect)
                        
                        let color: Color = hazardLevel > 0.8 ? .red : (hazardLevel > 0.4 ? .yellow : .green)
                        
                        // 绘制具有科技感的锁定角标
                        drawTargetCorners(context: context, rect: rect, color: color)
                        
                        // 标签信息
                        context.draw(Text("\(detection.label.uppercased())").font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. 顶层：数字化驾驶舱仪表面板
            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("FSD V4.0 MASTER").font(.system(.headline, design: .monospaced)).foregroundColor(.white)
                        HStack {
                            StatusTag(text: "AI", active: engine.isModelLoaded)
                            StatusTag(text: "NPU", active: true)
                            StatusTag(text: "GPS", active: engine.isGpsActive)
                        }
                        Text(engine.headingInfo).font(.system(size: 11, design: .monospaced)).foregroundColor(.white.opacity(0.8))
                    }
                    Spacer()
                    // 主时速表
                    VStack(alignment: .trailing) {
                        Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 55, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                        Text("KM/H").font(.caption).bold().foregroundColor(.yellow)
                    }
                }
                .padding().background(LinearGradient(colors: [.black.opacity(0.8), .clear], startPoint: .top, endPoint: .bottom))
                
                Spacer()
                
                // 底部：实时工程监测数据
                HStack {
                    VStack(alignment: .leading) {
                        Text("LATENCY: \(Int(engine.latency * 1000))ms").foregroundColor(.cyan)
                        Text("FPS: \(engine.fps)").foregroundColor(.green)
                    }
                    Spacer()
                    // 陀螺仪实时角度 (Pitch)
                    Text("PITCH: \(String(format: "%.1f°", engine.pitch * 180 / .pi))").foregroundColor(.white.opacity(0.6))
                }
                .font(.system(size: 10, design: .monospaced)).padding().background(Color.black.opacity(0.3))
                
                if let err = engine.errorMessage {
                    Text(err).font(.system(size: 9)).padding(5).background(Color.red).foregroundColor(.white).frame(maxWidth: .infinity)
                }
            }
        }
        .onAppear { engine.startAllSystems() }
    }
    
    // 绘制 3D 稳定路径
    func drawStabilizedPath(context: GraphicsContext, size: CGSize, attitude: CMAttitude?) {
        var path = Path()
        let rollOffset = CGFloat(attitude?.roll ?? 0) * size.width * 0.2
        
        path.move(to: CGPoint(x: size.width * 0.1 + rollOffset, y: size.height))
        path.addLine(to: CGPoint(x: size.width * 0.45, y: size.height * 0.6))
        path.addLine(to: CGPoint(x: size.width * 0.55, y: size.height * 0.6))
        path.addLine(to: CGPoint(x: size.width * 0.9 + rollOffset, y: size.height))
        
        context.fill(path, with: .linearGradient(Gradient(colors: [.blue.opacity(0.4), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.6)))
    }
    
    // 绘制锁定角标 (替代死板的方框)
    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color) {
        let len = 10.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + len)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + len, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - len, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + len))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - len)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - len, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + len, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - len))
        }, with: .color(color), lineWidth: 2)
    }
}

// MARK: - 核心计算引擎 (整合所有传感器)
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: CLLocationSpeed = 0
    @Published var headingInfo: String = "CALIBRATING..."
    @Published var isModelLoaded = false
    @Published var isGpsActive = false
    @Published var latency: TimeInterval = 0
    @Published var fps: Int = 0
    @Published var pitch: Double = 0
    @Published var deviceAttitude: CMAttitude?
    @Published var errorMessage: String?
    
    let captureSession = AVCaptureSession()
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private var requests = [VNRequest]()
    private var lastTime = Date()
    
    struct Detection { let label: String; let boundingBox: CGRect }

    func startAllSystems() {
        setupModel()
        checkPermissions()
        setupMotion()
    }

    private func setupModel() {
        let modelName = "yolov8l"
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") ?? 
                        Bundle.main.url(forResource: modelName, withExtension: "mlpackage") else {
            self.errorMessage = "BRAIN ERROR: Check weight.bin"
            return
        }

        do {
            let finalURL = url.pathExtension == "mlpackage" ? try MLModel.compileModel(at: url) : url
            let model = try MLModel(contentsOf: finalURL, configuration: MLModelConfiguration())
            let visionModel = try VNCoreMLModel(for: model)
            
            let request = VNCoreMLRequest(model: visionModel) { [weak self] req, _ in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        let duration = Date().timeIntervalSince(self?.lastTime ?? Date())
                        self?.latency = duration
                        self?.fps = Int(1.0 / duration)
                        self?.detections = results.map { Detection(label: $0.labels.first?.identifier ?? "?", boundingBox: $0.boundingBox) }
                    }
                }
            }
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            self.isModelLoaded = true
        } catch {
            self.errorMessage = "CORE ERROR: \(error.localizedDescription)"
        }
    }

    func setupMotion() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
            motionManager.startDeviceMotionUpdates(to: .main) { motion, _ in
                self.deviceAttitude = motion?.attitude
                self.pitch = motion?.attitude.pitch ?? 0
            }
        }
    }

    func checkPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        locationManager.startUpdatingHeading()
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
        lastTime = Date()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    func calculateHazard(rect: CGRect) -> Double {
        // 简单模拟 TTC (Time to Collision) 逻辑：宽度越大越危险
        return Double(rect.width)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = max(0, locations.last?.speed ?? 0)
        isGpsActive = true
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        let index = Int((newHeading.magneticHeading + 22.5) / 45.0) & 7
        headingInfo = "DIR: \(directions[index]) \(Int(newHeading.magneticHeading))°"
    }
}

struct StatusTag: View {
    let text: String; let active: Bool
    var body: some View {
        Text(text).font(.system(size: 8, weight: .black)).padding(3).background(active ? Color.blue : Color.red).foregroundColor(.white).cornerRadius(3)
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

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

struct FSDMasterView: View {
    @StateObject private var engine = FSDCoreEngine()
    
    var body: some View {
        ZStack {
            // 1. 底层：横屏相机流
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 2. 中层：AR 增强现实
            GeometryReader { geo in
                Canvas { context, size in
                    // 绘制 3D 轨迹线 (针对横屏视角优化)
                    drawHorizontal3DPath(context: context, size: size, attitude: engine.deviceAttitude)
                    
                    // 绘制 AI 目标
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        let color: Color = hazard > 0.75 ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: hazard > 0.75)
                        
                        let label = "\(detection.label.uppercased())"
                        context.draw(Text(label).font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. 顶层：横屏专用 HUD 布局
            VStack {
                // 顶部状态条
                HStack {
                    VStack(alignment: .leading) {
                        Text("FSD V4.0 HORIZONTAL MASTER").font(.system(size: 12, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                        Text(engine.headingInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white)
                    }
                    Spacer()
                    HStack(spacing: 15) {
                        StatusTag(text: "AI: \(engine.detections.count)", active: engine.isModelLoaded)
                        StatusTag(text: "NPU", active: true)
                        StatusTag(text: "FPS: \(engine.fps)", active: engine.fps > 20)
                    }
                }
                .padding(.horizontal, 40).padding(.top, 10)
                
                Spacer()
                
                // 底部核心数据区
                HStack(alignment: .bottom) {
                    // 左侧：系统参数
                    VStack(alignment: .leading) {
                        Text("LATENCY: \(Int(engine.latency * 1000))MS").foregroundColor(.green)
                        Text("PITCH: \(String(format: "%.1f°", engine.pitch * 180 / .pi))").foregroundColor(.white.opacity(0.7))
                    }
                    .font(.system(size: 10, design: .monospaced))
                    
                    Spacer()
                    
                    // 中间：碰撞警告 (仅危险时显示)
                    if engine.detections.contains(where: { engine.calculateHazard(rect: engine.convertRect($0.boundingBox, to: UIScreen.main.bounds.size)) > 0.75 }) {
                        Text("⚠️ COLLISION RISK")
                            .font(.system(size: 18, weight: .black))
                            .foregroundColor(.white).padding(8).background(Color.red).cornerRadius(5)
                            .padding(.bottom, 20)
                    }
                    
                    Spacer()
                    
                    // 右侧：大号数字时速
                    VStack(alignment: .trailing) {
                        Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 70, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                        Text("KM/H").font(.system(size: 14, weight: .bold)).foregroundColor(.yellow)
                    }
                }
                .padding(.horizontal, 40).padding(.bottom, 20)
            }
        }
        .onAppear { engine.startSystems() }
    }
    
    // 横屏 3D 路径 (透视点上移，更符合车载视角)
    func drawHorizontal3DPath(context: GraphicsContext, size: CGSize, attitude: CMAttitude?) {
        let roll = CGFloat(attitude?.roll ?? 0) * 200
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + roll, y: size.height * 0.45), 
                   control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), 
                   control2: CGPoint(x: size.width * 0.4, y: size.height * 0.6))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), 
                   control1: CGPoint(x: size.width * 0.6 + roll, y: size.height * 0.6), 
                   control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.blue.opacity(0.4), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.45)))
    }

    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l = isDanger ? 20.0 : 12.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: isDanger ? 4 : 2)
    }
}

// MARK: - 核心计算引擎 (Landscape 优化版)
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: CLLocationSpeed = 0
    @Published var headingInfo: String = "DIR: --"
    @Published var isModelLoaded = false
    @Published var isGpsActive = false
    @Published var latency: TimeInterval = 0
    @Published var fps: Int = 0
    @Published var pitch: Double = 0
    @Published var deviceAttitude: CMAttitude?
    
    private let focusLabels = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
    let captureSession = AVCaptureSession()
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private var requests = [VNRequest]()
    private var lastTime = Date()
    
    struct Detection { let label: String; let boundingBox: CGRect }

    func startSystems() {
        setupModel()
        setupPermissions()
        setupMotion()
    }

    private func setupModel() {
        let name = "yolov8l"
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { return }
        do {
            let model = try MLModel(contentsOf: url, configuration: MLModelConfiguration())
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        let diff = Date().timeIntervalSince(self?.lastTime ?? Date())
                        self?.latency = diff
                        self?.fps = Int(1.0 / diff)
                        self?.detections = results.compactMap { res in
                            let label = res.labels.first?.identifier ?? ""
                            return (self?.focusLabels.contains(label) == true) ? Detection(label: label, boundingBox: res.boundingBox) : nil
                        }
                    }
                }
            }
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            self.isModelLoaded = true
        } catch { print("Error: \(error)") }
    }

    private func setupPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        locationManager.startUpdatingHeading()
    }

    private func setupMotion() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1/30
            motionManager.startDeviceMotionUpdates(to: .main) { motion, _ in 
                self.deviceAttitude = motion?.attitude
                self.pitch = motion?.attitude.pitch ?? 0
            }
        }
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        try? captureSession.addInput(AVCaptureDeviceInput(device: device))
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        captureSession.addOutput(output)
        
        // 【关键修复】：锁定相机输出为横屏
        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .landscapeRight
        }
        
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lastTime = Date()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // 注意：横屏下 orientation 需设置为 .up，因为我们在 setupCamera 已翻转了 connection
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    // 【横屏映射算法】
    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        // 横屏下画面通常是 16:9
        let imageAspectRatio: CGFloat = 1920.0 / 1080.0 
        let viewAspectRatio: CGFloat = size.width / size.height
        
        var scale: CGFloat = 1.0
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0
        
        if viewAspectRatio > imageAspectRatio {
            scale = size.width
            offsetY = (size.height - size.width / imageAspectRatio) / 2
        } else {
            scale = size.height * imageAspectRatio
            offsetX = (size.width - size.height * imageAspectRatio) / 2
        }

        // Vision 的 Y轴反转
        return CGRect(
            x: rect.minX * scale + offsetX,
            y: (1.0 - rect.maxY) * (scale / imageAspectRatio) + offsetY,
            width: rect.width * scale,
            height: rect.height * (scale / imageAspectRatio)
        )
    }

    func calculateHazard(rect: CGRect) -> Double {
        return Double(rect.width / UIScreen.main.bounds.width * 2.8) 
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = max(0, locations.last?.speed ?? 0)
        isGpsActive = true
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        headingInfo = "HEADING: \(dirs[Int((newHeading.magneticHeading + 22.5) / 45.0) & 7]) \(Int(newHeading.magneticHeading))°"
    }
}

// 辅助组件保持不变...
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
        // 【关键修复】：预览层也需要锁定横屏
        layer.connection?.videoOrientation = .landscapeRight
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

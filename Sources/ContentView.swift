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
            // 1. 相机底图
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 2. AR 渲染层
            GeometryReader { geo in
                Canvas { context, size in
                    // --- 功能 A: 3D 动态轨迹引导 ---
                    drawHorizontal3DPath(context: context, size: size, attitude: engine.deviceAttitude)
                    
                    // --- 功能 B: 车道感知边界线 ---
                    drawLaneMarkers(context: context, size: size)
                    
                    // --- 功能 C: AI 目标锁定与测距 ---
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        let isDanger = hazard > 0.75
                        let color: Color = isDanger ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        // 绘制锁定角标
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 计算模拟距离 (基于框体比例反算)
                        let distance = Int(120 / (rect.width / size.width * 15))
                        let label = "\(detection.label.uppercased()) | \(distance)m"
                        
                        context.draw(Text(label).font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. HUD 交互层
            HUDOverlay(engine: engine)
        }
        .onAppear { engine.startSystems() }
    }
    
    // 渲染：3D 路径 (带陀螺仪补偿)
    func drawHorizontal3DPath(context: GraphicsContext, size: CGSize, attitude: CMAttitude?) {
        let roll = CGFloat(attitude?.roll ?? 0) * 200
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.15, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + roll, y: size.height * 0.45), 
                   control1: CGPoint(x: size.width * 0.25, y: size.height * 0.8), 
                   control2: CGPoint(x: size.width * 0.4, y: size.height * 0.6))
        p.addCurve(to: CGPoint(x: size.width * 0.85, y: size.height), 
                   control1: CGPoint(x: size.width * 0.6 + roll, y: size.height * 0.6), 
                   control2: CGPoint(x: size.width * 0.75, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.blue.opacity(0.4), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.45)))
    }
    
    // 渲染：侧边车道线
    func drawLaneMarkers(context: GraphicsContext, size: CGSize) {
        let leftLane = Path { p in
            p.move(to: CGPoint(x: 40, y: size.height * 0.95))
            p.addLine(to: CGPoint(x: size.width * 0.35, y: size.height * 0.5))
        }
        let rightLane = Path { p in
            p.move(to: CGPoint(x: size.width - 40, y: size.height * 0.95))
            p.addLine(to: CGPoint(x: size.width * 0.65, y: size.height * 0.5))
        }
        context.stroke(leftLane, with: .color(.white.opacity(0.2)), lineWidth: 1)
        context.stroke(rightLane, with: .color(.white.opacity(0.2)), lineWidth: 1)
    }

    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l = isDanger ? 22.0 : 14.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: isDanger ? 4 : 2)
    }
}

// MARK: - HUD 叠加组件
struct HUDOverlay: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        VStack {
            HStack {
                VStack(alignment: .leading) {
                    Text("FSD V4.0 MASTER ACTIVE").font(.system(size: 12, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                    Text(engine.headingInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white)
                }
                Spacer()
                HStack(spacing: 12) {
                    StatusTag(text: "AI: \(engine.detections.count)", active: engine.isModelLoaded)
                    StatusTag(text: "NPU", active: true)
                    StatusTag(text: "FPS: \(engine.fps)", active: engine.fps > 20)
                }
            }
            .padding(.horizontal, 30).padding(.top, 15)
            
            Spacer()
            
            if engine.detections.contains(where: { engine.calculateHazard(rect: engine.convertRect($0.boundingBox, to: UIScreen.main.bounds.size)) > 0.75 }) {
                Text("⚠️ COLLISION RISK ⚠️")
                    .font(.system(size: 20, weight: .black))
                    .foregroundColor(.white).padding().background(Color.red).cornerRadius(10)
                    .transition(.opacity).animation(.easeInOut, value: true)
                    .padding(.bottom, 20)
            }
            
            HStack(alignment: .bottom) {
                VStack(alignment: .leading) {
                    Text("LATENCY: \(Int(engine.latency * 1000))MS").foregroundColor(.green)
                    Text("PITCH: \(String(format: "%.1f°", engine.pitch * 180 / .pi))")
                }
                .font(.system(size: 10, design: .monospaced)).foregroundColor(.white.opacity(0.8))
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 75, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                    Text("KM/H").font(.system(size: 14, weight: .bold)).foregroundColor(.yellow)
                }
            }
            .padding(.horizontal, 30).padding(.bottom, 20)
        }
    }
}

// MARK: - 核心引擎
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: CLLocationSpeed = 0
    @Published var headingInfo: String = "DIR: --"
    @Published var isModelLoaded = false
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
        } catch { print("AI Load Error") }
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
        
        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .landscapeRight
        }
        
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lastTime = Date()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        // 横屏 16:9 精准对齐
        let imageAspectRatio: CGFloat = 1920.0 / 1080.0 
        let viewAspectRatio: CGFloat = size.width / size.height
        let scale = (viewAspectRatio > imageAspectRatio) ? size.width : size.height * imageAspectRatio
        let offsetX = (size.width - scale) / 2.0
        let offsetY = (size.height - scale / imageAspectRatio) / 2.0

        return CGRect(
            x: rect.minX * scale + offsetX,
            y: (1.0 - rect.maxY) * (scale / imageAspectRatio) + offsetY,
            width: rect.width * scale,
            height: rect.height * (scale / imageAspectRatio)
        )
    }

    func calculateHazard(rect: CGRect) -> Double {
        return Double(rect.width / UIScreen.main.bounds.width * 3.0) 
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = max(0, locations.last?.speed ?? 0)
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        headingInfo = "DIR: \(dirs[Int((newHeading.magneticHeading + 22.5) / 45.0) & 7]) \(Int(newHeading.magneticHeading))°"
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
        layer.connection?.videoOrientation = .landscapeRight
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

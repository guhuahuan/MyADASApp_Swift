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

// MARK: - 主交互视图
struct FSDMasterView: View {
    @StateObject private var engine = FSDCoreEngine()
    
    var body: some View {
        ZStack {
            // 1. 底层：强制旋转 90° 的全屏相机预览
            CameraPreview(session: engine.captureSession)
                .edgesIgnoringSafeArea(.all)
            
            // 2. 中层：AI 动态渲染与 AR 引导
            GeometryReader { geo in
                Canvas { context, size in
                    // 绘制 AR 3D 轨迹线 (基于实时陀螺仪数据)
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    // 绘制 AI 识别目标
                    for detection in engine.detections {
                        // 坐标从 Vision (0-1) 映射到屏幕像素
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        
                        // 计算碰撞危险等级
                        let hazard = engine.calculateHazard(rect: rect)
                        let isDanger = hazard > 0.7
                        let color: Color = isDanger ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        // 绘制锁定角标
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 距离估算算法 (基于景深与目标大小)
                        let dist = Int(140 / (rect.width / size.width * 11))
                        context.draw(Text("\(detection.label.uppercased()) \(dist)M")
                            .font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. 顶层：数字化 HUD 驾驶仪表
            HUDOverlay(engine: engine)
        }
        .onAppear { engine.startSystems() }
        .statusBar(hidden: true) // 隐藏状态栏增加沉浸感
    }

    // AR 路径绘制逻辑
    func drawARPath(context: GraphicsContext, size: CGSize, roll: Double) {
        let shift = CGFloat(roll) * 150.0
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + shift, y: size.height * 0.55), 
                   control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), 
                   control2: CGPoint(x: size.width * 0.4, y: size.height * 0.65))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), 
                   control1: CGPoint(x: size.width * 0.6 + shift, y: size.height * 0.65), 
                   control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.blue.opacity(0.35), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.55)))
    }

    // 工业级锁定框绘制
    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l = isDanger ? 18.0 : 12.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: isDanger ? 3 : 2)
    }
}

// MARK: - HUD 覆盖视图
struct HUDOverlay: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        VStack {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("FSD MASTER V4.0").font(.system(size: 14, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                    Text(engine.headingInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white)
                    HStack(spacing: 8) {
                        StatusTag(text: "YOLOv8l", active: engine.isModelLoaded)
                        StatusTag(text: "FPS: \(engine.fps)", active: engine.fps > 15)
                    }
                }
                Spacer()
                // 时速仪表
                VStack(alignment: .trailing) {
                    Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 75, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                    Text("KM/H").font(.system(size: 14, weight: .bold)).foregroundColor(.yellow).offset(y: -10)
                }
            }
            .padding(.horizontal, 40).padding(.top, 25)
            
            Spacer()
            
            // 底部数据流
            HStack {
                Text("LATENCY: \(Int(engine.latency * 1000))ms").foregroundColor(.cyan)
                Spacer()
                Text("COORD_MAPPING: FIXED_90").foregroundColor(.gray)
                Spacer()
                Text("G-FORCE: \(String(format: "%.2f", engine.roll))").foregroundColor(.white)
            }
            .font(.system(size: 10, design: .monospaced))
            .padding(.horizontal, 20).padding(.vertical, 8)
            .background(Color.black.opacity(0.5))
        }
    }
}

// MARK: - 核心处理机
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: Double = 0
    @Published var headingInfo: String = "HEADING: --"
    @Published var isModelLoaded = false
    @Published var fps: Int = 0
    @Published var latency: TimeInterval = 0
    @Published var roll: Double = 0
    
    // 驾驶聚焦识别标签
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
        // 查找编译后的模型包
        guard let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc") else { 
            print("❌ Model not found in bundle")
            return 
        }
        do {
            let model = try MLModel(contentsOf: url, configuration: MLModelConfiguration())
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        let diff = Date().timeIntervalSince(self?.lastTime ?? Date())
                        self?.latency = diff
                        self?.fps = Int(1.0 / (diff > 0 ? diff : 0.033))
                        self?.detections = results.compactMap { res in
                            let label = res.labels.first?.identifier ?? ""
                            // 只显示过滤后的核心驾驶目标
                            return (self?.focusLabels.contains(label) == true) ? Detection(label: label, boundingBox: res.boundingBox) : nil
                        }
                    }
                }
            }
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            self.isModelLoaded = true
        } catch { print("AI Initialization Error: \(error)") }
    }

    private func setupPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        locationManager.startUpdatingHeading()
    }

    private func setupMotion() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1/30
            motionManager.startDeviceMotionUpdates(to: .main) { motion, _ in
                self.roll = motion?.attitude.roll ?? 0
            }
        }
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        try? captureSession.addInput(AVCaptureDeviceInput(device: device))
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        captureSession.addOutput(output)
        
        // --- 全功能修正：物理旋转数据流 ---
        if let conn = output.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) {
                conn.videoRotationAngle = 90 // 对应横屏视角
            }
        }
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput b: CMSampleBuffer, from conn: AVCaptureConnection) {
        lastTime = Date()
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        // 既然数据流已经旋转 90 度，Vision 这里的 orientation 必须传 .up
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    // 核心转换逻辑：将识别到的坐标点转换为 SwiftUI 屏幕像素
    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        // Vision 坐标 (0,0) 在左下角 -> 转换到 SwiftUI 的左上角
        return CGRect(
            x: rect.minX * size.width,
            y: (1.0 - rect.maxY) * size.height,
            width: rect.width * size.width,
            height: rect.height * size.height
        )
    }

    func calculateHazard(rect: CGRect) -> Double {
        // 画面下半部分且面积较大的物体判定为危险
        return rect.minY > UIScreen.main.bounds.height * 0.5 ? Double(rect.width / UIScreen.main.bounds.width * 2.5) : 0
    }

    // GPS & 航向角回调
    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    func locationManager(_ m: CLLocationManager, didUpdateHeading h: CLHeading) {
        let dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        headingInfo = "航向: \(dirs[Int((h.magneticHeading + 22.5) / 45.0) & 7]) \(Int(h.magneticHeading))°"
    }
}

// MARK: - 相机预览层 (支持横屏旋转)
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        // 强制预览层对齐横屏旋转角度
        if let conn = layer.connection, conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
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

struct StatusTag: View {
    let text: String; let active: Bool
    var body: some View {
        Text(text).font(.system(size: 8, weight: .black)).padding(4).background(active ? Color.blue : Color.red).foregroundColor(.white).cornerRadius(4)
    }
}

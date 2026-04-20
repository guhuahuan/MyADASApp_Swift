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
            // 1. 底层：工业相机实时流
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 2. 中层：AR 增强现实叠加层 (精准锁定)
            GeometryReader { geo in
                Canvas { context, size in
                    // --- 功能 A: 动态 3D 路径 (带陀螺仪补偿) ---
                    draw3DPath(context: context, size: size, attitude: engine.deviceAttitude)
                    
                    // --- 功能 B: AI 目标精准锁定 (修复偏移) ---
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        let color: Color = hazard > 0.8 ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        // 绘制锁定角标
                        drawTargetCorners(context: context, rect: rect, color: color)
                        
                        // 标签与模拟测距 (基于像素占比估算)
                        let distance = Int(100 / (rect.width / size.width * 10))
                        let label = "\(detection.label.uppercased()) | \(distance)m"
                        context.draw(Text(label).font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. 顶层：数字化行车仪表盘 (仪表 UI)
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
                    VStack(alignment: .trailing) {
                        Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 55, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                        Text("KM/H").font(.caption).bold().foregroundColor(.yellow)
                    }
                }
                .padding().background(LinearGradient(colors: [.black.opacity(0.8), .clear], startPoint: .top, endPoint: .bottom))
                
                Spacer()
                
                // 底部工程数据区
                HStack {
                    VStack(alignment: .leading) {
                        Text("LATENCY: \(Int(engine.latency * 1000))ms").foregroundColor(.cyan)
                        Text("FPS: \(engine.fps)").foregroundColor(.green)
                    }
                    Spacer()
                    Text("PITCH: \(String(format: "%.1f°", engine.pitch * 180 / .pi))").foregroundColor(.white.opacity(0.6))
                }
                .font(.system(size: 10, design: .monospaced)).padding().background(Color.black.opacity(0.3))
                
                if let err = engine.errorMessage {
                    Text(err).font(.system(size: 9)).padding(5).background(Color.red).foregroundColor(.white).frame(maxWidth: .infinity)
                }
            }
        }
        .onAppear { engine.startSystems() }
    }
    
    // 渲染 3D 引导路径
    func draw3DPath(context: GraphicsContext, size: CGSize, attitude: CMAttitude?) {
        var p = Path()
        let rollOffset = CGFloat(attitude?.roll ?? 0) * 120 // 补偿车身倾斜
        p.move(to: CGPoint(x: size.width * 0.1 + rollOffset, y: size.height))
        p.addLine(to: CGPoint(x: size.width * 0.45, y: size.height * 0.6))
        p.addLine(to: CGPoint(x: size.width * 0.55, y: size.height * 0.6))
        p.addLine(to: CGPoint(x: size.width * 0.9 + rollOffset, y: size.height))
        context.fill(p, with: .linearGradient(Gradient(colors: [.blue.opacity(0.4), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.6)))
    }

    // 绘制科技感锁定角标
    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color) {
        let l = 12.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: 2.5)
    }
}

// MARK: - 核心计算与 AI 引擎
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
    @Published var errorMessage: String?
    
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
        let modelName = "yolov8l"
        // 三级路径搜索，防止加载失败
        let paths = [
            Bundle.main.url(forResource: modelName, withExtension: "mlmodelc"),
            Bundle.main.url(forResource: modelName, withExtension: "mlpackage"),
            Bundle.main.bundleURL.appendingPathComponent("\(modelName).mlmodelc")
        ].compactMap { $0 }
        
        guard let url = paths.first else {
            self.errorMessage = "ERROR: \(modelName).mlmodelc not found"
            return
        }

        do {
            let finalURL = url.pathExtension == "mlpackage" ? try MLModel.compileModel(at: url) : url
            let config = MLModelConfiguration()
            config.computeUnits = .all // 强制开启 NPU 加速
            let model = try MLModel(contentsOf: finalURL, configuration: config)
            let vnModel = try VNCoreMLModel(for: model)
            
            let request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        let diff = Date().timeIntervalSince(self?.lastTime ?? Date())
                        self?.latency = diff
                        self?.fps = Int(1.0 / diff)
                        self?.detections = results.map { Detection(label: $0.labels.first?.identifier ?? "?", boundingBox: $0.boundingBox) }
                    }
                }
            }
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            self.isModelLoaded = true
        } catch {
            self.errorMessage = "CORE ML ERROR: \(error.localizedDescription)"
        }
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

        // 【关键修复 1】: 强制视频流输出方向与 UI 一致，防止识别框 90 度偏转
        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lastTime = Date()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    // 【关键修复 2】: 精准坐标转换算法 (补齐 AspectFill 造成的裁切偏移)
    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        // iPhone 相机标准竖屏比例 9:16
        let imageSize = CGSize(width: 1080, height: 1920)
        let scale = max(size.width / imageSize.width, size.height / imageSize.height)
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        let offsetX = (size.width - scaledWidth) / 2.0
        let offsetY = (size.height - scaledHeight) / 2.0

        // Vision 坐标系 Y轴反转
        let invertedY = 1.0 - rect.maxY

        return CGRect(
            x: rect.minX * scaledWidth + offsetX,
            y: invertedY * scaledHeight + offsetY,
            width: rect.width * scaledWidth,
            height: rect.height * scaledHeight
        )
    }

    func calculateHazard(rect: CGRect) -> Double {
        return Double(rect.width) // 简单的碰撞紧迫度算法
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = max(0, locations.last?.speed ?? 0)
        isGpsActive = true
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        let i = Int((newHeading.magneticHeading + 22.5) / 45.0) & 7
        headingInfo = "DIR: \(dirs[i]) \(Int(newHeading.magneticHeading))°"
    }
}

// MARK: - 状态标签组件
struct StatusTag: View {
    let text: String; let active: Bool
    var body: some View {
        Text(text).font(.system(size: 8, weight: .black)).padding(3).background(active ? Color.blue : Color.red).foregroundColor(.white).cornerRadius(3)
    }
}

// MARK: - 相机预览渲染引擎
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

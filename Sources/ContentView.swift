import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation
import CoreMotion

// MARK: - 程序入口
@main
struct FSD_Ultimate_App: App {
    var body: some Scene {
        WindowGroup {
            FSDUltimateView()
        }
    }
}

// MARK: - 数据模型
struct TrackedObject: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    var rect: CGRect // 屏幕物理像素坐标
}

// MARK: - 主视图层
struct FSDUltimateView: View {
    @StateObject private var engine = FSDCoreEngine()
    @State private var isSettingsOpen = false
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 实时相机预览
            CameraPreviewLayer(session: engine.captureSession)
                .ignoresSafeArea()
            
            // 2. AR 渲染看板 (不精简：含轨迹线、动态框、距离标签)
            GeometryReader { geo in
                Canvas { context, size in
                    // 绘制随陀螺仪摆动的轨迹线
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    for obj in engine.trackedObjects {
                        let hazard = engine.calculateHazard(rect: obj.rect, size: size)
                        let isDanger = hazard > engine.hazardThreshold
                        let color: Color = isDanger ? .red : (hazard > engine.hazardThreshold * 0.5 ? .yellow : .green)
                        
                        // 绘制工业识别框 (带四个角点加强)
                        drawDetectionBox(context: context, rect: obj.rect, color: color, isDanger: isDanger)
                        
                        // 物理测距显示 (基于变焦补偿)
                        let distance = (engine.distanceK * engine.zoomFactor) / (obj.rect.width / size.width + 0.0001)
                        context.draw(Text("\(obj.label.uppercased()) \(Int(distance))M").font(.system(size: 14, weight: .black)).foregroundColor(color),
                                     at: CGPoint(x: obj.rect.minX, y: obj.rect.minY - 22))
                    }
                }
            }
            
            // 3. HUD 仪表与交互控制
            VStack {
                // 顶部状态栏：含时速、系统状态
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("FSD V10.0 PRODUCTION").font(.system(size: 12, weight: .black)).foregroundColor(.cyan)
                        StatusIndicator(isReady: engine.isModelReady)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: -8) {
                        Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 60, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                        Text("KM/H").font(.system(size: 12, weight: .bold)).foregroundColor(.yellow)
                    }
                }.padding(.top, 50).padding(.horizontal, 25)
                
                Spacer()
                
                // 性能看板
                PerformanceMetrics(fps: engine.fps, latency: engine.inferenceTime)
                
                if isSettingsOpen {
                    TuningDashboard(engine: engine, isOpen: $isSettingsOpen)
                        .transition(.move(edge: .bottom))
                } else {
                    Button(action: { withAnimation { isSettingsOpen = true } }) {
                        Image(systemName: "slider.horizontal.3").padding().background(Color.cyan).foregroundColor(.black).clipShape(Circle()).shadow(radius: 10)
                    }.padding(.bottom, 30)
                }
            }
        }
        .onAppear { engine.startup() }
    }
    
    // AR 绘图辅助函数
    func drawARPath(context: GraphicsContext, size: CGSize, roll: Double) {
        let drift = CGFloat(roll) * 160.0
        var p = Path(); p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + drift, y: size.height * 0.55), control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), control2: CGPoint(x: size.width * 0.45, y: size.height * 0.65))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), control1: CGPoint(x: size.width * 0.55 + drift, y: size.height * 0.65), control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.cyan.opacity(0.35), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.55)))
    }
    
    func drawDetectionBox(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l: CGFloat = 16; let w: CGFloat = isDanger ? 4 : 2
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: w)
    }
}

// MARK: - 核心感知引擎 (全量功能封装)
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var currentSpeed: Double = 0
    @Published var roll: Double = 0
    @Published var fps: Int = 0
    @Published var inferenceTime: Double = 0
    @Published var isModelReady = false
    
    // 工业级参数
    @Published var zoomFactor: Double = 1.8 { didSet { updateZoom() } }
    @Published var distanceK: Double = 7.0
    @Published var hazardThreshold: Double = 0.5
    
    let captureSession = AVCaptureSession()
    private var videoInput: AVCaptureDeviceInput?
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let synthesizer = AVSpeechSynthesizer()
    private var requests = [VNRequest]()
    
    // 追踪平滑与滞后补偿
    private var lastRects: [String: CGRect] = [:]
    private let alpha: CGFloat = 0.45 
    private var lastFrameTime = Date()
    private var lastSpeechTime = Date()

    func startup() {
        configureCamera(); configureModel(); configureSensors()
    }

    private func configureCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input); self.videoInput = input }
            try device.lockForConfiguration()
            device.videoZoomFactor = CGFloat(zoomFactor)
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.setExposureTargetBias(-0.5, completionHandler: nil) // 应对强光
            }
            device.unlockForConfiguration()
        } catch { print(error) }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue", qos: .userInteractive))
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        captureSession.commitConfiguration()
        DispatchQueue.global().async { self.captureSession.startRunning() }
    }

    private func updateZoom() {
        guard let device = videoInput?.device else { return }
        try? device.lockForConfiguration()
        device.videoZoomFactor = CGFloat(max(1.0, min(zoomFactor, 5.0)))
        device.unlockForConfiguration()
    }

    private func configureModel() {
        guard let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: url, configuration: MLModelConfiguration())) else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            let start = Date()
            if let results = req.results as? [VNRecognizedObjectObservation] {
                self?.handleDetections(results)
                DispatchQueue.main.async { self?.inferenceTime = Date().timeIntervalSince(start) * 1000 }
            }
        }
        request.imageCropAndScaleOption = .centerCrop
        self.requests = [request]
        DispatchQueue.main.async { self.isModelReady = true }
    }

    private func handleDetections(_ observations: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async {
            let screenSize = UIScreen.main.bounds.size
            let videoSize = CGSize(width: 1920, height: 1080)
            
            self.trackedObjects = observations.compactMap { obs in
                let label = obs.labels.first?.identifier ?? ""
                let targets = ["car", "person", "motorcycle", "bus", "truck", "bicycle"]
                guard targets.contains(label), obs.confidence > 0.35 else { return nil }
                
                // 核心：高精度 AspectFill 坐标对齐
                let rect = obs.boundingBox
                let x = rect.minX
                let y = 1.0 - rect.maxY
                
                let screenAspect = screenSize.width / screenSize.height
                let videoAspect = videoSize.height / videoSize.width
                
                var correctedRect = CGRect(x: x, y: y, width: rect.width, height: rect.height)
                if screenAspect > videoAspect {
                    let factor = videoAspect / screenAspect
                    correctedRect.origin.y = (y - (1 - factor) / 2) / factor
                    correctedRect.size.height = rect.height / factor
                } else {
                    let factor = screenAspect / videoAspect
                    correctedRect.origin.x = (x - (1 - factor) / 2) / factor
                    correctedRect.size.width = rect.width / factor
                }
                
                let physicalRect = CGRect(
                    x: correctedRect.minX * screenSize.width,
                    y: correctedRect.minY * screenSize.height,
                    width: correctedRect.size.width * screenSize.width,
                    height: correctedRect.size.height * screenSize.height
                )
                
                // EMA 平滑处理
                let smoothed = self.applyEMA(for: label, new: physicalRect)
                return TrackedObject(label: label, confidence: obs.confidence, rect: smoothed)
            }
            self.updateFPS()
            self.runSafetyLogic()
        }
    }
    
    private func applyEMA(for key: String, new: CGRect) -> CGRect {
        guard let old = lastRects[key] else { lastRects[key] = new; return new }
        let smoothed = CGRect(
            x: old.minX * (1-alpha) + new.minX * alpha,
            y: old.minY * (1-alpha) + new.minY * alpha,
            width: old.width * (1-alpha) + new.width * alpha,
            height: old.height * (1-alpha) + new.height * alpha
        )
        lastRects[key] = smoothed
        return smoothed
    }

    func calculateHazard(rect: CGRect, size: CGSize) -> Double {
        let area = (rect.width * rect.height) / (size.width * size.height)
        let centerBias = 1.0 - abs((rect.midX / size.width) - 0.5) * 2.0
        return rect.midY > size.height * 0.45 ? Double(area * 25.0 * centerBias) : 0
    }

    private func runSafetyLogic() {
        let isDanger = trackedObjects.contains { calculateHazard(rect: $0.rect, size: UIScreen.main.bounds.size) > hazardThreshold }
        if isDanger && Date().timeIntervalSince(lastSpeechTime) > 3.0 {
            let u = AVSpeechUtterance(string: "注意前方"); u.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            synthesizer.speak(u); lastSpeechTime = Date()
        }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    private func updateFPS() { fps = Int(1.0 / Date().timeIntervalSince(lastFrameTime)); lastFrameTime = Date() }
    private func configureSensors() {
        locationManager.delegate = self; locationManager.requestWhenInUseAuthorization(); locationManager.startUpdatingLocation()
        if motionManager.isDeviceMotionAvailable {
            motionManager.startDeviceMotionUpdates(to: .main) { m, _ in self.roll = m?.attitude.roll ?? 0 }
        }
    }
}

// MARK: - 辅助组件 (全部保留)
struct StatusIndicator: View {
    let isReady: Bool
    var body: some View {
        HStack { Circle().fill(isReady ? Color.green : Color.red).frame(width: 8, height: 8)
            Text(isReady ? "SYSTEM ACTIVE" : "INIT...").font(.system(size: 8, weight: .bold)).foregroundColor(.white.opacity(0.6))
        }
    }
}

struct PerformanceMetrics: View {
    let fps: Int; let latency: Double
    var body: some View {
        HStack(spacing: 15) {
            Label("\(fps) FPS", systemImage: "bolt.fill")
            Label("\(Int(latency))ms", systemImage: "cpu")
        }.font(.system(size: 9, weight: .bold, design: .monospaced)).padding(8).background(Color.black.opacity(0.6)).foregroundColor(.green).cornerRadius(6).padding(.bottom, 10)
    }
}

struct TuningDashboard: View {
    @ObservedObject var engine: FSDCoreEngine; @Binding var isOpen: Bool
    var body: some View {
        VStack(spacing: 20) {
            Text("工业级感知控制台").font(.headline).foregroundColor(.white)
            VStack {
                TuningRow(name: "光学变焦 (Zoom)", val: $engine.zoomFactor, range: 1.0...4.5)
                TuningRow(name: "测距校准 (K-Factor)", val: $engine.distanceK, range: 3.0...15.0)
                TuningRow(name: "避障灵敏 (Sensitivity)", val: $engine.hazardThreshold, range: 0.1...1.5)
            }
            Button("保存并关闭") { withAnimation { isOpen = false } }
                .padding().frame(maxWidth: .infinity).background(Color.cyan).foregroundColor(.black).cornerRadius(12).font(.bold(.body)())
        }
        .padding(25).background(BlurView(style: .systemUltraThinMaterialDark)).cornerRadius(25, corners: [.topLeft, .topRight]).ignoresSafeArea()
    }
}

struct TuningRow: View {
    let name: String; @Binding var val: Double; let range: ClosedRange<Double>
    var body: some View {
        VStack {
            HStack { Text(name).font(.caption); Spacer(); Text(String(format: "%.1f", val)).monospacedDigit() }.foregroundColor(.white.opacity(0.8))
            Slider(value: $val, in: range).accentColor(.cyan)
        }
    }
}

struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let v = UIView(frame: UIScreen.main.bounds); let l = AVCaptureVideoPreviewLayer(session: session)
        l.videoGravity = .resizeAspectFill; l.frame = v.layer.bounds; v.layer.addSublayer(l); return v
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct BlurView: UIViewRepresentable {
    var style: UIBlurEffect.Style
    func makeUIView(context: Context) -> UIVisualEffectView { UIVisualEffectView(effect: UIBlurEffect(style: style)) }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape( RoundedCorner(radius: radius, corners: corners) )
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity; var corners: UIRectCorner = .allCorners
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(roundedRect: rect, byRoundingCorners: corners, cornerRadii: CGSize(width: radius, height: radius))
        return Path(path.cgPath)
    }
}

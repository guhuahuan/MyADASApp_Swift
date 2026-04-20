import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation
import CoreMotion

@main
struct FSD_Final_App: App {
    var body: some Scene {
        WindowGroup {
            FSDMainView()
        }
    }
}

// MARK: - 顶层交互视图
struct FSDMainView: View {
    @StateObject private var engine = FSDCoreEngine()
    @State private var isPanelOpen = false
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 物理光学层
            CameraPreviewView(session: engine.captureSession)
                .ignoresSafeArea()
            
            // 2. 增强现实感知层 (AR)
            GeometryReader { geo in
                Canvas { context, size in
                    // 绘制动态辅助线
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect, size: size)
                        let isDanger = hazard > engine.hazardThreshold
                        let color: Color = isDanger ? .red : (hazard > engine.hazardThreshold * 0.6 ? .yellow : .green)
                        
                        // 绘制工业识别框
                        drawBox(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 测距算法：[K * Zoom] / 宽度比例
                        let dist = (engine.distanceK * engine.zoomFactor) / (rect.width / size.width + 0.0001)
                        let labelText = "\(detection.label.uppercased()) \(Int(dist))M"
                        
                        context.draw(Text(labelText).font(.system(size: 14, weight: .black)).foregroundColor(color),
                                     at: CGPoint(x: rect.minX, y: rect.minY - 20))
                    }
                }
            }
            
            // 3. 工业 HUD 信息层
            VStack {
                TopHUD(engine: engine)
                Spacer()
                PerformanceTag(engine: engine)
                
                if isPanelOpen {
                    ControlPanel(engine: engine, isOpen: $isPanelOpen)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                } else {
                    SettingsButton(isOpen: $isPanelOpen)
                }
            }
        }
        .onAppear { engine.startup() }
    }

    func drawARPath(context: GraphicsContext, size: CGSize, roll: Double) {
        let drift = CGFloat(roll) * 160.0
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + drift, y: size.height * 0.55),
                   control1: CGPoint(x: size.width * 0.2, y: size.height * 0.85),
                   control2: CGPoint(x: size.width * 0.45, y: size.height * 0.65))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height),
                   control1: CGPoint(x: size.width * 0.55 + drift, y: size.height * 0.65),
                   control2: CGPoint(x: size.width * 0.8, y: size.height * 0.85))
        context.fill(p, with: .linearGradient(Gradient(colors: [.cyan.opacity(0.3), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.55)))
    }

    func drawBox(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let b = isDanger ? 3.0 : 1.5
        context.stroke(Path(roundedRect: rect, cornerRadius: 4), with: .color(color), lineWidth: b)
        // 绘制四角加重 (AOI 风格)
        let l: CGFloat = 15
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: b * 2)
    }
}

// MARK: - 核心感知算法引擎
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [FSDObject] = []
    @Published var currentSpeed: Double = 0
    @Published var roll: Double = 0
    @Published var fps: Int = 0
    @Published var inferenceMs: Double = 0
    
    // 工业调节位 (持久化建议)
    @Published var zoomFactor: Double = 1.8 { didSet { updateHardwareZoom() } }
    @Published var distanceK: Double = 6.8
    @Published var hazardThreshold: Double = 0.5
    @Published var isSystemReady = false
    
    let captureSession = AVCaptureSession()
    private var videoInput: AVCaptureDeviceInput?
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let speech = AVSpeechSynthesizer()
    private var requests = [VNRequest]()
    private var lastFrameDate = Date()
    private var lastAlertDate = Date()

    struct FSDObject { let label: String; let confidence: Float; let boundingBox: CGRect }

    func startup() {
        initModel(); checkPermissions(); initMotion()
    }

    private func checkPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.initCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func initCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input); self.videoInput = input }
            
            try device.lockForConfiguration()
            device.videoZoomFactor = CGFloat(zoomFactor)
            // 自动曝光偏置：针对南宁强光环境稍微减低曝光值，防止 AI 识别受限
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.setExposureTargetBias(-0.5, completionHandler: nil)
            }
            device.unlockForConfiguration()
        } catch { print("Camera Err") }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision.queue", qos: .userInteractive))
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    private func updateHardwareZoom() {
        guard let device = videoInput?.device else { return }
        try? device.lockForConfiguration()
        device.videoZoomFactor = CGFloat(min(max(zoomFactor, 1.0), 4.0))
        device.unlockForConfiguration()
    }

    private func initModel() {
        guard let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
              let coreMLModel = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: MLModelConfiguration())) else { return }
        
        let request = VNCoreMLRequest(model: coreMLModel) { [weak self] req, _ in
            let start = Date()
            if let results = req.results as? [VNRecognizedObjectObservation] {
                DispatchQueue.main.async {
                    self?.detections = results.compactMap { res in
                        let label = res.labels.first?.identifier ?? ""
                        // 重点：增加南宁“电驴”识别策略 (motorcycle/bicycle)
                        let whitelist = ["car", "person", "truck", "bus", "motorcycle", "bicycle"]
                        if whitelist.contains(label) && res.confidence > 0.32 {
                            return FSDObject(label: label, confidence: res.confidence, boundingBox: res.boundingBox)
                        }
                        return nil
                    }
                    self?.inferenceMs = Date().timeIntervalSince(start) * 1000
                    self?.updateFPS()
                    self?.alertLogic()
                }
            }
        }
        // 核心方案：CenterCrop 聚焦中心，彻底解决远距识别率低的问题
        request.imageCropAndScaleOption = .centerCrop 
        self.requests = [request]
        DispatchQueue.main.async { self.isSystemReady = true }
    }

    private func updateFPS() {
        fps = Int(1.0 / Date().timeIntervalSince(lastFrameDate))
        lastFrameDate = Date()
    }

    private func alertLogic() {
        let isHazardous = detections.contains { 
            calculateHazard(rect: convertRect($0.boundingBox, to: UIScreen.main.bounds.size), size: UIScreen.main.bounds.size) > hazardThreshold 
        }
        if isHazardous && Date().timeIntervalSince(lastAlertDate) > 3.0 {
            let utt = AVSpeechUtterance(string: "注意前方")
            utt.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            speech.speak(utt)
            lastAlertDate = Date()
        }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1.0 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func calculateHazard(rect: CGRect, size: CGSize) -> Double {
        let area = (rect.width * rect.height) / (size.width * size.height)
        let centerWeight = 1.0 - abs((rect.midX / size.width) - 0.5) * 2.0
        // 只对画面中下方区域的目标进行碰撞评估
        return rect.midY > size.height * 0.5 ? Double(area * 22.0 * centerWeight) : 0
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    private func initMotion() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.startDeviceMotionUpdates(to: .main) { m, _ in self.roll = m?.attitude.roll ?? 0 }
        }
    }
}

// MARK: - 工业 UI 组件集
struct TopHUD: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("FSD PRO MAX V8.0").font(.system(size: 12, weight: .black)).foregroundColor(.cyan)
                HStack {
                    Circle().fill(engine.isSystemReady ? Color.green : Color.red).frame(width: 8, height: 8)
                    Text(engine.isSystemReady ? "VISION ACTIVE" : "ENGINE BOOTING").font(.system(size: 8)).foregroundColor(.white.opacity(0.7))
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: -10) {
                Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 65, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                Text("KM/H").font(.system(size: 12, weight: .bold)).foregroundColor(.yellow)
            }
        }.padding(.top, 60).padding(.horizontal, 25)
    }
}

struct PerformanceTag: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        HStack {
            Label("\(engine.fps) FPS", systemImage: "bolt.fill")
            Label("\(Int(engine.inferenceMs))ms", systemImage: "cpu")
        }
        .font(.system(size: 9, weight: .bold, design: .monospaced))
        .padding(8).background(Color.black.opacity(0.6)).foregroundColor(.green).cornerRadius(5).padding(.bottom, 10)
    }
}

struct ControlPanel: View {
    @ObservedObject var engine: FSDCoreEngine
    @Binding var isOpen: Bool
    var body: some View {
        VStack(spacing: 20) {
            Capsule().fill(Color.gray.opacity(0.5)).frame(width: 40, height: 4).padding(.top, 10)
            Text("视觉传感器参数精调").font(.system(size: 14, weight: .bold)).foregroundColor(.white)
            
            TuningSliderRow(name: "硬件变焦 (Sensor Zoom)", val: $engine.zoomFactor, range: 1.0...3.5, color: .orange)
            TuningSliderRow(name: "距离补偿 (K-Factor)", val: $engine.distanceK, range: 3.0...15.0, color: .cyan)
            TuningSliderRow(name: "危险判定 (Sensitivity)", val: $engine.hazardThreshold, range: 0.1...1.5, color: .red)
            
            Button("保存配置并锁定") { withAnimation { isOpen = false } }
                .font(.system(size: 14, weight: .bold)).frame(maxWidth: .infinity).padding().background(Color.cyan).foregroundColor(.black).cornerRadius(12)
        }
        .padding(25).background(BlurBG(style: .systemUltraThinMaterialDark)).cornerRadius(25, corners: [.topLeft, .topRight]).ignoresSafeArea()
    }
}

struct TuningSliderRow: View {
    let name: String; @Binding var val: Double; let range: ClosedRange<Double>; let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text(name).font(.caption); Spacer(); Text(String(format: "%.1f", val)).monospacedDigit() }.foregroundColor(.white.opacity(0.8))
            Slider(value: $val, in: range).accentColor(color)
        }
    }
}

struct SettingsButton: View {
    @Binding var isOpen: Bool
    var body: some View {
        Button(action: { withAnimation { isOpen = true } }) {
            Image(systemName: "gauge.with.dots.needle.bottom.100percent").font(.title2).padding().background(Color.cyan).foregroundColor(.black).clipShape(Circle()).shadow(radius: 10)
        }.padding(.bottom, 40)
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let v = UIView(frame: UIScreen.main.bounds); let l = AVCaptureVideoPreviewLayer(session: session)
        l.videoGravity = .resizeAspectFill; l.frame = v.layer.bounds; v.layer.addSublayer(l); return v
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct BlurBG: UIViewRepresentable {
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

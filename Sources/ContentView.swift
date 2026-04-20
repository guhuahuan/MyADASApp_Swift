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
            FSDMasterView()
        }
    }
}

// MARK: - 主视图
struct FSDMasterView: View {
    @StateObject private var engine = FSDCoreEngine()
    @State private var showSettings = false
    
    var body: some View {
        ZStack {
            // 底层：相机流 (黑屏加固版)
            Color.black.edgesIgnoringSafeArea(.all)
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 中层：AI 渲染与 AR 绘图
            GeometryReader { geo in
                Canvas { context, size in
                    // 1. AR 轨迹线
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    // 2. 目标识别与测距
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect, size: size)
                        
                        let isDanger = hazard > engine.hazardThreshold
                        let color: Color = isDanger ? .red : (hazard > engine.hazardThreshold/2 ? .yellow : .green)
                        
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 动态测距公式
                        let distance = engine.distanceK / (rect.width / size.width + 0.001)
                        context.draw(Text("\(Int(distance))M").font(.system(size: 16, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 25))
                    }
                }
            }
            
            // 顶层：HUD 仪表盘
            VStack {
                HUDHeader(engine: engine)
                Spacer()
                
                // 实时调优面板
                if showSettings {
                    TuningPanel(engine: engine, isPresented: $showSettings)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                } else {
                    // 控制按钮
                    HStack(spacing: 20) {
                        Button(action: { withAnimation { showSettings = true } }) {
                            Label("系统规格调优", systemImage: "slider.horizontal.3")
                                .font(.system(size: 12, weight: .bold))
                                .padding()
                                .background(Color.black.opacity(0.7))
                                .foregroundColor(.cyan)
                                .clipShape(Capsule())
                        }
                    }
                    .padding(.bottom, 40)
                }
            }
        }
        .onAppear { engine.startSystems() }
    }

    // AR 绘图逻辑
    func drawARPath(context: GraphicsContext, size: CGSize, roll: Double) {
        let shift = CGFloat(roll) * 130.0
        var p = Path(); p.move(to: CGPoint(x: size.width * 0.2, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + shift, y: size.height * 0.58), control1: CGPoint(x: size.width * 0.3, y: size.height * 0.85), control2: CGPoint(x: size.width * 0.45, y: size.height * 0.7))
        p.addCurve(to: CGPoint(x: size.width * 0.8, y: size.height), control1: CGPoint(x: size.width * 0.55 + shift, y: size.height * 0.7), control2: CGPoint(x: size.width * 0.7, y: size.height * 0.85))
        context.fill(p, with: .linearGradient(Gradient(colors: [.cyan.opacity(0.35), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.58)))
    }

    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l = isDanger ? 22.0 : 15.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: isDanger ? 4 : 2.5)
    }
}

// MARK: - 核心引擎
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: Double = 0
    @Published var roll: Double = 0
    @Published var distanceK: Double = 12.0
    @Published var hazardThreshold: Double = 0.6
    @Published var isModelReady = false
    
    let captureSession = AVCaptureSession()
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let synthesizer = AVSpeechSynthesizer() // 语音预警
    private var requests = [VNRequest]()
    private var lastAlertTime: Date = Date()
    
    struct Detection { let label: String; let boundingBox: CGRect }

    func startSystems() {
        setupModel(); setupPermissions(); setupMotion()
    }

    // 1. 相机与权限
    private func setupPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func setupCamera() {
        if captureSession.isRunning { return }
        captureSession.beginConfiguration()
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        captureSession.commitConfiguration()
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    // 2. AI 模型加载 (YOLO)
    private func setupModel() {
        guard let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: url, configuration: MLModelConfiguration())) else { return }
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                DispatchQueue.main.async {
                    self?.detections = results.compactMap { res in
                        let label = res.labels.first?.identifier ?? ""
                        return ["car", "person", "truck", "bus"].contains(label) ? Detection(label: label, boundingBox: res.boundingBox) : nil
                    }
                    self?.checkCollisionAlert()
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        self.requests = [request]
        DispatchQueue.main.async { self.isModelReady = true }
    }

    // 3. 碰撞判定与语音预警
    private func checkCollisionAlert() {
        let hasDanger = detections.contains { 
            calculateHazard(rect: convertRect($0.boundingBox, to: UIScreen.main.bounds.size), size: UIScreen.main.bounds.size) > hazardThreshold 
        }
        if hasDanger && Date().timeIntervalSince(lastAlertTime) > 3 {
            let utterance = AVSpeechUtterance(string: "注意前方")
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            utterance.rate = 0.6
            synthesizer.speak(utterance)
            lastAlertTime = Date()
        }
    }

    // 4. 数据采集回调
    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    // 5. 辅助计算
    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1.0 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func calculateHazard(rect: CGRect, size: CGSize) -> Double {
        let areaFactor = (rect.width * rect.height) / (size.width * size.height)
        let centerFactor = 1.0 - abs((rect.midX / size.width) - 0.5) * 2.0
        return rect.midY > size.height * 0.45 ? Double(areaFactor * 15.0 * centerFactor) : 0
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    private func setupMotion() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.startDeviceMotionUpdates(to: .main) { motion, _ in self.roll = motion?.attitude.roll ?? 0 }
        }
    }
}

// MARK: - UI 装饰组件
struct HUDHeader: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 5) {
                Text("FSD MASTER V6.0").font(.system(size: 12, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                StatusTag(text: engine.isModelReady ? "AI: ACTIVE" : "AI: LOADING", active: engine.isModelReady)
            }
            Spacer()
            VStack(alignment: .trailing) {
                Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 55, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                Text("KM/H").font(.system(size: 12, weight: .bold)).foregroundColor(.yellow).offset(y: -10)
            }
        }.padding(.top, 60).padding(.horizontal, 25)
    }
}

struct TuningPanel: View {
    @ObservedObject var engine: FSDCoreEngine
    @Binding var isPresented: Bool
    var body: some View {
        VStack(spacing: 25) {
            TuningSlider(label: "测距基准系数 (K)", value: $engine.distanceK, range: 5...25)
            TuningSlider(label: "碰撞报警灵敏度 (H)", value: $engine.hazardThreshold, range: 0.2...1.5)
            Button("锁定并返回") { withAnimation { isPresented = false } }
                .font(.system(size: 14, weight: .bold)).frame(maxWidth: .infinity).padding().background(Color.cyan).foregroundColor(.black).cornerRadius(12)
        }
        .padding(30).background(BlurView(style: .systemUltraThinMaterialDark).cornerRadius(25)).padding(.horizontal, 20).padding(.bottom, 40)
    }
}

struct TuningSlider: View {
    let label: String; @Binding var value: Double; let range: ClosedRange<Double>
    var body: some View {
        VStack(alignment: .leading) {
            HStack { Text(label).font(.caption.bold()); Spacer(); Text(String(format: "%.1f", value)).monospacedDigit() }.foregroundColor(.white)
            Slider(value: $value, in: range).accentColor(.cyan)
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let v = UIView(frame: UIScreen.main.bounds); let l = AVCaptureVideoPreviewLayer(session: session)
        l.videoGravity = .resizeAspectFill; l.frame = v.layer.bounds; v.layer.addSublayer(l); return v
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct StatusTag: View {
    let text: String; let active: Bool
    var body: some View {
        Text(text).font(.system(size: 8, weight: .black)).padding(4).background(active ? Color.cyan.opacity(0.3) : Color.red.opacity(0.3)).foregroundColor(active ? .cyan : .red).cornerRadius(4)
    }
}

struct BlurView: UIViewRepresentable {
    let style: UIBlurEffect.Style
    func makeUIView(context: Context) -> UIVisualEffectView { UIVisualEffectView(effect: UIBlurEffect(style: style)) }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation
import CoreMotion

@main
struct FSD_Precision_App: App {
    var body: some Scene {
        WindowGroup {
            FSDTrackView()
        }
    }
}

// MARK: - 核心追踪模型
struct FSDObject: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    var rect: CGRect // 物理坐标系下的矩形
}

// MARK: - 主视图
struct FSDTrackView: View {
    @StateObject private var engine = FSDCoreEngine()
    @State private var showPanel = false
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 相机底板
            CameraPreview(session: engine.captureSession)
                .ignoresSafeArea()
            
            // 2. 增强现实感知层
            GeometryReader { geo in
                let screenSize = geo.size
                
                Canvas { context, size in
                    // 绘制动态辅助线
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    for obj in engine.trackedObjects {
                        let isDanger = engine.calculateHazard(rect: obj.rect, size: size) > engine.hazardThreshold
                        let color: Color = isDanger ? .red : (obj.label == "person" ? .yellow : .green)
                        
                        // 绘制工业角框
                        drawCornerBox(context: context, rect: obj.rect, color: color, isDanger: isDanger)
                        
                        // 测距与标签
                        let distance = (engine.distanceK * engine.zoomFactor) / (obj.rect.width / size.width + 0.001)
                        context.draw(Text("\(obj.label.uppercased()) \(Int(distance))M").font(.system(size: 14, weight: .heavy)).foregroundColor(color),
                                     at: CGPoint(x: obj.rect.minX, y: obj.rect.minY - 20))
                    }
                }
            }
            
            // 3. 工业 HUD
            VStack {
                TopHUDView(engine: engine)
                Spacer()
                PerformanceBar(engine: engine)
                
                if showPanel {
                    TuningPanel(engine: engine, isPresented: $showPanel)
                        .transition(.move(edge: .bottom))
                } else {
                    ControlTrigger(showPanel: $showPanel)
                }
            }
        }
        .onAppear { engine.start() }
    }
    
    func drawARPath(context: GraphicsContext, size: CGSize, roll: Double) {
        let shift = CGFloat(roll) * 150.0
        var p = Path(); p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + shift, y: size.height * 0.55), control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), control2: CGPoint(x: size.width * 0.4, y: size.height * 0.65))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), control1: CGPoint(x: size.width * 0.6 + shift, y: size.height * 0.65), control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.cyan.opacity(0.3), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.55)))
    }
    
    func drawCornerBox(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l: CGFloat = 15; let w: CGFloat = isDanger ? 4 : 2
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: w)
    }
}

// MARK: - 核心感知引擎 (含坐标校准算法)
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var trackedObjects: [FSDObject] = []
    @Published var currentSpeed: Double = 0
    @Published var roll: Double = 0
    @Published var fps: Int = 0
    
    @Published var zoomFactor: Double = 1.8 { didSet { applyZoom() } }
    @Published var distanceK: Double = 7.0
    @Published var hazardThreshold: Double = 0.5
    
    let captureSession = AVCaptureSession()
    private var videoInput: AVCaptureDeviceInput?
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let speech = AVSpeechSynthesizer()
    private var requests = [VNRequest]()
    
    // 平滑滤波器参数 (EMA)
    private var lastRects: [String: CGRect] = [:] 
    private let smoothingAlpha: CGFloat = 0.4 // 0.1-1.0, 越小越丝滑但滞后越重
    
    func start() {
        setupCamera(); setupModel(); setupSensors()
    }
    
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080 
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input); self.videoInput = input }
            try device.lockForConfiguration()
            device.videoZoomFactor = CGFloat(zoomFactor)
            device.unlockForConfiguration()
        } catch { print(error) }
        
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "perception_queue", qos: .userInteractive))
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        captureSession.commitConfiguration()
        DispatchQueue.global().async { self.captureSession.startRunning() }
    }
    
    private func applyZoom() {
        guard let device = videoInput?.device else { return }
        try? device.lockForConfiguration()
        device.videoZoomFactor = CGFloat(max(1.0, min(zoomFactor, 5.0)))
        device.unlockForConfiguration()
    }
    
    private func setupModel() {
        guard let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: MLModelConfiguration())) else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                self?.processDetections(results)
            }
        }
        request.imageCropAndScaleOption = .centerCrop 
        self.requests = [request]
    }
    
    // MARK: - 核心修复：高精度坐标对齐算法
    private func processDetections(_ observations: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async {
            let screenSize = UIScreen.main.bounds.size
            let videoSize = CGSize(width: 1920, height: 1080) // 必须与 sessionPreset 一致
            
            self.trackedObjects = observations.compactMap { obs in
                let label = obs.labels.first?.identifier ?? ""
                let targets = ["car", "person", "motorcycle", "bus", "truck"]
                guard targets.contains(label), obs.confidence > 0.35 else { return nil }
                
                // 1. 基础归一化坐标转换 (Vision -> iOS)
                let rect = obs.boundingBox
                let x = rect.minX
                let y = 1.0 - rect.maxY
                let w = rect.width
                let h = rect.height
                
                // 2. 补偿 AspectFill 裁剪误差
                // 计算视频在屏幕上填充后的实际缩放比例
                let screenAspect = screenSize.width / screenSize.height
                let videoAspect = videoSize.height / videoSize.width // 注意相机流是横向的
                
                var correctedRect = CGRect(x: x, y: y, width: w, height: h)
                
                if screenAspect > videoAspect {
                    let factor = videoAspect / screenAspect
                    correctedRect.origin.y = (y - (1 - factor) / 2) / factor
                    correctedRect.size.height = h / factor
                } else {
                    let factor = screenAspect / videoAspect
                    correctedRect.origin.x = (x - (1 - factor) / 2) / factor
                    correctedRect.size.width = w / factor
                }
                
                // 3. 映射到物理像素
                let physicalRect = CGRect(
                    x: correctedRect.minX * screenSize.width,
                    y: correctedRect.minY * screenSize.height,
                    width: correctedRect.size.width * screenSize.width,
                    height: correctedRect.size.height * screenSize.height
                )
                
                // 4. EMA 平滑滤波 (消除跳动)
                let smoothed = self.applyEMA(for: label, newRect: physicalRect)
                
                return FSDObject(label: label, confidence: obs.confidence, rect: smoothed)
            }
            self.checkAlerts()
        }
    }
    
    private func applyEMA(for label: String, newRect: CGRect) -> CGRect {
        guard let oldRect = lastRects[label] else {
            lastRects[label] = newRect
            return newRect
        }
        let smoothed = CGRect(
            x: oldRect.minX * (1 - smoothingAlpha) + newRect.minX * smoothingAlpha,
            y: oldRect.minY * (1 - smoothingAlpha) + newRect.minY * smoothingAlpha,
            width: oldRect.width * (1 - smoothingAlpha) + newRect.width * smoothingAlpha,
            height: oldRect.height * (1 - smoothingAlpha) + newRect.height * smoothingAlpha
        )
        lastRects[label] = smoothed
        return smoothed
    }

    func calculateHazard(rect: CGRect, size: CGSize) -> Double {
        let area = (rect.width * rect.height) / (size.width * size.height)
        let centerBias = 1.0 - abs((rect.midX / size.width) - 0.5) * 2.0
        return rect.midY > size.height * 0.5 ? Double(area * 25.0 * centerBias) : 0
    }
    
    private func checkAlerts() {
        let danger = trackedObjects.contains { calculateHazard(rect: $0.rect, size: UIScreen.main.bounds.size) > hazardThreshold }
        if danger && !speech.isSpeaking {
            let u = AVSpeechUtterance(string: "注意前方")
            u.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            u.rate = 0.58
            speech.speak(u)
        }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }
    
    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    private func setupSensors() {
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        if motionManager.isDeviceMotionAvailable {
            motionManager.startDeviceMotionUpdates(to: .main) { m, _ in self.roll = m?.attitude.roll ?? 0 }
        }
    }
}

// MARK: - 辅助 UI
struct TopHUDView: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("FSD PRECISION V9.0").font(.system(size: 10, weight: .black)).foregroundColor(.cyan)
                Text("CENTER-CROP ACTIVE").font(.system(size: 7)).foregroundColor(.white.opacity(0.5))
            }
            Spacer()
            Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 55, weight: .black, design: .monospaced)).foregroundColor(.yellow)
        }.padding(.top, 60).padding(.horizontal, 25)
    }
}

struct PerformanceBar: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        HStack {
            Image(systemName: "waveform.path.ecg").foregroundColor(.green)
            Text("EMA TRACKING STABLE").font(.system(size: 8, weight: .bold))
        }
        .padding(8).background(Color.black.opacity(0.7)).foregroundColor(.green).cornerRadius(5).padding(.bottom, 10)
    }
}

struct TuningPanel: View {
    @ObservedObject var engine: FSDCoreEngine; @Binding var isPresented: Bool
    var body: some View {
        VStack(spacing: 18) {
            SliderRow(title: "光学变焦", val: $engine.zoomFactor, range: 1.0...4.0)
            SliderRow(title: "测距校准", val: $engine.distanceK, range: 3.0...12.0)
            SliderRow(title: "碰撞灵敏", val: $engine.hazardThreshold, range: 0.1...1.2)
            Button("锁定参数") { withAnimation { isPresented = false } }
                .padding().frame(maxWidth: .infinity).background(Color.blue).foregroundColor(.white).cornerRadius(10)
        }
        .padding(25).background(VisualEffectBlur(style: .systemUltraThinMaterialDark)).cornerRadius(20).padding()
    }
}

struct SliderRow: View {
    let title: String; @Binding var val: Double; let range: ClosedRange<Double>
    var body: some View {
        VStack {
            HStack { Text(title); Spacer(); Text(String(format: "%.1f", val)) }.font(.caption).foregroundColor(.white)
            Slider(value: $val, in: range).accentColor(.cyan)
        }
    }
}

struct ControlTrigger: View {
    @Binding var showPanel: Bool
    var body: some View {
        Button(action: { withAnimation { showPanel = true } }) {
            Image(systemName: "dot.scope").padding().background(Color.cyan).foregroundColor(.black).clipShape(Circle())
        }.padding(.bottom, 40)
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let v = UIView(frame: UIScreen.main.bounds)
        let l = AVCaptureVideoPreviewLayer(session: session)
        l.videoGravity = .resizeAspectFill
        l.frame = v.layer.bounds
        v.layer.addSublayer(l)
        return v
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct VisualEffectBlur: UIViewRepresentable {
    var style: UIBlurEffect.Style
    func makeUIView(context: Context) -> UIVisualEffectView { UIVisualEffectView(effect: UIBlurEffect(style: style)) }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

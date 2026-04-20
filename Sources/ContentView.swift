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
    @State private var showPanel = false // 控制调优面板显示
    
    var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // AI 渲染层
            GeometryReader { geo in
                Canvas { context, size in
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        
                        // 使用实时调整的阈值
                        let isDanger = hazard > engine.hazardThreshold
                        let color: Color = isDanger ? .red : (hazard > engine.hazardThreshold/2 ? .yellow : .green)
                        
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 使用实时调整的距离系数 K
                        let pixelWidthRatio = rect.width / size.width
                        let distance = engine.distanceK / (pixelWidthRatio + 0.001)
                        let displayDist = min(max(Int(distance), 1), 150)
                        
                        context.draw(Text("\(detection.label.uppercased()) \(displayDist)M").font(.system(size: 11, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 18))
                    }
                }
            }
            
            // 顶层 HUD
            HUDOverlay(engine: engine)
            
            // --- 新增：路测实时调优面板 ---
            VStack {
                Spacer()
                if showPanel {
                    TuningPanel(engine: engine)
                        .transition(.move(edge: .bottom))
                }
                
                Button(action: { withAnimation { showPanel.toggle() } }) {
                    Label(showPanel ? "隐藏设置" : "实时调优", systemImage: "gearshape.fill")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(8)
                        .background(Color.black.opacity(0.6))
                        .foregroundColor(.cyan)
                        .cornerRadius(8)
                }
                .padding(.bottom, 40) // 避开底部手势条
            }
        }
        .onAppear { engine.startSystems() }
    }

    // 绘制逻辑保持不变 (省略部分重复绘图函数以保持简洁...)
    func drawARPath(context: GraphicsContext, size: CGSize, roll: Double) {
        let shift = CGFloat(roll) * 160.0
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + shift, y: size.height * 0.52), control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), control2: CGPoint(x: size.width * 0.4, y: size.height * 0.62))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), control1: CGPoint(x: size.width * 0.6 + shift, y: size.height * 0.62), control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.cyan.opacity(0.4), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.52)))
    }
    
    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l = isDanger ? 20.0 : 14.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: isDanger ? 3.5 : 2)
    }
}

// MARK: - 调优组件
struct TuningPanel: View {
    @ObservedObject var engine: FSDCoreEngine
    
    var body: some View {
        VStack(spacing: 15) {
            tuningSlider(title: "距离系数 (K)", value: $engine.distanceK, range: 5...25)
            tuningSlider(title: "警报阈值 (H)", value: $engine.hazardThreshold, range: 0.1...1.5)
        }
        .padding()
        .background(BlurView(style: .dark).cornerRadius(15))
        .padding(.horizontal, 40)
        .padding(.bottom, 10)
    }
    
    func tuningSlider(title: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title).foregroundColor(.white).font(.caption).bold()
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue)).foregroundColor(.cyan).font(.system(.caption, design: .monospaced))
            }
            Slider(value: value, in: range)
                .accentColor(.cyan)
        }
    }
}

// MARK: - 引擎增加可调属性
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: Double = 0
    @Published var headingInfo: String = "HEADING: --"
    @Published var isModelLoaded = false
    @Published var fps: Int = 0
    @Published var latency: TimeInterval = 0
    @Published var roll: Double = 0
    
    // --- 动态调优参数 ---
    @Published var distanceK: Double = 14.5        // 默认距离系数
    @Published var hazardThreshold: Double = 0.6   // 默认危险阈值
    
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
        guard let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc") else { return }
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
                            return (self?.focusLabels.contains(label) == true) ? Detection(label: label, boundingBox: res.boundingBox) : nil
                        }
                    }
                }
            }
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            self.isModelLoaded = true
        } catch { print("Model Init Error") }
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
                self.roll = motion?.attitude.roll ?? 0
            }
        }
    }

    private func setupCamera() {
        captureSession.beginConfiguration()
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        guard let input = try? AVCaptureDeviceInput(device: device) else { return }
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        if let conn = output.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }
        }
        captureSession.commitConfiguration()
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        lastTime = Date()
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1.0 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func calculateHazard(rect: CGRect) -> Double {
        let centerX = rect.midX / UIScreen.main.bounds.width
        let centerY = rect.midY / UIScreen.main.bounds.height
        let centerFactor = 1.0 - abs(centerX - 0.5) * 2.0
        return centerY > 0.5 ? Double(centerFactor * (rect.width / UIScreen.main.bounds.width) * 3.5) : 0
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    func locationManager(_ m: CLLocationManager, didUpdateHeading h: CLHeading) {
        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        headingInfo = "HEADING: \(dirs[Int((h.magneticHeading + 22.5) / 45.0) & 7]) \(Int(h.magneticHeading))°"
    }
}

// MARK: - 辅助组件
struct BlurView: UIViewRepresentable {
    let style: UIBlurEffect.Style
    func makeUIView(context: Context) -> UIVisualEffectView { UIVisualEffectView(effect: UIBlurEffect(style: style)) }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let v = UIView(frame: UIScreen.main.bounds); let l = AVCaptureVideoPreviewLayer(session: session)
        l.videoGravity = .resizeAspectFill; l.frame = v.layer.bounds
        if let conn = l.connection, conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }
        v.layer.addSublayer(l); return v
    }
    func updateUIView(_ uiView: UIView, context: Context) {
        if let l = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            l.frame = uiView.bounds
            if let conn = l.connection, conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }
        }
    }
}

struct HUDOverlay: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        VStack {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("FSD MASTER V4.2").font(.system(size: 14, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                    Text(engine.headingInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white)
                }
                Spacer()
                Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 60, weight: .black, design: .monospaced)).foregroundColor(.yellow)
            }
            .padding(.horizontal, 40).padding(.top, 25)
            Spacer()
        }
    }
}

struct StatusTag: View {
    let text: String; let active: Bool
    var body: some View {
        Text(text).font(.system(size: 8, weight: .black)).padding(3).background(active ? Color.blue : Color.red).foregroundColor(.white).cornerRadius(3)
    }
}

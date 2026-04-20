import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation
import CoreMotion

@main
struct FSD_V7_App: App {
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
            Color.black.edgesIgnoringSafeArea(.all)
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            GeometryReader { geo in
                Canvas { context, size in
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect, size: size)
                        let isDanger = hazard > engine.hazardThreshold
                        let color: Color = isDanger ? .red : (hazard > engine.hazardThreshold/2 ? .yellow : .green)
                        
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 距离计算：加入硬件变焦系数的补偿
                        // 变焦越大，物体看起来越大，需要用 zoomFactor 修正真实距离
                        let opticalCompensation = engine.zoomFactor
                        let distance = (engine.distanceK * opticalCompensation) / (rect.width / size.width + 0.001)
                        
                        context.draw(Text("\(Int(distance))M").font(.system(size: 16, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 25))
                    }
                }
            }
            
            VStack {
                HUDHeader(engine: engine)
                Spacer()
                if showSettings {
                    TuningPanel(engine: engine, isPresented: $showSettings)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                } else {
                    Button(action: { withAnimation { showSettings = true } }) {
                        Label("工业视觉调参", systemImage: "camera.macro")
                            .font(.system(size: 12, weight: .bold))
                            .padding()
                            .background(Color.black.opacity(0.8))
                            .foregroundColor(.cyan)
                            .clipShape(Capsule())
                    }
                    .padding(.bottom, 40)
                }
            }
        }
        .onAppear { engine.startSystems() }
    }

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

// MARK: - 核心引擎 (工业级光路与预处理)
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: Double = 0
    @Published var roll: Double = 0
    
    // 开放硬件级变焦控制，默认 1.8 倍（模拟长焦视野）
    @Published var zoomFactor: Double = 1.8 {
        didSet { updateCameraZoom() }
    }
    @Published var distanceK: Double = 6.0 // 基础K值
    @Published var hazardThreshold: Double = 0.4
    @Published var isModelReady = false
    
    let captureSession = AVCaptureSession()
    private var videoDeviceInput: AVCaptureDeviceInput?
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private let synthesizer = AVSpeechSynthesizer()
    private var requests = [VNRequest]()
    
    struct Detection { let label: String; let boundingBox: CGRect }

    func startSystems() {
        setupModel(); setupPermissions(); setupMotion()
    }

    private func setupPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080 
        
        // 【关键修复】使用主摄，并通过硬件级的 videoZoomFactor 来实现真正的远端放大
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { 
                captureSession.addInput(input) 
                self.videoDeviceInput = input
            }
            
            // 锁定自动对焦和初始焦距
            try device.lockForConfiguration()
            if device.isSmoothAutoFocusSupported { device.isSmoothAutoFocusEnabled = true }
            device.videoZoomFactor = CGFloat(min(zoomFactor, device.activeFormat.videoMaxZoomFactor))
            device.unlockForConfiguration()
            
        } catch { print("Camera config failed.") }
        
        let output = AVCaptureVideoDataOutput()
        // 丢弃延迟帧，防止运动模糊影响模型推理
        output.alwaysDiscardsLateVideoFrames = true 
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        
        captureSession.commitConfiguration()
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    // 动态更新物理镜头焦距
    private func updateCameraZoom() {
        guard let device = videoDeviceInput?.device else { return }
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = CGFloat(min(zoomFactor, device.activeFormat.videoMaxZoomFactor))
            device.unlockForConfiguration()
        } catch { print("Zoom failed") }
    }

    private func setupModel() {
        guard let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: url, configuration: MLModelConfiguration())) else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                DispatchQueue.main.async {
                    self?.detections = results.compactMap { res in
                        let label = res.labels.first?.identifier ?? ""
                        // 置信度保持 0.35，过滤绝对噪点，放行远端特征
                        if ["car", "person", "truck", "bus"].contains(label) && res.confidence > 0.35 {
                            return Detection(label: label, boundingBox: res.boundingBox)
                        }
                        return nil
                    }
                }
            }
        }
        // 维持中心裁切，配合硬件 Zoom，实现双重放大
        request.imageCropAndScaleOption = .centerCrop 
        self.requests = [request]
        DispatchQueue.main.async { self.isModelReady = true }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1.0 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func calculateHazard(rect: CGRect, size: CGSize) -> Double {
        let areaFactor = (rect.width * rect.height) / (size.width * size.height)
        let centerFactor = 1.0 - abs((rect.midX / size.width) - 0.5) * 2.0
        return rect.midY > size.height * 0.45 ? Double(areaFactor * 18.0 * centerFactor) : 0
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    private func setupMotion() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.startDeviceMotionUpdates(to: .main) { motion, _ in self.roll = motion?.attitude.roll ?? 0 }
        }
    }
}

// MARK: - 组件层
struct HUDHeader: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 5) {
                Text("FSD V7.0 OPTICAL").font(.system(size: 10, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                Text(engine.isModelReady ? "AI: RUNNING" : "AI: LOADING").font(.system(size: 7)).foregroundColor(engine.isModelReady ? .green : .red)
            }
            Spacer()
            Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 55, weight: .black, design: .monospaced)).foregroundColor(.yellow)
        }.padding(.top, 60).padding(.horizontal, 25)
    }
}

struct TuningPanel: View {
    @ObservedObject var engine: FSDCoreEngine
    @Binding var isPresented: Bool
    var body: some View {
        VStack(spacing: 20) {
            // 新增硬件光路变焦控制
            TuningSlider(label: "光学变焦 (Sensor Zoom)", value: $engine.zoomFactor, range: 1.0...3.0, color: .orange)
            TuningSlider(label: "测距基准系数 (K)", value: $engine.distanceK, range: 2...15, color: .cyan)
            TuningSlider(label: "预警灵敏阈值 (H)", value: $engine.hazardThreshold, range: 0.1...1.2, color: .red)
            
            Button("应用并返回") { withAnimation { isPresented = false } }
                .font(.system(size: 14, weight: .bold)).frame(maxWidth: .infinity).padding(12).background(Color.blue).foregroundColor(.white).cornerRadius(10)
        }
        .padding(25).background(BlurView(style: .systemUltraThinMaterialDark).cornerRadius(20)).padding(.horizontal, 15).padding(.bottom, 30)
    }
}

struct TuningSlider: View {
    let label: String; @Binding var value: Double; let range: ClosedRange<Double>; let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack { Text(label).font(.caption.bold()); Spacer(); Text(String(format: "%.1f", value)).monospacedDigit() }.foregroundColor(.white)
            Slider(value: $value, in: range).accentColor(color)
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

struct BlurView: UIViewRepresentable {
    let style: UIBlurEffect.Style
    func makeUIView(context: Context) -> UIVisualEffectView { UIVisualEffectView(effect: UIBlurEffect(style: style)) }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

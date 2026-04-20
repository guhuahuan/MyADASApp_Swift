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

// MARK: - 主视图
struct FSDMasterView: View {
    @StateObject private var engine = FSDCoreEngine()
    
    var body: some View {
        ZStack {
            // 1. 底层：横屏相机预览
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 2. 中层：AR 增强现实渲染
            GeometryReader { geo in
                Canvas { context, size in
                    // 绘制 3D 引导轨迹
                    drawAdvanced3DPath(context: context, size: size)
                    
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        let isDanger = hazard > 0.7
                        let color: Color = isDanger ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        // 计算估算距离 (基于框宽度的反比，仅作路测参考)
                        let estimatedDist = Int(140 / (rect.width / size.width * 11))
                        
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 绘制标签与距离
                        context.draw(Text("\(detection.label.uppercased()) \(estimatedDist)m")
                            .font(.system(size: 10, weight: .black)).foregroundColor(color), 
                            at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. 顶层：HUD 仪表盘
            HorizontalHUDOverlay(engine: engine)
        }
        .onAppear { engine.startSystems() }
    }
    
    func drawAdvanced3DPath(context: GraphicsContext, size: CGSize) {
        let rollShift = CGFloat(engine.roll) * 180.0
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + rollShift, y: size.height * 0.55), 
                   control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), 
                   control2: CGPoint(x: size.width * 0.4, y: size.height * 0.65))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), 
                   control1: CGPoint(x: size.width * 0.55 + rollShift, y: size.height * 0.65), 
                   control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.blue.opacity(0.35), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.55)))
    }

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

struct HorizontalHUDOverlay: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        VStack {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("FSD MASTER V4.0").font(.system(size: 14, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                    HStack(spacing: 12) {
                        StatusTag(text: "AI", active: engine.isModelLoaded)
                        StatusTag(text: "FPS: \(engine.fps)", active: engine.fps > 25)
                    }
                }
                Spacer()
                VStack {
                    Text("\(Int(engine.headingAngle))°").font(.system(size: 30, weight: .black)).foregroundColor(.white)
                    Text(engine.headingDir).font(.system(size: 10, weight: .black)).foregroundColor(.cyan)
                }.frame(width: 80)
            }
            .padding().background(LinearGradient(colors: [.black.opacity(0.8), .clear], startPoint: .top, endPoint: .bottom))
            Spacer()
            HStack(alignment: .bottom) {
                VStack {
                    Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 80, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                    Text("KM/H").font(.caption).bold().foregroundColor(.yellow)
                }
                .padding().background(LinearGradient(colors: [.black.opacity(0.8), .clear], startPoint: .bottom, endPoint: .top))
                Spacer()
                Text("LATENCY: \(Int(engine.latency * 1000))ms").font(.system(size: 10, design: .monospaced)).foregroundColor(.cyan).padding().background(Color.black.opacity(0.3))
            }
        }
    }
}

class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: Double = 0
    @Published var headingAngle: Double = 0
    @Published var headingDir: String = "N"
    @Published var isModelLoaded = false
    @Published var latency: TimeInterval = 0
    @Published var fps: Int = 0
    @Published var roll: Double = 0
    @Published var pitch: Double = 0
    
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
                        self?.fps = Int(1.0 / (diff > 0 ? diff : 0.03))
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
        } catch { print("AI Load Failed") }
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
        
        if let conn = output.connection(with: .video) {
            // 适配 iOS 17 的旋转 API
            if conn.isVideoRotationAngleSupported(90) {
                conn.videoRotationAngle = 90 // 对应 LandscapeRight
            }
        }
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput b: CMSampleBuffer, from conn: AVCaptureConnection) {
        lastTime = Date()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
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
        return CGRect(x: rect.minX * scale + offsetX, y: (1.0 - rect.maxY) * (scale / imageAspectRatio) + offsetY, width: rect.width * scale, height: rect.height * (scale / imageAspectRatio))
    }

    func calculateHazard(rect: CGRect) -> Double {
        return rect.minY > UIScreen.main.bounds.height * 0.5 ? Double(rect.width / UIScreen.main.bounds.width * 2.5) : 0.0
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    func locationManager(_ m: CLLocationManager, didUpdateHeading h: CLHeading) {
        headingAngle = h.magneticHeading
        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        headingDir = dirs[Int((h.magneticHeading + 22.5) / 45.0) & 7]
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
        // 适配 iOS 17 的预览层旋转
        if layer.connection?.isVideoRotationAngleSupported(90) ?? false {
            layer.connection?.videoRotationAngle = 90
        }
        layer.frame = view.layer.bounds
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

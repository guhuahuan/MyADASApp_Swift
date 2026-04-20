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
            // 背景设为黑色，防止闪烁
            Color.black.edgesIgnoringSafeArea(.all)
            
            // 1. 相机预览层 (加固版)
            CameraPreview(session: engine.captureSession)
                .edgesIgnoringSafeArea(.all)
            
            // 2. AI 渲染层
            GeometryReader { geo in
                Canvas { context, size in
                    drawARPath(context: context, size: size, roll: engine.roll)
                    
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        let isDanger = hazard > 0.7
                        let color: Color = isDanger ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        let dist = Int(140 / (rect.width / size.width * 11))
                        context.draw(Text("\(detection.label.uppercased()) \(dist)M").font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. HUD 仪表层
            HUDOverlay(engine: engine)
        }
        .onAppear {
            engine.startSystems()
        }
    }

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

// MARK: - HUD
struct HUDOverlay: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        VStack {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("FSD MASTER V4.0").font(.system(size: 14, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                    Text(engine.headingInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white)
                    HStack(spacing: 8) {
                        StatusTag(text: "CAM: \(engine.isCameraRunning ? "OK" : "ERR")", active: engine.isCameraRunning)
                        StatusTag(text: "AI: \(engine.isModelLoaded ? "READY" : "OFF")", active: engine.isModelLoaded)
                    }
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 70, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                    Text("KM/H").font(.caption).bold().foregroundColor(.yellow)
                }
            }
            .padding(.horizontal, 40).padding(.top, 25)
            Spacer()
            HStack {
                Text("FPS: \(engine.fps)").foregroundColor(.green)
                Spacer()
                Text("LATENCY: \(Int(engine.latency * 1000))ms").foregroundColor(.cyan)
            }
            .font(.system(size: 10, design: .monospaced)).padding().background(Color.black.opacity(0.4))
        }
    }
}

// MARK: - 计算引擎
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: Double = 0
    @Published var headingInfo: String = "航向: --"
    @Published var isModelLoaded = false
    @Published var isCameraRunning = false
    @Published var fps: Int = 0
    @Published var latency: TimeInterval = 0
    @Published var roll: Double = 0
    
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
        } catch { print("AI Load Failed") }
    }

    private func setupPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            if granted {
                DispatchQueue.main.async { self.setupCamera() }
            }
        }
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
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        
        if let conn = output.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }
        }
        captureSession.commitConfiguration()
        
        // 关键：在后台线程启动 Session，防止阻塞 UI
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
            DispatchQueue.main.async { self.isCameraRunning = self.captureSession.isRunning }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput b: CMSampleBuffer, from conn: AVCaptureConnection) {
        lastTime = Date()
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(
            x: rect.minX * size.width,
            y: (1.0 - rect.maxY) * size.height,
            width: rect.width * size.width,
            height: rect.height * size.height
        )
    }

    func calculateHazard(rect: CGRect) -> Double {
        return rect.minY > UIScreen.main.bounds.height * 0.5 ? Double(rect.width / UIScreen.main.bounds.width * 2.5) : 0
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = l.last?.speed ?? 0 }
    func locationManager(_ m: CLLocationManager, didUpdateHeading h: CLHeading) {
        let dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        headingInfo = "航向: \(dirs[Int((h.magneticHeading + 22.5) / 45.0) & 7]) \(Int(h.magneticHeading))°"
    }
}

// MARK: - 相机预览视图 (解决黑屏的关键组件)
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.layer.bounds
        
        // 旋转校准
        if let conn = layer.connection, conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
        
        view.layer.addSublayer(layer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // 关键：当 UI 布局刷新时，强制同步 Layer 的尺寸
        if let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            DispatchQueue.main.async {
                layer.frame = uiView.bounds
                if let conn = layer.connection, conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            }
        }
    }
}

struct StatusTag: View {
    let text: String; let active: Bool
    var body: some View {
        Text(text).font(.system(size: 8, weight: .black)).padding(3).background(active ? Color.blue : Color.red).foregroundColor(.white).cornerRadius(3)
    }
}

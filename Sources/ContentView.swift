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
    
    var body: some View {
        ZStack {
            // 1. 底层：横屏相机预览
            CameraPreview(session: engine.captureSession).edgesIgnoringSafeArea(.all)
            
            // 2. 中层：AR 增强渲染 (3D轨迹 + 碰撞预警)
            GeometryReader { geo in
                Canvas { context, size in
                    // 绘制 3D 引导线 (动态随陀螺仪偏移)
                    draw3DPath(context: context, size: size, attitude: engine.deviceAttitude)
                    
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        let hazard = engine.calculateHazard(rect: rect)
                        let isDanger = hazard > 0.75
                        let color: Color = isDanger ? .red : (hazard > 0.4 ? .yellow : .green)
                        
                        // 绘制锁定角标
                        drawTargetCorners(context: context, rect: rect, color: color, isDanger: isDanger)
                        
                        // 距离显示
                        let dist = Int(140 / (rect.width / size.width * 11))
                        context.draw(Text("\(detection.label.uppercased()) | \(dist)M").font(.system(size: 10, weight: .black)).foregroundColor(color), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 15))
                    }
                }
            }
            
            // 3. 顶层：HUD 数字化仪表
            HUDLayer(engine: engine)
        }
        .onAppear { engine.startSystems() }
    }

    func draw3DPath(context: GraphicsContext, size: CGSize, attitude: CMAttitude?) {
        let roll = CGFloat(attitude?.roll ?? 0) * 200
        var p = Path()
        p.move(to: CGPoint(x: size.width * 0.1, y: size.height))
        p.addCurve(to: CGPoint(x: size.width * 0.5 + roll, y: size.height * 0.45), 
                   control1: CGPoint(x: size.width * 0.2, y: size.height * 0.8), 
                   control2: CGPoint(x: size.width * 0.4, y: size.height * 0.6))
        p.addCurve(to: CGPoint(x: size.width * 0.9, y: size.height), 
                   control1: CGPoint(x: size.width * 0.6 + roll, y: size.height * 0.6), 
                   control2: CGPoint(x: size.width * 0.8, y: size.height * 0.8))
        context.fill(p, with: .linearGradient(Gradient(colors: [.blue.opacity(0.4), .clear]), startPoint: CGPoint(x: 0, y: size.height), endPoint: CGPoint(x: 0, y: size.height * 0.45)))
    }

    func drawTargetCorners(context: GraphicsContext, rect: CGRect, color: Color, isDanger: Bool) {
        let l = isDanger ? 24.0 : 14.0
        context.stroke(Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY + l)); p.addLine(to: CGPoint(x: rect.minX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.minX + l, y: rect.minY))
            p.move(to: CGPoint(x: rect.maxX - l, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + l))
            p.move(to: CGPoint(x: rect.maxX, y: rect.maxY - l)); p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.maxX - l, y: rect.maxY))
            p.move(to: CGPoint(x: rect.minX + l, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY)); p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - l))
        }, with: .color(color), lineWidth: isDanger ? 4 : 2)
    }
}

// MARK: - HUD 视图组件
struct HUDLayer: View {
    @ObservedObject var engine: FSDCoreEngine
    var body: some View {
        VStack {
            HStack(alignment: .top) {
                VStack(alignment: .leading) {
                    Text("FSD MASTER V4.0").font(.system(size: 14, weight: .black, design: .monospaced)).foregroundColor(.cyan)
                    Text(engine.headingInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white)
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 70, weight: .black, design: .monospaced)).foregroundColor(.yellow)
                    Text("KM/H").font(.caption).bold().foregroundColor(.yellow)
                }
            }
            .padding(.horizontal, 40).padding(.top, 20)
            
            Spacer()
            
            if engine.detections.contains(where: { engine.calculateHazard(rect: engine.convertRect($0.boundingBox, to: UIScreen.main.bounds.size)) > 0.75 }) {
                Text("⚠️ COLLISION RISK")
                    .font(.system(size: 20, weight: .black)).foregroundColor(.white).padding().background(Color.red).cornerRadius(10).padding(.bottom, 30)
            }
        }
    }
}

// MARK: - 引擎逻辑
class FSDCoreEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: CLLocationSpeed = 0
    @Published var headingInfo: String = "HEADING: --"
    @Published var isModelLoaded = false
    @Published var deviceAttitude: CMAttitude?
    
    private let focusLabels = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]
    let captureSession = AVCaptureSession()
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    private var requests = [VNRequest]()
    
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
        } catch { print("AI ERROR") }
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
            }
        }
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        try? captureSession.addInput(AVCaptureDeviceInput(device: device))
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        captureSession.addOutput(output)
        if let conn = output.connection(with: .video) { conn.videoOrientation = .landscapeRight }
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput b: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(b) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        let imgRatio: CGFloat = 1920.0 / 1080.0 
        let viewRatio: CGFloat = size.width / size.height
        let scale = (viewRatio > imgRatio) ? size.width : size.height * imgRatio
        let oX = (size.width - scale) / 2.0
        let oY = (size.height - scale / imgRatio) / 2.0
        return CGRect(x: rect.minX * scale + oX, y: (1.0 - rect.maxY) * (scale / imgRatio) + oY, width: rect.width * scale, height: rect.height * (scale / imgRatio))
    }

    func calculateHazard(rect: CGRect) -> Double { Double(rect.width / UIScreen.main.bounds.width * 2.8) }

    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) { currentSpeed = max(0, l.last?.speed ?? 0) }
    func locationManager(_ m: CLLocationManager, didUpdateHeading h: CLHeading) {
        let dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        headingInfo = "HEADING: \(dirs[Int((h.magneticHeading + 22.5) / 45.0) & 7]) \(Int(h.magneticHeading))°"
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.connection?.videoOrientation = .landscapeRight
        layer.frame = view.layer.bounds
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

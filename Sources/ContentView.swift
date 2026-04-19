@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreMotion
import CoreLocation
import CoreImage
import CoreImage.CIFilterBuiltins

// MARK: - 1. 核心数据结构
struct FSDV3Object: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var velocityVector: CGVector
    var boundingBox: CGRect
    var isSideHazard: Bool
    var isCritical: Bool
}

struct SafeModelBufferV3: @unchecked Sendable {
    let detModel: VNCoreMLModel
}

enum ADASCameraMode: String, Sendable {
    case ultraWide = "CITY: WIDE-ANGLE (0.5x)"
    case telephoto = "HIGHWAY: FOCUS (1x + ROI)"
}

// MARK: - 2. FSD V3 旗舰引擎
@MainActor
class FSDV3Engine: NSObject, ObservableObject {
    // UI 驱动属性
    @Published var cameraMode: ADASCameraMode = .ultraWide
    @Published var trackedObjects: [FSDV3Object] = []
    @Published var occupancyMask: CGImage?
    @Published var sideWarning: Edge? = nil
    @Published var currentSpeed: Double = 0.0
    @Published var speedLimit: Int = 0
    @Published var isNightMode: Bool = false
    
    // Zen Path & IMU 属性
    @Published var yawRate: Double = 0.0
    @Published var autoPitch: Double = 0.0
    
    // 硬件与计算
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.fsd.v3.queue", qos: .userInteractive)
    nonisolated(unsafe) private let ciContext = CIContext()
    private let motion = CMMotionManager()
    private let locationManager = CLLocationManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    
    nonisolated(unsafe) private var safeModel: SafeModelBufferV3?
    private var prevPositions: [String: CGPoint] = [:]
    private var currentInput: AVCaptureDeviceInput?
    
    // 图像增强滤镜
    nonisolated(unsafe) private let colorFilter = CIFilter.colorControls()
    nonisolated(unsafe) private let exposureFilter = CIFilter.exposureAdjust()

    override init() {
        super.init()
        setupSensors()
        loadModel()
        setupCamera()
        haptic.prepare()
    }

    // MARK: - 传感器逻辑
    private func setupSensors() {
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        
        if motion.isDeviceMotionAvailable {
            motion.deviceMotionUpdateInterval = 1/60
            motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                // Pitch: 修正地平线 | YawRate: 驱动 3D 路径弯曲
                self.autoPitch = 0.05 * data.attitude.pitch + 0.95 * self.autoPitch
                self.yawRate = 0.15 * data.rotationRate.z + 0.85 * self.yawRate
            }
        }
    }

    private func loadModel() {
        Task {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            if let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let mDet = try? MLModel(contentsOf: url, configuration: config),
               let vDet = try? VNCoreMLModel(for: mDet) {
                self.safeModel = SafeModelBufferV3(detModel: vDet)
            }
        }
    }

    // MARK: - 夜视增强 (Night Vision)
    private func applyNightVision(to image: CIImage) -> CIImage {
        let extent = image.extent
        guard let avgFilter = CIFilter(name: "CIAreaAverage", parameters: [kCIInputImageKey: image, kCIInputExtentKey: CIVector(cgRect: extent)])?.outputImage else { return image }
        
        var bitmap = [UInt8](repeating: 0, count: 4)
        ciContext.render(avgFilter, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)
        
        let luminance = Float(bitmap[0]) / 255.0
        Task { @MainActor in self.isNightMode = luminance < 0.35 }
        
        if luminance < 0.35 {
            exposureFilter.inputImage = image
            exposureFilter.ev = 1.3
            colorFilter.inputImage = exposureFilter.outputImage
            colorFilter.contrast = 1.2
            return colorFilter.outputImage ?? image
        }
        return image
    }

    // MARK: - 硬件联动
    private func setupCamera() {
        switchCamera(to: .ultraWide)
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        
        let capture = self.session
        DispatchQueue.global(qos: .userInitiated).async { capture.startRunning() }
    }

    func updateSpeed(_ speed: Double) {
        self.currentSpeed = speed
        let target: ADASCameraMode = speed > 65.0 ? .telephoto : .ultraWide
        if target != cameraMode {
            switchCamera(to: target)
            self.cameraMode = target
        }
    }

    private func switchCamera(to mode: ADASCameraMode) {
        session.beginConfiguration()
        if let input = currentInput { session.removeInput(input) }
        let type: AVCaptureDevice.DeviceType = (mode == .telephoto) ? .builtInWideAngleCamera : .builtInUltraWideCamera
        guard let dev = AVCaptureDevice.default(type, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev) else {
            session.commitConfiguration()
            return
        }
        if session.canAddInput(input) {
            session.addInput(input)
            currentInput = input
        }
        session.commitConfiguration()
    }
}

// MARK: - 3. 核心算法流
extension FSDV3Engine: AVCaptureVideoDataOutputSampleBufferDelegate, CLLocationManagerDelegate {
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        let speed = max(0, loc.speed * 3.6)
        Task { @MainActor in self.updateSpeed(speed) }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer),
              let container = self.safeModel else { return }
        
        let modeTask = Task { @MainActor in self.cameraMode }
        let rawCI = CIImage(cvPixelBuffer: pixel)
        let processedCI = applyNightVision(to: rawCI)
        
        // OCR 识别
        let ocrReq = VNRecognizeTextRequest { req, _ in
            if let results = req.results as? [VNRecognizedTextObservation] {
                Task { @MainActor in
                    for obs in results {
                        if let top = obs.topCandidates(1).first, let val = Int(top.string.filter({$0.isNumber})), val >= 20 {
                            self.speedLimit = val
                        }
                    }
                }
            }
        }

        // FSD 检测与 ROI 裁切
        let detReq = VNCoreMLRequest(model: container.detModel) { req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeWorld(results) }
            }
        }
        
        Task {
            if await modeTask.value == .telephoto {
                detReq.regionOfInterest = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
            }
        }

        // 占用网络
        let segReq = VNGenerateForegroundInstanceMaskRequest()
        
        try? VNImageRequestHandler(ciImage: processedCI, orientation: .right).perform([ocrReq, detReq, segReq])
        
        if let maskObs = segReq.results?.first {
            if let cg = self.ciContext.createCGImage(CIImage(cvPixelBuffer: maskObs.instanceMask), from: CIImage(cvPixelBuffer: maskObs.instanceMask).extent) {
                Task { @MainActor in self.occupancyMask = cg }
            }
        }
    }

    @MainActor
    private func analyzeWorld(_ observations: [VNRecognizedObjectObservation]) {
        var nextObjs: [FSDV3Object] = []
        var sideAlert: Edge? = nil
        
        for obs in observations {
            let label = obs.labels.first?.identifier ?? "target"
            let center = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
            
            let prev = prevPositions[label] ?? center
            let vector = CGVector(dx: (center.x - prev.x) * 25, dy: (center.y - prev.y) * 25)
            prevPositions[label] = center
            
            let dist = 1.2 / (Float(obs.boundingBox.minY) + Float(autoPitch) * 0.5 + 0.05)
            let isSide = dist < 7.0 && (obs.boundingBox.midX < 0.25 || obs.boundingBox.midX > 0.75)
            let isCrit = dist < 12.0 && obs.boundingBox.midX > 0.4 && obs.boundingBox.midX < 0.6
            
            if isSide { sideAlert = obs.boundingBox.midX < 0.5 ? .leading : .trailing }
            if isCrit && vector.dy < -0.05 { haptic.impactOccurred() }
            
            nextObjs.append(FSDV3Object(id: UUID(), label: label, distance: dist, velocityVector: vector, boundingBox: obs.boundingBox, isSideHazard: isSide, isCritical: isCrit))
        }
        self.trackedObjects = nextObjs
        self.sideWarning = sideAlert
    }
}

// MARK: - 4. 终极 UI 界面 (Zen Path 版)
struct FSDMasterViewV3: View {
    @StateObject var engine = FSDV3Engine()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 4.1 占用网络 (蓝格底座)
            if let mask = engine.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            // 4.2 3D Zen Path (3D 路径规划线条)
            GeometryReader { geo in
                Path { path in
                    let w = geo.size.width
                    let h = geo.size.height
                    path.move(to: CGPoint(x: w * 0.5, y: h))
                    
                    let vY = h * (0.45 + engine.autoPitch * 0.1)
                    let vX = w * 0.5 + (engine.yawRate * 150)
                    
                    let c1 = CGPoint(x: w * 0.5 + (engine.yawRate * 450), y: h * 0.8)
                    let c2 = CGPoint(x: w * 0.5 + (engine.yawRate * 200), y: h * 0.6)
                    
                    path.addCurve(to: CGPoint(x: vX, y: vY), control1: c1, control2: c2)
                }
                .stroke(
                    LinearGradient(colors: [.cyan.opacity(0.8), .blue.opacity(0)], startPoint: .bottom, endPoint: .top),
                    style: StrokeStyle(lineWidth: 65, lineCap: .round, lineJoin: .round)
                )
                .blur(radius: 12)
                .blendMode(.screen)
                .opacity(engine.currentSpeed > 5 ? 0.65 : 0)
                .animation(.interactiveSpring(), value: engine.yawRate)
            }

            // 4.3 侧方流光报警
            HStack {
                Rectangle().fill(LinearGradient(colors: [.red.opacity(engine.sideWarning == .leading ? 0.8 : 0), .clear], startPoint: .leading, endPoint: .trailing)).frame(width: 85)
                Spacer()
                Rectangle().fill(LinearGradient(colors: [.clear, .red.opacity(engine.sideWarning == .trailing ? 0.8 : 0)], startPoint: .leading, endPoint: .trailing)).frame(width: 85)
            }.ignoresSafeArea()
            
            // 4.4 目标框与矢量预测
            GeometryReader { geo in
                ForEach(engine.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // 预测虚线
                    Path { p in
                        let start = CGPoint(x: rect.midX, y: geo.size.height - rect.minY)
                        p.move(to: start)
                        p.addLine(to: CGPoint(x: start.x + obj.velocityVector.dx * 100, y: start.y - obj.velocityVector.dy * 150))
                    }.stroke(Color.yellow, style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
                    
                    // 动态着色框
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(obj.isCritical ? Color.red : (obj.isSideHazard ? Color.orange : Color.white), lineWidth: 1.5)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.minY)
                }
            }
            
            // 4.5 HUD 控制面板
            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading) {
                        Text("\(Int(engine.currentSpeed))").font(.system(size: 55, weight: .black, design: .monospaced))
                        Text("KM/H").font(.caption)
                    }
                    Spacer()
                    if engine.speedLimit > 0 {
                        ZStack {
                            Circle().strokeBorder(Color.red, lineWidth: 5).background(Circle().fill(.white))
                            Text("\(engine.speedLimit)").font(.system(size: 20, weight: .bold)).foregroundColor(.black)
                        }.frame(width: 55, height: 55)
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Image(systemName: engine.isNightMode ? "moon.stars.fill" : (engine.cameraMode == .telephoto ? "scope" : "eye.circle"))
                        Text(engine.cameraMode.rawValue).font(.system(size: 8, weight: .bold))
                    }.foregroundColor(engine.isNightMode ? .yellow : (engine.cameraMode == .telephoto ? .yellow : .cyan))
                }
                .padding(30).background(Color.black.opacity(0.4)).foregroundColor(.white)
                Spacer()
            }
        }
    }
}

@main
struct ADASFinalApp: App {
    var body: some Scene { WindowGroup { FSDMasterViewV3() } }
}

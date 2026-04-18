@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreMotion
import CoreLocation

// MARK: - 1. 核心数据结构
struct FSDTrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var velocityVector: CGVector   // 运动矢量预测
    var boundingBox: CGRect
    var isSideHazard: Bool         // 侧方盲区预警
    var isCritical: Bool           // 正向碰撞预警
}

struct SafeModelBuffer: @unchecked Sendable {
    let model: VNCoreMLModel
}

enum ADASCameraMode: String, Sendable {
    case ultraWide = "CITY: WIDE-ANGLE (0.5x)"
    case telephoto = "HIGHWAY: FOCUS (1x + ROI)"
}

// MARK: - 2. 核心感知引擎
@MainActor
class FSDEngine: NSObject, ObservableObject {
    @Published var cameraMode: ADASCameraMode = .ultraWide
    @Published var trackedObjects: [FSDTrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var sideWarning: Edge? = nil
    @Published var currentSpeed: Double = 0.0
    @Published var speedLimit: Int = 0
    
    // 硬件实例
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.fsd.ultra.queue", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    private let motion = CMMotionManager()
    private let locationManager = CLLocationManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private let synthesizer = AVSpeechSynthesizer()
    
    // 算法缓存与状态
    nonisolated(unsafe) private var safeModel: SafeModelBuffer?
    private var prevPositions: [String: CGPoint] = [:]
    private var autoPitch: Double = 0.0
    private var currentInput: AVCaptureDeviceInput?
    private var lastVoiceAlert = Date.distantPast

    override init() {
        super.init()
        setupSensors()
        loadCoreML()
        setupCamera()
        haptic.prepare()
    }

    private func setupSensors() {
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        
        if motion.isDeviceMotionAvailable {
            motion.deviceMotionUpdateInterval = 1/60
            motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                // 自动 Pitch 校准：结合 IMU 修正支架偏移
                self.autoPitch = 0.1 * data.attitude.pitch + 0.9 * self.autoPitch
            }
        }
    }

    private func loadCoreML() {
        Task {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let mlModel = try? MLModel(contentsOf: url, configuration: config),
               let vModel = try? VNCoreMLModel(for: mlModel) {
                self.safeModel = SafeModelBuffer(model: vModel)
            }
        }
    }

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
        // 动态切换：高速 (>65km/h) 开启长焦/ROI，低速开启超广角
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

// MARK: - 3. FSD 算法逻辑集成
extension FSDEngine: AVCaptureVideoDataOutputSampleBufferDelegate, CLLocationManagerDelegate {
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        let speed = max(0, loc.speed * 3.6)
        Task { @MainActor in self.updateSpeed(speed) }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer),
              let container = self.safeModel else { return }
        
        let modeTask = Task { @MainActor in self.cameraMode }
        
        // A. 语义理解 (OCR)
        let ocrReq = VNRecognizeTextRequest { req, _ in
            if let results = req.results as? [VNRecognizedTextObservation] {
                Task { @MainActor in self.parseSpeed(results) }
            }
        }
        
        // B. 目标检测与预测 (FSD)
        let detReq = VNCoreMLRequest(model: container.model) { req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeFSD(results) }
            }
        }
        
        // 高速模式 ROI 裁切
        Task {
            if await modeTask.value == .telephoto {
                detReq.regionOfInterest = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
            }
        }

        // C. 占用网络 (Segmentation)
        let segReq = VNGenerateForegroundInstanceMaskRequest()
        
        try? VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .right).perform([ocrReq, detReq, segReq])
        
        if let maskObs = segReq.results?.first {
            let ci = CIImage(cvPixelBuffer: maskObs.instanceMask)
            if let cg = self.context.createCGImage(ci, from: ci.extent) {
                Task { @MainActor in self.occupancyMask = cg }
            }
        }
    }

    @MainActor
    private func parseSpeed(_ observations: [VNRecognizedTextObservation]) {
        for obs in observations {
            if let top = obs.topCandidates(1).first, let val = Int(top.string.filter({$0.isNumber})), val >= 20 {
                self.speedLimit = val
            }
        }
    }

    @MainActor
    private func analyzeFSD(_ observations: [VNRecognizedObjectObservation]) {
        var nextObjs: [FSDTrackedObject] = []
        var sideAlert: Edge? = nil
        
        for obs in observations {
            let label = obs.labels.first?.identifier ?? "target"
            let center = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
            
            // 时序预测矢量
            let prev = prevPositions[label] ?? center
            let vector = CGVector(dx: (center.x - prev.x) * 20, dy: (center.y - prev.y) * 20)
            prevPositions[label] = center
            
            // 测距与预判
            let dist = 1.2 / (Float(obs.boundingBox.minY) + Float(autoPitch) * 0.5 + 0.05)
            let isSide = dist < 6.0 && (obs.boundingBox.midX < 0.22 || obs.boundingBox.midX > 0.78)
            let isCrit = dist < 12.0 && obs.boundingBox.midX > 0.4 && obs.boundingBox.midX < 0.6
            
            if isSide { sideAlert = obs.boundingBox.midX < 0.5 ? .leading : .trailing }
            if isCrit && vector.dy < -0.04 { // 幽灵刹车预警 (Y轴加速度异常)
                haptic.impactOccurred()
                triggerVoice("注意距离")
            }
            
            nextObjs.append(FSDTrackedObject(id: UUID(), label: label, distance: dist, velocityVector: vector, boundingBox: obs.boundingBox, isSideHazard: isSide, isCritical: isCrit))
        }
        self.trackedObjects = nextObjs
        self.sideWarning = sideAlert
    }
    
    private func triggerVoice(_ msg: String) {
        if Date().timeIntervalSince(lastVoiceAlert) > 5.0 {
            let utt = AVSpeechUtterance(string: msg)
            utt.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            synthesizer.speak(utt)
            lastVoiceAlert = Date()
        }
    }
}

// MARK: - 4. 终极 FSD UI 界面
struct FSDMasterView: View {
    @StateObject var engine = FSDEngine()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 占用网络
            if let mask = engine.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            // 侧方流光预警
            HStack {
                Rectangle().fill(LinearGradient(colors: [.red.opacity(engine.sideWarning == .leading ? 0.8 : 0), .clear], startPoint: .leading, endPoint: .trailing)).frame(width: 80)
                Spacer()
                Rectangle().fill(LinearGradient(colors: [.clear, .red.opacity(engine.sideWarning == .trailing ? 0.8 : 0)], startPoint: .leading, endPoint: .trailing)).frame(width: 80)
            }.ignoresSafeArea()
            
            GeometryReader { geo in
                ForEach(engine.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // FSD 预测虚线
                    Path { p in
                        let start = CGPoint(x: rect.midX, y: geo.size.height - rect.minY)
                        p.move(to: start)
                        p.addLine(to: CGPoint(x: start.x + obj.velocityVector.dx * 100, y: start.y - obj.velocityVector.dy * 150))
                    }
                    .stroke(Color.yellow, style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
                    
                    // 目标框
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(obj.isCritical ? Color.red : (obj.isSideHazard ? Color.orange : Color.white), lineWidth: 1.5)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // HUD
            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading) {
                        Text("\(Int(engine.currentSpeed))").font(.system(size: 50, weight: .black, design: .monospaced))
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
                        Image(systemName: engine.cameraMode == .telephoto ? "scope" : "eye.circle")
                        Text(engine.cameraMode.rawValue).font(.system(size: 8, weight: .bold))
                    }.foregroundColor(engine.cameraMode == .telephoto ? .yellow : .cyan)
                }
                .padding(30).background(LinearGradient(colors: [.black.opacity(0.5), .clear], startPoint: .top, endPoint: .bottom)).foregroundColor(.white)
                Spacer()
            }
        }
    }
}

@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { FSDMasterView() } }
}

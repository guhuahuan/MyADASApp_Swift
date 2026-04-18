@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreImage
import CoreMotion

// MARK: - 数据结构
struct SafeModelContainer: @unchecked Sendable {
    let visionModel: VNCoreMLModel
}

struct TrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var ttc: Float
    var isCutIn: Bool
    var boundingBox: CGRect
}

struct TrajectoryPath: Sendable {
    let points: [CGPoint]
}

// MARK: - ADAS 核心引擎
@MainActor
class ADASMasterViewModel: NSObject, ObservableObject {
    enum AlertLevel { case safe, warning, critical }
    
    @Published var trackedObjects: [TrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var alertStatus: AlertLevel = .safe
    @Published var currentPath: TrajectoryPath?
    @Published var debugInfo: String = ""
    
    private let captureSession = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.adas.ultra.process", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    nonisolated(unsafe) private var modelContainer: SafeModelContainer?
    private let synthesizer = AVSpeechSynthesizer()
    private var lastAlertTime = Date.distantPast

    // Core Motion 传感器
    private let motionManager = CMMotionManager()
    private var baselinePitch: Double = 0.0
    private var smoothPitchOffset: Double = 0.0
    private let alpha: Double = 0.15 

    override init() {
        super.init()
        setupSystem()
        startMotionUpdates()
    }

    // MARK: - 1. 系统初始化
    private func setupSystem() {
        Task {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let coreML = try? MLModel(contentsOf: modelURL, configuration: config),
               let visionModel = try? VNCoreMLModel(for: coreML) {
                self.modelContainer = SafeModelContainer(visionModel: visionModel)
            }
            
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
            
            DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
        }
    }

    // MARK: - 2. IMU 姿态补偿 (动态测距核心)
    private func startMotionUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                let pitch = data.attitude.pitch
                if self.baselinePitch == 0.0 { self.baselinePitch = pitch }
                let rawOffset = pitch - self.baselinePitch
                self.smoothPitchOffset = (self.alpha * rawOffset) + (1.0 - self.alpha) * self.smoothPitchOffset
            }
        }
    }

    // MARK: - 3. 3D 轨迹线投影 (Trajectory)
    func updateTrajectory() {
        let pitchAdj = Float(self.smoothPitchOffset)
        let f: Float = 0.85 
        let h_cam: Float = 1.2 

        var points: [CGPoint] = []
        for d in stride(from: 3.0, through: 30.0, by: 3.0) {
            let y_norm = 1.0 - ((h_cam / Float(d)) * f + pitchAdj * 0.6)
            let laneWidth = (3.5 / Float(d)) * f
            points.append(CGPoint(x: CGFloat(0.5 - laneWidth/2), y: CGFloat(y_norm)))
            points.append(CGPoint(x: CGFloat(0.5 + laneWidth/2), y: CGFloat(y_norm)))
        }
        self.currentPath = TrajectoryPath(points: points)
    }
}

// MARK: - 4. 视频流回调与感知算法整合
extension ADASMasterViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let segRequest = VNGenerateForegroundInstanceMaskRequest()
        guard let container = self.modelContainer else { return }
        
        let detRequest = VNCoreMLRequest(model: container.visionModel) { request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeEnvironment(results) }
            }
        }
        detRequest.imageCropAndScaleOption = .scaleFill
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([segRequest, detRequest])
        
        // 处理占用网络遮罩
        if let observation = segRequest.results?.first {
            let ciMask = CIImage(cvPixelBuffer: observation.instanceMask)
            if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                Task { @MainActor in 
                    self.occupancyMask = cgMask 
                    self.updateTrajectory()
                }
            }
        }
    }

    @MainActor
    private func analyzeEnvironment(_ observations: [VNRecognizedObjectObservation]) {
        var currentLevel: AlertLevel = .safe
        var newObjects: [TrackedObject] = []
        let pitchAdj = Float(self.smoothPitchOffset)

        for obs in observations {
            let label = obs.labels.first?.identifier ?? "Object"
            let xPos = Float(obs.boundingBox.midX)
            let yPosCompensated = Float(obs.boundingBox.origin.y) + (pitchAdj * 0.6)
            
            // 测距与 TTC 计算 (模拟)
            let dist = 1.2 / (yPosCompensated + 0.05)
            let ttc = dist / 15.0 // 简化模型：假设相对速度 15m/s
            
            // 侧向切入逻辑
            let isCutIn = (xPos < 0.25 || xPos > 0.75) && dist < 8.0
            
            newObjects.append(TrackedObject(id: UUID(), label: label, distance: dist, ttc: ttc, isCutIn: isCutIn, boundingBox: obs.boundingBox))

            // 预警决策
            if (dist < 4.0 || ttc < 1.8 || isCutIn) && (label == "car" || label == "motorcycle" || label == "person") {
                currentLevel = .critical
                voiceAlert(msg: isCutIn ? "侧方切入" : "注意距离")
            } else if dist < 8.0 {
                if currentLevel != .critical { currentLevel = .warning }
            }
        }
        
        self.trackedObjects = newObjects
        self.alertStatus = currentLevel
        self.debugInfo = "PITCH: \(String(format: "%.2f", smoothPitchOffset)) | OBJ: \(newObjects.count)"
    }

    private func voiceAlert(msg: String) {
        if Date().timeIntervalSince(lastAlertTime) > 4.0 {
            let utterance = AVSpeechUtterance(string: msg)
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            synthesizer.speak(utterance)
            lastAlertTime = Date()
        }
    }
}

// MARK: - 5. 视图层 (Tesla 风格 UI)
struct ADASView: View {
    @StateObject var viewModel = ADASMasterViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 占用网络层
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(viewModel.alertStatus == .critical ? .red.opacity(0.4) : .cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            GeometryReader { geo in
                // 3D 轨迹线
                if let path = viewModel.currentPath {
                    Path { p in
                        let w = geo.size.width
                        let h = geo.size.height
                        p.move(to: CGPoint(x: path.points[0].x * w, y: path.points[0].y * h))
                        for i in stride(from: 2, to: path.points.count, by: 2) { p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h)) }
                        p.move(to: CGPoint(x: path.points[1].x * w, y: path.points[1].y * h))
                        for i in stride(from: 3, to: path.points.count, by: 2) { p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h)) }
                    }
                    .stroke(viewModel.alertStatus == .critical ? Color.red : Color.cyan, lineWidth: 3)
                }
                
                // 目标框与距离显示
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let color: Color = obj.isCutIn ? .orange : (obj.distance < 5 ? .red : .green)
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(String(format: "%.1fm", obj.distance))")
                            .font(.system(size: 10, weight: .black, design: .monospaced))
                            .padding(2).background(color).foregroundColor(.white)
                        Rectangle().stroke(color, lineWidth: 2).frame(width: rect.width, height: rect.height)
                    }
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 底部数据面板
            VStack {
                Spacer()
                HStack {
                    Text(viewModel.debugInfo).foregroundColor(.cyan)
                    Spacer()
                    Circle().fill(viewModel.alertStatus == .critical ? Color.red : Color.green).frame(width: 10, height: 10)
                    Text(viewModel.alertStatus == .critical ? "DANGER" : "SYSTEM READY")
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .padding().background(.ultraThinMaterial)
            }
        }
    }
}

@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { ADASView() } }
}

@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreImage
import CoreMotion

// MARK: - 基础模型
struct SafeModelContainer: @unchecked Sendable {
    let visionModel: VNCoreMLModel
}

struct TrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var boundingBox: CGRect
}

struct TrajectoryPath: Sendable {
    let points: [CGPoint]
}

// MARK: - ADAS 主引擎
@MainActor
class ADASMasterViewModel: NSObject, ObservableObject {
    enum AlertLevel { case safe, warning, critical }
    
    @Published var trackedObjects: [TrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var alertStatus: AlertLevel = .safe
    @Published var currentPath: TrajectoryPath?
    @Published var pitchDebug: Double = 0.0
    
    private let captureSession = AVCaptureSession()
    private let queue = DispatchQueue(label: "adas.final.queue", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    nonisolated(unsafe) private var modelContainer: SafeModelContainer?
    private let synthesizer = AVSpeechSynthesizer()
    
    // Core Motion 属性
    private let motionManager = CMMotionManager()
    private var baselinePitch: Double = 0.0
    private var smoothPitchOffset: Double = 0.0
    private let alpha: Double = 0.15 

    override init() {
        super.init()
        setupSystem()
        startMotionUpdates()
    }

    // MARK: - 系统初始化
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

    // MARK: - IMU 姿态补偿逻辑
    private func startMotionUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                let currentPitch = data.attitude.pitch
                if self.baselinePitch == 0.0 { self.baselinePitch = currentPitch }
                let rawOffset = currentPitch - self.baselinePitch
                self.smoothPitchOffset = (self.alpha * rawOffset) + (1.0 - self.alpha) * self.smoothPitchOffset
                self.pitchDebug = self.smoothPitchOffset
            }
        }
    }

    // MARK: - 3D 轨迹线投影算法 (核心 Feature)
    func updateTrajectoryProjection() {
        let pitchAdjustment = Float(self.smoothPitchOffset)
        let f: Float = 0.8 
        let cameraHeight: Float = 1.2 

        var projectedPoints: [CGPoint] = []
        for distance in stride(from: 2.0, through: 25.0, by: 2.5) {
            let d = Float(distance)
            let worldY = (cameraHeight / d) * f
            let compensatedY = worldY + (pitchAdjustment * 0.6)
            let normalizedY = 1.0 - compensatedY 
            
            let laneWidthAtDist = (3.5 / d) * f
            let leftX = 0.5 - (laneWidthAtDist / 2.0)
            let rightX = 0.5 + (laneWidthAtDist / 2.0)
            
            projectedPoints.append(CGPoint(x: CGFloat(leftX), y: CGFloat(normalizedY)))
            projectedPoints.append(CGPoint(x: CGFloat(rightX), y: CGFloat(normalizedY)))
        }
        self.currentPath = TrajectoryPath(points: projectedPoints)
    }
}

// MARK: - 视频流回调处理
extension ADASMasterViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let segRequest = VNGenerateForegroundInstanceMaskRequest()
        
        guard let container = self.modelContainer else { return }
        let detRequest = VNCoreMLRequest(model: container.visionModel) { request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeWorld(results) }
            }
        }
        detRequest.imageCropAndScaleOption = .scaleFill
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([segRequest, detRequest])
        
        if let observation = segRequest.results?.first {
            let maskBuffer = observation.instanceMask
            let ciMask = CIImage(cvPixelBuffer: maskBuffer)
            if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                Task { @MainActor in 
                    self.occupancyMask = cgMask 
                    self.updateTrajectoryProjection() // 每帧同步更新轨迹
                }
            }
        }
    }

    @MainActor
    private func analyzeWorld(_ observations: [VNRecognizedObjectObservation]) {
        var currentAlert: AlertLevel = .safe
        var newObjects: [TrackedObject] = []
        let pitchAdjustment = Float(self.smoothPitchOffset)

        for obs in observations {
            let label = obs.labels.first?.identifier ?? "Target"
            let compensatedY = Float(obs.boundingBox.origin.y) + (pitchAdjustment * 0.6)
            let dist = 1.2 / (compensatedY + 0.05)
            
            newObjects.append(TrackedObject(id: UUID(), label: label, distance: dist, boundingBox: obs.boundingBox))

            if dist < 3.5 && (label == "car" || label == "motorcycle" || label == "person") {
                currentAlert = .critical
            } else if dist < 7.0 {
                if currentAlert != .critical { currentAlert = .warning }
            }
        }
        self.trackedObjects = newObjects
        self.alertStatus = currentAlert
    }
}

// MARK: - UI 视图
struct ContentView: View {
    @StateObject var viewModel = ADASMasterViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 占用网络层
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(viewModel.alertStatus == .critical ? .red.opacity(0.4) : .cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            // 2. 3D 轨迹线投影层
            GeometryReader { geo in
                if let path = viewModel.currentPath {
                    Path { p in
                        let w = geo.size.width
                        let h = geo.size.height
                        
                        // 绘制左侧线
                        if path.points.count > 1 {
                            p.move(to: CGPoint(x: path.points[0].x * w, y: path.points[0].y * h))
                            for i in stride(from: 2, to: path.points.count, by: 2) {
                                p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h))
                            }
                            
                            // 绘制右侧线
                            p.move(to: CGPoint(x: path.points[1].x * w, y: path.points[1].y * h))
                            for i in stride(from: 3, to: path.points.count, by: 2) {
                                p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h))
                            }
                        }
                    }
                    .stroke(viewModel.alertStatus == .critical ? Color.red : Color.cyan, lineWidth: 3)
                }
                
                // 3. 物体检测框
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let color: Color = obj.distance < 4 ? .red : .green
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(String(format: "%.1f", obj.distance))M")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .padding(2)
                            .background(color)
                            .foregroundColor(.white)
                        Rectangle().stroke(color, lineWidth: 2).frame(width: rect.width, height: rect.height)
                    }
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 底部信息栏
            VStack {
                Spacer()
                HStack {
                    Text("IMU: \(String(format: "%.2f", viewModel.pitchDebug))").foregroundColor(.yellow)
                    Spacer()
                    Text("ADAS ACTIVE").bold().foregroundColor(.cyan)
                }
                .font(.caption2.monospaced())
                .padding().background(.ultraThinMaterial)
            }
        }
    }
}

@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

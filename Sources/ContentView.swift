@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreMotion
import CoreLocation

// MARK: - 1. FSD 增强型数据结构
struct FSDTrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var velocityVector: CGVector // 模拟 FSD 路径预测
    var boundingBox: CGRect
    var isSideHazard: Bool
}

// MARK: - 2. 线程安全容器
struct ModelBuffer: @unchecked Sendable {
    let model: VNCoreMLModel
}

// MARK: - 3. FSD 核心引擎
@MainActor
class FSDEngine: NSObject, ObservableObject {
    @Published var trackedObjects: [FSDTrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var sideWarning: Edge? = nil // 用于流光预警
    @Published var currentSpeed: Double = 0.0
    @Published var autoPitch: Double = 0.0
    
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.fsd.vision.queue", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    private let motion = CMMotionManager()
    
    // 状态追踪缓存
    nonisolated(unsafe) private var safeModel: ModelBuffer?
    private var prevPositions: [String: CGPoint] = [:]
    private var lastFrameTime = Date()

    override init() {
        super.init()
        setupHardware()
        loadFSDModel()
    }

    private func loadFSDModel() {
        Task {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let mlModel = try? MLModel(contentsOf: url, configuration: config),
               let vModel = try? VNCoreMLModel(for: mlModel) {
                self.safeModel = ModelBuffer(model: vModel)
            }
        }
    }

    private func setupHardware() {
        // IMU 自动校准
        if motion.isDeviceMotionAvailable {
            motion.deviceMotionUpdateInterval = 1/60
            motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                // FSD 自愈逻辑：结合 IMU 动态修正地平线
                self.autoPitch = 0.1 * data.attitude.pitch + 0.9 * self.autoPitch
            }
        }
        
        Task {
            guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: dev) else { return }
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: queue)
            if session.canAddInput(input) { session.addInput(input) }
            if session.canAddOutput(output) { session.addOutput(output) }
            
            let capture = self.session
            DispatchQueue.global(qos: .userInitiated).async { capture.startRunning() }
        }
    }
}

// MARK: - 4. 预测与感知算法 (FSD Logic)
extension FSDEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer),
              let container = self.safeModel else { return }
        
        let segReq = VNGenerateForegroundInstanceMaskRequest()
        let detReq = VNCoreMLRequest(model: container.model) { req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeFSD(results) }
            }
        }
        
        try? VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .right).perform([segReq, detReq])
        
        if let maskObs = segReq.results?.first {
            if let cgMask = try? self.context.createCGImage(CIImage(cvPixelBuffer: maskObs.instanceMask), from: CIImage(cvPixelBuffer: maskObs.instanceMask).extent) {
                Task { @MainActor in self.occupancyMask = cgMask }
            }
        }
    }

    @MainActor
    private func analyzeFSD(_ observations: [VNRecognizedObjectObservation]) {
        var nextObjects: [FSDTrackedObject] = []
        var sideAlert: Edge? = nil
        
        for obs in observations {
            let label = obs.labels.first?.identifier ?? "obj"
            let currentCenter = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
            
            // A. 运动矢量计算
            let prev = prevPositions[label] ?? currentCenter
            let vector = CGVector(dx: (currentCenter.x - prev.x) * 15, dy: (currentCenter.y - prev.y) * 15)
            prevPositions[label] = currentCenter
            
            // B. 测距 (含自适应 Pitch)
            let dist = 1.2 / (Float(obs.boundingBox.minY) + Float(autoPitch) * 0.5 + 0.05)
            
            // C. 边缘流光预警判断 (特斯拉盲区逻辑)
            if dist < 6.0 {
                if obs.boundingBox.midX < 0.2 { sideAlert = .leading }
                if obs.boundingBox.midX > 0.8 { sideAlert = .trailing }
            }
            
            nextObjects.append(FSDTrackedObject(id: UUID(), label: label, distance: dist, velocityVector: vector, boundingBox: obs.boundingBox, isSideHazard: dist < 5.0))
        }
        self.trackedObjects = nextObjects
        self.sideWarning = sideAlert
    }
}

// MARK: - 5. FSD 视觉界面
struct FSDContentView: View {
    @StateObject var engine = FSDEngine()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 占用网络格栅 (Occupancy Grid)
            if let mask = engine.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.blue.opacity(0.25))
                    .ignoresSafeArea()
                    .overlay(
                        // 语义化增强：格栅化视觉
                        Rectangle().stroke(Color.blue.opacity(0.1), lineWidth: 0.5)
                    )
            }
            
            // 侧方流光预警 (Tesla Blind Spot Visual)
            HStack {
                Rectangle().fill(LinearGradient(colors: [.red.opacity(engine.sideWarning == .leading ? 0.7 : 0), .clear], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 80)
                Spacer()
                Rectangle().fill(LinearGradient(colors: [.clear, .red.opacity(engine.sideWarning == .trailing ? 0.7 : 0)], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 80)
            }.ignoresSafeArea()
            
            GeometryReader { geo in
                ForEach(engine.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // 运动预测矢量线
                    Path { p in
                        let start = CGPoint(x: rect.midX, y: geo.size.height - rect.minY)
                        p.move(to: start)
                        p.addLine(to: CGPoint(x: start.x + obj.velocityVector.dx * 100, y: start.y - obj.velocityVector.dy * 100))
                    }
                    .stroke(Color.yellow, style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
                    
                    // 3D 风格框
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(obj.isSideHazard ? Color.red : Color.white, lineWidth: 1.5)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // FSD HUD
            VStack {
                HStack {
                    Text("FSD CALIBRATED").font(.system(size: 12, weight: .bold, design: .monospaced)).foregroundColor(.blue)
                    Spacer()
                }.padding()
                Spacer()
            }
        }
    }
}

// MARK: - 6. 程序入口 (修复 _main 错误)
@main
struct ADASApp: App {
    var body: some Scene {
        WindowGroup {
            FSDContentView()
        }
    }
}

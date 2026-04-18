@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreImage
import CoreMotion

// MARK: - 线程安全容器
struct SafeModelContainer: @unchecked Sendable {
    let visionModel: VNCoreMLModel
}

struct TrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var boundingBox: CGRect
}

// MARK: - ADAS 核心引擎
@MainActor
class ADASMasterViewModel: NSObject, ObservableObject {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var alertStatus: AlertLevel = .safe
    @Published var pitchDebug: Double = 0.0 // 用于调试实时俯仰角
    
    enum AlertLevel { case safe, warning, critical }

    private let captureSession = AVCaptureSession()
    private let queue = DispatchQueue(label: "adas.worker.queue", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    nonisolated(unsafe) private var modelContainer: SafeModelContainer?
    private let synthesizer = AVSpeechSynthesizer()
    
    // Core Motion 动力
    private let motionManager = CMMotionManager()
    private var baselinePitch: Double = 0.0
    private var smoothPitchOffset: Double = 0.0
    private let alpha: Double = 0.15 // 低通滤波系数
    
    override init() {
        super.init()
        setupSystem()
        startMotionUpdates()
    }

    // MARK: - IMU 姿态感知
    private func startMotionUpdates() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0 // 60Hz 采样，保证平滑
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                
                let currentPitch = data.attitude.pitch
                if self.baselinePitch == 0.0 { self.baselinePitch = currentPitch }
                
                // 实时偏差计算
                let rawOffset = currentPitch - self.baselinePitch
                
                // 关键：低通滤波处理，过滤掉引擎高频振动，保留路面颠簸趋势
                self.smoothPitchOffset = (self.alpha * rawOffset) + (1.0 - self.alpha) * self.smoothPitchOffset
                self.pitchDebug = self.smoothPitchOffset
            }
        }
    }

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

    // 重置校准基准 (当你把手机卡在架子上停稳后点一下)
    func resetCalibration() {
        self.baselinePitch = 0.0
    }
}

// MARK: - 视觉推理与补偿算法
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
        
        // 占用网络渲染
        if let observation = segRequest.results?.first {
            let maskBuffer = observation.instanceMask
            let ciMask = CIImage(cvPixelBuffer: maskBuffer)
            if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                Task { @MainActor in self.occupancyMask = cgMask }
            }
        }
    }

    @MainActor
    private func analyzeWorld(_ observations: [VNRecognizedObjectObservation]) {
        var currentAlert: AlertLevel = .safe
        var newObjects: [TrackedObject] = []
        
        // 获取实时 IMU 补偿值
        let pitchAdjustment = Float(self.smoothPitchOffset)

        for obs in observations {
            let label = obs.labels.first?.identifier ?? "Target"
            
            // 原始 Y 轴中心点
            let originalY = Float(obs.boundingBox.origin.y)
            
            // --- 核心补偿公式 ---
            // 1. 如果手机向上仰（pitch > 0），视觉上物体会下移，y变小，我们需要加上这个偏移
            // 2. 0.6 是针对 iPhone 镜头焦距的转换增益系数
            let compensatedY = originalY + (pitchAdjustment * 0.6)
            
            // 测距算法 (H=1.2米)
            let dist = 1.2 / (compensatedY + 0.05)
            
            let tracked = TrackedObject(id: UUID(), label: label, distance: dist, boundingBox: obs.boundingBox)
            newObjects.append(tracked)

            // TTC 与报警逻辑
            if dist < 3.0 && (label == "car" || label == "motorcycle") {
                currentAlert = .critical
                triggerVoiceAlert(label: label, d: dist)
            } else if dist < 6.0 {
                currentAlert = .warning
            }
        }
        
        self.trackedObjects = newObjects
        self.alertStatus = currentAlert
    }

    private func triggerVoiceAlert(label: String, d: Float) {
        if synthesizer.isSpeaking { return }
        let msg = "注意前方 \(Int(d)) 米 \(label == "motorcycle" ? "摩托车" : "汽车")"
        let utterance = AVSpeechUtterance(string: msg)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = 0.55
        synthesizer.speak(utterance)
    }
}

// MARK: - UI 界面
struct ContentView: View {
    @StateObject var viewModel = ADASMasterViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 占用网络
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(viewModel.alertStatus == .critical ? .red.opacity(0.4) : .cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            // 检测框渲染
            GeometryReader { geo in
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let color: Color = obj.distance < 4 ? .red : .green
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(String(format: "%.1f", obj.distance))M")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .padding(2)
                            .background(color)
                            .foregroundColor(.white)
                        
                        Rectangle()
                            .stroke(color, lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                    }
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 顶部调试信息
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("IMU PITCH: \(String(format: "%.3f", viewModel.pitchDebug))")
                        Text("CALIBRATION: ACTIVE")
                    }
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.yellow)
                    Spacer()
                    Button("校准 (RESET)") {
                        viewModel.resetCalibration()
                    }
                    .font(.system(size: 10, weight: .bold))
                    .padding(8)
                    .background(Color.yellow)
                    .foregroundColor(.black)
                    .cornerRadius(5)
                }
                .padding()
                Spacer()
            }
            
            // 碰撞警告视觉
            if viewModel.alertStatus == .critical {
                VStack {
                    Text("⚠️ BRAKE ⚠️")
                        .font(.system(size: 60, weight: .black))
                        .foregroundColor(.red)
                        .padding()
                        .background(Color.white.opacity(0.8))
                        .cornerRadius(20)
                }
            }
        }
    }
}

// MARK: - 入口
@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

import SwiftUI
import Vision
import AVFoundation
import CoreImage

// MARK: - 线程安全容器
struct SafeModelContainer: Sendable {
    let visionModel: VNCoreMLModel
}

@MainActor
class ADASUltraViewModel: NSObject, ObservableObject {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var isCriticalWarning: Bool = false
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.ultra.worker", qos: .userInteractive)
    private let context = CIContext()
    private let synthesizer = AVSpeechSynthesizer()
    private var lastAlertTime = Date.distantPast

    // 使用 nonisolated(unsafe) 确保后台线程可以访问模型进行推理
    nonisolated(unsafe) private var modelContainer: SafeModelContainer?

    override init() {
        super.init()
        setupSystem()
    }

    private func setupSystem() {
        Task {
            // 1. 加载并编译 YOLOv8L
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let coreML = try? MLModel(contentsOf: modelURL, configuration: config),
               let visionModel = try? VNCoreMLModel(for: coreML) {
                self.modelContainer = SafeModelContainer(visionModel: visionModel)
            }

            // 2. 配置相机输入
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(videoDataOutput) { captureSession.addOutput(videoDataOutput) }
            
            // 3. 异步启动相机 (解决 Swift 6 警告)
            let session = self.captureSession
            Task.detached {
                await session.startRunning()
            }
        }
    }
}

// MARK: - 后台处理引擎
extension ADASUltraViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        var requests: [VNRequest] = []

        // 功能 3: 实例分割 (用于构建占用网络)
        let segmentationRequest = VNGenerateForegroundInstanceMaskRequest()
        requests.append(segmentationRequest)

        // 功能 1: YOLO 检测
        if let container = self.modelContainer {
            let detectionRequest = VNCoreMLRequest(model: container.visionModel) { request, _ in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    Task { @MainActor in self.processDetections(results) }
                }
            }
            detectionRequest.imageCropAndScaleOption = .scaleFill
            requests.append(detectionRequest)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform(requests)

        // 修复功能 3：正确获取分割遮罩
        if let observation = segmentationRequest.results?.first {
            // 在 iOS 17+ 中，我们直接使用 instanceMask
            let maskBuffer = observation.instanceMask
            let ciMask = CIImage(cvPixelBuffer: maskBuffer)
            if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                Task { @MainActor in
                    self.occupancyMask = cgMask
                }
            }
        }
    }

    @MainActor
    private func processDetections(_ observations: [VNRecognizedObjectObservation]) {
        var newObjects: [TrackedObject] = []
        var danger = false

        for obs in observations {
            // 简单的几何测距公式
            let dist = 1.2 / (Float(obs.boundingBox.origin.y) + 0.05)
            let label = obs.labels.first?.identifier ?? "Object"
            
            newObjects.append(TrackedObject(id: UUID(), label: label, distance: dist, boundingBox: obs.boundingBox))

            // 功能 4: 语音预警 (3米内且是车/人)
            if dist < 3.0 && (label == "car" || label == "motorcycle" || label == "person") {
                danger = true
                triggerVoice(label: label, d: dist)
            }
        }
        self.trackedObjects = newObjects
        self.isCriticalWarning = danger
    }

    @MainActor
    private func triggerVoice(label: String, d: Float) {
        if Date().timeIntervalSince(lastAlertTime) > 5.0 {
            let msg = "注意，\(Int(d))米内有\(label == "motorcycle" ? "摩托车" : "障碍物")"
            let utterance = AVSpeechUtterance(string: msg)
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            utterance.rate = 0.55
            synthesizer.speak(utterance)
            lastAlertTime = Date()
        }
    }
}

// MARK: - 模型定义
struct TrackedObject: Identifiable {
    let id: UUID
    var label: String
    var distance: Float
    var boundingBox: CGRect
}

// MARK: - 特斯拉 SFD 风格 UI
struct ContentView: View {
    @StateObject var viewModel = ADASUltraViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 占用网络 (Occupancy Mask) 渲染层
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.35)) // 用特斯拉标志性的科技蓝表示占用空间
                    .ignoresSafeArea()
            }
            
            // 2. 目标检测与距离层
            GeometryReader { geo in
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let color: Color = obj.distance < 4 ? .red : .green
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(Int(obj.distance))M")
                            .font(.system(size: 10, weight: .black, design: .monospaced))
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
            
            // 3. 碰撞视觉警告
            if viewModel.isCriticalWarning {
                Color.red.opacity(0.3).ignoresSafeArea()
                Text("BRAKE")
                    .font(.system(size: 80, weight: .black))
                    .foregroundColor(.white)
                    .italic()
            }
            
            // 4. 底部状态仪表
            VStack {
                Spacer()
                HStack {
                    Text("VISION: YOLOv8L")
                    Spacer()
                    Text("SFD: ACTIVE")
                }
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundColor(.cyan)
                .padding()
                .background(.ultraThinMaterial)
            }
        }
    }
}

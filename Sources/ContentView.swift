@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreImage

// MARK: - 线程安全容器 (解决 Sendable 报错)
struct SafeModelContainer: @unchecked Sendable {
    let visionModel: VNCoreMLModel
}

// MARK: - 追踪对象模型
struct TrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var boundingBox: CGRect
}

// MARK: - ADAS 深度进化引擎
@MainActor
class ADASUltraViewModel: NSObject, ObservableObject {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var isCriticalWarning: Bool = false
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.ultra.worker", qos: .userInteractive)
    
    // 关键修复：CIContext 不再属于 MainActor，允许后台线程直接使用
    nonisolated(unsafe) private let context = CIContext()
    
    private let synthesizer = AVSpeechSynthesizer()
    private var lastAlertTime = Date.distantPast

    // 关键修复：模型容器
    nonisolated(unsafe) private var modelContainer: SafeModelContainer?

    override init() {
        super.init()
        setupSystem()
    }

    private func setupSystem() {
        Task {
            // 1. 加载 YOLOv8L
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let coreML = try? MLModel(contentsOf: modelURL, configuration: config),
               let visionModel = try? VNCoreMLModel(for: coreML) {
                self.modelContainer = SafeModelContainer(visionModel: visionModel)
            }

            // 2. 配置相机
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(videoDataOutput) { captureSession.addOutput(videoDataOutput) }
            
            // 3. 启动相机 (修复异步调用警告)
            let session = self.captureSession
            DispatchQueue.global(qos: .userInitiated).async {
                session.startRunning()
            }
        }
    }
}

// MARK: - 后台推理引擎
extension ADASUltraViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let requests: [VNRequest]
        
        // 1. 准备请求
        let segmentationRequest = VNGenerateForegroundInstanceMaskRequest()
        
        if let container = self.modelContainer {
            let detectionRequest = VNCoreMLRequest(model: container.visionModel) { request, _ in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    Task { @MainActor in self.processDetections(results) }
                }
            }
            detectionRequest.imageCropAndScaleOption = .scaleFill
            requests = [segmentationRequest, detectionRequest]
        } else {
            requests = [segmentationRequest]
        }

        // 2. 执行推理
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform(requests)

        // 3. 占用网络提取 (修复 CIContext Actor 隔离报错)
        if let observation = segmentationRequest.results?.first {
            let maskBuffer = observation.instanceMask
            let ciMask = CIImage(cvPixelBuffer: maskBuffer)
            // 在此后台线程使用 context，因为 context 标记为 nonisolated(unsafe)
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
            let dist = 1.2 / (Float(obs.boundingBox.origin.y) + 0.05)
            let label = obs.labels.first?.identifier ?? "Object"
            
            newObjects.append(TrackedObject(id: UUID(), label: label, distance: dist, boundingBox: obs.boundingBox))

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
            let labelCN = (label == "motorcycle" ? "摩托车" : (label == "car" ? "汽车" : "行人"))
            let msg = "注意前方 \(Int(d)) 米有 \(labelCN)"
            let utterance = AVSpeechUtterance(string: msg)
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            utterance.rate = 0.55
            synthesizer.speak(utterance)
            lastAlertTime = Date()
        }
    }
}

// MARK: - 特斯拉风格 UI
struct ADASView: View {
    @StateObject var viewModel = ADASUltraViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.35))
                    .ignoresSafeArea()
            }
            
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
            
            if viewModel.isCriticalWarning {
                Color.red.opacity(0.3).ignoresSafeArea()
                Text("BRAKE")
                    .font(.system(size: 80, weight: .black))
                    .foregroundColor(.white)
                    .italic()
            }
        }
    }
}

// MARK: - 应用入口 (修复 Linker 错误)
@main
struct ADASApp: App {
    var body: some Scene {
        WindowGroup {
            ADASView()
        }
    }
}

import SwiftUI
import Vision
import AVFoundation
import CoreImage

// MARK: - 追踪对象状态
struct TrackedObject: Identifiable {
    let id: UUID
    var label: String
    var confidence: Float
    var boundingBox: CGRect
    var distance: Float
    var lastSeen: Date
}

// MARK: - ADAS 深度进化引擎
@MainActor
class ADASUltraViewModel: NSObject, ObservableObject {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var laneMask: CGImage?
    @Published var isWarning: Bool = false
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.ultra.queue", qos: .userInteractive)
    
    // 语音合成器
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var lastSpeechDate = Date.distantPast
    
    // 线程安全模型容器
    nonisolated(unsafe) private var model: VNCoreMLModel?
    private let context = CIContext()

    override init() {
        super.init()
        setupSystem()
    }

    private func setupSystem() {
        Task {
            // 1. 加载模型
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let coreMLModel = try? MLModel(contentsOf: modelURL, configuration: config) {
                self.model = try? VNCoreMLModel(for: coreMLModel)
            }

            // 2. 相机配置
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(videoDataOutput) { captureSession.addOutput(videoDataOutput) }
            
            captureSession.startRunning()
        }
    }
}

// MARK: - 并发处理逻辑
extension ADASUltraViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        var requests: [VNRequest] = []

        // 功能 3: 语义分割 (可行驶区域)
        let segmentationRequest = VNDetectGenerateSegmentationRequest(completionHandler: { request, _ in
            if let observation = request.results?.first as? VNPixelBufferObservation {
                let mask = observation.pixelBuffer
                let ciMask = CIImage(cvPixelBuffer: mask)
                if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                    Task { @MainActor in self.laneMask = cgMask }
                }
            }
        })
        segmentationRequest.revision = VNDetectGenerateSegmentationRequestRevision1
        requests.append(segmentationRequest)

        // 功能 1: YOLO 检测 + 追踪基础
        if let visionModel = self.model {
            let detectionRequest = VNCoreMLRequest(model: visionModel) { request, _ in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    Task { @MainActor in self.processAndTrack(results) }
                }
            }
            detectionRequest.imageCropAndScaleOption = .scaleFill
            requests.append(detectionRequest)
        }

        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform(requests)
    }

    // 功能 1 & 4: 追踪平滑与语音播报逻辑
    @MainActor
    private func processAndTrack(_ observations: [VNRecognizedObjectObservation]) {
        var newTrackedObjects: [TrackedObject] = []
        var dangerDetected = false

        for obs in observations {
            // 测距计算 (基于之前讨论的高度公式)
            let distance = 1.2 / (Float(obs.boundingBox.origin.y) + 0.05)
            
            let newObject = TrackedObject(
                id: UUID(),
                label: obs.labels.first?.identifier ?? "Unknown",
                confidence: obs.labels.first?.confidence ?? 0,
                boundingBox: obs.boundingBox,
                distance: distance,
                lastSeen: Date()
            )
            newTrackedObjects.append(newObject)

            // 功能 4: 语音逻辑 (3米内报警，且每5秒最多说一次)
            if distance < 3.0 && (newObject.label == "person" || newObject.label == "car" || newObject.label == "motorcycle") {
                dangerDetected = true
                triggerVoiceAlert(label: newObject.label, dist: distance)
            }
        }
        
        self.trackedObjects = newTrackedObjects
        self.isWarning = dangerDetected
    }

    @MainActor
    private func triggerVoiceAlert(label: String, dist: Float) {
        let now = Date()
        if now.timeIntervalSince(lastSpeechDate) > 5.0 {
            let translatedLabel = label == "motorcycle" ? "摩托车" : (label == "car" ? "汽车" : "障碍物")
            let utterance = AVSpeechUtterance(string: "注意，前方\(Int(dist))米有\(translatedLabel)")
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            utterance.rate = 0.55
            speechSynthesizer.speak(utterance)
            lastSpeechDate = now
        }
    }
}

// MARK: - 特斯拉 SFD 风格 UI
struct ContentView: View {
    @StateObject var viewModel = ADASUltraViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 核心视图：路面分割层 (语义空间)
            if let mask = viewModel.laneMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.blue.opacity(0.3)) // 将可行驶区域渲染为淡蓝色
                    .ignoresSafeArea()
            }
            
            // 目标追踪层
            GeometryReader { geo in
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let color: Color = obj.distance < 4 ? .red : .green
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(Int(obj.distance))M")
                            .font(.system(size: 12, weight: .black, design: .monospaced))
                            .background(color)
                            .foregroundColor(.white)
                        
                        Rectangle()
                            .stroke(color, lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                    }
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 功能 4: 视觉警告指示器
            if viewModel.isWarning {
                Rectangle()
                    .stroke(Color.red, lineWidth: 10)
                    .ignoresSafeArea()
                    .overlay(
                        Text("COLLISION WARNING")
                            .font(.system(size: 30, weight: .black))
                            .foregroundColor(.red)
                            .padding()
                            .background(Color.black.opacity(0.7))
                    )
            }
            
            // 底部状态栏
            VStack {
                Spacer()
                HStack {
                    VStack(alignment: .leading) {
                        Text("SYSTEM: YOLOv8L + SEGMENTATION")
                        Text("TRACKING: ACTIVE")
                    }
                    Spacer()
                    Circle()
                        .fill(viewModel.isWarning ? Color.red : Color.green)
                        .frame(width: 12, height: 12)
                }
                .font(.system(size: 10, design: .monospaced))
                .padding()
                .background(.ultraThinMaterial)
                .foregroundColor(.white)
            }
        }
    }
}

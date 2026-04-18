import SwiftUI
import Vision
import AVFoundation
import CoreImage

// MARK: - 追踪对象模型
struct TrackedObject: Identifiable {
    let id: UUID
    var label: String
    var distance: Float
    var boundingBox: CGRect
}

@MainActor
class ADASUltraViewModel: NSObject, ObservableObject {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var roadMask: CGImage?
    @Published var isCriticalWarning: Bool = false
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.ultra.processing", qos: .userInteractive)
    private let context = CIContext()
    private let synthesizer = AVSpeechSynthesizer()
    private var lastAlertTime = Date.distantPast

    // 线程安全模型容器
    nonisolated(unsafe) private var model: VNCoreMLModel?

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
               let coreML = try? MLModel(contentsOf: modelURL, configuration: config) {
                self.model = try? VNCoreMLModel(for: coreML)
            }

            // 2. 相机输入
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(videoDataOutput) { captureSession.addOutput(videoDataOutput) }
            
            // 3. 启动
            Task.detached {
                self.captureSession.startRunning()
            }
        }
    }
}

// MARK: - 后台推理逻辑 (修复报错点)
extension ADASUltraViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        var requests: [VNRequest] = []

        // 修复功能 3: 使用正确的 iOS 17+ 前景分割请求
        // 这可以识别出路上的车辆和行人，从而构建“占用地图”
        let segmentationRequest = VNGenerateForegroundInstanceMaskRequest() 
        requests.append(segmentationRequest)

        // 功能 1: 物体检测
        if let visionModel = self.model {
            let detectionRequest = VNCoreMLRequest(model: visionModel) { request, _ in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    Task { @MainActor in self.processDetections(results) }
                }
            }
            detectionRequest.imageCropAndScaleOption = .scaleFill
            requests.append(detectionRequest)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        try? handler.perform(requests)

        // 处理分割结果
        if let maskObservation = segmentationRequest.results?.first {
            if let maskBuffer = try? maskObservation.generateScaledMaskForCorrespondingRect(
                from: CGRect(x: 0, y: 0, width: 1, height: 1), 
                count: 1, 
                pixelBuffer: pixelBuffer
            ) {
                let ciMask = CIImage(cvPixelBuffer: maskBuffer)
                if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                    Task { @MainActor in self.roadMask = cgMask }
                }
            }
        }
    }

    @MainActor
    private func processDetections(_ observations: [VNRecognizedObjectObservation]) {
        var newObjects: [TrackedObject] = []
        var hasDanger = false

        for obs in observations {
            let dist = 1.2 / (Float(obs.boundingBox.origin.y) + 0.05)
            let label = obs.labels.first?.identifier ?? "Target"
            
            let obj = TrackedObject(id: UUID(), label: label, distance: dist, boundingBox: obs.boundingBox)
            newObjects.append(obj)

            // 功能 4: 语音预警逻辑 (距离 < 3米 触发)
            if dist < 3.0 {
                hasDanger = true
                speakAlert(label: label, distance: dist)
            }
        }
        self.trackedObjects = newObjects
        self.isCriticalWarning = hasDanger
    }

    @MainActor
    private func speakAlert(label: String, distance: Float) {
        let now = Date()
        if now.timeIntervalSince(lastAlertTime) > 5.0 { // 冷却时间 5 秒，防止轰炸
            let text = "注意前方 \(Int(distance)) 米 \(label)"
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            utterance.rate = 0.55
            synthesizer.speak(utterance)
            lastAlertTime = now
        }
    }
}

// MARK: - UI 布局
struct ContentView: View {
    @StateObject var viewModel = ADASUltraViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 语义分割背景层 (路面占用识别)
            if let mask = viewModel.roadMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.4))
                    .ignoresSafeArea()
            }
            
            // 2. 追踪框层
            GeometryReader { geo in
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let isClose = obj.distance < 5
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(Int(obj.distance))M")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .padding(2)
                            .background(isClose ? Color.red : Color.green)
                            .foregroundColor(.white)
                        
                        Rectangle()
                            .stroke(isClose ? Color.red : Color.green, lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                    }
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 3. 碰撞视觉预警
            if viewModel.isCriticalWarning {
                Color.red.opacity(0.2).ignoresSafeArea()
                VStack {
                    Text("⚠️ 立即减速 ⚠️")
                        .font(.system(size: 40, weight: .black))
                        .foregroundColor(.red)
                        .padding()
                        .background(Color.white.opacity(0.8))
                        .cornerRadius(20)
                }
            }
            
            // 4. 底部状态
            VStack {
                Spacer()
                Text("iPhone 15 ADAS PRO | YOLOv8L + Instance Segmentation")
                    .font(.caption2.monospaced())
                    .foregroundColor(.white.opacity(0.6))
                    .padding(.bottom, 10)
            }
        }
    }
}

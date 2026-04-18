import SwiftUI
import Vision
import AVFoundation
import CoreML

// 1. 扩展检测模型，包含运动状态
struct TrackedObject: Identifiable {
    let id: UUID
    var label: String
    var currentBox: CGRect
    var velocity: CGPoint // 运动矢量
    var history: [CGPoint] // 历史中心点
    var ttc: Double // 预计碰撞时间
}

@MainActor
class GlobalTrafficController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var trackedObjects: [TrackedObject] = []
    @Published var systemAlertLevel: Double = 0.0 // 全局风险值 0-1
    
    let captureSession = AVCaptureSession()
    private var model: VNCoreMLModel?
    private let videoOutput = AVCaptureVideoDataOutput()
    
    override init() {
        super.init()
        loadModel()
        setupCapture()
    }

    private func loadModel() {
        if let modelURL = Bundle.main.url(forResource: "yolov8s", withExtension: "mlmodelc") {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let m = try? MLModel(contentsOf: modelURL, configuration: config) {
                self.model = try? VNCoreMLModel(for: m)
            }
        }
    }

    func setupCapture() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        captureSession.addInput(input)
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "traffic_queue"))
        captureSession.addOutput(videoOutput)
        captureSession.commitConfiguration()
        DispatchQueue.global().async { self.captureSession.startRunning() }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer), let model = self.model else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            self?.analyzeGlobalScene(req.results as? [VNRecognizedObjectObservation] ?? [])
        }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([request])
    }

    private func analyzeGlobalScene(_ observations: [VNRecognizedObjectObservation]) {
        let targetLabels = ["person", "car", "bus", "truck", "motorbike"]
        
        var currentRisk = 0.0
        let newFrameObjects = observations.compactMap { obs -> TrackedObject? in
            let label = obs.labels.first?.identifier ?? "obj"
            guard targetLabels.contains(label) else { return nil }
            
            let box = obs.boundingBox
            let center = CGPoint(x: box.midX, y: box.midY)
            
            // 简单的轨迹预判逻辑：寻找上一帧最近的目标进行匹配 (Simple IOU/Distance Match)
            let prev = self.trackedObjects.first(where: { 
                abs($0.currentBox.midX - box.midX) < 0.1 && abs($0.currentBox.midY - box.midY) < 0.1 
            })
            
            let velocity = prev != nil ? CGPoint(x: center.x - prev!.history.last!.x, y: center.y - prev!.history.last!.y) : .zero
            var history = prev?.history ?? []
            history.append(center)
            if history.count > 10 { history.removeFirst() }
            
            // 计算风险指数：靠近中心且速度快的目标风险更高
            let distToCenter = sqrt(pow(center.x - 0.5, 2) + pow(center.y - 0.5, 2))
            if distToCenter < 0.3 { currentRisk += (box.width * box.height) * 2 }

            return TrackedObject(id: prev?.id ?? UUID(), label: label, currentBox: box, velocity: velocity, history: history, ttc: 0)
        }

        Task { @MainActor in
            self.trackedObjects = newFrameObjects
            self.systemAlertLevel = min(currentRisk, 1.0)
            if self.systemAlertLevel > 0.7 { AudioServicesPlaySystemSound(1016) } // 连续报警音
        }
    }
}

// 3. UI 界面：全局雷达感官
struct ContentView: View {
    @StateObject var controller = GlobalTrafficController()
    
    var body: some View {
        ZStack {
            CameraPreviewHolder(session: controller.captureSession).ignoresSafeArea()
            
            // 全局分析层
            Canvas { context, size in
                for obj in controller.trackedObjects {
                    let rect = CGRect(
                        x: obj.currentBox.minX * size.width,
                        y: (1 - obj.currentBox.maxY) * size.height,
                        width: obj.currentBox.width * size.width,
                        height: obj.currentBox.height * size.height
                    )
                    
                    // 1. 绘制当前框
                    let color = Color(hue: 0.3 - (controller.systemAlertLevel * 0.3), saturation: 1, brightness: 1)
                    context.stroke(Path(rect), with: .color(color), lineWidth: 2)
                    
                    // 2. 绘制预测轨迹线 (未来 5 帧)
                    var predictionPath = Path()
                    predictionPath.move(to: CGPoint(x: rect.midX, y: rect.midY))
                    predictionPath.addLine(to: CGPoint(
                        x: rect.midX + (obj.velocity.x * size.width * 5),
                        y: rect.midY - (obj.velocity.y * size.height * 5)
                    ))
                    context.stroke(predictionPath, with: .color(color.opacity(0.6)), style: StrokeStyle(lineWidth: 1, dash: [5]))
                }
                
                // 3. 绘制全场风险雷达边界
                let scanRect = CGRect(x: size.width*0.2, y: size.height*0.2, width: size.width*0.6, height: size.height*0.6)
                context.stroke(Path(scanRect), with: .color(.white.opacity(0.2)), lineWidth: 1)
            }
            
            // 风险仪表盘
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("系统预判等级").font(.system(.caption, design: .monospaced))
                        Capsule().fill(.gray.opacity(0.3)).frame(width: 100, height: 6)
                            .overlay(alignment: .leading) {
                                Capsule().fill(controller.systemAlertLevel > 0.6 ? .red : .green)
                                    .frame(width: 100 * CGFloat(controller.systemAlertLevel), height: 6)
                            }
                    }
                    Spacer()
                    Text("OBJECTS: \(controller.trackedObjects.count)").font(.system(.title3, design: .monospaced).bold())
                }
                .padding().background(.black.opacity(0.5)).foregroundColor(.white).cornerRadius(15).padding(20)
                Spacer()
            }
        }
    }
}

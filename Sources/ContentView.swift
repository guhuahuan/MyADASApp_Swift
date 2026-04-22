import SwiftUI
import AVFoundation
import Vision
import CoreML

// MARK: - App 入口
@main
struct FSD_Mini_App: App {
    var body: some Scene {
        WindowGroup {
            MainDetectionView()
        }
    }
}

// MARK: - 核心感知引擎
class FSDCoreEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [VNRecognizedObjectObservation] = []
    @Published var modelStatus: String = "初始化中..."
    @Published var isReady: Bool = false
    
    let session = AVCaptureSession()
    private var requests = [VNRequest]()
    
    // 根据你提供的元数据：0: person, 2: car, 5: bus, 7: truck
    private let targetLabels = ["person", "car", "bus", "truck"]

    func startup() {
        configureCamera()
        configureModel()
    }

    private func configureCamera() {
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { 
            self.modelStatus = "❌ 无法访问相机"
            return 
        }
        if session.canAddInput(input) { session.addInput(input) }
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        if session.canAddOutput(output) { session.addOutput(output) }
        
        DispatchQueue.global().async { self.session.startRunning() }
    }

    private func configureModel() {
        // 1. 尝试定位编译后的模型
        guard let modelURL = Bundle.main.url(forResource: "yolo26x", withExtension: "mlmodelc") else {
            DispatchQueue.main.async { 
                self.modelStatus = "❌ 找不到 yolo26x.mlmodelc"
                self.isReady = false
            }
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // 强制开启 Neural Engine
            
            let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: config))
            
            let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        // 过滤目标并将阈值设为 0.3
                        self?.detections = results.filter { obs in
                            let label = obs.labels.first?.identifier ?? ""
                            return (self?.targetLabels.contains(label) ?? false) && obs.confidence > 0.3
                        }
                    }
                }
            }
            
            // 适配你的 imgsz [640, 640]
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            
            DispatchQueue.main.async {
                self.modelStatus = "✅ YOLO26x 加载成功"
                self.isReady = true
            }
        } catch {
            DispatchQueue.main.async { 
                self.modelStatus = "❌ 加载失败: \(error.localizedDescription)"
                self.isReady = false
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        try? handler.perform(self.requests)
    }
}

// MARK: - 主视图层
struct MainDetectionView: View {
    @StateObject private var engine = FSDCoreEngine()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // 相机预览
                CameraPreviewView(session: engine.session)
                    .ignoresSafeArea()
                
                // 绘制层
                Canvas { context, size in
                    for obs in engine.detections {
                        // 坐标转换
                        let rect = VNImageRectForNormalizedRect(obs.boundingBox, Int(size.width), Int(size.height))
                        let correctedRect = CGRect(x: rect.minX, y: size.height - rect.maxY, width: rect.width, height: rect.height)
                        
                        let label = obs.labels.first?.identifier ?? ""
                        let isPerson = label == "person"
                        let color: Color = isPerson ? .cyan : .green
                        
                        // 画框
                        context.stroke(Path(correctedRect), with: .color(color), lineWidth: 3)
                        
                        // 画标签
                        let text = Text("\(label.uppercased()) \(Int(obs.confidence * 100))%")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(.black)
                        
                        context.fill(Path(CGRect(x: correctedRect.minX, y: correctedRect.minY - 20, width: 100, height: 20)), with: .color(color))
                        context.draw(text, at: CGPoint(x: correctedRect.minX + 5, y: correctedRect.minY - 10), anchor: .leading)
                    }
                }
                
                // 状态指示灯（用于排查模型加载）
                VStack {
                    HStack {
                        HStack(spacing: 8) {
                            Circle()
                                .fill(engine.isReady ? Color.green : Color.red)
                                .frame(width: 10, height: 10)
                            Text(engine.modelStatus)
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(.white)
                        }
                        .padding(8)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(8)
                        Spacer()
                    }
                    .padding(.top, 50)
                    .padding(.leading, 20)
                    Spacer()
                }
            }
        }
        .onAppear { engine.startup() }
    }
}

// MARK: - UIKit 桥接组件
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.frame = view.frame
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

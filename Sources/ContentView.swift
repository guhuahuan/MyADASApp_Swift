import SwiftUI
import AVFoundation
import Vision
import CoreML

// MARK: - App 入口
@main
struct MiniDetectorApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - 核心引擎
class DetectionEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [VNRecognizedObjectObservation] = []
    
    let session = AVCaptureSession()
    private var requests = [VNRequest]()
    
    // 根据元数据，yolo26x 标签是小写
    private let targetLabels = ["person", "car", "truck", "bus", "motorcycle"]

    func setup() {
        // 1. 相机配置 (720P 兼顾性能与识别距离)
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        if session.canAddInput(input) { session.addInput(input) }
        
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_processing_queue"))
        if session.canAddOutput(output) { session.addOutput(output) }
        
        // 2. 加载 yolo26x 模型
        configureModel()
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
        }
    }

    private func configureModel() {
        // 获取本地编译后的模型路径
        guard let modelURL = Bundle.main.url(forResource: "yolo26x", withExtension: "mlmodelc") else {
            print("❌ 找不到 yolo26x.mlmodelc")
            return
        }
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // 强制开启 Neural Engine 加速
            
            let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: config))
            
            let request = VNCoreMLRequest(model: model) { [weak self] req, error in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        // 过滤目标并设置 0.25 的置信度阈值
                        self?.detections = results.filter { obs in
                            let label = obs.labels.first?.identifier ?? ""
                            return (self?.targetLabels.contains(label) ?? false) && obs.confidence > 0.25
                        }
                    }
                }
            }
            
            // 关键：针对 imgsz [640, 640] 进行填充缩放，不剪裁边缘
            request.imageCropAndScaleOption = .scaleFill
            self.requests = [request]
            
        } catch {
            print("❌ 模型初始化失败: \(error)")
        }
    }

    // 相机回调
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 由于是前置/后置摄像头，通常需要指定 orientation 为 .up
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform(self.requests)
        } catch {
            print("❌ 推理失败: \(error)")
        }
    }
}

// MARK: - 渲染层
struct ContentView: View {
    @StateObject private var engine = DetectionEngine()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // 1. 底层相机流
                CameraView(session: engine.session)
                    .ignoresSafeArea()
                
                // 2. 识别框层
                Canvas { context, size in
                    for obs in engine.detections {
                        // 坐标转换：Vision(0~1, Y向上) -> SwiftUI(绝对像素, Y向下)
                        let rect = VNImageRectForNormalizedRect(obs.boundingBox, Int(size.width), Int(size.height))
                        let correctedRect = CGRect(x: rect.minX, y: size.height - rect.maxY, width: rect.width, height: rect.height)
                        
                        let label = obs.labels.first?.identifier ?? ""
                        let isPerson = label == "person"
                        let themeColor: Color = isPerson ? .cyan : .green
                        
                        // 绘制外框
                        context.stroke(Path(correctedRect), with: .color(themeColor), lineWidth: 3)
                        
                        // 绘制标签背景
                        let text = Text(label.uppercased())
                            .font(.system(size: 12, weight: .black, design: .monospaced))
                            .foregroundColor(.black)
                        
                        let textPos = CGPoint(x: correctedRect.minX + 5, y: correctedRect.minY + 10)
                        context.fill(Path(CGRect(x: correctedRect.minX, y: correctedRect.minY - 20, width: 80, height: 20)), with: .color(themeColor))
                        context.draw(text, at: textPos, anchor: .topLeading)
                    }
                }
            }
        }
        .onAppear { engine.setup() }
    }
}

// MARK: - UIKit 相机桥接
struct CameraView: UIViewRepresentable {
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

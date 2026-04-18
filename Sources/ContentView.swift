import SwiftUI
import Vision
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins

// MARK: - 线程安全容器
// 用于跨越 Actor 边界传递模型
struct SafeModelContainer: Sendable {
    let visionModel: VNCoreMLModel
}

@MainActor
class ADASProViewModel: NSObject, ObservableObject {
    @Published var detections: [VNRecognizedObjectObservation] = []
    @Published var laneBEVImage: CGImage?
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.worker.queue", qos: .userInteractive)
    private let context = CIContext()

    // 关键修复：使用 nonisolated(unsafe) 允许后台线程访问
    // 这是在 2026 年处理高性能 AI 推理的常用“硬核”手段
    nonisolated(unsafe) private var modelContainer: SafeModelContainer?

    override init() {
        super.init()
        setupSystem()
    }

    private func setupSystem() {
        Task {
            // 1. 动态加载并编译模型 (解决找不到 yolov8l 类的问题)
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            
            guard let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc") ?? 
                    Bundle.main.url(forResource: "yolov8l", withExtension: "mlpackage") else {
                print("❌ 找不到模型文件")
                return
            }

            do {
                let compiledURL = try await MLModel.compileModel(at: modelURL)
                let coreMLModel = try MLModel(contentsOf: compiledURL, configuration: config)
                let visionModel = try VNCoreMLModel(for: coreMLModel)
                self.modelContainer = SafeModelContainer(visionModel: visionModel)
            } catch {
                print("❌ 模型初始化失败: \(error)")
            }

            // 2. 相机设置
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(videoDataOutput) { captureSession.addOutput(videoDataOutput) }
            
            captureSession.startRunning() 
        }
    }
}

// MARK: - 后台推理逻辑
extension ADASProViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 1. 车道重建 (BEV)
        reconstructLanesInternal(from: pixelBuffer)

        // 2. 物体检测 (使用线程安全的 container)
        guard let container = self.modelContainer else { return }
        
        let request = VNCoreMLRequest(model: container.visionModel) { request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in
                    self.detections = results
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([request])
    }

    // 内部车道重建逻辑 (运行在后台线程)
    nonisolated private func reconstructLanesInternal(from pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let filter = CIFilter.perspectiveCorrection()
        filter.inputImage = ciImage
        
        let w = ciImage.extent.width
        let h = ciImage.extent.height
        
        // 特斯拉风格透视参数 (针对 iPhone 15 安装高度优化)
        filter.topLeft = CGPoint(x: w * 0.35, y: h * 0.65)
        filter.topRight = CGPoint(x: w * 0.65, y: h * 0.65)
        filter.bottomLeft = CGPoint(x: w * 0.05, y: h * 0.2)
        filter.bottomRight = CGPoint(x: w * 0.95, y: h * 0.2)

        if let output = filter.outputImage,
           let cgImage = context.createCGImage(output, from: output.extent) {
            Task { @MainActor in
                self.laneBEVImage = cgImage
            }
        }
    }
}

// MARK: - 特斯拉 FSD 风格界面
struct ContentView: View {
    @StateObject var viewModel = ADASProViewModel()
    
    var body: some View {
        ZStack(alignment: .bottom) {
            Color.black.ignoresSafeArea()
            
            // 2D 增强现实检测层
            GeometryReader { geo in
                ForEach(viewModel.detections, id: \.uuid) { obs in
                    let rect = VNImageRectForNormalizedRect(obs.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    ZStack {
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(Color.red, lineWidth: 2)
                        Text("\(obs.labels.first?.identifier.uppercased() ?? "")")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .background(Color.red)
                            .foregroundColor(.white)
                            .offset(y: -rect.height/2 - 10)
                    }
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }

            // 特斯拉 3D 向量空间 (BEV) 视图
            VStack(spacing: 0) {
                HStack {
                    Image(systemName: "dot.radiowaves.up.forward")
                    Text("LANE RECONSTRUCTION & OCCUPANCY")
                }
                .font(.system(size: 10, weight: .black, design: .monospaced))
                .foregroundColor(.cyan)
                .padding(.bottom, 8)

                if let bev = viewModel.laneBEVImage {
                    ZStack {
                        Image(decorative: bev, scale: 1.0)
                            .resizable()
                            .frame(width: 320, height: 160)
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.cyan.opacity(0.4), lineWidth: 2))
                        
                        // 自车渲染
                        Image(systemName: "car.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white)
                            .shadow(color: .cyan, radius: 8)
                            .offset(y: 40)
                    }
                } else {
                    RoundedRectangle(cornerRadius: 20)
                        .fill(Color.gray.opacity(0.2))
                        .frame(width: 320, height: 160)
                        .overlay(Text("等待环境感知...").foregroundColor(.gray))
                }
            }
            .padding(.bottom, 50)
        }
    }
}

@main
struct MyADASApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

import SwiftUI
import Vision
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins

// MARK: - 车道重建引擎 (BEV Reconstruction)
@MainActor
class LaneReconstructionEngine: ObservableObject {
    @Published var laneBEVImage: CGImage?
    private let context = CIContext()

    func reconstructLanes(from pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = ciImage.extent.width
        let height = ciImage.extent.height

        // 1. 使用核心滤镜：透视变换 (Perspective Correction)
        let filter = CIFilter.perspectiveCorrection()
        filter.inputImage = ciImage
        
        // 关键修复：使用 CGPoint 代替 CIVector
        // 定义一个梯形区域，将其“拉平”为长方形。
        // 这四个点定义了你车头前方的路面区域。
        filter.topLeft = CGPoint(x: width * 0.35, y: height * 0.65)
        filter.topRight = CGPoint(x: width * 0.65, y: height * 0.65)
        filter.bottomLeft = CGPoint(x: width * 0.05, y: height * 0.2)
        filter.bottomRight = CGPoint(x: width * 0.95, y: height * 0.2)

        guard let output = filter.outputImage else { return }

        // 2. 增强对比度，突出车道线 (类似特斯拉的二值化占用网络)
        let colorFilter = CIFilter.colorControls()
        colorFilter.inputImage = output
        colorFilter.contrast = 2.5
        colorFilter.brightness = -0.2
        
        if let finalImage = colorFilter.outputImage,
           let cgImage = context.createCGImage(finalImage, from: finalImage.extent) {
            self.laneBEVImage = cgImage
        }
    }
}

// MARK: - ADAS 综合模型
@MainActor
class ADASProViewModel: NSObject, ObservableObject {
    @Published var detections: [VNRecognizedObjectObservation] = []
    @Published var laneEngine = LaneReconstructionEngine()
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.analysis.queue", qos: .userInteractive)
    private var model: VNCoreMLModel?

    override init() {
        super.init()
        setupSystem()
    }

    private func setupSystem() {
        Task {
            // 模型加载逻辑 (动态编译方法)
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let coreMLModel = try? MLModel(contentsOf: modelURL, configuration: config) {
                self.model = try? VNCoreMLModel(for: coreMLModel)
            }

            // 相机配置
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(videoDataOutput) { captureSession.addOutput(videoDataOutput) }
            
            captureSession.startRunning() 
        }
    }
}

extension ADASProViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // 1. 车道线 BEV 重建
        Task { @MainActor in
            self.laneEngine.reconstructLanes(from: pixelBuffer)
        }

        // 2. 物体检测 (YOLOv8L)
        guard let model = self.model else { return }
        let request = VNCoreMLRequest(model: model) { request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.detections = results }
            }
        }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([request])
    }
}

// MARK: - 特斯拉 SFD 渲染界面
struct ContentView: View {
    @StateObject var viewModel = ADASProViewModel()
    
    var body: some View {
        ZStack(alignment: .bottom) {
            // 背景层：原始相机流（此处可用简单的黑背景模拟，重点在 BEV）
            Color.black.ignoresSafeArea()
            
            // 顶层：2D 检测框
            GeometryReader { geo in
                ForEach(viewModel.detections, id: \.uuid) { obs in
                    let rect = VNImageRectForNormalizedRect(obs.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.red, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }

            // 核心功能：特斯拉风格 BEV 向量空间
            VStack(spacing: 0) {
                HStack {
                    Image(systemName: "view.2d")
                    Text("3D VECTOR SPACE RECONSTRUCTION")
                        .font(.system(size: 10, weight: .black, design: .monospaced))
                }
                .foregroundColor(.cyan)
                .padding(.bottom, 5)

                if let bev = viewModel.laneEngine.laneBEVImage {
                    ZStack {
                        Image(decorative: bev, scale: 1.0)
                            .resizable()
                            .frame(width: 320, height: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 15))
                            .overlay(RoundedRectangle(cornerRadius: 15).stroke(Color.cyan.opacity(0.5), lineWidth: 2))
                        
                        // 自车图标 (模拟特斯拉渲染)
                        VStack {
                            Spacer()
                            Image(systemName: "car.fill")
                                .font(.system(size: 30))
                                .foregroundColor(.white)
                                .shadow(color: .cyan, radius: 10)
                                .padding(.bottom, 10)
                        }
                    }
                    .frame(width: 320, height: 180)
                }
            }
            .padding(.bottom, 40)
            .padding(.horizontal)
            .background(LinearGradient(colors: [.clear, .black.opacity(0.8)], startPoint: .top, endPoint: .bottom))
        }
    }
}

@main
struct MyADASApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

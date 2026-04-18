import SwiftUI
import Vision
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins

// MARK: - 车道重建与渲染引擎
@MainActor
class LaneReconstructionEngine: ObservableObject {
    @Published var laneBEVImage: CGImage? // 鸟瞰图车道
    
    private let context = CIContext()
    
    // 将相机画面转换为鸟瞰图，并提取车道
    func reconstructLanes(from pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // 1. 动态生成可行驶区域分割图 (利用 iOS 26 新版 Vision)
        // (这里为了代码简洁，使用 CI 滤镜模拟道路灰度提取，实际应调用语义分割请求)
        let roadFilter = CIFilter.colorThreshold()
        roadFilter.inputImage = ciImage
        roadFilter.threshold = 0.3 // 假设柏油路灰度
        guard let roadMask = roadFilter.outputImage else { return }
        
        // 2. 执行透视变换 (Perspective Transform / Homography)
        // 将梯形的道路区域拉伸为长方形的鸟瞰图
        let perspectiveTransform = CIFilter.perspectiveTransform()
        perspectiveTransform.inputImage = roadMask
        
        // 定义原图中的梯形顶点 (根据 iPhone 15 安装高度调整)
        let width = ciImage.extent.width
        let height = ciImage.extent.height
        perspectiveTransform.topLeft = CIVector(x: width * 0.4, y: height * 0.6)
        perspectiveTransform.topRight = CIVector(x: width * 0.6, y: height * 0.6)
        perspectiveTransform.bottomLeft = CIVector(x: 0, y: height * 0.1)
        perspectiveTransform.bottomRight = CIVector(x: width, y: height * 0.1)
        
        guard let bevOutput = perspectiveTransform.outputImage else { return }
        
        // 3. 渲染为 CGImage 用于 SwiftUI 显示
        if let cgImage = context.createCGImage(bevOutput, from: bevOutput.extent) {
            self.laneBEVImage = cgImage
        }
    }
}

@MainActor
class ADASProViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [(observation: VNRecognizedObjectObservation, distance: Float)] = []
    
    private let captureSession = AVCaptureSession()
    private var model: VNCoreMLModel?
    private let queue = DispatchQueue(label: "video-processing", qos: .userInteractive)
    
    // 引入车道引擎
    @ObservedObject var laneEngine = LaneReconstructionEngine()

    override init() {
        super.init()
        setupEngine()
    }

    private func setupEngine() {
        Task {
            // 模型和相机配置省略... (参考前一版代码)
            await captureSession.startRunning()
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 核心功能 1：YOLO 物体检测
        // (检测代码省略...)
        
        // 核心功能 2：实时车道重建
        Task { @MainActor in
            self.laneEngine.reconstructLanes(from: pixelBuffer)
        }
    }
}

// MARK: - 特斯拉风仪表盘界面
struct ContentView: View {
    @StateObject var viewModel = ADASProViewModel()
    
    var body: some View {
        ZStack(alignment: .bottom) {
            // 1. 全屏相机背景 + YOLO 检测框 (参考前一版代码)
            Color.black.ignoresSafeArea() 
            
            // 2. 底部车道渲染窗口 (BEV Bird's Eye View)
            VStack {
                HStack {
                    Image(systemName: "car.top.radiowaves.rear.left.and.rear.right")
                    Text("VECTOR SPACE | LANE RECONSTRUCTION")
                }
                .font(.system(size: 10, weight: .black, design: .monospaced))
                .foregroundColor(.white)
                .padding(.top, 8)
                
                if let bevImage = viewModel.laneEngine.laneBEVImage {
                    Image(decorative: bevImage, scale: 1.0)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .overlay {
                            // 绘制自车（上帝视角中心）
                            Rectangle()
                                .fill(Color.blue)
                                .frame(width: 30, height: 50)
                                .cornerRadius(4)
                                .offset(y: 40) // 位于 BEV 图底部中心
                        }
                        .frame(width: 300, height: 150)
                        .background(Color.black)
                        .cornerRadius(12)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.3), lineWidth: 1))
                        .padding(.bottom, 30)
                } else {
                    ProgressView("计算车道中...").tint(.white).frame(width: 300, height: 150)
                }
            }
            .background(Color.black.opacity(0.8))
            .cornerRadius(16)
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
    }
}

import SwiftUI
import Vision
import AVFoundation

// MARK: - ADAS 核心逻辑控制器
@MainActor
class ADASViewModel: NSObject, ObservableObject {
    @Published var detectedObjects: [VNRecognizedObjectObservation] = []
    @Published var isModelLoading = true
    
    private var captureSession = AVCaptureSession()
    private var videoDeviceInput: AVCaptureDeviceInput?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "com.user.adas.video-queue", qos: .userInteractive)
    
    private var model: VNCoreMLModel?

    override init() {
        super.init()
        Task {
            await setupModel()
            setupCamera()
        }
    }
    
    // 加载刚刚上传的 yolov8l 模型
    private func setupModel() async {
        do {
            // 注意：确保项目中模型文件名完全匹配 yolov8l
            let config = MLModelConfiguration()
            config.computeUnits = .all // 强制开启 Neural Engine (ANE)
            
            let coreMLModel = try yolov8l(configuration: config).model
            let visionModel = try VNCoreMLModel(for: coreMLModel)
            
            self.model = visionModel
            self.isModelLoading = false
        } catch {
            print("模型加载失败: \(error)")
        }
    }
    
    private func setupCamera() {
        captureSession.beginConfiguration()
        
        // 获取后置摄像头
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
            if captureSession.canAddInput(videoDeviceInput) {
                captureSession.addInput(videoDeviceInput)
            }
            
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
                videoDataOutput.alwaysDiscardsLateVideoFrames = true
                videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            }
            
            // 针对 2026 年的高清屏幕优化分辨率
            captureSession.sessionPreset = .hd1920x1080
            captureSession.commitConfiguration()
            
            Task.detached {
                self.captureSession.startRunning()
            }
        } catch {
            print("相机设置失败: \(error)")
        }
    }
}

// MARK: - 实时推理委托
extension ADASViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let model = model, !isModelLoading else { return }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                DispatchQueue.main.async {
                    self.detectedObjects = results
                }
            }
        }
        
        // 保持推理方向与手机竖屏一致
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("推理失败: \(error)")
        }
    }
}

// MARK: - UI 界面
struct ContentView: View {
    @StateObject private var viewModel = ADASViewModel()
    
    var body: some View {
        ZStack {
            // 相机预览层（此处建议在实际工程中嵌入 UIViewRepresentable 的预览层）
            Color.black.ignoresSafeArea()
            
            if viewModel.isModelLoading {
                VStack {
                    ProgressView()
                        .tint(.white)
                    Text("正在加载 YOLOv8L 引擎...")
                        .foregroundColor(.white)
                        .padding()
                }
            } else {
                // 绘制检测框
                GeometryReader { geometry in
                    ForEach(viewModel.detectedObjects, id: \.uuid) { obj in
                        let box = obj.boundingBox
                        let width = box.width * geometry.size.width
                        let height = box.height * geometry.size.height
                        let x = box.origin.x * geometry.size.width
                        let y = (1 - box.origin.y - box.height) * geometry.size.height
                        
                        Rectangle()
                            .path(in: CGRect(x: x, y: y, width: width, height: height))
                            .stroke(Color.green, lineWidth: 2)
                        
                        if let label = obj.labels.first {
                            Text("\(label.identifier) \(Int(label.confidence * 100))%")
                                .position(x: x + width/2, y: y - 10)
                                .foregroundColor(.green)
                                .font(.caption.bold())
                        }
                    }
                }
                
                // 状态指示器
                VStack {
                    HStack {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 10, height: 10)
                        Text("iOS 26 ADAS 实时监控 (YOLOv8L)")
                            .font(.system(size: 12, weight: .black, design: .monospaced))
                            .foregroundColor(.white)
                    }
                    .padding(8)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(8)
                    .padding(.top, 50)
                    Spacer()
                }
            }
        }
        .statusBarHidden()
    }
}

// MARK: - 程序入口
@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

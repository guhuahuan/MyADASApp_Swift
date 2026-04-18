import SwiftUI
import Vision
import AVFoundation
import CoreML

// MARK: - ADAS 核心逻辑控制器
@MainActor
class ADASViewModel: NSObject, ObservableObject {
    @Published var detectedObjects: [VNRecognizedObjectObservation] = []
    @Published var isModelLoading = true
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "com.user.adas.video-queue", qos: .userInteractive)
    
    // 关键修复：模型必须能够安全地在后台访问
    private var model: VNCoreMLModel?

    override init() {
        super.init()
        Task {
            await setupModel()
            await setupCamera()
        }
    }
    
    private func setupModel() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            
            // 修复点 1：使用更稳健的方式实例化模型
            // 如果你的文件名是 yolov8l.mlpackage，生成的类名通常是 yolov8l
            let coreMLModel = try yolov8l(configuration: config).model
            let visionModel = try VNCoreMLModel(for: coreMLModel)
            
            self.model = visionModel
            self.isModelLoading = false
        } catch {
            print("模型加载失败: \(error)")
        }
    }
    
    private func setupCamera() async {
        captureSession.beginConfiguration()
        
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
            
            captureSession.sessionPreset = .hd1920x1080
            captureSession.commitConfiguration()
            
            // 修复点 2：iOS 26 中 startRunning 必须异步调用且 await
            if !captureSession.isRunning {
                await captureSession.startRunning()
            }
        } catch {
            print("相机设置失败: \(error)")
        }
    }

    // 内部辅助方法：允许后台线程更新 UI
    fileprivate func updateDetections(_ observations: [VNRecognizedObjectObservation]) {
        self.detectedObjects = observations
    }
}

// MARK: - 实时推理委托 (适配 Swift 6 隔离规则)
extension ADASViewModel: @preconcurrency AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // 修复点 3：标记为 nonisolated，解决 Swift 6 的 Actor 隔离警告
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // 这里的逻辑在后台线程执行
        guard let model = self.model else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                // 修复点 4：从后台切回主线程更新 UI
                Task { @MainActor in
                    self.updateDetections(results)
                }
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        
        try? handler.perform([request])
    }
}

// MARK: - UI 界面
struct ContentView: View {
    @StateObject private var viewModel = ADASViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            if viewModel.isModelLoading {
                VStack {
                    ProgressView().tint(.white)
                    Text("载入 YOLOv8L 强力引擎...").foregroundColor(.white).padding()
                }
            } else {
                GeometryReader { geometry in
                    ForEach(viewModel.detectedObjects, id: \.uuid) { obj in
                        let box = obj.boundingBox
                        let width = box.width * geometry.size.width
                        let height = box.height * geometry.size.height
                        let x = box.origin.x * geometry.size.width
                        let y = (1 - box.origin.y - box.height) * geometry.size.height
                        
                        Rectangle()
                            .path(in: CGRect(x: x, y: y, width: width, height: height))
                            .stroke(Color.red, lineWidth: 2) // Large模型用红色，更显霸气
                        
                        if let label = obj.labels.first {
                            Text("\(label.identifier) \(Int(label.confidence * 100))%")
                                .position(x: x + width/2, y: y - 10)
                                .foregroundColor(.red)
                                .font(.system(size: 14, weight: .bold))
                        }
                    }
                }
                
                VStack {
                    HStack {
                        Image(systemName: "eye.fill")
                        Text("iOS 26 ADAS PRO | YOLOv8L")
                    }
                    .font(.caption.monospaced())
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .padding(.top, 50)
                    Spacer()
                }
            }
        }
        .statusBarHidden()
    }
}

@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

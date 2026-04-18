import SwiftUI
import Vision
import AVFoundation
import CoreML

// MARK: - 线程安全模型容器
// 解决 Swift 6 隔离问题的关键：创建一个可在线程间传递的容器
struct ModelContainer: Sendable {
    let visionModel: VNCoreMLModel
}

@MainActor
class ADASViewModel: NSObject, ObservableObject {
    @Published var detectedObjects: [VNRecognizedObjectObservation] = []
    @Published var isModelLoading = true
    
    private let captureSession = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "com.user.adas.video-queue", qos: .userInteractive)
    
    // 关键修复：改为非隔离存储，允许后台访问
    nonisolated(unsafe) private var modelContainer: ModelContainer?

    override init() {
        super.init()
        Task {
            await setupModel()
            await setupCamera()
        }
    }
    
    private func setupModel() async {
        // 动态寻找并编译模型，解决 "cannot find yolov8l in scope"
        guard let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc") ?? 
                Bundle.main.url(forResource: "yolov8l", withExtension: "mlpackage") else {
            print("找不到模型文件")
            return
        }
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            // 动态编译并加载
            let compiledURL = try await MLModel.compileModel(at: modelURL)
            let coreMLModel = try MLModel(contentsOf: compiledURL, configuration: config)
            let visionModel = try VNCoreMLModel(for: coreMLModel)
            
            self.modelContainer = ModelContainer(visionModel: visionModel)
            self.isModelLoading = false
        } catch {
            print("动态模型加载失败: \(error)")
        }
    }
    
    private func setupCamera() async {
        captureSession.beginConfiguration()
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        
        do {
            let input = try AVCaptureDeviceInput(device: videoDevice)
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
                videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            }
            captureSession.sessionPreset = .hd1920x1080
            captureSession.commitConfiguration()
            
            // 兼容 iOS 26 的启动调用
            Task.detached {
                await self.captureSession.startRunning()
            }
        } catch {
            print("相机启动失败")
        }
    }
}

// MARK: - 实时推理委托
extension ADASViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // 这里的 self.modelContainer 现在是可以安全跨线程访问的
        guard let container = self.modelContainer else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNCoreMLRequest(model: container.visionModel) { request, error in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in
                    self.detectedObjects = results
                }
            }
        }
        
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }
}

// MARK: - 简洁 UI 
struct ContentView: View {
    @StateObject private var viewModel = ADASViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            if viewModel.isModelLoading {
                ProgressView("正在编译 YOLOv8L 引擎...").tint(.white).foregroundColor(.white)
            } else {
                GeometryReader { geo in
                    ForEach(viewModel.detectedObjects, id: \.uuid) { obj in
                        let box = obj.boundingBox
                        let rect = CGRect(
                            x: box.origin.x * geo.size.width,
                            y: (1 - box.origin.y - box.height) * geo.size.height,
                            width: box.width * geo.size.width,
                            height: box.height * geo.size.height
                        )
                        
                        RoundedRectangle(cornerRadius: 4)
                            .path(in: rect)
                            .stroke(Color.red, lineWidth: 3)
                        
                        Text("\(obj.labels.first?.identifier ?? "??")")
                            .position(x: rect.midX, y: rect.minY - 15)
                            .foregroundColor(.red)
                    }
                }
            }
        }
        .statusBarHidden()
    }
}

@main
struct MyADASApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

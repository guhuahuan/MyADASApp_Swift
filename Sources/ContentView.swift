import SwiftUI
import Vision
import AVFoundation

// 核心逻辑控制器：处理摄像头流与 Vision 推理
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detectedRects: [VNRectangleObservation] = []
    
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.adas.vision.queue")
    
    override init() {
        super.init()
        setupCaptureSession()
    }
    
    private func setupCaptureSession() {
        // 在模拟器环境下，硬件相关的设置需要非常小心
        #if targetEnvironment(simulator)
        print("运行在模拟器环境，跳过真实摄像头初始化")
        #else
        captureSession.beginConfiguration()
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            captureSession.commitConfiguration()
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
        #endif
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 创建矩形检测请求
        let request = VNDetectRectanglesRequest { [weak self] request, error in
            guard let results = request.results as? [VNRectangleObservation] else { return }
            DispatchQueue.main.async {
                self?.detectedRects = results
            }
        }
        
        request.minimumConfidence = 0.5
        request.maximumObservations = 5
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}

// 预览组件：在模拟器中显示占位符，真机显示摄像头
struct CameraPreviewView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        
        #if targetEnvironment(simulator)
        let label = UILabel()
        label.text = "模拟器模式: 摄像头不可用"
        label.textColor = .lightGray
        label.textAlignment = .center
        label.frame = view.bounds
        view.addSubview(label)
        #endif
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var adas = ADASController()
    
    var body: some View {
        ZStack {
            CameraPreviewView()
                .ignoresSafeArea()
            
            // 渲染层
            Canvas { context, size in
                for rect in adas.detectedRects {
                    let width = rect.boundingBox.width * size.width
                    let height = rect.boundingBox.height * size.height
                    let x = rect.boundingBox.minX * size.width
                    let y = (1 - rect.boundingBox.maxY) * size.height
                    
                    let drawRect = CGRect(x: x, y: y, width: width, height: height)
                    context.stroke(Path(drawRect), with: .color(.green), lineWidth: 2)
                }
            }
            
            VStack {
                Text("ADAS 监控中")
                    .font(.headline)
                    .padding()
                    .background(.thinMaterial)
                    .cornerRadius(10)
                    .padding(.top, 20)
                Spacer()
            }
        }
    }
}

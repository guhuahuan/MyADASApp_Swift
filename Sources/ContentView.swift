import SwiftUI
import Vision
import AVFoundation

// 整个类标记为 @MainActor，确保 UI 属性安全更新
@MainActor
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
        #if targetEnvironment(simulator)
        print("Simulator detected: Skipping camera hardware init.")
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
        
        // 异步启动，避免阻塞初始化线程
        Task {
            self.captureSession.startRunning()
        }
        #endif
    }
    
    // 关键修正：标记为 nonisolated 以满足协议要求，内部手动切换到 MainActor
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNDetectRectanglesRequest { request, error in
            // 提取结果并转义到主线程
            if let results = request.results as? [VNRectangleObservation] {
                let capturedResults = results // 捕获局部变量
                Task { @MainActor in
                    self.detectedRects = capturedResults
                }
            }
        }
        
        request.minimumConfidence = 0.5
        request.maximumObservations = 5
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}

// 预览视图，处理模拟器兼容性
struct CameraPreviewView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        
        #if targetEnvironment(simulator)
        let label = UILabel()
        label.text = "ADAS 视觉待命 (模拟器)"
        label.textColor = .green
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
            
            // 绘图层：使用 Canvas 提高渲染效率
            Canvas { context, size in
                for rect in adas.detectedRects {
                    let drawRect = calculateRect(rect.boundingBox, in: size)
                    context.stroke(Path(drawRect), with: .color(.green), lineWidth: 2)
                    
                    // 模拟检测标签
                    let text = context.resolve(Text("DETECTED").font(.system(size: 10, weight: .bold)))
                    context.draw(text, at: CGPoint(x: drawRect.midX, y: drawRect.minY - 8))
                }
            }
            
            VStack {
                Text("Vision 内置检测模式")
                    .font(.caption)
                    .padding(6)
                    .background(Capsule().fill(Color.black.opacity(0.6)))
                    .foregroundColor(.green)
                    .padding(.top, 50)
                Spacer()
            }
        }
    }
    
    // 视觉坐标系转换逻辑
    private func calculateRect(_ box: CGRect, in size: CGSize) -> CGRect {
        let w = box.width * size.width
        let h = box.height * size.height
        let x = box.minX * size.width
        let y = (1 - box.maxY) * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

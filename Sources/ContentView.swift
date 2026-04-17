import SwiftUI
import Vision
import AVFoundation

// 使用 @MainActor 确保所有属性更新都在主线程安全进行
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
        print("模拟器环境：仅加载 UI 框架")
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
        
        Task.detached(priority: .userInitiated) {
            self.captureSession.startRunning()
        }
        #endif
    }
    
    // 关键修正：非隔离回调处理
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNDetectRectanglesRequest { request, error in
            guard let results = request.results as? [VNRectangleObservation] else { return }
            
            // 使用 Task 回到 Main Actor 更新 UI
            Task { @MainActor in
                self.detectedRects = results
            }
        }
        
        request.minimumConfidence = 0.5
        request.maximumObservations = 5
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}

struct CameraPreviewView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        
        #if targetEnvironment(simulator)
        let label = UILabel()
        label.text = "ADAS 模拟器模式"
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
            
            // 实时视觉叠加层
            Canvas { context, size in
                for rect in adas.detectedRects {
                    let drawRect = projectRect(rect.boundingBox, to: size)
                    context.stroke(Path(drawRect), with: .color(.green), lineWidth: 2)
                }
            }
            
            VStack {
                StatusBadge(rectCount: adas.detectedRects.count)
                Spacer()
            }
        }
    }
    
    // 坐标转换工具函数
    private func projectRect(_ box: CGRect, to size: CGSize) -> CGRect {
        let w = box.width * size.width
        let h = box.height * size.height
        let x = box.minX * size.width
        let y = (1 - box.maxY) * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

struct StatusBadge: View {
    let rectCount: Int
    var body: some View {
        Text(rectCount > 0 ? "检测到 \(rectCount) 个目标" : "正在扫描...")
            .font(.system(.caption, design: .monospaced))
            .padding(8)
            .background(.black.opacity(0.7))
            .foregroundColor(.green)
            .cornerRadius(8)
            .padding(.top, 40)
    }
}

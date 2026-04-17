import SwiftUI
import Vision
import AVFoundation

@MainActor
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    // 核心修改：将 [VNRectangleObservation] 改为 [CGRect]，因为 CGRect 是 Sendable 的安全值类型
    @Published var detectedRects: [CGRect] = []
    
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
        
        Task {
            self.captureSession.startRunning()
        }
        #endif
    }
    
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNDetectRectanglesRequest { request, error in
            if let results = request.results as? [VNRectangleObservation] {
                // 核心修改：在后台线程提取纯值类型 CGRect (Sendable)，丢弃无法跨线程的 Vision 对象
                let boundingBoxes = results.map { $0.boundingBox }
                
                Task { @MainActor in
                    // 传递 Sendable 数组，彻底消除 Swift 6 编译报错
                    self.detectedRects = boundingBoxes
                }
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
            
            Canvas { context, size in
                // 核心修改：迭代纯 CGRect 数组
                for box in adas.detectedRects {
                    let drawRect = calculateRect(box, in: size)
                    context.stroke(Path(drawRect), with: .color(.green), lineWidth: 2)
                    
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
    
    private func calculateRect(_ box: CGRect, in size: CGSize) -> CGRect {
        let w = box.width * size.width
        let h = box.height * size.height
        let x = box.minX * size.width
        let y = (1 - box.maxY) * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

import SwiftUI
import Vision
import AVFoundation

// 核心逻辑控制器：处理摄像头流与 Vision 推理
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detectedRects: [VNRectangleObservation] = []
    
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "adas.vision.queue")
    
    override init() {
        super.init()
        setupCaptureSession()
    }
    
    private func setupCaptureSession() {
        captureSession.beginConfiguration()
        
        // 1. 输入：默认摄像头
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else { return }
        
        if captureSession.canAddInput(videoInput) { captureSession.addInput(videoInput) }
        
        // 2. 输出：视频帧数据
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        
        captureSession.commitConfiguration()
        
        // 注意：在真机上需要这一行，但模拟器环境下 Session 不会真的启动
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    // 每一帧画面都会触发这个回调
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 3. 创建 Vision 请求：检测矩形（作为车辆/障碍物的临时方案）
        let request = VNDetectRectanglesRequest { [weak self] request, error in
            guard let results = request.results as? [VNRectangleObservation] else { return }
            DispatchQueue.main.async {
                self->detectedRects = results
            }
        }
        
        // 设置检测参数
        request.minimumConfidence = 0.5
        request.maximumObservations = 5
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}

struct ContentView: View {
    @StateObject private var adas = ADASController()
    
    var body: some View {
        ZStack {
            // 背景层：模拟摄像头预览
            Color.black.ignoresSafeArea()
            
            VStack {
                Text("ADAS 系统运行中 (系统内置检测模式)")
                    .font(.caption)
                    .foregroundColor(.green)
                    .padding(5)
                    .background(.black.opacity(0.5))
                Spacer()
            }
            
            // 渲染层：根据检测到的目标画框
            Canvas { context, size in
                for rect in adas.detectedRects {
                    // Vision 坐标系（左下角 0,0）转 SwiftUI 坐标系（左上角 0,0）
                    let width = rect.boundingBox.width * size.width
                    let height = rect.boundingBox.height * size.height
                    let x = rect.boundingBox.minX * size.width
                    let y = (1 - rect.boundingBox.maxY) * size.height
                    
                    let drawRect = CGRect(x: x, y: y, width: width, height: height)
                    
                    context.stroke(Path(drawRect), with: .color(.green), lineWidth: 2)
                    
                    let label = context.resolve(Text("目标已锁定").font(.system(size: 10)).foregroundColor(.green))
                    context.draw(label, at: CGPoint(x: drawRect.midX, y: drawRect.minY - 10))
                }
            }
        }
    }
}

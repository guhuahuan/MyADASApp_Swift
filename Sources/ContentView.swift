import CoreML
import Vision

class DetectionEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [VNRecognizedObjectObservation] = []
    let session = AVCaptureSession()
    
    // 强制指定模型输出名为 "var_2187" (根据你元数据中的输出 ID)
    private var requests = [VNRequest]()

    func setup() {
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        session.addInput(input)
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video_queue"))
        session.addOutput(output)

        // --- 核心修复：加载模型 ---
        guard let modelURL = Bundle.main.url(forResource: "yolo26x", withExtension: "mlmodelc") else { return }
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // 开启 GPU/ANE 加速
            let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL, configuration: config))
            
            let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
                if let results = req.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        // 过滤逻辑：yolo26x 的标签是小写的
                        let allowed = ["person", "car", "truck", "bus"]
                        self?.detections = results.filter { 
                            let label = $0.labels.first?.identifier ?? ""
                            return allowed.contains(label) && $0.confidence > 0.25 // 降低阈值测试
                        }
                    }
                }
            }
            
            // 关键：针对 imgsz [640, 640] 的对齐
            request.imageCropAndScaleOption = .scaleFill 
            self.requests = [request]
            
        } catch {
            print("模型初始化失败: \(error)")
        }

        DispatchQueue.global().async { self.session.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        try? handler.perform(self.requests)
    }
}

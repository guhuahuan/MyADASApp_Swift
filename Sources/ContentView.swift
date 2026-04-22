import SwiftUI
import AVFoundation
import Vision
import CoreML

@main
struct FSD_App: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

class DetectionEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [VNRecognizedObjectObservation] = []
    let session = AVCaptureSession()
    private var requests = [VNRequest]()

    func setup() {
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        if session.canAddInput(input) { session.addInput(input) }
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video_queue"))
        if session.canAddOutput(output) { session.addOutput(output) }
        
        // 关键：对应 yolo26x
        guard let modelURL = Bundle.main.url(forResource: "yolo26x", withExtension: "mlmodelc"),
              let visionModel = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else { 
            print("模型加载失败"); return 
        }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                DispatchQueue.main.async {
                    // 只保留人 (person) 和 车 (car/truck/bus)
                    let allowed = ["person", "car", "truck", "bus"]
                    self?.detections = results.filter { allowed.contains($0.labels.first?.identifier ?? "") }
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop
        self.requests = [request]
        
        DispatchQueue.global().async { self.session.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(requests)
    }
}

struct ContentView: View {
    @StateObject private var engine = DetectionEngine()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                CameraView(session: engine.session)
                ForEach(0..<engine.detections.count, id: \.self) { i in
                    let obs = engine.detections[i]
                    let rect = VNImageRectForNormalizedRect(obs.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let correctedRect = CGRect(x: rect.minX, y: geo.size.height - rect.maxY, width: rect.width, height: rect.height)
                    
                    let label = obs.labels.first?.identifier ?? ""
                    let isPerson = label == "person"
                    
                    Rectangle()
                        .path(in: correctedRect)
                        .stroke(isPerson ? Color.blue : Color.green, lineWidth: 3)
                        .overlay(
                            Text(label.uppercased())
                                .font(.caption.bold())
                                .foregroundColor(.white)
                                .background(isPerson ? Color.blue : Color.green)
                                .position(x: correctedRect.minX + 25, y: correctedRect.minY - 10)
                        )
                }
            }
        }
        .ignoresSafeArea()
        .onAppear { engine.setup() }
    }
}

struct CameraView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.frame = view.frame
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

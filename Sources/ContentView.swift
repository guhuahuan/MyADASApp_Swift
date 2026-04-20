import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreLocation

// MARK: - App Entry
@main
struct FSD_V4_App: App {
    var body: some Scene {
        WindowGroup {
            FSDMainView()
        }
    }
}

// MARK: - UI Layer
struct FSDMainView: View {
    @StateObject private var engine = ADASLogicEngine()
    
    var body: some View {
        ZStack {
            // 1. 相机底图
            CameraPreview(session: engine.captureSession)
                .edgesIgnoringSafeArea(.all)
            
            // 2. AI 识别层
            GeometryReader { geo in
                Canvas { context, size in
                    for detection in engine.detections {
                        let rect = engine.convertRect(detection.boundingBox, to: size)
                        context.stroke(Path(rect), with: .color(.cyan), lineWidth: 2)
                        
                        var title = detection.label
                        // --- 修复后的逻辑 ---
                        if engine.currentSpeed > 0.5 {
                            title += " \(String(format: "%.1f", engine.currentSpeed * 3.6))km/h"
                        }
                        
                        context.draw(Text(title).font(.caption).bold().foregroundColor(.cyan), 
                                     at: CGPoint(x: rect.minX, y: rect.minY - 10))
                    }
                }
            }
            
            // 3. 仪表盘覆盖层
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("FSD V4 ADAS").font(.title3).bold().foregroundColor(.white)
                        Text(engine.isModelLoaded ? "AI ACTIVE" : "AI LOADING...")
                            .font(.caption).foregroundColor(engine.isModelLoaded ? .green : .red)
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("\(Int(engine.currentSpeed * 3.6))").font(.system(size: 40, weight: .bold)).foregroundColor(.yellow)
                        Text("KM/H").font(.caption).foregroundColor(.yellow)
                    }
                }
                .padding()
                .background(LinearGradient(gradient: Gradient(colors: [.black.opacity(0.7), .clear]), startPoint: .top, endPoint: .bottom))
                
                Spacer()
                
                if let err = engine.errorMessage {
                    Text(err).font(.system(size: 12, design: .monospaced))
                        .padding(8).background(Color.red).foregroundColor(.white).cornerRadius(5)
                }
            }
        }
        .onAppear { engine.checkPermissions() }
    }
}

// MARK: - Logic Engine
class ADASLogicEngine: NSObject, ObservableObject, CLLocationManagerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detections: [Detection] = []
    @Published var currentSpeed: CLLocationSpeed = 0
    @Published var isModelLoaded = false
    @Published var errorMessage: String?
    
    let captureSession = AVCaptureSession()
    private let locationManager = CLLocationManager()
    private var requests = [VNRequest]()
    
    struct Detection {
        let label: String
        let boundingBox: CGRect
    }

    override init() {
        super.init()
        setupModel()
    }

    private func setupModel() {
        let possibleURLs = [
            Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
            Bundle.main.url(forResource: "yolov8l.mlpackage/Data/com.apple.CoreML/model", withExtension: "mlmodelc")
        ]
        
        guard let modelURL = possibleURLs.compactMap({ $0 }).first else {
            self.errorMessage = "ERROR: YOLOv8L.mlmodelc not found"
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let visionModel = try VNCoreMLModel(for: model)
            
            let objectRecognition = VNCoreMLRequest(model: visionModel) { (request, error) in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        self.detections = results.map { Detection(label: $0.labels.first?.identifier ?? "?", boundingBox: $0.boundingBox) }
                    }
                }
            }
            objectRecognition.imageCropAndScaleOption = .scaleFill
            self.requests = [objectRecognition]
            self.isModelLoaded = true
        } catch {
            self.errorMessage = "Model Load Failed: \(error.localizedDescription)"
        }
    }

    func checkPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { if $0 { self.setupCamera() } }
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func setupCamera() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        try? captureSession.addInput(AVCaptureDeviceInput(device: device))
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "vision_queue"))
        captureSession.addOutput(output)
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform(self.requests)
    }

    func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(x: rect.minX * size.width, y: (1 - rect.maxY) * size.height, width: rect.width * size.width, height: rect.height * size.height)
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentSpeed = locations.last?.speed ?? 0
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.layer.bounds
        view.layer.addSublayer(layer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

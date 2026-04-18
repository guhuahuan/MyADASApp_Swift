@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreMotion
import CoreLocation

// MARK: - 1. FSD 增强型数据结构
struct FSDTrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var velocityVector: CGVector
    var boundingBox: CGRect
    var isSideHazard: Bool
}

struct SafeModelBuffer: @unchecked Sendable {
    let model: VNCoreMLModel
}

enum ADASCameraMode: Sendable {
    case ultraWide // 城市模式
    case telephoto // 高速模式
}

// MARK: - 2. FSD 双摄联动引擎
@MainActor
class DualCamFSDEngine: NSObject, ObservableObject {
    @Published var cameraMode: ADASCameraMode = .ultraWide
    @Published var trackedObjects: [FSDTrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var sideWarning: Edge? = nil
    @Published var currentSpeed: Double = 0.0
    
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.fsd.dualcam.queue", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    private let locationManager = CLLocationManager()
    
    nonisolated(unsafe) private var safeModel: SafeModelBuffer?
    private var prevPositions: [String: CGPoint] = [:]
    private var currentInput: AVCaptureDeviceInput?

    override init() {
        super.init()
        setupLocation()
        loadModel()
        setupInitialCamera()
    }

    private func setupLocation() {
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func loadModel() {
        Task {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            if let url = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let mlModel = try? MLModel(contentsOf: url, configuration: config),
               let vModel = try? VNCoreMLModel(for: mlModel) {
                self.safeModel = SafeModelBuffer(model: vModel)
            }
        }
    }

    private func setupInitialCamera() {
        switchCamera(to: .ultraWide)
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        
        let capture = self.session
        DispatchQueue.global(qos: .userInitiated).async { capture.startRunning() }
    }

    // 动态镜头切换核心逻辑
    func updateSystemSpeed(_ speed: Double) {
        self.currentSpeed = speed
        let targetMode: ADASCameraMode = speed > 65.0 ? .telephoto : .ultraWide
        if targetMode != cameraMode {
            switchCamera(to: targetMode)
            cameraMode = targetMode
        }
    }

    private func switchCamera(to mode: ADASCameraMode) {
        session.beginConfiguration()
        if let input = currentInput { session.removeInput(input) }
        
        // 如果是高速模式，首选主摄(Wide)获取高像素以进行中心裁切，城市模式用超广角
        let deviceType: AVCaptureDevice.DeviceType = (mode == .telephoto) ? .builtInWideAngleCamera : .builtInUltraWideCamera
        
        guard let device = AVCaptureDevice.default(deviceType, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            session.commitConfiguration()
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
            currentInput = input
        }
        session.commitConfiguration()
    }
}

// MARK: - 3. 视觉处理与 ROI 动态调整
extension DualCamFSDEngine: AVCaptureVideoDataOutputSampleBufferDelegate, CLLocationManagerDelegate {
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        let speedKmH = max(0, location.speed * 3.6)
        Task { @MainActor in self.updateSystemSpeed(speedKmH) }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer),
              let container = self.safeModel else { return }
        
        let mode = Task { @MainActor in self.cameraMode }
        
        let detReq = VNCoreMLRequest(model: container.model) { req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeFSD(results) }
            }
        }

        // 关键逻辑：如果是高速模式，强制 Vision 关注画面中心区域（虚拟长焦）
        Task {
            let currentMode = await mode
            if currentMode == .telephoto {
                detReq.regionOfInterest = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
            }
        }

        let segReq = VNGenerateForegroundInstanceMaskRequest()
        try? VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .right).perform([segReq, detReq])
        
        if let maskObs = segReq.results?.first {
            if let cgMask = try? self.context.createCGImage(CIImage(cvPixelBuffer: maskObs.instanceMask), from: CIImage(cvPixelBuffer: maskObs.instanceMask).extent) {
                Task { @MainActor in self.occupancyMask = cgMask }
            }
        }
    }

    @MainActor
    private func analyzeFSD(_ observations: [VNRecognizedObjectObservation]) {
        var nextObjects: [FSDTrackedObject] = []
        var sideAlert: Edge? = nil
        
        for obs in observations {
            let label = obs.labels.first?.identifier ?? "obj"
            let currentCenter = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
            
            let prev = prevPositions[label] ?? currentCenter
            let vector = CGVector(dx: (currentCenter.x - prev.x) * 20, dy: (currentCenter.y - prev.y) * 20)
            prevPositions[label] = currentCenter
            
            let dist = 1.2 / (Float(obs.boundingBox.minY) + 0.05)
            
            if dist < 6.0 {
                if obs.boundingBox.midX < 0.25 { sideAlert = .leading }
                if obs.boundingBox.midX > 0.75 { sideAlert = .trailing }
            }
            
            nextObjects.append(FSDTrackedObject(id: UUID(), label: label, distance: dist, velocityVector: vector, boundingBox: obs.boundingBox, isSideHazard: dist < 5.0))
        }
        self.trackedObjects = nextObjects
        self.sideWarning = sideAlert
    }
}

// MARK: - 4. 终极 FSD 交互界面
struct FSDUltimateView: View {
    @StateObject var engine = DualCamFSDEngine()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 占用网络预览
            if let mask = engine.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            // 侧方流光报警
            HStack {
                Rectangle().fill(LinearGradient(colors: [.red.opacity(engine.sideWarning == .leading ? 0.8 : 0), .clear], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 70)
                Spacer()
                Rectangle().fill(LinearGradient(colors: [.clear, .red.opacity(engine.sideWarning == .trailing ? 0.8 : 0)], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 70)
            }.ignoresSafeArea()
            
            GeometryReader { geo in
                ForEach(engine.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // 预测矢量线
                    Path { p in
                        let start = CGPoint(x: rect.midX, y: geo.size.height - rect.minY)
                        p.move(to: start)
                        p.addLine(to: CGPoint(x: start.x + obj.velocityVector.dx * 80, y: start.y - obj.velocityVector.dy * 80))
                    }
                    .stroke(Color.yellow, style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
                    
                    // 特斯拉风格目标框
                    RoundedRectangle(cornerRadius: 2)
                        .stroke(obj.isSideHazard ? Color.red : Color.white, lineWidth: 1)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // HUD 信息面板
            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text("\(Int(engine.currentSpeed))").font(.system(size: 40, weight: .black, design: .monospaced))
                        Text("KM/H").font(.caption).foregroundColor(.gray)
                    }
                    Spacer()
                    Image(systemName: engine.cameraMode == .telephoto ? "scope" : "eye.circle")
                        .foregroundColor(engine.cameraMode == .telephoto ? .yellow : .cyan)
                    Text(engine.cameraMode == .telephoto ? "HIGH-SPEED (ROI FOCUS)" : "CITY (WIDE-ANGLE)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                }
                .padding()
                .background(Color.black.opacity(0.4))
                .foregroundColor(.white)
                Spacer()
            }
        }
    }
}

@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { FSDUltimateView() } }
}

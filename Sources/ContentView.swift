@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreMotion
import CoreLocation

// MARK: - 1. 数据结构定义
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
    case ultraWide // 城市：超广角
    case telephoto // 高速：主摄 ROI 裁切（模拟长焦）
}

// MARK: - 2. 核心引擎
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
            // 请确保项目中已导入 yolov8l.mlmodelc
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
        
        // 高速用主摄(1x)配合ROI裁切，城市用超广角(0.5x)
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

// MARK: - 3. 视觉处理与多模式适配
extension DualCamFSDEngine: AVCaptureVideoDataOutputSampleBufferDelegate, CLLocationManagerDelegate {
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        let speedKmH = max(0, location.speed * 3.6)
        Task { @MainActor in self.updateSystemSpeed(speedKmH) }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer),
              let container = self.safeModel else { return }
        
        // 显式捕获 Task 句柄以供后续 await 访问
        let modeTask = Task { @MainActor in self.cameraMode }
        
        let detReq = VNCoreMLRequest(model: container.model) { req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeFSD(results) }
            }
        }

        // 异步配置：高速模式下收缩 ROI（感兴趣区域）以实现虚拟长焦
        Task {
            let mode = await modeTask.value
            if mode == .telephoto {
                detReq.regionOfInterest = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
            }
        }

        let segReq = VNGenerateForegroundInstanceMaskRequest()
        try? VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .right).perform([segReq, detReq])
        
        if let maskObs = segReq.results?.first {
            let maskCI = CIImage(cvPixelBuffer: maskObs.instanceMask)
            if let cgMask = self.context.createCGImage(maskCI, from: maskCI.extent) {
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
            let vector = CGVector(dx: (currentCenter.x - prev.x) * 25, dy: (currentCenter.y - prev.y) * 25)
            prevPositions[label] = currentCenter
            
            // 测距算法
            let dist = 1.2 / (Float(obs.boundingBox.minY) + 0.05)
            
            // 侧方盲区报警
            if dist < 7.0 {
                if obs.boundingBox.midX < 0.22 { sideAlert = .leading }
                if obs.boundingBox.midX > 0.78 { sideAlert = .trailing }
            }
            
            nextObjects.append(FSDTrackedObject(id: UUID(), label: label, distance: dist, velocityVector: vector, boundingBox: obs.boundingBox, isSideHazard: dist < 6.0))
        }
        self.trackedObjects = nextObjects
        self.sideWarning = sideAlert
    }
}

// MARK: - 4. 终极 UI 界面
struct FSDUltimateView: View {
    @StateObject var engine = DualCamFSDEngine()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. 占用网络预览 (Occupancy)
            if let mask = engine.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.cyan.opacity(0.35))
                    .ignoresSafeArea()
            }
            
            // 2. 特斯拉风格侧方流光预警 (Side Warnings)
            HStack {
                Rectangle().fill(LinearGradient(colors: [.red.opacity(engine.sideWarning == .leading ? 0.8 : 0), .clear], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 80)
                Spacer()
                Rectangle().fill(LinearGradient(colors: [.clear, .red.opacity(engine.sideWarning == .trailing ? 0.8 : 0)], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 80)
            }.ignoresSafeArea()
            
            // 3. 目标追踪与矢量线 (FSD Vectors)
            GeometryReader { geo in
                ForEach(engine.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // 预测矢量线 (Yellow Predictive Path)
                    Path { p in
                        let start = CGPoint(x: rect.midX, y: geo.size.height - rect.minY)
                        p.move(to: start)
                        p.addLine(to: CGPoint(x: start.x + obj.velocityVector.dx * 120, y: start.y - obj.velocityVector.dy * 120))
                    }
                    .stroke(Color.yellow, style: StrokeStyle(lineWidth: 2.5, dash: [6, 4]))
                    
                    // 目标框
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(obj.isSideHazard ? Color.red : Color.white, lineWidth: 1.5)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 4. HUD 仪表盘
            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading) {
                        Text("\(Int(engine.currentSpeed))")
                            .font(.system(size: 50, weight: .black, design: .monospaced))
                        Text("KM/H").font(.caption).foregroundColor(.gray)
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Image(systemName: engine.cameraMode == .telephoto ? "scope" : "eye.circle")
                            .font(.title2)
                        Text(engine.cameraMode == .telephoto ? "HIGH-SPEED FOCUS" : "CITY WIDE-ANGLE")
                            .font(.system(size: 10, weight: .bold))
                    }.foregroundColor(engine.cameraMode == .telephoto ? .yellow : .cyan)
                }
                .padding(30)
                .background(Color.black.opacity(0.4))
                .foregroundColor(.white)
                Spacer()
            }
        }
    }
}

// MARK: - 5. 程序入口
@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { FSDUltimateView() } }
}

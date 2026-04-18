@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreImage
import CoreMotion
import CoreLocation

// MARK: - 线程安全容器
struct SafeModelContainer: @unchecked Sendable {
    let visionModel: VNCoreMLModel
}

struct TrackedObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var ttc: Float
    var isCutIn: Bool
    var boundingBox: CGRect
}

struct TrajectoryPath: Sendable {
    let points: [CGPoint]
}

// MARK: - ADAS 终极引擎
@MainActor
class ADASMasterViewModel: NSObject, ObservableObject {
    enum AlertLevel { case safe, warning, critical }
    
    @Published var trackedObjects: [TrackedObject] = []
    @Published var occupancyMask: CGImage?
    @Published var alertStatus: AlertLevel = .safe
    @Published var currentPath: TrajectoryPath?
    @Published var currentSpeed: Double = 0.0
    @Published var speedLimit: Int = 0
    @Published var debugInfo: String = ""
    
    private let captureSession = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.adas.ultra.engine", qos: .userInteractive)
    nonisolated(unsafe) private let context = CIContext()
    private let motionManager = CMMotionManager()
    private let locationManager = CLLocationManager()
    private let hapticGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private let synthesizer = AVSpeechSynthesizer()
    
    // 关键修复：使用非隔离安全变量
    nonisolated(unsafe) private var safeModel: SafeModelContainer?
    private var lastDistances: [String: Float] = [:]
    private var baselinePitch: Double = 0.0
    private var smoothPitch: Double = 0.0
    private var lastVoiceAlert = Date.distantPast
    
    override init() {
        super.init()
        setupSensors()
        setupVision()
        hapticGenerator.prepare()
    }

    private func setupSensors() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let data = data, let self = self else { return }
                if self.baselinePitch == 0.0 { self.baselinePitch = data.attitude.pitch }
                self.smoothPitch = 0.15 * (data.attitude.pitch - self.baselinePitch) + 0.85 * self.smoothPitch
            }
        }
        
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    private func setupVision() {
        Task {
            let config = MLModelConfiguration()
            config.computeUnits = .all 
            if let modelURL = Bundle.main.url(forResource: "yolov8l", withExtension: "mlmodelc"),
               let coreML = try? MLModel(contentsOf: modelURL, configuration: config),
               let visionModel = try? VNCoreMLModel(for: coreML) {
                self.safeModel = SafeModelContainer(visionModel: visionModel)
            }
            
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }
            
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: queue)
            if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
            
            // 修复：在非阻塞全局队列启动，但不直接引用 self.captureSession
            let session = self.captureSession
            DispatchQueue.global(qos: .userInitiated).async {
                session.startRunning()
            }
        }
    }
}

// MARK: - 核心算法与回调
extension ADASMasterViewModel: AVCaptureVideoDataOutputSampleBufferDelegate, CLLocationManagerDelegate {
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        if let location = locations.last {
            let speedKmH = max(0, location.speed * 3.6)
            Task { @MainActor in self.currentSpeed = speedKmH }
        }
    }

    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // OCR 识别
        let ocrRequest = VNRecognizeTextRequest { request, _ in
            if let results = request.results as? [VNRecognizedTextObservation] {
                // 修复：回到主线程解析
                Task { @MainActor in self.parseSpeedSigns(results) }
            }
        }
        
        // 目标检测
        guard let container = self.safeModel else { return }
        let detRequest = VNCoreMLRequest(model: container.visionModel) { request, _ in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                Task { @MainActor in self.analyzeWorld(results) }
            }
        }
        
        let segRequest = VNGenerateForegroundInstanceMaskRequest()
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([segRequest, ocrRequest, detRequest])
        
        if let observation = segRequest.results?.first {
            let ciMask = CIImage(cvPixelBuffer: observation.instanceMask)
            if let cgMask = self.context.createCGImage(ciMask, from: ciMask.extent) {
                Task { @MainActor in 
                    self.occupancyMask = cgMask 
                    self.updateTrajectory()
                }
            }
        }
    }

    @MainActor
    private func parseSpeedSigns(_ observations: [VNRecognizedTextObservation]) {
        for obs in observations {
            if let topCandidate = obs.topCandidates(1).first {
                let text = topCandidate.string.filter { $0.isNumber }
                if let val = Int(text), val >= 20 && val <= 120 {
                    self.speedLimit = val
                }
            }
        }
    }

    @MainActor
    private func analyzeWorld(_ observations: [VNRecognizedObjectObservation]) {
        var highestAlert: AlertLevel = .safe
        var newObjects: [TrackedObject] = []
        let isHighway = currentSpeed > 75.0
        
        for obs in observations {
            let label = obs.labels.first?.identifier ?? "Target"
            let yCompensated = Float(obs.boundingBox.origin.y) + Float(smoothPitch) * 0.65
            let distance = 1.2 / (yCompensated + 0.05)
            
            // 简单 TTC 逻辑
            let prevDist = lastDistances[label] ?? distance
            let velocity = (prevDist - distance) * 30.0 
            let ttc = distance / (velocity > 0 ? velocity : 0.1)
            let isCutIn = (obs.boundingBox.midX < 0.2 || obs.boundingBox.midX > 0.8) && distance < 12.0
            
            let dangerDist: Float = isHighway ? 45.0 : 6.0
            if distance < dangerDist || (isHighway && ttc < 2.5) {
                highestAlert = .critical
                if velocity > 5.0 { hapticGenerator.impactOccurred() }
                triggerVoice("危险")
            }
            
            newObjects.append(TrackedObject(id: UUID(), label: label, distance: distance, ttc: ttc, isCutIn: isCutIn, boundingBox: obs.boundingBox))
            lastDistances[label] = distance
        }
        
        self.trackedObjects = newObjects
        self.alertStatus = highestAlert
        self.debugInfo = "SPD: \(Int(currentSpeed)) | PITCH: \(String(format: "%.2f", smoothPitch))"
    }

    @MainActor
    private func updateTrajectory() {
        let isHighway = currentSpeed > 70.0
        let h_cam: Float = 1.2
        let f: Float = 0.85
        let pAdj = Float(smoothPitch)
        
        var pts: [CGPoint] = []
        let range = isHighway ? stride(from: 5.0, through: 60.0, by: 5.0) : stride(from: 2.0, through: 25.0, by: 2.5)
        
        for d in range {
            let yNorm = 1.0 - ((h_cam / Float(d)) * f + pAdj * 0.6)
            let laneW = (3.5 / Float(d)) * f
            pts.append(CGPoint(x: CGFloat(0.5 - laneW/2), y: CGFloat(yNorm)))
            pts.append(CGPoint(x: CGFloat(0.5 + laneW/2), y: CGFloat(yNorm)))
        }
        self.currentPath = TrajectoryPath(points: pts)
    }

    private func triggerVoice(_ msg: String) {
        if Date().timeIntervalSince(lastVoiceAlert) > 4.0 {
            let utterance = AVSpeechUtterance(string: msg)
            utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
            synthesizer.speak(utterance)
            lastVoiceAlert = Date()
        }
    }
}

// MARK: - 视图层 (ADAS HUD)
struct ADASContentView: View {
    @StateObject var viewModel = ADASMasterViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(viewModel.alertStatus == .critical ? .red.opacity(0.4) : .cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            GeometryReader { geo in
                // 3D 轨迹线投影
                if let path = viewModel.currentPath, path.points.count > 1 {
                    Path { p in
                        let w = geo.size.width
                        let h = geo.size.height
                        p.move(to: CGPoint(x: path.points[0].x * w, y: path.points[0].y * h))
                        for i in stride(from: 2, to: path.points.count, by: 2) { p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h)) }
                        p.move(to: CGPoint(x: path.points[1].x * w, y: path.points[1].y * h))
                        for i in stride(from: 3, to: path.points.count, by: 2) { p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h)) }
                    }
                    .stroke(viewModel.alertStatus == .critical ? Color.red : Color.cyan, lineWidth: 4)
                }
                
                // 目标检测框
                ForEach(viewModel.trackedObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    let color: Color = obj.isCutIn ? .orange : (obj.distance < 10 ? .red : .green)
                    
                    VStack(alignment: .leading, spacing: 0) {
                        Text("\(obj.label.uppercased()) \(String(format: "%.1fm", obj.distance))")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .padding(2).background(color).foregroundColor(.white)
                        Rectangle().stroke(color, lineWidth: 2).frame(width: rect.width, height: rect.height)
                    }
                    .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 仪表盘 HUD
            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading) {
                        Text("\(Int(viewModel.currentSpeed))")
                            .font(.system(size: 60, weight: .black, design: .monospaced))
                            .foregroundColor(.white)
                        Text("KM/H").font(.caption).foregroundColor(.gray)
                    }
                    Spacer()
                    if viewModel.speedLimit > 0 {
                        ZStack {
                            Circle().strokeBorder(Color.red, lineWidth: 6).background(Circle().fill(.white))
                            Text("\(viewModel.speedLimit)").font(.system(size: 22, weight: .bold)).foregroundColor(.black)
                        }.frame(width: 60, height: 60)
                    }
                }.padding(30)
                Spacer()
                Text(viewModel.debugInfo).font(.system(size: 10, design: .monospaced)).foregroundColor(.white).padding(5).background(.black.opacity(0.5))
            }
        }
    }
}

@main
struct ADASApp: App {
    var body: some Scene { WindowGroup { ADASContentView() } }
}

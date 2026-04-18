@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreLocation

// MARK: - 镜头配置枚举
enum ADASCameraMode {
    case ultraWide // 城市：大视野、占用网络优先
    case telephoto // 高速：远距、运动矢量优先
    
    var deviceType: AVCaptureDevice.DeviceType {
        switch self {
        case .ultraWide: return .builtInUltraWideCamera
        case .telephoto: return .builtInWideAngleCamera // iPhone 15 主摄具备极高像素，适合做远距虚拟长焦
        }
    }
}

@MainActor
class DualCamFSDEngine: NSObject, ObservableObject {
    @Published var cameraMode: ADASCameraMode = .ultraWide
    @Published var fsdObjects: [FSDTrackedObject] = []
    
    private let session = AVCaptureSession()
    private var currentInput: AVCaptureDeviceInput?
    private let queue = DispatchQueue(label: "com.dualcam.fsd", qos: .userInteractive)
    
    // 速度监控与镜头自动切换
    func updateCameraSystem(speed: Double) {
        let newMode: ADASCameraMode = speed > 65.0 ? .telephoto : .ultraWide
        
        if newMode != self.cameraMode {
            self.cameraMode = newMode
            self.switchCamera(to: newMode)
        }
    }

    private func switchCamera(to mode: ADASCameraMode) {
        session.beginConfiguration()
        if let currentInput = currentInput { session.removeInput(currentInput) }
        
        guard let device = AVCaptureDevice.default(mode.deviceType, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            session.commitConfiguration()
            return
        }
        
        if session.canAddInput(input) {
            session.addInput(input)
            self.currentInput = input
        }
        
        // 针对高速模式优化：锁定高帧率以增强运动矢量准确度
        try? device.lockForConfiguration()
        if mode == .telephoto {
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
        }
        device.unlockForConfiguration()
        
        session.commitConfiguration()
    }
}

// MARK: - 5. 视觉逻辑增强 (Vision ROI 动态调整)
extension DualCamFSDEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 创建不同的请求区域
        let request = VNCoreMLRequest(model: model) { req, _ in
             // 处理逻辑...
        }
        
        // 高速模式下：只关注画面中心 50% 区域（提升远距像素密度）
        if Task { @MainActor in self.cameraMode } == .telephoto {
            request.regionOfInterest = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        } else {
            request.regionOfInterest = CGRect(x: 0, y: 0, width: 1, height: 1)
        }
        
        try? VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .right).perform([request])
    }
}

// MARK: - UI 层 (增加镜头状态显示)
struct DualCamFSDView: View {
    @StateObject var engine = DualCamFSDEngine()
    
    var body: some View {
        ZStack {
            // 基础视频流预览...
            
            VStack {
                HStack {
                    // 镜头状态指示器 (特斯拉风格图标)
                    Image(systemName: engine.cameraMode == .telephoto ? "scope" : "eye.circle")
                        .foregroundColor(engine.cameraMode == .telephoto ? .yellow : .cyan)
                    Text(engine.cameraMode == .telephoto ? "HIGH-SPEED MODE (TELE)" : "CITY MODE (WIDE)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                    Spacer()
                }.padding().background(Color.black.opacity(0.3))
                Spacer()
            }
            
            // 渲染矢量线和检测框...
        }
    }
}

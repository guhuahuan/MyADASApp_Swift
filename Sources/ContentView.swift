@preconcurrency import SwiftUI
@preconcurrency import Vision
import AVFoundation
import CoreImage
import CoreMotion

// MARK: - 3D 轨迹线模型
struct TrajectoryPath: Sendable {
    let points: [CGPoint] // 屏幕坐标系下的 3D 投影点序列
}

@MainActor
class ADASMasterViewModel: NSObject, ObservableObject {
    // ... 原有属性保持不变 ...
    @Published var occupancyMask: CGImage?
    @Published var alertStatus: AlertLevel = .safe
    
    // 轨迹线属性
    @Published var currentPath: TrajectoryPath?
    
    // ... Core Motion 属性保持不变 ...
    nonisolated(unsafe) private var context = CIContext()
    
    override init() {
        super.init()
        setupSystem()
        startMotionUpdates()
    }

    // MARK: - 核心算法：3D 轨迹线投影 (Feature 1)
    private func updateTrajectoryProjection() {
        // 利用 IMU 补偿后的 Pitch 角
        let compensatedPitch = Float(self.baselinePitch + self.smoothPitchOffset)
        
        // 相机内参模拟 (针对 iPhone 15 焦距)
        let f: Float = 0.8 // 归一化焦距系数
        let cameraHeight: Float = 1.2 // H = 1.2米

        var projectedPoints: [CGPoint] = []
        
        // 我们预测未来 2 米到 30 米的路径
        for distance in stride(from: 2.0, through: 30.0, by: 2.0) {
            let d = Float(distance)
            
            // 1. 计算物体底部的 3D Y 坐标 (Geometry-based)
            // 公式：y = H / D * f
            let worldY = (cameraHeight / d) * f
            
            // 2. 补偿手机俯仰角造成的视觉偏移
            let compensatedY = worldY + compensatedPitch * 0.4 // 0.4 为补偿增益

            // 3. 将 3D 物理坐标转换为归一化屏幕坐标 (0.0 - 1.0)
            let normalizedY = 1.0 - compensatedY // 屏幕底为 0，顶为 1

            // 计算该距离处的车道宽度 (假设标准车道宽 3.5 米，自车占中心)
            let laneWidthAtDist = (3.5 / d) * f
            
            // 左右两条线的 X 坐标
            let leftX = 0.5 - (laneWidthAtDist / 2.0)
            let rightX = 0.5 + (laneWidthAtDist / 2.0)
            
            projectedPoints.append(CGPoint(x: CGFloat(leftX), y: CGFloat(normalizedY)))
            projectedPoints.append(CGPoint(x: CGFloat(rightX), y: CGFloat(normalizedY)))
        }
        
        // 整理为左右两条线的点序列
        self.currentPath = TrajectoryPath(points: projectedPoints)
    }

    // ... setupSystem 保持不变 ...
}

// MARK: - 推理回调 (整合轨迹更新)
extension ADASMasterViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 原有检测和分割请求保持不变 ...
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([segRequest, detRequest])
        
        // 占用网络渲染逻辑保持不变 ...

        // 功能 1: 在每一帧推理后，更新 3D 轨迹线投影
        Task { @MainActor in
            self.updateTrajectoryProjection()
        }
    }
    
    // ... analyzeWorld 逻辑保持不变 ...
}

// MARK: - UI 布局 (特斯拉 4.0 界面)
struct ContentView: View {
    @StateObject var viewModel = ADASMasterViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 相机背景和检测框省略... (CameraPreview + Detections)
            
            // 1. 占用网络层 (Feature 3)
            if let mask = viewModel.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(viewModel.alertStatus == .critical ? .red.opacity(0.4) : .cyan.opacity(0.3))
                    .ignoresSafeArea()
            }
            
            // 2. 功能 1: 3D 轨迹线投影渲染
            GeometryReader { geo in
                if let path = viewModel.currentPath {
                    Path { p in
                        let w = geo.size.width
                        let h = geo.size.height
                        
                        // 绘制左侧轨迹线
                        p.move(to: CGPoint(x: path.points[0].x * w, y: path.points[0].y * h))
                        for i in stride(from: 2, to: path.points.count, by: 2) {
                            p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h))
                        }
                        
                        // 绘制右侧轨迹线
                        p.move(to: CGPoint(x: path.points[1].x * w, y: path.points[1].y * h))
                        for i in stride(from: 3, to: path.points.count, by: 2) {
                            p.addLine(to: CGPoint(x: path.points[i].x * w, y: path.points[i].y * h))
                        }
                    }
                    // 动态车道线颜色：青色为安全，红色为预警
                    .stroke(viewModel.alertStatus == .critical ? Color.red : Color.cyan, lineWidth: 3)
                }
            }
            
            // 3. 功能 4: 盲区流光预警省略... (Lateral Alerts)
            
            // 底部特斯拉风状态面板省略... (Bottom Panel)
        }
    }
}

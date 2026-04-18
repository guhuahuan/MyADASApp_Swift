@preconcurrency import SwiftUI
@preconcurrency import Vision
import CoreMotion

// MARK: - FSD 增强型数据结构
struct FSDObject: Identifiable, Sendable {
    let id: UUID
    var label: String
    var distance: Float
    var velocityVector: CGVector // 运动矢量线
    var boundingBox: CGRect
    var isSideHazard: Bool // 侧方威胁
}

@MainActor
class FSDViewModel: NSObject, ObservableObject {
    @Published var fsdObjects: [FSDObject] = []
    @Published var sideWarning: Edge? // nil, .leading, or .trailing
    @Published var occupancyMask: CGImage?
    @Published var autoCalibratedPitch: Double = 0.0
    
    private let motion = CMMotionManager()
    private var lastFrameObjects: [String: (center: CGPoint, dist: Float, time: Date)] = [:]
    
    // 1. 运动预测向量逻辑 (Reference Tesla FSD Path Prediction)
    func calculateVectors(for obs: VNRecognizedObjectObservation, currentDist: Float) -> CGVector {
        let currentCenter = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
        let label = obs.labels.first?.identifier ?? "unknown"
        
        guard let prev = lastFrameObjects[label] else {
            lastFrameObjects[label] = (currentCenter, currentDist, Date())
            return .zero
        }
        
        // 计算位移矢量
        let dx = (currentCenter.x - prev.center.x) * 10.0 // 放大位移便于视觉呈现
        let dy = (currentCenter.y - prev.center.y) * 10.0
        
        lastFrameObjects[label] = (currentCenter, currentDist, Date())
        return CGVector(dx: dx, dy: dy)
    }

    // 2. 边缘流光预警 (Blind Spot / Side Alert)
    func checkSideHazards(obs: VNRecognizedObjectObservation, dist: Float) -> Edge? {
        let x = obs.boundingBox.midX
        if dist < 5.0 {
            if x < 0.15 { return .leading } // 左侧贴身
            if x > 0.85 { return .trailing } // 右侧贴身
        }
        return nil
    }

    // 3. 自动校准逻辑 (Self-Calibration)
    // 模拟 FSD 在行驶初期寻找地平消失点
    func autoCalibrate(objects: [VNRecognizedObjectObservation]) {
        // 寻找远方车辆的聚集点作为消失点参考
        let remoteCars = objects.filter { $0.boundingBox.origin.y > 0.4 && $0.boundingBox.origin.y < 0.6 }
        if remoteCars.count > 3 {
            let avgY = remoteCars.map { $0.boundingBox.midY }.reduce(0, +) / CGFloat(remoteCars.count)
            let pitchError = Double(avgY - 0.5) * 0.1
            self.autoCalibratedPitch -= pitchError // 逐渐逼近真实地平线
        }
    }

    @MainActor
    func processFSD(observations: [VNRecognizedObjectObservation]) {
        var newFSDObjects: [FSDObject] = []
        var activeSideWarning: Edge? = nil
        
        autoCalibrate(objects: observations)
        
        for obs in observations {
            let dist = 1.2 / (Float(obs.boundingBox.minY) + Float(autoCalibratedPitch) + 0.05)
            let vector = calculateVectors(for: obs, currentDist: dist)
            let side = checkSideHazards(obs: obs, dist: dist)
            
            if side != nil { activeSideWarning = side }
            
            newFSDObjects.append(FSDObject(
                id: UUID(),
                label: obs.labels.first?.identifier ?? "",
                distance: dist,
                velocityVector: vector,
                boundingBox: obs.boundingBox,
                isSideHazard: side != nil
            ))
        }
        self.fsdObjects = newFSDObjects
        self.sideWarning = activeSideWarning
    }
}

// MARK: - 视觉呈现层 (特斯拉风格 UI)
struct FSDContainerView: View {
    @StateObject var vm = FSDViewModel()
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 占用网络格栅 (Occupancy Grid)
            if let mask = vm.occupancyMask {
                Image(decorative: mask, scale: 1.0)
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.blue.opacity(0.2))
                    .overlay(
                        // 叠加格栅纹理
                        Image("grid_pattern").resizable(resizingMode: .tile).opacity(0.1)
                    )
                    .ignoresSafeArea()
            }
            
            // 边缘流光预警 (特斯拉侧方变红效果)
            HStack {
                Rectangle().fill(LinearGradient(colors: [.red.opacity(vm.sideWarning == .leading ? 0.6 : 0), .clear], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 60)
                Spacer()
                Rectangle().fill(LinearGradient(colors: [.clear, .red.opacity(vm.sideWarning == .trailing ? 0.6 : 0)], startPoint: .leading, endPoint: .trailing))
                    .frame(width: 60)
            }.ignoresSafeArea()
            
            GeometryReader { geo in
                ForEach(vm.fsdObjects) { obj in
                    let rect = VNImageRectForNormalizedRect(obj.boundingBox, Int(geo.size.width), Int(geo.size.height))
                    
                    // 动态矢量线 (Predictive Path)
                    Path { path in
                        let start = CGPoint(x: rect.midX, y: geo.size.height - rect.minY)
                        path.move(to: start)
                        path.addLine(to: CGPoint(x: start.x + obj.velocityVector.dx * 50, y: start.y - obj.velocityVector.dy * 50))
                    }
                    .stroke(Color.yellow, style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [5, 3]))
                    
                    // 简化后的 FSD 物体框
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(obj.isSideHazard ? Color.red : Color.white.opacity(0.6), lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: geo.size.height - rect.midY)
                }
            }
            
            // 顶部状态栏
            VStack {
                Text("AUTOPILOT CALIBRATED")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.blue)
                    .padding(4).background(Color.black.opacity(0.4))
                Spacer()
            }
        }
    }
}

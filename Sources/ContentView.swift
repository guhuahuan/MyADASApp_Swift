import SwiftUI
import Vision
import AVFoundation

// 核心逻辑控制器：负责摄像头和推理
class ADASController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var detectedObjects: [VNRecognizedObjectObservation] = []
    private var sequenceHandler = VNSequenceRequestHandler()
    
    // 这里未来加载你的 YOLO CoreML 模型
    func setupVision() {
        // 示例：let model = try? VNCoreMLModel(for: YourYOLOModel().model)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 在这里执行推理请求
        // 推理结果更新到 @Published 变量，驱动 UI 刷新
    }
}

struct ContentView: View {
    @StateObject private var adas = ADASController()
    
    var body: some View {
        ZStack {
            CameraPreviewHolder() // 之前的摄像头组件
                .ignoresSafeArea()
            
            // 动态检测框绘制
            Canvas { context, size in
                for object in adas.detectedObjects {
                    // 将 Vision 的归一化坐标转换为屏幕坐标
                    let rect = VNImageRectForNormalizedRect(object.boundingBox, Int(size.width), Int(size.height))
                    
                    context.stroke(Path(rect), with: .color(.green), lineWidth: 2)
                    
                    let label = object.labels.first?.identifier ?? "未知"
                    let text = context.resolve(Text(label).bold().foregroundColor(.green))
                    context.draw(text, at: CGPoint(x: rect.midX, y: rect.minY - 15))
                }
            }
            
            // 安全预警叠加层
            VStack {
                if adas.detectedObjects.count > 0 {
                    Text("警告：前方检测到目标")
                        .padding()
                        .background(.red.opacity(0.8))
                        .cornerRadius(12)
                        .transition(.move(edge: .top))
                }
                Spacer()
            }
            .padding(.top, 50)
        }
    }
}

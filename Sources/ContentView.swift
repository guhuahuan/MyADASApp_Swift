import SwiftUI
import AVFoundation

// 摄像头预览包装器
struct CameraPreviewHolder: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black // 模拟摄像头背景
        
        let label = UILabel()
        label.text = "摄像头画面占位"
        label.textColor = .white
        label.textAlignment = .center
        label.frame = view.bounds
        view.addSubview(label)
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct ContentView: View {
    var body: some View {
        ZStack {
            // 底层：摄像头预览
            CameraPreviewHolder()
                .ignoresSafeArea()
            
            // 中层：ADAS 叠加信息
            VStack {
                HStack {
                    HStack {
                        Circle()
                            .fill(.green)
                            .frame(width: 10, height: 10)
                        Text("ADAS 系统已激活")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.white)
                    }
                    .padding(10)
                    .background(BlurView(style: .systemUltraThinMaterialDark))
                    .cornerRadius(10)
                    
                    Spacer()
                }
                .padding()
                
                Spacer()
            }
            
            // 顶层：绘图层（YOLO 检测框模拟）
            Canvas { context, size in
                let rect = CGRect(x: size.width * 0.2, y: size.height * 0.35, width: size.width * 0.6, height: size.height * 0.4)
                
                // 画检测框
                context.stroke(Path(rect), with: .color(.red), lineWidth: 3)
                
                // 画标签文本 - 修正后的语法
                let resolvedText = context.resolve(Text("前方车辆 - 距离: 15m").bold().foregroundColor(.red))
                context.draw(resolvedText, at: CGPoint(x: rect.midX, y: rect.minY - 20))
            }
        }
    }
}

// 辅助视图：毛玻璃效果
struct BlurView: UIViewRepresentable {
    var style: UIBlurEffect.Style
    func makeUIView(context: Context) -> UIVisualEffectView {
        return UIVisualEffectView(effect: UIBlurEffect(style: style))
    }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

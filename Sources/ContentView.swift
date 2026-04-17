import SwiftUI

struct ContentView: View {
    var body: some View {
        ZStack {
            // 背景设为黑色，模拟摄像头层
            Color.black.ignoresSafeArea()
            
            VStack {
                Text("ADAS 系统已启动")
                    .font(.headline)
                    .foregroundColor(.green)
                    .padding()
                    .background(Capsule().fill(Color.black.opacity(0.8)))
                
                Spacer()
                
                // 未来在这里叠加 YOLO 识别框
                RoundedRectangle(cornerRadius: 15)
                    .stroke(Color.red, lineWidth: 2)
                    .frame(width: 200, height: 150)
                    .overlay(
                        Text("前方障碍物检测区")
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(5),
                        alignment: .topLeading
                    )
                
                Spacer()
                
                Button(action: {
                    print("开始录制/分析")
                }) {
                    Label("开始分析", systemImage: "play.circle.fill")
                        .font(.title)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(15)
                }
            }
            .padding()
        }
    }
}

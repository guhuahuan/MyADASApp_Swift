import SwiftUI
import Foundation

struct ContentView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack {
                Text("ADAS 系统已启动")
                    .font(.headline)
                    .foregroundColor(.green)
                    .padding()
                    .background(Capsule().fill(Color.black.opacity(0.8)))
                
                Spacer()
                
                RoundedRectangle(cornerRadius: 15)
                    .stroke(Color.red, lineWidth: 2)
                    .frame(width: 200, height: 150)
                
                Spacer()
                
                Button(action: {
                    print("开始录制/分析")
                }) {
                    Text("开始分析")
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

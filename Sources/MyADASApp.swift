import SwiftUI

@main
struct MyADASApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                // 强制使用暗色模式，有效降低工业/车载场景下的屏幕眩光
                // Enforce dark mode to reduce screen glare in industrial/automotive scenarios
                .preferredColorScheme(.dark) 
        }
    }
}

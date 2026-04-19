// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MyADASApp_Swift",
    platforms: [.iOS(.v17)],
    products: [
        .executable(name: "MyADASApp_Swift", targets: ["MyADASApp_Swift"])
    ],
    targets: [
        .executableTarget(
            name: "MyADASApp_Swift",
            path: "Sources" // 确保你的 ContentView.swift 在 Sources 文件夹里
        )
    ]
)

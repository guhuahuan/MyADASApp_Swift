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
            dependencies: [],
            path: "Sources",
            resources: [
                // 关键点：改为你新上传的模型文件名
                .process("yolo26x.mlpackage")
            ]
        )
    ]
)

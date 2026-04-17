// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MyADASApp",
    platforms: [.iOS(.v17)],
    products: [
        .executable(name: "MyADASApp", targets: ["MyADASApp"])
    ],
    targets: [
        .executableTarget(
            name: "MyADASApp",
            path: "Sources"
        )
    ]

// swift-tools-version: 6.0
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
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-sdk", "/Applications/Xcode_16.4.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk", "-target", "arm64-apple-ios17.0-simulator"])
            ]
        )
    ]
)

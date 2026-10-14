// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AudioRef",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "AudioRef", targets: ["AudioRef"]),
        .executable(name: "audioref-capture", targets: ["audioref-capture"]),
    ],
    targets: [
        .target(name: "CaptureCore", path: "Sources/CaptureCore"),
        .executableTarget(name: "AudioRef", dependencies: ["CaptureCore"], path: "Sources/AudioRef"),
        .executableTarget(name: "audioref-capture", dependencies: ["CaptureCore"], path: "Sources/audioref-capture"),
    ]
)

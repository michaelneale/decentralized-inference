// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MeshLLMMenuBar",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "MeshLLMMenuBar", targets: ["MeshLLMMenuBar"])
    ],
    targets: [
        .executableTarget(
            name: "MeshLLMMenuBar",
            path: "Sources",
            resources: [
                .copy("Resources")
            ]
        )
    ]
)

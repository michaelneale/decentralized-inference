// membench-fingerprint.swift — Memory bandwidth fingerprint for Apple Silicon
// Run with: swiftc -O membench-fingerprint.swift -o membench-fingerprint && ./membench-fingerprint

import Metal
import Foundation
import Darwin

struct ChipBandwidth {
    let key: String
    let variant: String
    let gbps: Double
}

let ratedBandwidth: [ChipBandwidth] = [
    // M5
    .init(key: "M5",       variant: "all",                       gbps: 153),
    .init(key: "M5 Pro",   variant: "all",                       gbps: 307),
    .init(key: "M5 Max",   variant: "18-core CPU / 32-core GPU", gbps: 460),
    .init(key: "M5 Max",   variant: "18-core CPU / 40-core GPU", gbps: 614),

    // M4
    .init(key: "M4",       variant: "all",                       gbps: 120),
    .init(key: "M4 Pro",   variant: "all",                       gbps: 273),
    .init(key: "M4 Max",   variant: "14-core CPU / 32-core GPU", gbps: 410),
    .init(key: "M4 Max",   variant: "16-core CPU / 40-core GPU", gbps: 546),

    // M3
    .init(key: "M3",       variant: "all",                       gbps: 100),
    .init(key: "M3 Pro",   variant: "all",                       gbps: 150),
    .init(key: "M3 Max",   variant: "14-core CPU / 30-core GPU", gbps: 300),
    .init(key: "M3 Max",   variant: "16-core CPU / 40-core GPU", gbps: 400),
    .init(key: "M3 Ultra", variant: "all",                       gbps: 819),

    // M2
    .init(key: "M2",       variant: "all",                       gbps: 100),
    .init(key: "M2 Pro",   variant: "all",                       gbps: 200),
    .init(key: "M2 Max",   variant: "all",                       gbps: 400),
    .init(key: "M2 Ultra", variant: "all",                       gbps: 800),

    // M1
    .init(key: "M1",       variant: "all",                       gbps:  68),
    .init(key: "M1 Pro",   variant: "all",                       gbps: 200),
    .init(key: "M1 Max",   variant: "all",                       gbps: 400),
    .init(key: "M1 Ultra", variant: "all",                       gbps: 800),
]

func physicalCPUCount() -> Int {
    var count: Int32 = 0
    var size = MemoryLayout<Int32>.size
    sysctlbyname("hw.physicalcpu", &count, &size, nil, 0)
    return Int(count)
}

func ratedFor(_ deviceName: String) -> (gbps: Double, estimated: Bool)? {
    let candidates = ratedBandwidth
        .filter { deviceName.contains($0.key) }
        .sorted { $0.key.count > $1.key.count }

    guard !candidates.isEmpty else { return nil }

    let bestKey  = candidates[0].key
    let matching = candidates.filter { $0.key == bestKey }

    if matching.allSatisfy({ $0.variant == "all" }) {
        return (matching[0].gbps, false)
    }

    let cpuCores     = physicalCPUCount()
    let cpuPattern   = "\(cpuCores)-core CPU"
    let cpuMatches   = matching.filter { $0.variant.contains(cpuPattern) }

    if cpuMatches.count == 1 {
        return (cpuMatches[0].gbps, false)
    }

    // CPU count alone is ambiguous (e.g. M5 Max: both variants are 18-core CPU, differ only in GPU count).
    let lowestVariant = matching.min(by: { $0.gbps < $1.gbps })!
    return (lowestVariant.gbps, true)
}

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void memread(
    device const float4*  src  [[buffer(0)]],
    device       float*   sink [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float4 v = src[id];
    if (v.x == 9999999.0f) sink[0] = v.x;
}
"""

let jsonMode = CommandLine.arguments.contains("--json")

guard let device = MTLCreateSystemDefaultDevice() else {
    if jsonMode {
        print(#"{"error":"Metal device not available"}"#)
    } else {
        print("Metal device not available")
    }
    exit(1)
}

let library  = try! device.makeLibrary(source: shaderSource, options: nil)
let pso      = try! device.makeComputePipelineState(function: library.makeFunction(name: "memread")!)
let queue    = device.makeCommandQueue()!
let sink     = device.makeBuffer(length: 16, options: .storageModeShared)!

let bufferBytes  = 512 * 1024 * 1024  // 512 MB — safely above SLC on all current/near-future Apple Silicon
let float4Bytes  = MemoryLayout<SIMD4<Float>>.stride
let elementCount = bufferBytes / float4Bytes

let buf = device.makeBuffer(length: bufferBytes, options: .storageModeShared)!
let ptr = buf.contents().bindMemory(to: Float.self, capacity: bufferBytes / 4)
for i in stride(from: 0, to: bufferBytes / 4, by: max(1, bufferBytes / 4 / 1024)) {
    ptr[i] = Float(i % 256)
}

let tpg  = MTLSize(width: pso.maxTotalThreadsPerThreadgroup, height: 1, depth: 1)
let grid = MTLSize(width: elementCount, height: 1, depth: 1)

func dispatch() -> Double {
    let cmd = queue.makeCommandBuffer()!
    var elapsed = 0.0
    cmd.addCompletedHandler { b in elapsed = b.gpuEndTime - b.gpuStartTime }
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(buf, offset: 0, index: 0)
    enc.setBuffer(sink, offset: 0, index: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    return elapsed
}

for _ in 0..<3 { _ = dispatch() }

let runs = 20
let wallStart = Date()
var gbps: [Double] = []
for _ in 0..<runs {
    let t = dispatch()
    gbps.append(Double(bufferBytes) / t / 1e9)
}

let sorted = gbps.sorted()
let p50    = sorted[runs / 2]
let p90    = sorted[Int(Double(runs) * 0.90) - 1]
let noise  = (p90 - p50) / p90 * 100.0

let deviceName  = device.name
let ratedResult  = ratedFor(deviceName)
let rated        = ratedResult?.gbps
let estimated    = ratedResult?.estimated ?? false
let efficiency   = rated.map { p90 / $0 * 100.0 }
let runtimeSecs = Date().timeIntervalSince(wallStart)

if jsonMode {
    var fields: [String] = [
        #""device":"\#(deviceName)""#,
        #""buffer_mb":512"#,
        String(format: #""runs":%d"#, runs),
        String(format: #""p50_gbps":%.2f"#, p50),
        String(format: #""p90_gbps":%.2f"#, p90),
        String(format: #""noise_pct":%.2f"#, noise),
        String(format: #""runtime_s":%.3f"#, runtimeSecs),
    ]
    if let r = rated {
        fields.append(String(format: #""rated_gbps":%.0f"#, r))
        fields.append(#""rated_estimated":\#(estimated ? "true" : "false")"#)
    }
    if let e = efficiency {
        fields.append(String(format: #""efficiency_pct":%.2f"#, e))
    }
    print("[{\(fields.joined(separator: ","))}]")
} else {
    let estimatedTag = estimated ? " (estimated)" : ""
    let ratedStr = rated.map { String(format: "%.0f GB/s rated\(estimatedTag)", $0) } ?? "rated: unknown"
    let effStr   = efficiency.map { String(format: "  efficiency: %.1f%%", $0) } ?? ""

    print("=== Memory Bandwidth Fingerprint ===")
    print("Device : \(deviceName)  (\(ratedStr))")
    print("Buffer : 512 MB read-only  (\(runs) runs)")
    print(String(format: "p50    : %.1f GB/s", p50))
    print(String(format: "p90    : %.1f GB/s\(effStr)", p90))
    print(String(format: "noise  : %.1f%%  (p90-p50 spread — lower is better)", noise))
    print(String(format: "runtime: %.2fs", runtimeSecs))
}

// membench-fingerprint.hip — Memory bandwidth fingerprint for AMD GPUs (ROCm/HIP)
// Build:  hipcc -O3 -std=c++17 -o membench-fingerprint-hip membench-fingerprint.hip
//         Requires ROCm installed (typically /opt/rocm). No static-link equivalent exists.
//         Runtime dep: /opt/rocm/lib/libamdhip64.so
// Run:    ./membench-fingerprint-hip [--json]
//
// Cross-check rated bandwidth with:
//   rocm-smi --showmeminfo vram          (CLI)
//   rocm-smi -a --json                   (all metrics as JSON)
//   amdsmi_get_gpu_vram_info()            (C++ API, libamd_smi)
//
// HBM note: hipDeviceProp_t.memoryClockRate may already report the effective
// (post-DDR-doubled) rate on Instinct/CDNA cards. The DDR factor is set to ×1
// for detected HBM devices (gfx90x, gfx94x) and ×2 for GDDR (RDNA/RX series).
// Verify against rocm-smi if rated_estimated is true.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER_BYTES (512 * 1024 * 1024)  // 512 MB — safely above L2/LLC on all current AMD GPUs
#define WARMUP_RUNS  3
#define TIMED_RUNS   20
#define BLOCK_SIZE   256

// Read-only kernel. __ldg maps to scalar cache on RDNA/CDNA.
// Wavefront: 64 threads on GCN/CDNA, 32 on RDNA — gridDim accounts for warpSize.
__global__ void memread(const float4* __restrict__ src, float* sink, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float4 v = __ldg(&src[id]);
        if (v.x == 9999999.0f) atomicAdd(sink, v.x);  // prevent dead-code elimination
    }
}

static void check(hipError_t err, const char* ctx) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s: %s\n", ctx, hipGetErrorString(err));
        exit(1);
    }
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

// Detect HBM (Instinct/CDNA) devices by gcnArchName prefix.
//   CDNA1 (MI100):        gfx908
//   CDNA2 (MI200 series): gfx90a
//   CDNA3 (MI300 series): gfx940, gfx941, gfx942
// GDDR devices (RDNA / RX series): gfx1xxx — DDR factor of ×2 applies.
static int isHBMDevice(const char* gcnArchName) {
    return (strncmp(gcnArchName, "gfx90", 5) == 0 ||
            strncmp(gcnArchName, "gfx94", 5) == 0);
}

int main(int argc, char** argv) {
    int jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = 1;
    }

    int deviceCount = 0;
    check(hipGetDeviceCount(&deviceCount), "hipGetDeviceCount");
    if (deviceCount == 0) {
        if (jsonMode) printf("{\"error\":\"No HIP/ROCm devices found\"}\n");
        else          printf("No HIP/ROCm devices found\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        check(hipSetDevice(dev), "hipSetDevice");

        hipDeviceProp_t props;
        check(hipGetDeviceProperties(&props, dev), "hipGetDeviceProperties");

        // memoryClockRate is in kHz, memoryBusWidth in bits.
        // DDR factor: ×2 for GDDR6 (RX/RDNA); ×1 for HBM (Instinct/CDNA, where
        // the clock already represents the effective doubled rate).
        int hbm          = isHBMDevice(props.gcnArchName);
        double ddrFactor  = hbm ? 1.0 : 2.0;
        double ratedGBps  = (double)props.memoryBusWidth
                          * (double)props.memoryClockRate   // kHz
                          * ddrFactor / 8.0 / 1e6;

        int elementCount = BUFFER_BYTES / sizeof(float4);
        int gridSize     = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float4* dSrc;
        float*  dSink;
        check(hipMalloc(&dSrc,  BUFFER_BYTES), "hipMalloc src");
        check(hipMalloc(&dSink, sizeof(float)), "hipMalloc sink");
        check(hipMemset(dSrc,  0, BUFFER_BYTES), "hipMemset src");
        check(hipMemset(dSink, 0, sizeof(float)), "hipMemset sink");

        hipEvent_t evStart, evStop;
        check(hipEventCreate(&evStart), "hipEventCreate start");
        check(hipEventCreate(&evStop),  "hipEventCreate stop");

        auto dispatch = [&]() -> double {
            check(hipEventRecord(evStart, 0), "hipEventRecord start");
            memread<<<gridSize, BLOCK_SIZE>>>(dSrc, dSink, elementCount);
            check(hipEventRecord(evStop, 0), "hipEventRecord stop");
            check(hipEventSynchronize(evStop), "hipEventSync");
            float ms = 0.0f;
            check(hipEventElapsedTime(&ms, evStart, evStop), "hipEventElapsed");
            return (double)BUFFER_BYTES / (ms / 1000.0) / 1e9;
        };

        for (int i = 0; i < WARMUP_RUNS; i++) dispatch();

        struct timespec wallStart, wallEnd;
        clock_gettime(CLOCK_MONOTONIC, &wallStart);

        double samples[TIMED_RUNS];
        for (int i = 0; i < TIMED_RUNS; i++) samples[i] = dispatch();

        clock_gettime(CLOCK_MONOTONIC, &wallEnd);
        double runtimeSecs = (wallEnd.tv_sec  - wallStart.tv_sec)
                           + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e9;

        qsort(samples, TIMED_RUNS, sizeof(double), cmp_double);
        double p50      = samples[TIMED_RUNS / 2];
        double p90      = samples[(int)(TIMED_RUNS * 0.90) - 1];
        double noisePct = (p90 - p50) / p90 * 100.0;
        double effPct   = p90 / ratedGBps * 100.0;

        if (jsonMode) {
            if (dev == 0) printf("[");
            printf("{\"device\":\"%s\","
                   "\"gcn_arch\":\"%s\","
                   "\"buffer_mb\":512,"
                   "\"runs\":%d,"
                   "\"p50_gbps\":%.2f,"
                   "\"p90_gbps\":%.2f,"
                   "\"noise_pct\":%.2f,"
                   "\"runtime_s\":%.3f,"
                   "\"rated_gbps\":%.0f,"
                   "\"rated_estimated\":%s,"
                   "\"efficiency_pct\":%.2f,"
                   "\"bus_width_bits\":%d,"
                   "\"mem_clock_mhz\":%.0f,"
                   "\"hbm\":%s}",
                   props.name, props.gcnArchName, TIMED_RUNS,
                   p50, p90, noisePct, runtimeSecs,
                   ratedGBps,
                   hbm ? "true" : "false",   // HBM DDR factor is unverified → rated_estimated
                   effPct,
                   props.memoryBusWidth,
                   props.memoryClockRate / 1000.0,
                   hbm ? "true" : "false");
            if (dev < deviceCount - 1) printf(",");
            else printf("]\n");
        } else {
            printf("=== Memory Bandwidth Fingerprint (ROCm/HIP) ===\n");
            printf("Device  : %s  [%s]\n", props.name, props.gcnArchName);
            if (hbm) {
                printf("Memory  : HBM  (rated_estimated=true — verify DDR factor with rocm-smi)\n");
            }
            printf("Rated   : %.0f GB/s  (%d-bit @ %.0f MHz, DDR factor x%.0f)\n",
                   ratedGBps, props.memoryBusWidth,
                   props.memoryClockRate / 1000.0, ddrFactor);
            printf("Buffer  : 512 MB read-only  (%d runs)\n", TIMED_RUNS);
            printf("p50     : %.1f GB/s\n", p50);
            printf("p90     : %.1f GB/s  efficiency: %.1f%%\n", p90, effPct);
            printf("noise   : %.1f%%  (p90-p50 spread -- lower is better)\n", noisePct);
            printf("runtime : %.2fs\n", runtimeSecs);
            if (dev < deviceCount - 1) printf("\n");
        }

        hipFree(dSrc);
        hipFree(dSink);
        hipEventDestroy(evStart);
        hipEventDestroy(evStop);
    }

    return 0;
}

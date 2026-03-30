// membench-fingerprint.cu — Memory bandwidth fingerprint for NVIDIA GPUs
// Build: nvcc -O3 -o membench-fingerprint-cuda membench-fingerprint.cu
// Run:   ./membench-fingerprint-cuda [--json]

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>

#define BUFFER_BYTES (512 * 1024 * 1024)  // 512 MB — safely above L2/LLC on all current NVIDIA GPUs
#define WARMUP_RUNS  3
#define TIMED_RUNS   20
#define BLOCK_SIZE   256

__global__ void memread(const float4* __restrict__ src, float* sink, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float4 v = src[id];
        if (v.x == 9999999.0f) atomicAdd(sink, v.x);
    }
}

static void check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
        exit(1);
    }
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    int jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = 1;
    }

    int deviceCount = 0;
    check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (jsonMode) printf("{\"error\":\"No CUDA devices found\"}\n");
        else          printf("No CUDA devices found\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        check(cudaSetDevice(dev), "cudaSetDevice");

        cudaDeviceProp props;
        check(cudaGetDeviceProperties(&props, dev), "cudaGetDeviceProperties");

        // Theoretical peak: bus_width(bits) * clock(kHz) * 2 (DDR) / 8 (bits→bytes) / 1e6 (→GB/s)
        int memClockKHz = 0;
        check(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev), "memClockRate");
        double ratedGBps = (double)props.memoryBusWidth
                         * (double)memClockKHz
                         * 2.0 / 8.0 / 1e6;

        int elementCount = BUFFER_BYTES / sizeof(float4);
        int gridSize     = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float4* dSrc;
        float*  dSink;
        check(cudaMalloc(&dSrc,  BUFFER_BYTES), "cudaMalloc src");
        check(cudaMalloc(&dSink, sizeof(float)), "cudaMalloc sink");
        check(cudaMemset(dSrc,  0, BUFFER_BYTES), "cudaMemset src");
        check(cudaMemset(dSink, 0, sizeof(float)), "cudaMemset sink");

        cudaEvent_t evStart, evStop;
        check(cudaEventCreate(&evStart), "eventCreate start");
        check(cudaEventCreate(&evStop),  "eventCreate stop");

        auto dispatch = [&]() -> double {
            check(cudaEventRecord(evStart), "eventRecord start");
            memread<<<gridSize, BLOCK_SIZE>>>(dSrc, dSink, elementCount);
            check(cudaEventRecord(evStop), "eventRecord stop");
            check(cudaEventSynchronize(evStop), "eventSync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, evStart, evStop), "eventElapsed");
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
                   "\"buffer_mb\":512,"
                   "\"runs\":%d,"
                   "\"p50_gbps\":%.2f,"
                   "\"p90_gbps\":%.2f,"
                   "\"noise_pct\":%.2f,"
                   "\"runtime_s\":%.3f,"
                   "\"rated_gbps\":%.0f,"
                   "\"rated_estimated\":false,"
                   "\"efficiency_pct\":%.2f,"
                   "\"bus_width_bits\":%d,"
                   "\"mem_clock_mhz\":%.0f}",
                   props.name, TIMED_RUNS,
                   p50, p90, noisePct, runtimeSecs,
                   ratedGBps, effPct,
                   props.memoryBusWidth,
                   memClockKHz / 1000.0);
            if (dev < deviceCount - 1) printf(",");
            else printf("]\n");
        } else {
            printf("=== Memory Bandwidth Fingerprint ===\n");
            printf("Device : %s  (%.0f GB/s rated)\n", props.name, ratedGBps);
            printf("Bus    : %d-bit @ %.0f MHz\n",
                   props.memoryBusWidth, memClockKHz / 1000.0);
            printf("Buffer : 512 MB read-only  (%d runs)\n", TIMED_RUNS);
            printf("p50    : %.1f GB/s\n", p50);
            printf("p90    : %.1f GB/s  efficiency: %.1f%%\n", p90, effPct);
            printf("noise  : %.1f%%  (p90-p50 spread -- lower is better)\n", noisePct);
            printf("runtime: %.2fs\n", runtimeSecs);
            if (dev < deviceCount - 1) printf("\n");
        }

        cudaFree(dSrc);
        cudaFree(dSink);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
    }

    return 0;
}

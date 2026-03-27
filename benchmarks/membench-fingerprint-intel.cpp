// membench-fingerprint-intel.cpp — Memory bandwidth fingerprint for Intel Arc / Xe GPUs
// Build: icpx -O3 -fsycl -o membench-fingerprint-intel membench-fingerprint-intel.cpp
// Run:   ./membench-fingerprint-intel [--json]
//
// TODO: Unvalidated — no Intel Arc hardware available at time of writing.
//       Verify output format, xpu-smi field names, and SYCL queue behaviour
//       against real hardware before distributing.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
#include <string>

#define BUFFER_BYTES (512 * 1024 * 1024)  // 512 MB — safely above L2/LLC on all current Intel Arc GPUs
#define WARMUP_RUNS  3

#define BLOCK_SIZE   256

// TODO: validate that ext::intel::info::device::memory_bus_width and
//       ext::intel::info::device::memory_clock_rate are available and
//       return correct values on Arc A-series and B-series hardware.
static double ratedBandwidthGBps(const sycl::device& dev) {
    try {
        auto busWidth  = dev.get_info<sycl::ext::intel::info::device::memory_bus_width>();
        auto clockRate = dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
        // bus_width(bits) * clock(kHz) * 2 (DDR) / 8 (bits→bytes) / 1e6 (→GB/s)
        return (double)busWidth * (double)clockRate * 2.0 / 8.0 / 1e6;
    } catch (...) {
        return -1.0;
    }
}

static int cmpDouble(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    int jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = 1;
    }

    std::vector<sycl::device> gpus;
    for (const auto& dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (dev.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
            gpus.push_back(dev);
        }
    }

    if (gpus.empty()) {
        if (jsonMode) printf("{\"error\":\"No Intel GPU devices found\"}\n");
        else          printf("No Intel GPU devices found\n");
        return 1;
    }

    int deviceCount = (int)gpus.size();

    if (jsonMode) printf("[");

    for (int dev = 0; dev < deviceCount; dev++) {
        sycl::queue q(gpus[dev], sycl::property::queue::enable_profiling{});

        std::string deviceName = gpus[dev].get_info<sycl::info::device::name>();
        double ratedGBps       = ratedBandwidthGBps(gpus[dev]);
        bool   ratedEstimated  = (ratedGBps < 0);

        int elementCount = BUFFER_BYTES / sizeof(sycl::float4);

        sycl::float4* dSrc  = sycl::malloc_device<sycl::float4>(elementCount, q);
        float*        dSink = sycl::malloc_device<float>(1, q);

        // TODO: validate that memset reaches device memory correctly on Arc hardware.
        q.memset(dSrc,  0, BUFFER_BYTES).wait();
        q.memset(dSink, 0, sizeof(float)).wait();

        auto dispatch = [&]() -> double {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(elementCount), [=](sycl::id<1> id) {
                    sycl::float4 v = dSrc[id];
                    if (v.x() == 9999999.0f)
                        sycl::atomic_ref<float,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(dSink[0]) += v.x();
                });
            });
            ev.wait();
            auto start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end   = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            double ns  = (double)(end - start);
            return (double)BUFFER_BYTES / (ns / 1e9) / 1e9;
        };

        for (int i = 0; i < WARMUP_RUNS; i++) dispatch();

        struct timespec wallStart, wallEnd;
        clock_gettime(CLOCK_MONOTONIC, &wallStart);

        // TODO: restore TIMED_RUNS once build is validated
        const int timedRuns = 20;
        double samples[20];
        for (int i = 0; i < timedRuns; i++) samples[i] = dispatch();

        clock_gettime(CLOCK_MONOTONIC, &wallEnd);
        double runtimeSecs = (wallEnd.tv_sec  - wallStart.tv_sec)
                           + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e9;

        qsort(samples, timedRuns, sizeof(double), cmpDouble);
        double p50      = samples[timedRuns / 2];
        double p90      = samples[(int)(timedRuns * 0.90) - 1];
        double noisePct = (p90 - p50) / p90 * 100.0;

        if (jsonMode) {
            printf("{\"device\":\"%s\","
                   "\"buffer_mb\":512,"
                   "\"runs\":%d,"
                   "\"p50_gbps\":%.2f,"
                   "\"p90_gbps\":%.2f,"
                   "\"noise_pct\":%.2f,"
                   "\"runtime_s\":%.3f,",
                   deviceName.c_str(), timedRuns,
                   p50, p90, noisePct, runtimeSecs);

            if (ratedGBps > 0) {
                double effPct = p90 / ratedGBps * 100.0;
                printf("\"rated_gbps\":%.0f,"
                       "\"rated_estimated\":%s,"
                       "\"efficiency_pct\":%.2f}",
                       ratedGBps,
                       ratedEstimated ? "true" : "false",
                       effPct);
            } else {
                printf("\"rated_gbps\":null,\"rated_estimated\":true,\"efficiency_pct\":null}");
            }

            if (dev < deviceCount - 1) printf(",");
            else printf("]\n");
        } else {
            printf("=== Memory Bandwidth Fingerprint ===\n");
            if (ratedGBps > 0) {
                double effPct = p90 / ratedGBps * 100.0;
                printf("Device : %s  (%.0f GB/s rated%s)\n",
                       deviceName.c_str(), ratedGBps,
                       ratedEstimated ? " estimated" : "");
                printf("p90    : %.1f GB/s  efficiency: %.1f%%\n", p90, effPct);
            } else {
                printf("Device : %s  (rated: unknown)\n", deviceName.c_str());
                printf("p90    : %.1f GB/s\n", p90);
            }
            printf("Buffer : 512 MB read-only  (%d runs)\n", timedRuns);
            printf("p50    : %.1f GB/s\n", p50);
            printf("noise  : %.1f%%  (p90-p50 spread -- lower is better)\n", noisePct);
            printf("runtime: %.2fs\n", runtimeSecs);
            if (dev < deviceCount - 1) printf("\n");
        }

        sycl::free(dSrc,  q);
        sycl::free(dSink, q);
    }

    return 0;
}

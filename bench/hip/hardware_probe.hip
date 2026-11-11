// Phase 0 hardware characterization for gfx1151: allocation/copy bandwidth,
// rocBLAS GEMM, and a rocWMMA FP16 correctness smoke test.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void check(hipError_t err, const char * what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

double seconds_between(hipEvent_t start, hipEvent_t stop) {
    float ms = 0;
    check(hipEventElapsedTime(&ms, start, stop), "event elapsed");
    return ms / 1000.0;
}

void bandwidth_row(const char * name, uint64_t bytes, double seconds, int reps) {
    const double gbps = double(bytes) * reps / seconds / 1e9;
    std::printf("%-22s %8.1f MiB  %8.2f GB/s\n", name, bytes / 1048576.0, gbps);
}

__global__ void wmma_tile(const __half * a, const __half * b, float * c) {
    using namespace rocwmma;
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

bool test_wmma() {
    constexpr int n = 16;
    std::vector<__half> a(n * n), b(n * n);
    std::vector<float> expected(n * n, 0);
    for (int i = 0; i < n * n; ++i) {
        a[i] = __float2half(0.01f * float(i % 13) - 0.05f);
        b[i] = __float2half(0.02f * float(i % 7) - 0.03f);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                expected[i * n + j] += __half2float(a[i * n + k]) * __half2float(b[k * n + j]);
            }
        }
    }
    __half * da = nullptr;
    __half * db = nullptr;
    float * dc = nullptr;
    check(hipMalloc(&da, a.size() * sizeof(__half)), "wmma malloc a");
    check(hipMalloc(&db, b.size() * sizeof(__half)), "wmma malloc b");
    check(hipMalloc(&dc, expected.size() * sizeof(float)), "wmma malloc c");
    check(hipMemcpy(da, a.data(), a.size() * sizeof(__half), hipMemcpyHostToDevice), "wmma h2d a");
    check(hipMemcpy(db, b.data(), b.size() * sizeof(__half), hipMemcpyHostToDevice), "wmma h2d b");
    wmma_tile<<<1, 32>>>(da, db, dc);
    check(hipGetLastError(), "wmma launch");
    check(hipDeviceSynchronize(), "wmma sync");
    std::vector<float> got(n * n);
    check(hipMemcpy(got.data(), dc, got.size() * sizeof(float), hipMemcpyDeviceToHost), "wmma d2h");
    bool ok = true;
    for (int i = 0; i < n * n; ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-3f) {
            std::fprintf(stderr, "wmma mismatch at %d: got %f expected %f\n", i, got[i], expected[i]);
            ok = false;
            break;
        }
    }
    check(hipFree(dc), "free c");
    check(hipFree(db), "free b");
    check(hipFree(da), "free a");
    return ok;
}

void test_rocblas() {
    constexpr int n = 1024;
    rocblas_handle handle = nullptr;
    if (rocblas_create_handle(&handle) != rocblas_status_success) {
        std::fprintf(stderr, "rocblas_create_handle failed\n");
        std::exit(1);
    }
    float * a = nullptr;
    float * b = nullptr;
    float * c = nullptr;
    check(hipMalloc(&a, size_t(n) * n * 4), "rocblas malloc a");
    check(hipMalloc(&b, size_t(n) * n * 4), "rocblas malloc b");
    check(hipMalloc(&c, size_t(n) * n * 4), "rocblas malloc c");
    check(hipMemset(a, 0x3c, size_t(n) * n * 4), "rocblas memset a");
    check(hipMemset(b, 0x3c, size_t(n) * n * 4), "rocblas memset b");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    rocblas_status status = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                          n, n, n, &alpha, a, n, b, n, &beta, c, n);
    if (status != rocblas_status_success) {
        std::fprintf(stderr, "rocblas_sgemm failed: %d\n", int(status));
        std::exit(1);
    }
    check(hipDeviceSynchronize(), "rocblas sync");
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event create");
    check(hipEventCreate(&stop), "event create");
    check(hipEventRecord(start), "event record");
    for (int i = 0; i < 5; ++i) {
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                      n, n, n, &alpha, a, n, b, n, &beta, c, n);
    }
    check(hipEventRecord(stop), "event record");
    check(hipDeviceSynchronize(), "rocblas timing sync");
    const double seconds = seconds_between(start, stop);
    const double tflops = 2.0 * n * n * n * 5 / seconds / 1e12;
    std::printf("rocblas_sgemm 1024^3       %.3f s  %.2f TFLOP/s\n", seconds, tflops);
    check(hipEventDestroy(stop), "destroy event");
    check(hipEventDestroy(start), "destroy event");
    check(hipFree(c), "free c");
    check(hipFree(b), "free b");
    check(hipFree(a), "free a");
    rocblas_destroy_handle(handle);
}

void measure_memory(hipDeviceProp_t & prop) {
    const uint64_t bytes = 256ull << 20;
    std::vector<char> host(bytes);
    for (uint64_t i = 0; i < bytes; i += 4096) host[i] = char(i);
    char * device = nullptr;
    char * pinned = nullptr;
    check(hipMalloc(&device, bytes), "malloc device");
    check(hipHostMalloc(&pinned, bytes), "malloc pinned");
    std::memcpy(pinned, host.data(), bytes);
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event create");
    check(hipEventCreate(&stop), "event create");
    const int reps = 5;
    check(hipEventRecord(start), "record");
    for (int i = 0; i < reps; ++i) check(hipMemcpy(device, pinned, bytes, hipMemcpyHostToDevice), "h2d");
    check(hipEventRecord(stop), "record");
    check(hipDeviceSynchronize(), "sync");
    bandwidth_row("hipMemcpy H2D", bytes, seconds_between(start, stop), reps);

    check(hipEventRecord(start), "record");
    for (int i = 0; i < reps; ++i) check(hipMemcpy(pinned, device, bytes, hipMemcpyDeviceToHost), "d2h");
    check(hipEventRecord(stop), "record");
    check(hipDeviceSynchronize(), "sync");
    bandwidth_row("hipMemcpy D2H", bytes, seconds_between(start, stop), reps);

    check(hipEventRecord(start), "record");
    for (int i = 0; i < reps; ++i) check(hipMemset(device, i & 0xff, bytes), "memset");
    check(hipEventRecord(stop), "record");
    check(hipDeviceSynchronize(), "sync");
    bandwidth_row("hipMemset", bytes, seconds_between(start, stop), reps);

    // Copy inside device memory (read + write).
    char * second = nullptr;
    check(hipMalloc(&second, bytes), "malloc second");
    check(hipEventRecord(start), "record");
    for (int i = 0; i < reps; ++i) check(hipMemcpy(second, device, bytes, hipMemcpyDeviceToDevice), "d2d");
    check(hipEventRecord(stop), "record");
    check(hipDeviceSynchronize(), "sync");
    bandwidth_row("hipMemcpy D2D (r+w)", bytes, seconds_between(start, stop), reps);
    check(hipFree(second), "free second");

    // Host-pinned mapped memory visible to the device.
    char * mapped = nullptr;
    check(hipHostMalloc(&mapped, bytes, hipHostMallocMapped | hipHostMallocPortable), "host mapped malloc");
    void * mapped_device = nullptr;
    check(hipHostGetDevicePointer(&mapped_device, mapped, 0), "host mapped pointer");
    check(hipEventRecord(start), "record");
    for (int i = 0; i < reps; ++i) check(hipMemset(mapped_device, i & 0xff, bytes), "mapped memset");
    check(hipEventRecord(stop), "record");
    check(hipDeviceSynchronize(), "sync");
    bandwidth_row("mapped memset", bytes, seconds_between(start, stop), reps);
    check(hipHostFree(mapped), "free mapped");

    // Allocation latency.
    std::vector<void *> blocks;
    const int allocReps = 64;
    const uint64_t allocBytes = 16ull << 20;
    check(hipEventRecord(start), "record");
    for (int i = 0; i < allocReps; ++i) {
        void * p = nullptr;
        check(hipMalloc(&p, allocBytes), "alloc loop");
        blocks.push_back(p);
    }
    check(hipEventRecord(stop), "record");
    check(hipDeviceSynchronize(), "sync");
    std::printf("hipMalloc 16 MiB x%d       %.3f ms total\n", allocReps,
                seconds_between(start, stop) * 1000.0);
    for (void * p : blocks) check(hipFree(p), "free block");
    std::printf("device pool bytes          %llu MiB\n",
                static_cast<unsigned long long>(prop.totalGlobalMem / 1048576));

    check(hipEventDestroy(stop), "destroy event");
    check(hipEventDestroy(start), "destroy event");
    check(hipHostFree(pinned), "free pinned");
    check(hipFree(device), "free device");
}

} // namespace

int main() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 77;
    }
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0), "device properties");
    std::printf("device: %s (%s) CUs=%d clock=%d MHz\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.clockRate / 1000);
    std::printf("memory: %.1f GiB  wave=%d  L2=%d KiB\n",
                prop.totalGlobalMem / 1073741824.0, prop.warpSize, prop.l2CacheSize / 1024);
    measure_memory(prop);
    test_rocblas();
    if (!test_wmma()) {
        std::fprintf(stderr, "rocWMMA smoke test failed\n");
        return 1;
    }
    std::printf("rocWMMA 16x16x16 FP16      ok\n");
    return 0;
}

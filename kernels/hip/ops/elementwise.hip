#include "../yuuka_kernels.h"

namespace {

__global__ void add_f32_kernel(float * dst, const float * a, const float * b, uint64_t n) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] + b[i];
    }
}

} // namespace

namespace yuuka {

void launch_add_f32(float * dst, const float * a, const float * b, uint64_t n,
                    hipStream_t stream) {
    if (n == 0) {
        return;
    }
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
    add_f32_kernel<<<blocks, threads, 0, stream>>>(dst, a, b, n);
}

} // namespace yuuka

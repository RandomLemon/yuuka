#include "../common.cuh"
#include "../yuuka_kernels.h"

namespace {

__global__ void rope_f32_kernel(float * data, int heads, int stride, int head_dim,
                                int rot_dim, int pos, float base) {
    (void)head_dim;
    const int head = blockIdx.x;
    const int i = threadIdx.x;
    const int half = rot_dim / 2;
    if (head >= heads || i >= half) {
        return;
    }
    const float theta = static_cast<float>(pos) * powf(base, -2.0f * i / rot_dim);
    const float c = cosf(theta);
    const float s = sinf(theta);
    float * head_ptr = data + head * stride;
    const float x0 = head_ptr[i];
    const float x1 = head_ptr[i + half];
    head_ptr[i] = x0 * c - x1 * s;
    head_ptr[i + half] = x0 * s + x1 * c;
}

} // namespace

namespace yuuka {

void launch_rope_f32(float * data, int heads, int stride, int head_dim, int rot_dim,
                     int pos, float base, hipStream_t stream) {
    if (rot_dim <= 0 || heads <= 0) {
        return;
    }
    rope_f32_kernel<<<heads, rot_dim / 2, 0, stream>>>(
        data, heads, stride, head_dim, rot_dim, pos, base);
}

} // namespace yuuka

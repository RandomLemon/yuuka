#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

// Remapped decode matvec: one wave per output row, sixteen K values per lane.
//
// The previous streaming kernel tied the grid to the output width in blocks of
// 32 channels and read two bytes of Q4_K weights per lane, so a narrow
// projection launched too few blocks to cover DRAM latency and every lane had
// little data in flight. This kernel keeps the weights packed and changes the
// shape of the mapping instead:
//   - grid = ceil(n_out / (kRowWarps * kRows)): one block covers four output
//     rows at the shipped kRows=1, so a 1024-output projection launches 256
//     blocks instead of 32. Two rows per wave was measured slower because it
//     added no memory-level parallelism and reduced the block count;
//   - a lane owns sixteen consecutive K values and decodes them from one
//     eight- to sixteen-byte span of the packed row, one whole 32-value
//     sub-block of a K-quant or half of a Q8_0/Q4_0/Q5_0 block;
//   - the lane's sixteen activation values are read once per K step from the
//     activation row and reused for every row the wave handles, so the
//     activation is read from L1 once per warp step and never staged in shared
//     memory, which removes the per-K-step barrier entirely;
//   - the only reduction is one wave shuffle tree per row in the epilogue.
//
// kRows is a compile-time row count per wave so the accumulators stay in
// registers: the block covers kRowWarps * kRows output rows and walks K once.
// The launcher only selects this kernel for a 512-value K step, which every
// decode projection in the supported models satisfies, and for one row block.
// The shipped kRows is one; the pair form keeps the same template so a
// two-matrix launch uses the same row mapping.

namespace {

constexpr int kRowWarps = 4;
constexpr int kRowThreads = 32 * kRowWarps;
constexpr int kSpanValues = 16;              // K values one lane owns per step
constexpr int kStepValues = 32 * kSpanValues; // 512 K values per wave step
constexpr int kRowsPerWave = 1;
constexpr int kRowsPerBlock = kRowWarps * kRowsPerWave;

// load_activation_span reads the lane's sixteen activation values. The
// activation is read once per warp per K step and reused for every row the wave
// handles; it is small enough to stay in L1, which is why the kernel stages
// nothing in shared memory and needs no barrier inside K.
__device__ __forceinline__ void load_activation_span(const float * x, int64_t k,
                                                    float (&xv)[kSpanValues]) {
    #pragma unroll
    for (int i = 0; i < kSpanValues / 4; ++i) {
        const float4 v = __ldg(reinterpret_cast<const float4 *>(x + k + 4 * i));
        xv[4 * i + 0] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
    }
}

// gemv_row_kernel is out[row] = dot(W[row], x) + res[row], one wave per row and
// kRows rows per wave.
template <uint32_t DType, int kRows>
__global__ void gemv_row_kernel(float * out, const float * x, const uint8_t * w,
                                const float * res, int64_t n_in, int64_t n_out,
                                int64_t row_bytes) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t steps = n_in / kStepValues;

    float acc[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = 0.0f;
    }
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t k = step * kStepValues + lane * kSpanValues;
        float xv[kSpanValues];
        load_activation_span(x, k, xv);
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int64_t row = first + r * kRowWarps;
            if (row >= n_out) {
                continue;
            }
            float wv[kSpanValues];
            decode_span16<DType>(w + row * row_bytes, k, wv);
            #pragma unroll
            for (int i = 0; i < kSpanValues; ++i) {
                acc[r] = fmaf(wv[i], xv[i], acc[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int64_t row = first + r * kRowWarps;
        const float total = warp_reduce_sum(acc[r]);
        if (lane == 0 && row < n_out) {
            out[row] = res != nullptr ? total + res[row] : total;
        }
    }
}

// gemv_row_pair_kernel walks one activation row against two packed matrices, so
// the gate/up projection reads the activation once. Both matrices must share
// the dtype and row stride; the output widths may differ.
template <uint32_t DType, int kRows, bool kSwiglu>
__global__ void gemv_row_pair_kernel(float * out0, float * out1, const float * x,
                                     const uint8_t * w0, const uint8_t * w1,
                                     int64_t n_in, int64_t n_out0, int64_t n_out1,
                                     int64_t row_bytes) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t steps = n_in / kStepValues;

    float acc0[kRows];
    float acc1[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
    }
    #pragma unroll 2
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t k = step * kStepValues + lane * kSpanValues;
        float xv[kSpanValues];
        load_activation_span(x, k, xv);
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int64_t row = first + r * kRowWarps;
            if (row < n_out0) {
                float wv[kSpanValues];
                decode_span16<DType>(w0 + row * row_bytes, k, wv);
                #pragma unroll
                for (int i = 0; i < kSpanValues; ++i) {
                    acc0[r] = fmaf(wv[i], xv[i], acc0[r]);
                }
            }
            if (row < n_out1) {
                float wv[kSpanValues];
                decode_span16<DType>(w1 + row * row_bytes, k, wv);
                #pragma unroll
                for (int i = 0; i < kSpanValues; ++i) {
                    acc1[r] = fmaf(wv[i], xv[i], acc1[r]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int64_t row = first + r * kRowWarps;
        const float total0 = warp_reduce_sum(acc0[r]);
        const float total1 = warp_reduce_sum(acc1[r]);
        if (lane != 0) {
            continue;
        }
        if (row < n_out0) {
            out0[row] = kSwiglu
                            ? (total0 / (1.0f + expf(-total0))) * total1
                            : total0;
        }
        if (!kSwiglu && row < n_out1) {
            out1[row] = total1;
        }
    }
}

// gemv_pair_f32_32_kernel is the narrow FP32 beta/alpha pair: one block per
// output row, 256 threads, two independent accumulators over the same strided
// activation values, and the same halving reduction tree as gemv_f32_kernel.
// A wave-per-row pair would launch four blocks at this width, so the per-row
// form is the one that fills the device.
__global__ void gemv_pair_f32_32_kernel(float * out0, float * out1, const float * x,
                                        const float * w0, const float * w1,
                                        int64_t n_in) {
    const int64_t row = blockIdx.x;
    const float * row0 = w0 + row * n_in;
    const float * row1 = w1 + row * n_in;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        const float xv = x[i];
        sum0 += row0[i] * xv;
        sum1 += row1[i] * xv;
    }
    __shared__ float scratch0[256];
    __shared__ float scratch1[256];
    scratch0[threadIdx.x] = sum0;
    scratch1[threadIdx.x] = sum1;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            scratch0[threadIdx.x] += scratch0[threadIdx.x + stride];
            scratch1[threadIdx.x] += scratch1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out0[row] = scratch0[0];
        out1[row] = scratch1[0];
    }
}

// dot4 is the signed int8 dot product of the four digits packed in each operand;
// the supplied activation digits and the Q6_K/Q8_0 weight digits are signed, so
// both operands are read as signed. It mirrors gemv_i8.hip.
__device__ __forceinline__ int dot4_row(int32_t a, int32_t b) {
    return __builtin_amdgcn_sudot4(true, a, true, b, 0, false);
}

// gemv_row_i8_kernel is the remapped mapping over a Q8_1 activation row: the
// same grid, lane span, and wave-per-row shape as gemv_row_kernel, with each
// four-value weight span dotted against its four activation digits by one
// v_dot4. The per-sub-block scale and the Q4_K/Q5_K minimum are applied to a
// 4-value span, and the activation scale is one float per 32-value block, so
// the lane's sixteen values share it.
template <uint32_t DType, int kRows>
__global__ void gemv_row_i8_kernel(float * out, const int8_t * qs, const float2 * ds,
                                   const uint8_t * w, int64_t n_in, int64_t n_out,
                                   int64_t row_bytes) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t steps = n_in / kStepValues;
    constexpr bool kHasBias = DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K;

    float acc[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = 0.0f;
    }
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t k = step * kStepValues + lane * kSpanValues;
        int32_t a4[kSpanValues / 4];
        #pragma unroll
        for (int i = 0; i < kSpanValues / 4; ++i) {
            __builtin_memcpy(&a4[i], qs + k + 4 * i, 4);
        }
        const float a_scale = ds[k / 32].x;
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int64_t row = first + r * kRowWarps;
            if (row >= n_out) {
                continue;
            }
            const uint8_t * base = w + row * row_bytes;
            #pragma unroll
            for (int i = 0; i < kSpanValues / 4; ++i) {
                int32_t q = 0;
                float scale = 0.0f;
                float bias = 0.0f;
                decode_span4_i8<DType>(base, k + 4 * i, q, scale, bias);
                acc[r] = fmaf(a_scale * scale, static_cast<float>(dot4_row(q, a4[i])),
                              acc[r]);
                if (kHasBias && bias != 0.0f) {
                    acc[r] = fmaf(a_scale * bias,
                                  static_cast<float>(dot4_row(a4[i], 0x01010101)),
                                  acc[r]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int64_t row = first + r * kRowWarps;
        const float total = warp_reduce_sum(acc[r]);
        if (lane == 0 && row < n_out) {
            out[row] = total;
        }
    }
}

template <uint32_t DType>
void launch_gemv_row_i8_dtype(float * out, const int8_t * qs, const float2 * ds,
                              const uint8_t * w, int64_t n_in, int64_t n_out,
                              int64_t row_bytes, hipStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(
        (n_out + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_row_i8_kernel<DType, kRowsPerWave><<<blocks, kRowThreads, 0, stream>>>(
        out, qs, ds, w, n_in, n_out, row_bytes);
}

template <uint32_t DType>
void launch_gemv_row_dtype(float * out, const float * x, const uint8_t * w,
                           const float * res, int64_t n_in, int64_t n_out,
                           int64_t row_bytes, hipStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(
        (n_out + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_row_kernel<DType, kRowsPerWave><<<blocks, kRowThreads, 0, stream>>>(
        out, x, w, res, n_in, n_out, row_bytes);
}

template <uint32_t DType, bool kSwiglu>
void launch_gemv_row_pair_dtype(float * out0, float * out1, const float * x,
                                const uint8_t * w0, const uint8_t * w1, int64_t n_in,
                                int64_t n_out0, int64_t n_out1, int64_t row_bytes,
                                hipStream_t stream) {
    const int64_t widest = n_out0 > n_out1 ? n_out0 : n_out1;
    const unsigned blocks = static_cast<unsigned>(
        (widest + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_row_pair_kernel<DType, kRowsPerWave, kSwiglu>
        <<<blocks, kRowThreads, 0, stream>>>(out0, out1, x, w0, w1, n_in, n_out0,
                                             n_out1, row_bytes);
}

bool gemv_row_supported(uint32_t dtype, uint64_t n_in) {
    // The kernel walks K in 512-value steps; every decode projection of the
    // supported models satisfies this, and any other width keeps the previous
    // streaming kernel.
    if (n_in % kStepValues != 0) {
        return false;
    }
    switch (dtype) {
        case YUUKA_DTYPE_F32:
        case YUUKA_DTYPE_F16:
        case YUUKA_DTYPE_Q4_0:
        case YUUKA_DTYPE_Q5_0:
        case YUUKA_DTYPE_Q8_0:
        case YUUKA_DTYPE_Q4_K:
        case YUUKA_DTYPE_Q5_K:
        case YUUKA_DTYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

} // namespace

namespace yuuka {

bool gemv_row_available(uint32_t dtype, uint64_t n_in) {
    return gemv_row_supported(dtype, n_in);
}

void launch_gemv_row_f32out(float * out, const float * x, const void * w,
                            const float * res, uint64_t n_in, uint64_t n_out,
                            uint32_t dtype, hipStream_t stream) {
    const auto * weights = static_cast<const uint8_t *>(w);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto rb = static_cast<int64_t>(yuuka_packed_row_bytes(dtype, n_in));
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_gemv_row_dtype<YUUKA_DTYPE_F32>(out, x, weights, res, in, on, rb,
                                                   stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_gemv_row_dtype<YUUKA_DTYPE_F16>(out, x, weights, res, in, on, rb,
                                                   stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q4_0>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q5_0>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q8_0>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q4_K>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q5_K>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        default:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q6_K>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
    }
}

void launch_gemv_row_pair_f32out(float * out0, float * out1, const float * x,
                                 const void * w0, const void * w1, uint64_t n_in,
                                 uint64_t n_out0, uint64_t n_out1, uint32_t dtype,
                                 int swiglu, hipStream_t stream) {
    const auto * w0b = static_cast<const uint8_t *>(w0);
    const auto * w1b = static_cast<const uint8_t *>(w1);
    const auto in = static_cast<int64_t>(n_in);
    const auto o0 = static_cast<int64_t>(n_out0);
    const auto o1 = static_cast<int64_t>(n_out1);
    const auto rb = static_cast<int64_t>(yuuka_packed_row_bytes(dtype, n_in));
    const bool sigmoid = swiglu != 0;
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            if (!sigmoid && o0 == 32 && o1 == 32 && in % 256 == 0 &&
                in <= INT32_MAX) {
                gemv_pair_f32_32_kernel<<<32, 256, 0, stream>>>(
                    out0, out1, x, reinterpret_cast<const float *>(w0b),
                    reinterpret_cast<const float *>(w1b), in);
                break;
            }
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_F32, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_F32, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_F16:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_F16, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_F16, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_0, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_0, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_0, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_0, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q8_0, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q8_0, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_K, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_K, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_K, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_K, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        default:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q6_K, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q6_K, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
    }
}

void launch_gemv_row_i8_f32out(float * out, const void * qs, const void * ds,
                                const void * w, uint64_t n_in, uint64_t n_out,
                                uint32_t dtype, hipStream_t stream) {
    const auto * weights = static_cast<const uint8_t *>(w);
    const auto * quantized = static_cast<const int8_t *>(qs);
    const auto * scales = static_cast<const float2 *>(ds);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto rb = static_cast<int64_t>(yuuka_packed_row_bytes(dtype, n_in));
    switch (dtype) {
        case YUUKA_DTYPE_Q8_0:
            launch_gemv_row_i8_dtype<YUUKA_DTYPE_Q8_0>(out, quantized, scales, weights,
                                                       in, on, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_gemv_row_i8_dtype<YUUKA_DTYPE_Q4_K>(out, quantized, scales, weights,
                                                       in, on, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_gemv_row_i8_dtype<YUUKA_DTYPE_Q5_K>(out, quantized, scales, weights,
                                                       in, on, rb, stream);
            break;
        default:
            launch_gemv_row_i8_dtype<YUUKA_DTYPE_Q6_K>(out, quantized, scales, weights,
                                                       in, on, rb, stream);
            break;
    }
}

} // namespace yuuka

#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

// Streaming LM head for the prefill graph.
//
// The prefill graph ends with one RMSNorm row against the output matrix, which
// is the widest matrix in the model (vocab x hidden) and is read exactly once
// per prompt chunk. The operator is therefore pure weight bandwidth: the
// single-row GEMV reaches 46 GB/s on gfx1151 because it launches one small
// block per output channel and pays a block reduction per row, while the
// streaming form below keeps every lane busy on 16 bytes of packed weights per
// step and reached 226 GB/s on the same matrix.
//
// Shape: out[m][n] = sum_k W[n][k] * x[m][k] with n in [0, n_out) and m in
// [0, rows). rows is small by construction: the chunk tail needs the last row
// only, and speculative verification needs at most kMaxRows candidate rows.
// Every extra row is nearly free because the weight row is already in
// registers, so the kernel is written for the whole row block instead of one
// row at a time; sixteen rows cost 5.9 ms against 2.3 ms for one on the
// 2560 -> 248320 Q6_K matrix, where sixteen separate GEMV submissions cost
// 181 ms.
//
// Mapping, chosen so that every load instruction consumes a whole 32-byte
// sector of the four-byte lanes it reads:
//   - one wave owns four output channels and walks the K axis itself;
//   - a lane owns four consecutive K values of each of its four channels, so a
//     quantization block is covered by 32 lanes without overlap and the packed
//     reads of the 32 lanes are contiguous;
//   - the activation row block for the current 256-value K step is staged in
//     shared memory once per block, so the activation is read from L2 once per
//     block instead of once per output channel;
//   - a lane accumulates the four values of every channel for every row in
//     registers, and the final wave reduction is one shfl tree per channel.
//
// The kernel keeps no intermediate tile and therefore needs no K-quant
// dequantization buffer: decode_span4 in quant.cuh decodes the four values of a
// lane straight into registers.

namespace {

constexpr int kWarpsPerBlock = 8;
// One wave owns four output channels. Eight was measured on gfx1151 and is 28%
// slower in the decode step: the wider lane block spills the
// weight registers it has to keep live per K step, so the staging and barrier
// cost that a wider wave would amortize is smaller than the cost of the spill.
constexpr int kChansPerWarp = 4;
constexpr int kChansPerWarpDecode = kChansPerWarp;
constexpr int kThreads = 32 * kWarpsPerBlock;
constexpr int kStepK = 256;  // values staged per step, one Q4_K/Q5_K/Q6_K block
constexpr int kMaxRows = 16;  // matches the speculative verification cap

// lm_head_kernel dots KStep-wide K steps of kChansPerWarp packed weight rows
// against up to MaxRows activation rows. MaxRows is a compile-time bound
// because the accumulators must stay in registers: the single-row instantiation
// uses 69 VGPRs and the sixteen-row instantiation 127, where a runtime bound
// would index the accumulator array dynamically and spill it to scratch.
// lm_head_kernel is the streaming single-row-block matvec. res, when non-null,
// is added to every output element: that is the residual add of the sub-layer the
// projection ends, folded into the epilogue instead of taking its own launch.
template <uint32_t DType, int MaxRows, int kChans>
__global__ void lm_head_kernel(float * out, const float * x, const uint8_t * w,
                               const float * res, int64_t n_in, int64_t n_out,
                               int64_t rows, int64_t row_bytes) {
    extern __shared__ __align__(16) float xs[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t ch0 =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kChans;
    // A warp whose channels are all past the output width must still reach the
    // barriers that stage the activation tile, so it skips its own work
    // instead of returning.
    const bool active = ch0 < n_out;
    const int j0 = 4 * lane;
    const int64_t steps = n_in / kStepK;
    // The complete 16-row tile can stage four contiguous activations per
    // lane instead of issuing one scalar load per lane for each row.
    constexpr bool kKQuant = DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K ||
                             DType == YUUKA_DTYPE_Q6_K;
    const bool vector_stage = kKQuant && MaxRows == kMaxRows && rows == kMaxRows &&
                              reinterpret_cast<uintptr_t>(x) % alignof(float4) == 0;
    // An inactive wave keeps a valid pointer so the guarded decode below never
    // forms one past the weight buffer.
    const uint8_t * base = active ? w + ch0 * row_bytes : w;

    float acc[kChans][MaxRows];
    #pragma unroll
    for (int c = 0; c < kChans; ++c) {
        #pragma unroll
        for (int m = 0; m < MaxRows; ++m) {
            acc[c][m] = 0.0f;
        }
    }

    for (int64_t step = 0; step < steps; ++step) {
        if (vector_stage) {
            constexpr int kVectorsPerRow = kStepK / 4;
            auto * tile = reinterpret_cast<float4 *>(xs);
            for (int idx = threadIdx.x; idx < kMaxRows * kVectorsPerRow; idx += kThreads) {
                const int m = idx / kVectorsPerRow;
                const int k = (idx % kVectorsPerRow) * 4;
                const float * src = x + m * n_in + step * kStepK + k;
                tile[idx] = *reinterpret_cast<const float4 *>(src);
            }
        } else {
            for (int64_t m = 0; m < rows; ++m) {
                const float * src = x + m * n_in + step * kStepK;
                for (int idx = static_cast<int>(threadIdx.x); idx < kStepK;
                     idx += kThreads) {
                    xs[m * kStepK + idx] = src[idx];
                }
            }
        }
        __syncthreads();
        // Each step is a whole number of 128-value halves: a lane covers four
        // consecutive values in each, which is the layout decode_span4 expects.
        #pragma unroll
        for (int half = 0; half < kStepK / 128; ++half) {
            const int base_j = half * 128 + j0;
            if (active) {
                float wv[kChans][4];
                #pragma unroll
                for (int c = 0; c < kChans; ++c) {
                    if (ch0 + c >= n_out) {
                        break;
                    }
                    decode_span4<DType>(base + c * row_bytes,
                                        step * kStepK + base_j, wv[c]);
                }
                #pragma unroll
                for (int m = 0; m < MaxRows; ++m) {
                    if (m >= rows) {
                        continue;
                    }
                    const float4 xv = *reinterpret_cast<const float4 *>(
                        xs + m * kStepK + base_j);
                    #pragma unroll
                    for (int c = 0; c < kChans; ++c) {
                        if (ch0 + c >= n_out) {
                            break;
                        }
                        acc[c][m] = fmaf(wv[c][0], xv.x, acc[c][m]);
                        acc[c][m] = fmaf(wv[c][1], xv.y, acc[c][m]);
                        acc[c][m] = fmaf(wv[c][2], xv.z, acc[c][m]);
                        acc[c][m] = fmaf(wv[c][3], xv.w, acc[c][m]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }
    #pragma unroll
    for (int c = 0; c < kChans; ++c) {
        if (ch0 + c >= n_out) {
            break;
        }
        #pragma unroll
        for (int m = 0; m < MaxRows; ++m) {
            if (m >= rows) {
                continue;
            }
            const float total = warp_reduce_sum(acc[c][m]);
            if (lane == 0) {
                const int64_t idx = m * n_out + ch0 + c;
                out[idx] = res != nullptr ? total + res[idx] : total;
            }
        }
    }
}

// lm_head_pair_kernel is the two-matrix form: one block walks the same K axis of
// both matrices and each channel range of both, so two projections that read the
// same activation row share one staging pass and one launch. The two matrices may
// differ in packed dtype and in output width, which is why the decoders are
// dispatched at run time: a block still sees one dtype per span, and the branch
// costs a few instructions per four weights. kSwiglu writes silu(a)*b of the two
// dots; otherwise both rows are written.
template <uint32_t DType, bool kSwiglu>
__global__ void lm_head_pair_kernel(float * out0, float * out1, const float * x,
                                    const uint8_t * w0, const uint8_t * w1,
                                    int64_t n_in, int64_t n_out0, int64_t n_out1,
                                    int64_t row_bytes0, int64_t row_bytes1) {
    extern __shared__ __align__(16) float xs[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t ch0 =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kChansPerWarp;
    const bool active0 = ch0 < n_out0;
    const bool active1 = ch0 < n_out1;
    const int j0 = 4 * lane;
    const int64_t steps = n_in / kStepK;
    const uint8_t * base0 = active0 ? w0 + ch0 * row_bytes0 : w0;
    const uint8_t * base1 = active1 ? w1 + ch0 * row_bytes1 : w1;

    float acc0[kChansPerWarp];
    float acc1[kChansPerWarp];
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        acc0[c] = 0.0f;
        acc1[c] = 0.0f;
    }

    for (int64_t step = 0; step < steps; ++step) {
        const float * src = x + step * kStepK;
        for (int idx = static_cast<int>(threadIdx.x); idx < kStepK; idx += kThreads) {
            xs[idx] = src[idx];
        }
        __syncthreads();
        #pragma unroll
        for (int half = 0; half < kStepK / 128; ++half) {
            const int base_j = half * 128 + j0;
            const float4 xv =
                *reinterpret_cast<const float4 *>(xs + base_j);
            if (active0) {
                #pragma unroll
                for (int c = 0; c < kChansPerWarp; ++c) {
                    if (ch0 + c >= n_out0) {
                        break;
                    }
                    float wv[4];
                    decode_span4<DType>(base0 + c * row_bytes0, step * kStepK + base_j,
                                        wv);
                    acc0[c] = fmaf(wv[0], xv.x, acc0[c]);
                    acc0[c] = fmaf(wv[1], xv.y, acc0[c]);
                    acc0[c] = fmaf(wv[2], xv.z, acc0[c]);
                    acc0[c] = fmaf(wv[3], xv.w, acc0[c]);
                }
            }
            if (active1) {
                #pragma unroll
                for (int c = 0; c < kChansPerWarp; ++c) {
                    if (ch0 + c >= n_out1) {
                        break;
                    }
                    float wv[4];
                    decode_span4<DType>(base1 + c * row_bytes1, step * kStepK + base_j,
                                        wv);
                    acc1[c] = fmaf(wv[0], xv.x, acc1[c]);
                    acc1[c] = fmaf(wv[1], xv.y, acc1[c]);
                    acc1[c] = fmaf(wv[2], xv.z, acc1[c]);
                    acc1[c] = fmaf(wv[3], xv.w, acc1[c]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        // Every reduction runs for the whole wave: a shuffle inside a lane guard
        // would leave the other lanes out of it.
        const float total0 = active0 && ch0 + c < n_out0 ? warp_reduce_sum(acc0[c]) : 0.0f;
        const float total1 = active1 && ch0 + c < n_out1 ? warp_reduce_sum(acc1[c]) : 0.0f;
        if (lane != 0) {
            continue;
        }
        if (active0 && ch0 + c < n_out0) {
            out0[ch0 + c] = kSwiglu ? (total0 / (1.0f + expf(-total0))) * total1 : total0;
        }
        if (!kSwiglu && active1 && ch0 + c < n_out1) {
            out1[ch0 + c] = total1;
        }
    }
}

template <uint32_t DType>
void launch_lm_head_tile(float * out, const float * x, const uint8_t * w,
                         const float * res, int64_t n_in, int64_t n_out, int64_t rows,
                         int64_t row_bytes, hipStream_t stream) {
    const size_t shared = static_cast<size_t>(rows) * kStepK * sizeof(float);
    if (rows == 1) {
        constexpr int kChans = kChansPerWarpDecode;
        const unsigned blocks = static_cast<unsigned>(
            (n_out + kWarpsPerBlock * kChans - 1) / (kWarpsPerBlock * kChans));
        lm_head_kernel<DType, 1, kChans><<<blocks, kThreads, shared, stream>>>(
            out, x, w, res, n_in, n_out, rows, row_bytes);
    } else {
        constexpr int kChans = kChansPerWarp;
        const unsigned blocks = static_cast<unsigned>(
            (n_out + kWarpsPerBlock * kChans - 1) / (kWarpsPerBlock * kChans));
        lm_head_kernel<DType, kMaxRows, kChans><<<blocks, kThreads, shared, stream>>>(
            out, x, w, res, n_in, n_out, rows, row_bytes);
    }
}

// launch_lm_head_pair runs one of the two pair forms. Both matrices are walked
// by every block, so the grid covers the wider of the two output widths.
template <uint32_t DType>
void launch_lm_head_pair_tile(float * out0, float * out1, const float * x,
                              const uint8_t * w0, const uint8_t * w1, int64_t n_in,
                              int64_t n_out0, int64_t n_out1, int64_t row_bytes,
                              bool swiglu, hipStream_t stream) {
    const int64_t widest = n_out0 > n_out1 ? n_out0 : n_out1;
    const unsigned blocks = static_cast<unsigned>(
        (widest + kWarpsPerBlock * kChansPerWarp - 1) / (kWarpsPerBlock * kChansPerWarp));
    const size_t shared = size_t(kStepK) * sizeof(float);
    if (swiglu) {
        lm_head_pair_kernel<DType, true><<<blocks, kThreads, shared, stream>>>(
            out0, out1, x, w0, w1, n_in, n_out0, n_out1, row_bytes, row_bytes);
    } else {
        lm_head_pair_kernel<DType, false><<<blocks, kThreads, shared, stream>>>(
            out0, out1, x, w0, w1, n_in, n_out0, n_out1, row_bytes, row_bytes);
    }
}

void launch_lm_head_pair(float * out0, float * out1, const float * x,
                         const uint8_t * w0, const uint8_t * w1, int64_t n_in,
                         int64_t n_out0, int64_t n_out1, int64_t row_bytes,
                         uint32_t dtype, bool swiglu, hipStream_t stream) {
    // The pair form follows the single-row dispatch: a 512-value K step takes
    // the remapped kernel and keeps the activation row shared by both matrices.
    if (yuuka::gemv_row_available(dtype, static_cast<uint64_t>(n_in))) {
        yuuka::launch_gemv_row_pair_f32out(
            out0, out1, x, w0, w1, static_cast<uint64_t>(n_in),
            static_cast<uint64_t>(n_out0), static_cast<uint64_t>(n_out1), dtype,
            swiglu ? 1 : 0, stream);
        return;
    }
    (void)row_bytes;
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_lm_head_pair_tile<YUUKA_DTYPE_F32>(out0, out1, x, w0, w1, n_in,
                                                      n_out0, n_out1, row_bytes, swiglu,
                                                      stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_lm_head_pair_tile<YUUKA_DTYPE_F16>(out0, out1, x, w0, w1, n_in,
                                                      n_out0, n_out1, row_bytes, swiglu,
                                                      stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_lm_head_pair_tile<YUUKA_DTYPE_Q4_0>(out0, out1, x, w0, w1, n_in,
                                                       n_out0, n_out1, row_bytes, swiglu,
                                                       stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_lm_head_pair_tile<YUUKA_DTYPE_Q5_0>(out0, out1, x, w0, w1, n_in,
                                                       n_out0, n_out1, row_bytes, swiglu,
                                                       stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_lm_head_pair_tile<YUUKA_DTYPE_Q8_0>(out0, out1, x, w0, w1, n_in,
                                                       n_out0, n_out1, row_bytes, swiglu,
                                                       stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_lm_head_pair_tile<YUUKA_DTYPE_Q4_K>(out0, out1, x, w0, w1, n_in,
                                                       n_out0, n_out1, row_bytes, swiglu,
                                                       stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_lm_head_pair_tile<YUUKA_DTYPE_Q5_K>(out0, out1, x, w0, w1, n_in,
                                                       n_out0, n_out1, row_bytes, swiglu,
                                                       stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_lm_head_pair_tile<YUUKA_DTYPE_Q6_K>(out0, out1, x, w0, w1, n_in,
                                                       n_out0, n_out1, row_bytes, swiglu,
                                                       stream);
            break;
        default:
            break;
    }
}

} // namespace

namespace {

// dispatch_lm_head_tile selects the dtype instantiation and launches the
// streaming form, with res selecting the residual epilogue when non-null.
void dispatch_lm_head_tile(float * out, const float * x, const uint8_t * w,
                           const float * res, int64_t n_in, int64_t n_out, int64_t rows,
                           int64_t row_bytes, uint32_t dtype, hipStream_t stream) {
    // One row block on a 512-value K step takes the remapped kernel: one wave
    // per output row, wide packed spans, no K-step barrier. Other row counts and
    // K widths keep the streaming tile below.
    if (rows == 1 && yuuka::gemv_row_available(dtype, static_cast<uint64_t>(n_in))) {
        yuuka::launch_gemv_row_f32out(out, x, w, res, static_cast<uint64_t>(n_in),
                               static_cast<uint64_t>(n_out), dtype, stream);
        return;
    }
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_lm_head_tile<YUUKA_DTYPE_F32>(out, x, w, res, n_in, n_out, rows,
                                                 row_bytes, stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_lm_head_tile<YUUKA_DTYPE_F16>(out, x, w, res, n_in, n_out, rows,
                                                 row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_lm_head_tile<YUUKA_DTYPE_Q4_0>(out, x, w, res, n_in, n_out, rows,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_lm_head_tile<YUUKA_DTYPE_Q5_0>(out, x, w, res, n_in, n_out, rows,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_lm_head_tile<YUUKA_DTYPE_Q8_0>(out, x, w, res, n_in, n_out, rows,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_lm_head_tile<YUUKA_DTYPE_Q4_K>(out, x, w, res, n_in, n_out, rows,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_lm_head_tile<YUUKA_DTYPE_Q5_K>(out, x, w, res, n_in, n_out, rows,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_lm_head_tile<YUUKA_DTYPE_Q6_K>(out, x, w, res, n_in, n_out, rows,
                                                  row_bytes, stream);
            break;
        default:
            break;
    }
}

} // namespace

namespace yuuka {

void launch_lm_head_f32out(float * out, const float * x, const void * w,
                           uint64_t n_in, uint64_t n_out, uint64_t rows,
                           uint32_t dtype, hipStream_t stream) {
    launch_lm_head_residual_f32out(out, x, w, nullptr, n_in, n_out, rows, dtype,
                                   stream);
}

void launch_lm_head_pair_f32out(float * out0, float * out1, const float * x,
                                const void * w0, const void * w1, uint64_t n_in,
                                uint64_t n_out0, uint64_t n_out1, uint32_t dtype,
                                int swiglu, hipStream_t stream) {
    if (n_in == 0 || n_out0 == 0 || n_out1 == 0 || n_in % kStepK != 0) {
        return;
    }
    const uint64_t row_bytes = yuuka_packed_row_bytes(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    launch_lm_head_pair(out0, out1, x, static_cast<const uint8_t *>(w0),
                        static_cast<const uint8_t *>(w1), static_cast<int64_t>(n_in),
                        static_cast<int64_t>(n_out0), static_cast<int64_t>(n_out1),
                        static_cast<int64_t>(row_bytes), dtype, swiglu != 0, stream);
}

void launch_lm_head_residual_f32out(float * out, const float * x, const void * w,
                                    const float * res, uint64_t n_in, uint64_t n_out,
                                    uint64_t rows, uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || rows == 0 || rows > kMaxRows ||
        n_in % kStepK != 0) {
        return;
    }
    const uint64_t row_bytes = yuuka_packed_row_bytes(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * weights = static_cast<const uint8_t *>(w);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto m = static_cast<int64_t>(rows);
    const auto rb = static_cast<int64_t>(row_bytes);
    dispatch_lm_head_tile(out, x, weights, res, in, on, m, rb, dtype, stream);
}

} // namespace yuuka

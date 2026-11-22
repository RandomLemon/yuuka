#include "../common.cuh"
#include "../kv.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

namespace {

// The WMMA prefill attention kernel tiles a 16-row query block against 16-key
// KV tiles. One workgroup is four waves, one wave per query head of the group
// that shares a KV head, so the staged K/V tile serves all four heads. Each
// wave keeps the whole 16 x head_dim output block in accumulator registers and
// runs the online softmax in the accumulator fragment layout, which keeps the
// per-row max, the rescale factor, and the probability values on the lane that
// owns them.
constexpr int kWmmaHeads = 4;
constexpr int kWmmaQTile = 16;
constexpr int kWmmaKTile = 16;
constexpr int kWmmaThreads = kWmmaHeads * 32;
constexpr int kPrefixSeedKeys = 512;
// A short chunk leaves too few 16-row query tiles to fill the device, so the
// tiled kernel wins below this many rows and at short context. The crossover
// and its measurements are recorded in kernels.md.
constexpr int kWmmaMinRows = 64;
constexpr int kWmmaMinPrefix = 256;
// A 16x16 FP32 accumulator spreads 256 values over one wave, eight per lane.
// Lane l holds element e at output row 2*e + (l >> 4), column l & 15.
constexpr int kWmmaLaneElems = 8;

using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half,
                                 rocwmma::row_major>;
using frag_b_col = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::col_major>;
using frag_b_row = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::row_major>;
using frag_acc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

// wmma_paged_row maps a logical KV position to its pool row through the block
// table.
__device__ __forceinline__ int64_t wmma_paged_row(const int32_t * block_table,
                                                 int block_size, int pos) {
    const int block = block_table[pos / block_size];
    return static_cast<int64_t>(block) * block_size + (pos % block_size);
}

// stage_kv_tile gathers one 16-key tile of K and V through the block table and
// writes it as FP16. An F16 pool is copied, an F32 pool is converted, and keys
// outside the tile's causal bound are zeroed. Each key row segment is
// contiguous in the pool, so the copy moves four values per step.
template <uint32_t DType, int HD, bool KeyOnly = false>
__device__ __forceinline__ void stage_kv_tile(
    __half * kh, __half * vh, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int kv_dim, int kvh, int k0,
    int valid) {
    constexpr int kVec = 4;
    constexpr int kVecs = HD / kVec;
    const int tid = threadIdx.x;
    constexpr int keys = KeyOnly ? 2 * kWmmaKTile : kWmmaKTile;
    for (int idx = tid; idx < keys * kVecs; idx += kWmmaThreads) {
        const int kk = idx / kVecs;
        const int d = (idx % kVecs) * kVec;
        __half * kdst = kh + kk * HD + d;
        if (kk >= valid) {
            *reinterpret_cast<uint2 *>(kdst) = make_uint2(0u, 0u);
            if constexpr (!KeyOnly) {
                *reinterpret_cast<uint2 *>(vh + kk * HD + d) = make_uint2(0u, 0u);
            }
            continue;
        }
        const int64_t row = wmma_paged_row(block_table, block_size, k0 + kk);
        const int64_t col = static_cast<int64_t>(kvh) * HD + d;
        if constexpr (DType == YUUKA_DTYPE_F16) {
            const __half * ksrc =
                static_cast<const __half *>(k_pool) + row * kv_dim + col;
            *reinterpret_cast<uint2 *>(kdst) =
                *reinterpret_cast<const uint2 *>(ksrc);
            if constexpr (!KeyOnly) {
                const __half * vsrc =
                    static_cast<const __half *>(v_pool) + row * kv_dim + col;
                *reinterpret_cast<uint2 *>(vh + kk * HD + d) =
                    *reinterpret_cast<const uint2 *>(vsrc);
            }
        } else {
            static_assert(!KeyOnly);
            const float * ksrc =
                static_cast<const float *>(k_pool) + row * kv_dim + col;
            const float * vsrc =
                static_cast<const float *>(v_pool) + row * kv_dim + col;
            const float4 k4 = *reinterpret_cast<const float4 *>(ksrc);
            const float4 v4 = *reinterpret_cast<const float4 *>(vsrc);
            __half2 * kout = reinterpret_cast<__half2 *>(kdst);
            __half2 * vout = reinterpret_cast<__half2 *>(vh + kk * HD + d);
            kout[0] = __floats2half2_rn(k4.x, k4.y);
            kout[1] = __floats2half2_rn(k4.z, k4.w);
            vout[0] = __floats2half2_rn(v4.x, v4.y);
            vout[1] = __floats2half2_rn(v4.z, v4.w);
        }
    }
}

// attention_prefill_paged_wmma_kernel is the FP16 MMA paged prefill attention
// path. Q, K, and V are staged as FP16, both products accumulate in FP32, and
// the causal mask is applied inside the tile before the online softmax update.
//
// The score product runs as S^T = K Q^T so that both operands are row major in
// shared memory: K keeps the key-row layout it is gathered in, and Q is stored
// transposed once per workgroup instead of once per key tile. A score lane then
// owns one query row and eight keys, so a row reduction is eight register
// operations and one cross-lane step instead of a four-step butterfly, and the
// running maximum and sum are scalars.
template <uint32_t DType, int HD, bool Dense, bool Partial = false,
          bool MaxOnly = false, int GroupsOfFour = 1>
__launch_bounds__(kWmmaThreads, 2)
__global__ void attention_prefill_paged_wmma_kernel(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int heads, int kv_dim,
    int rows, int position, float scale, float * range_max, int splits) {
    constexpr int kOutFrags = HD / 16;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    // One workgroup serves four logical query heads. At a 4:1 GQA ratio those
    // four share the workgroup's KV head; at 8:1 two consecutive workgroups
    // share it, so the grouping template keeps the KV index separate from the
    // logical head used for Q, output, gate, and partial offsets.
    const int q_head_base = kWmmaHeads * blockIdx.x;
    const int kvh = blockIdx.x / GroupsOfFour;
    const int head = q_head_base + wave;
    const int q_base = blockIdx.y * kWmmaQTile;
    const int head_stride = Dense ? HD : 2 * HD;
    const int row_stride = heads * head_stride;

    extern __shared__ __align__(16) unsigned char smem[];
    // qh holds Q transposed, [dim][query row] per head, which is the matrix_b
    // row-major layout of the score product.
    __half * qh = reinterpret_cast<__half *>(smem);
    __half * kh = qh + kWmmaHeads * kWmmaQTile * HD;
    constexpr int keys = MaxOnly ? 2 * kWmmaKTile : kWmmaKTile;
    __half * vh = kh + keys * HD;
    __half * ph = vh + kWmmaKTile * HD;
    const int ph_stride = kWmmaQTile * kWmmaKTile;

    for (int idx = tid; idx < kWmmaHeads * kWmmaQTile * HD;
         idx += kWmmaThreads) {
        const int h = idx / (kWmmaQTile * HD);
        const int d = (idx / kWmmaQTile) % HD;
        const int r = idx % kWmmaQTile;
        const int q_row = q_base + r;
        float value = 0.0f;
        if (q_row < rows) {
            value = q[static_cast<int64_t>(q_row) * row_stride +
                      (q_head_base + h) * head_stride + d];
        }
        qh[h * kWmmaQTile * HD + d * kWmmaQTile + r] = __float2half(value);
    }
    __syncthreads();

    frag_acc acc[kOutFrags];
    #pragma unroll
    for (int n = 0; n < kOutFrags; ++n) {
        rocwmma::fill_fragment(acc[n], 0.0f);
    }

    const int lane_key = lane >> 4;
    const int lane_row = lane & 15;
    const int q_last = min(q_base + kWmmaQTile, rows) - 1;
    const int q_pos = position + min(q_base + lane_row, q_last);
    const int max_pos = position + q_last;
    const __half * q_head_ptr = qh + wave * kWmmaQTile * HD;
    __half * p_tile = ph + wave * ph_stride;
    // The softmax runs in base-2 units so an exponent is a single scaled exp.
    const float log2_scale = scale * 1.4426950408889634f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t row_head = static_cast<int64_t>(q_base + lane_row) * heads + head;

    int key_begin = 0;
    int key_end = max_pos + 1;
    if constexpr (Partial) {
        const int tiles = max_pos / kWmmaKTile + 1;
        key_begin = int(int64_t(blockIdx.z) * tiles / splits) * kWmmaKTile;
        const int64_t next_tile = int64_t(blockIdx.z + 1) * tiles / splits;
        key_end = int(min(int64_t(max_pos) + 1, next_tile * kWmmaKTile));
        if constexpr (MaxOnly) {
            key_end = min(key_end, kPrefixSeedKeys);
        } else if (range_max != nullptr && blockIdx.z > 0 && q_base + lane_row < rows) {
            // Use a shared leading-prefix maximum to limit the change in
            // FP16 probability rounding for the later key ranges.
            row_max = range_max[row_head];
        }
    }
    for (int k0 = key_begin; k0 < key_end; k0 += keys) {
        const int valid = min(keys, key_end - k0);
        stage_kv_tile<DType, HD, MaxOnly>(kh, vh, k_pool, v_pool, block_table, block_size,
                                 kv_dim, kvh, k0, valid);
        __syncthreads();

        // S^T: rows are keys, columns are query rows.
        frag_acc score;
        rocwmma::fill_fragment(score, 0.0f);
        frag_acc next_score;
        if constexpr (MaxOnly) {
            rocwmma::fill_fragment(next_score, 0.0f);
        }
        frag_a k_frag;
        frag_b_row q_frag;
        #pragma unroll
        for (int d = 0; d < HD; d += 16) {
            rocwmma::load_matrix_sync(k_frag, kh + d, HD);
            rocwmma::load_matrix_sync(q_frag, q_head_ptr + d * kWmmaQTile,
                                      kWmmaQTile);
            rocwmma::mma_sync(score, k_frag, q_frag, score);
            if constexpr (MaxOnly) {
                rocwmma::load_matrix_sync(k_frag, kh + kWmmaKTile * HD + d, HD);
                rocwmma::mma_sync(next_score, k_frag, q_frag, next_score);
            }
        }

        float prob[kWmmaLaneElems];
        float mx = -INFINITY;
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int key = 2 * e + lane_key;
            float value = score.x[e] * log2_scale;
            if (key >= valid || k0 + key > q_pos) {
                value = -INFINITY;
            }
            prob[e] = value;
            mx = fmaxf(mx, value);
            if constexpr (MaxOnly) {
                const int next_key = key + kWmmaKTile;
                if (next_key < valid && k0 + next_key <= q_pos) {
                    mx = fmaxf(mx, next_score.x[e] * log2_scale);
                }
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        const float updated = fmaxf(row_max, mx);
        if constexpr (MaxOnly) {
            row_max = updated;
            __syncthreads();
            continue;
        }
        const float rescale = Partial && updated == -INFINITY
            ? 1.0f : exp2f(row_max - updated);
        row_max = updated;
        float sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int key = 2 * e + lane_key;
            const float p = Partial && updated == -INFINITY ? 0.0f : exp2f(prob[e] - updated);
            sum += p;
            p_tile[lane_row * kWmmaKTile + key] = __float2half(p);
        }
        sum += __shfl_xor(sum, 16);
        row_sum = row_sum * rescale + sum;
        // The accumulator holds query rows in its own lane mapping, so the
        // rescale factor is fetched from the lane that owns each of those rows.
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const float alpha = __shfl(rescale, 2 * e + lane_key);
            #pragma unroll
            for (int n = 0; n < kOutFrags; ++n) {
                acc[n].x[e] *= alpha;
            }
        }
        __syncwarp();

        frag_a p_frag;
        rocwmma::load_matrix_sync(p_frag, p_tile, kWmmaKTile);
        frag_b_row v_frag;
        #pragma unroll
        for (int n = 0; n < kOutFrags; ++n) {
            rocwmma::load_matrix_sync(v_frag, vh + n * 16, HD);
            rocwmma::mma_sync(acc[n], p_frag, v_frag, acc[n]);
        }
        __syncthreads();
    }

    if constexpr (MaxOnly) {
        if (lane_key == 0 && q_base + lane_row < rows) {
            range_max[row_head] = row_max;
        }
        return;
    }
    if constexpr (Partial) {
        const int64_t split_offset = static_cast<int64_t>(blockIdx.z) * rows * heads * (HD + 2);
        if (lane_key == 0 && q_base + lane_row < rows) {
            float * partial = out + split_offset +
                (static_cast<int64_t>(q_base + lane_row) * heads + head) * (HD + 2);
            partial[0] = row_sum > 0.0f ? row_max : -INFINITY;
            partial[1] = row_sum;
        }
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int q_row = q_base + 2 * e + lane_key;
            if (q_row < rows) {
                float * partial = out + split_offset +
                    (static_cast<int64_t>(q_row) * heads + head) * (HD + 2) + 2;
                #pragma unroll
                for (int n = 0; n < kOutFrags; ++n) {
                    partial[n * 16 + lane_row] = acc[n].x[e];
                }
            }
        }
        return;
    }
    const float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int n = 0; n < kOutFrags; ++n) {
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int row = 2 * e + lane_key;
            const int q_row = q_base + row;
            if (q_row >= rows) {
                continue;
            }
            const int col = n * 16 + lane_row;
            const int64_t out_index = static_cast<int64_t>(q_row) * heads * HD +
                                      static_cast<int64_t>(head) * HD + col;
            float value = acc[n].x[e] * __shfl(inv_sum, row);
            if constexpr (!Dense) {
                const float gate =
                    q[static_cast<int64_t>(q_row) * row_stride +
                      static_cast<int64_t>(head) * head_stride + HD + col];
                value = value / (1.0f + expf(-gate));
            }
            out[out_index] = value;
        }
    }
}

// Compose the unnormalized values in ascending key-range order. Empty
// partials contribute zero and can carry an infinite negative maximum.
__global__ void merge_attention_partials_f32(float * out, const float * q,
        const float * partial, int rows, int heads, int splits) {
    constexpr int hd = 256;
    constexpr int record = hd + 2;
    const int lane = threadIdx.x & 31;
    const int64_t row_head = int64_t(blockIdx.x) * kWmmaHeads + threadIdx.x / 32;
    if (row_head >= int64_t(rows) * heads) return;
    const int64_t stride = static_cast<int64_t>(rows) * heads * record;
    const float * first = partial + row_head * record;
    float weights[4] = {};
    float denominator = 0.0f;
    if (lane == 0) {
        float max_score = -INFINITY;
        for (int i = 0; i < splits; ++i) {
            const float * part = first + i * stride;
            if (part[1] > 0.0f) max_score = fmaxf(max_score, part[0]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i < splits) {
                const float * part = first + i * stride;
                weights[i] = part[1] > 0.0f ? exp2f(part[0] - max_score) : 0.0f;
                denominator += weights[i] * part[1];
            }
        }
    }
    denominator = __shfl(denominator, 0);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        weights[i] = __shfl(weights[i], 0);
    }
    for (int d = lane; d < hd; d += 32) {
        float value = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i < splits) {
                value += weights[i] * first[i * stride + d + 2];
            }
        }
        const float gate = q[row_head * 2 * hd + hd + d];
        out[row_head * hd + d] = (value * (1.0f / denominator)) / (1.0f + expf(-gate));
    }
}

template <uint32_t DType, int HD, int GroupsOfFour = 1>
hipError_t launch_attention_wmma_dtype(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int heads, int kv_dim,
    int rows, int position, float scale, int dense, dim3 grid, size_t shared,
    hipStream_t stream) {
    if (dense) {
        static const hipError_t attribute = hipFuncSetAttribute(
            reinterpret_cast<const void *>(
                &attention_prefill_paged_wmma_kernel<DType, HD, true, false,
                                                     false, GroupsOfFour>),
            hipFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (attribute != hipSuccess) return attribute;
        attention_prefill_paged_wmma_kernel<DType, HD, true, false, false,
                                            GroupsOfFour>
            <<<grid, kWmmaThreads, shared, stream>>>(out, q, k_pool, v_pool,
                                                     block_table, block_size,
                                                     heads, kv_dim, rows,
                                                     position, scale, nullptr, 1);
    } else {
        static const hipError_t attribute = hipFuncSetAttribute(
            reinterpret_cast<const void *>(
                &attention_prefill_paged_wmma_kernel<DType, HD, false, false,
                                                     false, GroupsOfFour>),
            hipFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (attribute != hipSuccess) return attribute;
        attention_prefill_paged_wmma_kernel<DType, HD, false, false, false,
                                            GroupsOfFour>
            <<<grid, kWmmaThreads, shared, stream>>>(out, q, k_pool, v_pool,
                                                     block_table, block_size,
                                                     heads, kv_dim, rows,
                                                     position, scale, nullptr, 1);
    }
    return hipGetLastError();
}

}  // namespace

namespace yuuka {

template <uint32_t DType, int HD, int GroupsOfFour>
hipError_t launch_attention_prefill_split_group(
        float * out, const float * q, const void * k_pool, const void * v_pool,
        const int32_t * block_table, int block_size, int heads, int kv_dim,
        int rows, int position, float scale, float * partial, int splits,
        hipStream_t stream) {
    constexpr size_t shared =
        ((kWmmaHeads * kWmmaQTile + 2 * kWmmaKTile) * HD +
         kWmmaHeads * kWmmaQTile * kWmmaKTile) * sizeof(__half);
    static const hipError_t attribute = hipFuncSetAttribute(
        reinterpret_cast<const void *>(
            &attention_prefill_paged_wmma_kernel<DType, HD, false, true, false,
                                                 GroupsOfFour>),
        hipFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (attribute != hipSuccess) return attribute;
    const dim3 grid((heads + kWmmaHeads - 1) / kWmmaHeads,
                    (rows + kWmmaQTile - 1) / kWmmaQTile, splits);
    float * range_max = nullptr;
    if (splits > 1) {
        constexpr size_t max_shared = (kWmmaHeads * kWmmaQTile + 2 * kWmmaKTile) *
            HD * sizeof(__half);
        static const hipError_t max_attribute = hipFuncSetAttribute(
            reinterpret_cast<const void *>(
                &attention_prefill_paged_wmma_kernel<DType, HD, false, true, true,
                                                     GroupsOfFour>),
            hipFuncAttributeMaxDynamicSharedMemorySize, max_shared);
        if (max_attribute != hipSuccess) return max_attribute;
        range_max = partial + static_cast<int64_t>(splits) * rows * heads * (HD + 2);
        const dim3 max_grid(grid.x, grid.y);
        attention_prefill_paged_wmma_kernel<DType, HD, false, true, true,
                                            GroupsOfFour>
            <<<max_grid, kWmmaThreads, max_shared, stream>>>(
                nullptr, q, k_pool, nullptr, block_table, block_size, heads,
                kv_dim, rows, position, scale, range_max, splits);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
    }
    attention_prefill_paged_wmma_kernel<DType, HD, false, true, false,
                                        GroupsOfFour>
        <<<grid, kWmmaThreads, shared, stream>>>(
            partial, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, range_max, splits);
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    const int blocks = (rows * heads - 1) / kWmmaHeads + 1;
    merge_attention_partials_f32<<<blocks, kWmmaThreads, 0, stream>>>(
        out, q, partial, rows, heads, splits);
    return hipGetLastError();
}

hipError_t launch_attention_prefill_paged_split_f16(float * out, const float * q,
        const void * k_pool, const void * v_pool, const int32_t * block_table,
        int block_size, int kv_heads, int heads, int kv_dim, int rows,
        int position, float scale, float * partial, int splits, hipStream_t stream) {
    constexpr int head_dim = 256;
    const int groups = heads == kv_heads * 4 ? 1
                       : heads == kv_heads * 8 ? 2
                                               : 0;
    if (groups == 0) return hipErrorInvalidValue;
    if (groups == 2) {
        return launch_attention_prefill_split_group<YUUKA_DTYPE_F16, head_dim, 2>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, partial, splits, stream);
    }
    return launch_attention_prefill_split_group<YUUKA_DTYPE_F16, head_dim, 1>(
        out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
        position, scale, partial, splits, stream);
}

size_t attention_wmma_shared_bytes(int head_dim) {
    const size_t hd = static_cast<size_t>(head_dim);
    return (kWmmaHeads * kWmmaQTile * hd + 2 * kWmmaKTile * hd) * sizeof(__half) +
           kWmmaHeads * kWmmaQTile * kWmmaKTile * sizeof(__half);
}

bool attention_wmma_preferred(int rows, int position) {
    return rows >= kWmmaMinRows || position + rows >= kWmmaMinPrefix;
}

bool attention_wmma_supported(int heads, int kv_heads, int head_dim,
                              int block_size, uint32_t dtype) {
    if (block_size <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return false;
    }
    if (dtype != YUUKA_DTYPE_F32 && dtype != YUUKA_DTYPE_F16) {
        return false;
    }
    const bool ratio4 = heads == kv_heads * kWmmaHeads;
    const bool ratio8 = dtype == YUUKA_DTYPE_F16 && head_dim == 256 &&
                        heads == kv_heads * 2 * kWmmaHeads;
    if (!ratio4 && !ratio8) {
        return false;
    }
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
        return false;
    }
    return attention_wmma_shared_bytes(head_dim) <= 64 * 1024;
}

hipError_t launch_attention_prefill_paged_wmma_f32(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int kv_heads, int heads,
    int head_dim, int kv_dim, int rows, int position, float scale, uint32_t dtype,
    int dense, hipStream_t stream) {
    const int groups = heads == kv_heads * kWmmaHeads
        ? 1
        : (dtype == YUUKA_DTYPE_F16 && head_dim == 256 &&
           heads == kv_heads * 2 * kWmmaHeads ? 2 : 0);
    if (groups == 0) {
        return hipErrorInvalidValue;
    }
    const size_t shared = attention_wmma_shared_bytes(head_dim);
    const dim3 grid(static_cast<unsigned>((heads + kWmmaHeads - 1) / kWmmaHeads),
                    static_cast<unsigned>((rows + kWmmaQTile - 1) / kWmmaQTile));
    if (dtype == YUUKA_DTYPE_F16) {
        if (head_dim == 64) {
            return launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 64>(
                out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
                rows, position, scale, dense, grid, shared, stream);
        }
        if (head_dim == 128) {
            return launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 128>(
                out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
                rows, position, scale, dense, grid, shared, stream);
        }
        if (groups == 2) {
            return launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 256, 2>(
                out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
                rows, position, scale, dense, grid, shared, stream);
        }
        return launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 256>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
            rows, position, scale, dense, grid, shared, stream);
    }
    if (head_dim == 64) {
        return launch_attention_wmma_dtype<YUUKA_DTYPE_F32, 64>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, dense, grid, shared, stream);
    }
    if (head_dim == 128) {
        return launch_attention_wmma_dtype<YUUKA_DTYPE_F32, 128>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, dense, grid, shared, stream);
    }
    return launch_attention_wmma_dtype<YUUKA_DTYPE_F32, 256>(
        out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
        position, scale, dense, grid, shared, stream);
}

}  // namespace yuuka

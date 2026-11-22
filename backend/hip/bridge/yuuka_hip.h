#ifndef YUUKA_HIP_H
#define YUUKA_HIP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YUUKA_HIP_ABI_VERSION 12

#if defined(_WIN32)
#define YUUKA_API __declspec(dllexport)
#elif defined(__GNUC__)
#define YUUKA_API __attribute__((visibility("default")))
#else
#define YUUKA_API
#endif

/* Opaque handles. Ownership is documented per function. */
typedef struct yuuka_device yuuka_device;
typedef struct yuuka_buffer yuuka_buffer;
typedef struct yuuka_stream yuuka_stream;
typedef struct yuuka_event yuuka_event;

typedef enum yuuka_status {
    YUUKA_OK = 0,
    YUUKA_ERR_INVALID_ARGUMENT = 1,
    YUUKA_ERR_UNSUPPORTED = 2,
    YUUKA_ERR_DEVICE = 3,
    YUUKA_ERR_ALLOCATION = 4,
    YUUKA_ERR_INTERNAL = 5
} yuuka_status;

typedef struct yuuka_device_info {
    char name[256];
    char gcn_arch[64];
    char hip_version[64];
    uint64_t total_memory;
    uint32_t compute_units;
    uint32_t clock_mhz;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} yuuka_device_info;

/* Returns the ABI version compiled into the shared library. */
YUUKA_API uint32_t yuuka_abi_version(void);

/* Returns a thread-local message for the most recent failed call. */
YUUKA_API const char * yuuka_last_error(void);

/* Opens device index and returns a handle owned by the caller. */
YUUKA_API yuuka_status yuuka_device_open(int32_t index, yuuka_device ** out);
YUUKA_API void yuuka_device_close(yuuka_device * dev);
YUUKA_API yuuka_status yuuka_device_info_get(const yuuka_device * dev, yuuka_device_info * info);

/* Device allocations are owned by the caller and freed exactly once. */
YUUKA_API yuuka_status yuuka_buffer_alloc(yuuka_device * dev, uint64_t bytes, yuuka_buffer ** out);
YUUKA_API void yuuka_buffer_free(yuuka_buffer * buf);
YUUKA_API uint64_t yuuka_buffer_size(const yuuka_buffer * buf);
YUUKA_API void * yuuka_buffer_ptr(const yuuka_buffer * buf);

/* Explicit stream and event objects. Events are single-use completion signals. */
YUUKA_API yuuka_status yuuka_stream_create(yuuka_device * dev, yuuka_stream ** out);
YUUKA_API void yuuka_stream_destroy(yuuka_stream * stream);
YUUKA_API yuuka_status yuuka_stream_synchronize(yuuka_stream * stream);
YUUKA_API yuuka_status yuuka_event_create(yuuka_device * dev, yuuka_event ** out);
YUUKA_API void yuuka_event_destroy(yuuka_event * event);
YUUKA_API yuuka_status yuuka_event_wait(yuuka_event * event);

/* Host transfers are synchronous with respect to the host buffer. */
YUUKA_API yuuka_status yuuka_memcpy_h2d(yuuka_buffer * dst, const void * src, uint64_t bytes);
YUUKA_API yuuka_status yuuka_memcpy_d2h(void * dst, const yuuka_buffer * src, uint64_t bytes);
YUUKA_API yuuka_status yuuka_memcpy_d2h_at(void * dst, const yuuka_buffer * src,
                                           uint64_t offset, uint64_t bytes);
YUUKA_API yuuka_status yuuka_memset(yuuka_stream * stream, yuuka_buffer * dst, int value, uint64_t bytes);

/* Asynchronous device-to-device copy on stream. Offsets and length must fit
 * both buffers. Used to give a request a private prefix block. */
YUUKA_API yuuka_status yuuka_buffer_copy(yuuka_stream * stream,
                                yuuka_buffer * dst, uint64_t dst_offset,
                                const yuuka_buffer * src, uint64_t src_offset,
                                uint64_t bytes, yuuka_event * completion);

/* Uploads the two dynamic decode parameters: int32 token and int32 position.
 * Graph replay reads them from device memory so kernel arguments stay fixed. */
YUUKA_API yuuka_status yuuka_dynamic_set(yuuka_stream * stream, yuuka_buffer * dynamic,
                                int32_t token, uint32_t position);

/* Elementwise smoke operation used by the ABI test. */
YUUKA_API yuuka_status yuuka_add_f32(yuuka_stream * stream, yuuka_buffer * dst,
                           const yuuka_buffer * a, const yuuka_buffer * b,
                           uint64_t n, yuuka_event * completion);

/* Fused quantized matrix-vector product.
 *
 * out[n_out] = W[n_out, n_in] * x[n_in]
 * W uses the packed GGUF layout selected by dtype. Supported dtype values are
 * declared in yuuka_dtype. The operation is submitted on stream and signals
 * completion after the kernel. No host synchronization is performed.
 */
typedef enum yuuka_op_code {
    YUUKA_OP_GEMV = 1,
    YUUKA_OP_RMSNORM = 2,
    YUUKA_OP_L2NORM = 3,
    YUUKA_OP_GATED_RMSNORM = 4,
    YUUKA_OP_ROPE = 5,
    YUUKA_OP_ATTENTION = 6,
    YUUKA_OP_CONV1D = 7,
    YUUKA_OP_GDN = 8,
    YUUKA_OP_SILU_MUL = 9,
    YUUKA_OP_ADD = 10,
    YUUKA_OP_COPY = 11,
    YUUKA_OP_SCALE = 12,
    YUUKA_OP_EMBEDDING = 13,
    YUUKA_OP_KV_WRITE = 14,
    YUUKA_OP_RMSNORM_HEADS = 15,
    YUUKA_OP_GEMM = 16,
    YUUKA_OP_GEMV_BATCH = 17,
    YUUKA_OP_RMSNORM_BATCH = 18,
    YUUKA_OP_RMSNORM_HEADS_BATCH = 19,
    YUUKA_OP_ROPE_BATCH = 20,
    YUUKA_OP_KV_WRITE_BATCH = 21,
    YUUKA_OP_ATTENTION_PREFILL = 22,
    YUUKA_OP_RECURRENT_CHUNK = 23,
    YUUKA_OP_EMBEDDING_BATCH = 24,
    YUUKA_OP_CAST_F16 = 25,
    YUUKA_OP_KV_WRITE_PAGED = 26,
    YUUKA_OP_ATTENTION_PAGED = 27,
    YUUKA_OP_KV_WRITE_BATCH_PAGED = 28,
    YUUKA_OP_ATTENTION_PREFILL_PAGED = 29,
    YUUKA_OP_MOE_ROUTE = 30,
    YUUKA_OP_MOE_GATE_UP = 31,
    YUUKA_OP_MOE_DOWN = 32,
    YUUKA_OP_MOE_REDUCE_ADD = 33,
    YUUKA_OP_GEMM_SWIGLU = 34,
    YUUKA_OP_DEQUANT_F16 = 35,
    YUUKA_OP_QUANT_Q8_1 = 36,
    YUUKA_OP_LM_HEAD = 37,
    YUUKA_OP_GEMV_Q8_1 = 38,
    YUUKA_OP_RECURRENT_TOKEN = 39,
    YUUKA_OP_CONV_L2NORM = 40,
    YUUKA_OP_GEMV_RESIDUAL = 41,
    YUUKA_OP_GEMV_PAIR = 42,
    YUUKA_OP_SILU_MUL_QUANT_Q8_1 = 43,
    YUUKA_OP_MOE_GATE_UP_SELECTED = 44,
    YUUKA_OP_MOE_DOWN_SELECTED = 45,
    YUUKA_OP_MOE_SHARED_ADD = 46,
    YUUKA_OP_MOE_GATE_UP_Q8 = 47,
    YUUKA_OP_MOE_DOWN_Q8 = 48,
    YUUKA_OP_MOE_GATE_UP_MMQ = 49,
    YUUKA_OP_MOE_DOWN_MMQ = 50
} yuuka_op_code;

#define YUUKA_OP_FLAG_TAIL 0x1u
#define YUUKA_OP_FLAG_PREFILL_BLAS 0x2u
// Forces the tiled paged prefill attention kernel where the FP16 MMA kernel
// would otherwise be selected. Validation and measurement only.
#define YUUKA_OP_FLAG_PREFILL_ATTENTION_TILED 0x4u
// These values are mutually exclusive choices in the split-count field.
#define YUUKA_OP_FLAG_PREFILL_SPLIT_MASK 0x18u
#define YUUKA_OP_FLAG_PREFILL_SPLIT_1 0x08u
#define YUUKA_OP_FLAG_PREFILL_SPLIT_2 0x10u
#define YUUKA_OP_FLAG_PREFILL_SPLIT_4 0x18u

/* GEMM implementation selection carried in yuuka_op.u[1]. AUTO uses the
 * compiled default for the dtype. BLAS is the rocBLAS route and requires FP16
 * weights, an FP16 x, and an FP16 scratch buffer that receives the FP32-accumulated
 * product before it is cast to the FP32 result. WMMA is the custom rocWMMA
 * kernel, which accepts FP32 x and every supported weight dtype. MMQ is the
 * int8 route: Q4_K/Q5_K/Q6_K weights, Q8_1 activations in buf[1], their
 * float2 block scales in buf[3], and FP32 output. */
#define YUUKA_GEMM_PATH_AUTO 0u
#define YUUKA_GEMM_PATH_BLAS 1u
#define YUUKA_GEMM_PATH_WMMA 2u
#define YUUKA_GEMM_PATH_MMQ 3u

/* Int8 decode matvec: out[n_out] = W[n_out, n_in] * x. buf[0] holds the packed
 * weights, buf[3] the FP32 output, n[0] = n_in, n[1] = n_out, u[0] = dtype.
 * u[1] selects where the Q8_1 activation comes from:
 *
 *   0  buf[1] holds the int8 digits and buf[2] their float2 block scales, both
 *      written by YUUKA_OP_QUANT_Q8_1;
 *   1  buf[1] holds the FP32 activation row and the block quantizes it into
 *      shared memory itself, so no quantization operation is submitted.
 *
 * Q4_K, Q5_K, Q6_K, and Q8_0 are supported; rows are not, because speculative
 * verification keeps the FP32 streaming kernel. */
typedef struct yuuka_op {
    uint32_t code;
    uint32_t flags;
    yuuka_buffer * buf[16];
    uint64_t off[16];
    uint64_t n[6];
    uint32_t u[4];
    float f[4];
} yuuka_op;

typedef struct yuuka_plan yuuka_plan;
typedef struct yuuka_graph yuuka_graph;

typedef enum yuuka_dtype {
    YUUKA_DTYPE_F32 = 0,
    YUUKA_DTYPE_F16 = 1,
    YUUKA_DTYPE_Q4_0 = 2,
    YUUKA_DTYPE_Q5_0 = 6,
    YUUKA_DTYPE_Q8_0 = 8,
    YUUKA_DTYPE_Q4_K = 12,
    YUUKA_DTYPE_Q5_K = 13,
    YUUKA_DTYPE_Q6_K = 14,
    /* IQ2_XXS, IQ2_S, and IQ3_S are decode-supported only by the MoE expert
     * kernels and by yuuka_dequant_f16; every other kernel family rejects
     * them so a general matvec cannot read them as zero. */
    YUUKA_DTYPE_IQ2_XXS = 16,
    YUUKA_DTYPE_IQ3_S = 21,
    YUUKA_DTYPE_IQ2_S = 22
} yuuka_dtype;

YUUKA_API yuuka_status yuuka_gemv(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint32_t dtype,
                        yuuka_event * completion);

/* Batched matrix-vector product for prefill:
 *
 * out[batch, n_out] = W[n_out, n_in] * x[batch, n_in]
 *
 * x and out are contiguous row-major [batch, n_in] and [batch, n_out]. Only
 * F32 and F16 weights are supported by the Phase 1 baseline.
 */
YUUKA_API yuuka_status yuuka_gemm(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        uint32_t dtype, yuuka_event * completion);

/* Custom rocWMMA FP16/quantized GEMM:
 *
 * out[batch, n_out] = W[n_out, n_in] * x[batch, n_in]
 *
 * x is FP32 row-major; the kernel converts each tile to FP16 in shared memory.
 * w may be any supported dtype, including Q4_K, Q5_K, Q6_K, and Q8_0. This path
 * stages one weight tile at a time and never materializes a dequantized weight
 * matrix; the BLAS route below does materialize one FP16 weight tile per call
 * in a caller-owned scratch.
 */
YUUKA_API yuuka_status yuuka_gemm_wmma(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        uint32_t dtype, yuuka_event * completion);

/* Int8 MMQ prefill route:
 *
 * out[batch, n_out] = W[n_out, n_in] * x[batch, n_in]
 *
 * The two calls are the split form of one GEMM. yuuka_quantize_q8_1 converts
 * FP32 activations into Q8_1: qs holds batch*n int8 values and ds holds
 * batch*(n/32) float2 entries {scale, scale * integer block sum}; n must be a
 * multiple of 128. yuuka_gemm_mmq multiplies those activations by packed
 * Q4_K/Q5_K/Q6_K weights, decoding the weights to int8 and accumulating exact
 * int32 products before the FP32 weight scale is applied. n_in must be a
 * multiple of 256. Both calls submit on stream and record completion without a
 * host synchronization. */
YUUKA_API yuuka_status yuuka_quantize_q8_1(yuuka_stream * stream, yuuka_buffer * qs,
                        yuuka_buffer * ds, const yuuka_buffer * x,
                        uint64_t n, uint64_t batch, yuuka_event * completion);
YUUKA_API yuuka_status yuuka_gemm_mmq(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * qs, const yuuka_buffer * ds,
                        const yuuka_buffer * w, uint64_t n_in, uint64_t n_out,
                        uint64_t batch, uint32_t dtype, yuuka_event * completion);

/* rocBLAS prefill route:
 *
 * out[batch, n_out] = W[n_out, n_in] * x[batch, n_in]
 *
 * x16 must hold batch*n_in FP16 elements and w must hold FP16 weights. out16 is
 * a caller-owned FP16 scratch of n_out*batch elements: rocBLAS accumulates in
 * FP32 and rounds each output once into that scratch, and this call casts it
 * into out, which stays FP32.
 */
YUUKA_API yuuka_status yuuka_gemm_blas(yuuka_stream * stream, yuuka_buffer * out,
                        yuuka_buffer * out16, const yuuka_buffer * x16,
                        const yuuka_buffer * w, uint64_t n_in, uint64_t n_out,
                        uint64_t batch, yuuka_event * completion);

/* Streaming LM head for the prefill tail:
 *
 * out[rows, n_out] = W[n_out, n_in] * x[rows, n_in]
 *
 * x and out are row-major [rows, n_in] and [rows, n_out]. Every packed weight
 * element is read once and reused across all rows, so the call runs at the
 * streaming weight bandwidth; rows is limited to 16, which covers the last-row
 * chunk tail and the speculative verification row block. Every dtype in
 * yuuka_dtype is supported and n_in must be a multiple of 256. This is the
 * prefill form of the output projection only: the decode path keeps its GEMV.
 */
YUUKA_API yuuka_status yuuka_lm_head(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t rows,
                        uint32_t dtype, yuuka_event * completion);

/* Dequantizes a packed quantized weight matrix into an FP16 scratch:
 *
 * dst[n_out, n_in] = dequant(src[n_out, n_in])
 *
 * src is a packed Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, or IQ3_S matrix with dtype
 * row strides; n_in must be a multiple of 256 and dst must hold n_in*n_out*2
 * bytes. The decoded values are the FP16 values the prefill and MoE paths
 * decode, so feeding dst to yuuka_gemm_blas isolates the accumulation from the
 * dequantization.
 */
YUUKA_API yuuka_status yuuka_dequant_f16(yuuka_stream * stream, yuuka_buffer * dst,
                        const yuuka_buffer * src, uint64_t n_in, uint64_t n_out,
                        uint32_t dtype, yuuka_event * completion);

/* Converts n contiguous FP32 elements to FP16. The destination must hold
 * n*2 bytes. Used to prepare rocBLAS baseline inputs. */
YUUKA_API yuuka_status yuuka_cast_f32_to_f16(yuuka_stream * stream, yuuka_buffer * dst,
                        const yuuka_buffer * src, uint64_t n, yuuka_event * completion);

/* Batched quantized/unquantized GEMV:
 *
 * out[batch, n_out] = W[n_out, n_in] * x[batch, n_in]
 *
 * Unlike gemm, this path supports every decode dtype and reuses each packed
 * weight element across the batch. batch is limited to 16.
 */
YUUKA_API yuuka_status yuuka_gemv_batch(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        uint32_t dtype, yuuka_event * completion);

/* Plans batch generic device operations into one submission.
 *
 * GEMV, GEMM, GEMM_SWIGLU, GEMV_BATCH, LM_HEAD, the legacy single-token family (RMSNORM, L2NORM,
 * GATED_RMSNORM, RMSNORM_HEADS, ROPE, ATTENTION, CONV1D, GDN, SILU_MUL, ADD,
 * COPY, SCALE, and KV_WRITE), the fused delta-net pair (CONV_L2NORM and
 * RECURRENT_TOKEN), the fused decode matvec pair (GEMV_RESIDUAL and
 * GEMV_PAIR), and the chunked prefill family are validated when
 * added: RMSNORM_BATCH, RMSNORM_HEADS_BATCH, ROPE_BATCH, KV_WRITE_BATCH,
 * ATTENTION_PREFILL, RECURRENT_CHUNK, EMBEDDING_BATCH, CAST_F16,
 * DEQUANT_F16, QUANT_Q8_1, and SILU_MUL_QUANT_Q8_1. Unsupported
 * dtypes and widths that violate the quantization block layout fail with
 * YUUKA_ERR_UNSUPPORTED; invalid dimensions, batch sizes, and undersized
 * buffers fail with YUUKA_ERR_INVALID_ARGUMENT. A rejected operation is not
 * appended, so execution cannot silently skip it.
 *
 * Dynamic positions: ROPE_BATCH, KV_WRITE_BATCH, ATTENTION_PREFILL, and the
 * legacy single-token operators consume the position passed to
 * yuuka_plan_execute. Prefill plans pass the chunk start. ROPE_BATCH and
 * ATTENTION_PREFILL_PAGED add their descriptor's token displacement.
 * Attention and KV write batches reject ranges beyond their capacity.
 *
 * Buffer layouts for the prefill family:
 *   RMSNORM_BATCH:       buf0 src, buf1 weight, buf2 dst; n={cols,rows,
 *                        src_stride,dst_stride}
 *   RMSNORM_HEADS_BATCH: buf0 src, buf1 weight, buf2 dst; n={size,heads,
 *                        stride,rows}; rows are contiguous [rows,heads,stride]
 *   ROPE_BATCH:          buf0 data; n={heads,stride,head_dim,rot_dim,rows,
 *                        row_stride}; u[0]=token displacement
 *   KV_WRITE_BATCH:      buf0 k, buf1 v, buf2 k_cache, buf3 v_cache;
 *                        n={kv_dim,rows}; u[0]=capacity
 *   ATTENTION_PREFILL:   buf0 q, buf1 k_cache, buf2 v_cache, buf3 out;
 *                        n={kv_heads,heads,head_dim,kv_dim,rows}; u[0]=capacity;
 *                        u[3] must be zero (gated query layout only)
 *   RECURRENT_CHUNK:     buf0 conv_state, buf1 matrix_state, buf2 qkv, buf3 z,
 *                        buf4 alpha, buf5 beta, buf6 conv weights, buf7 dt,
 *                        buf8 a, buf9 norm, buf10 out, buf11 conv scratch;
 *                        optional buf12 triangular scratch or buf13 prepared
 *                        FP32 Q[rows,groups,128], K[rows,groups,128],
 *                        decay[rows,heads], beta[rows,heads] (state=128);
 *                        n={rows,groups,heads,state,conv_dim,inner};
 *                        u[0]=conv kernel; f[0]=eps; f[1]=q scale
 *   EMBEDDING_BATCH:     buf0 weight table, buf1 out, buf2 int32 tokens;
 *                        n={cols,rows,row_bytes,vocab}; u[0]=dtype
 *   CAST_F16:            buf0 FP32 src, buf1 FP16 dst; n[0]=elements
 *   DEQUANT_F16:         buf0 packed weights, buf1 FP16 dst;
 *                        n[0]=n_in, n[1]=n_out, u[0]=dtype
 *   QUANT_Q8_1:          buf0 FP32 src, buf1 int8 dst, buf2 float2 block
 *                        scales; n={n, batch}; n must be a multiple of 128
 *   SILU_MUL_QUANT_Q8_1: buf0 FP32 gate, buf1 FP32 up, buf2 int8 dst,
 *                        buf3 float2 block scales; n={width,rows}; offsets
 *                        must be aligned to 16, 16, 4 and 8 bytes
 *   GEMM with path MMQ:  buf0 packed weights, buf1 Q8_1 int8 activations,
 *                        buf2 FP32 out, buf3 float2 block scales;
 *                        n={n_in,n_out,rows}; u[0]=Q4_K/Q5_K/Q6_K;
 *                        n_in must be a multiple of 256
 *   GEMM_SWIGLU:         buf0 gate weights, buf1 up weights, buf2 FP32 x,
 *                        buf3 out; n={n_in,n_out,rows}; u[0]=dtype. Both
 *                        weights share dtype and shape; out holds
 *                        silu(W_gate*x) * (W_up*x) in FP32.
 *   LM_HEAD:             buf0 packed weights, buf1 FP32 x, buf2 FP32 out;
 *                        n={n_in,n_out,rows}; u[0]=dtype; n_in must be a
 *                        multiple of 256 and rows at most 16
 *   RECURRENT_TOKEN:     buf0 state, buf1 q, buf2 k, buf3 v, buf4 alpha,
 *                        buf5 beta, buf6 dt, buf7 a, buf8 out, buf9 norm
 *                        weight, buf10 gate; off[1..3] select q, k, and v;
 *                        n={groups,heads,state}; f[0]=eps. state is
 *                        [heads,state,state] FP32 in the chunked prefill
 *                        layout, q and k are [groups,state], v, out, and gate
 *                        are [heads,state], and out receives the gated RMSNorm
 *                        of the head rather than the raw delta-net result.
 *                        state must be 128, and the operation fails with
 *                        YUUKA_ERR_UNSUPPORTED for any other width.
 *   GEMV_RESIDUAL:       buf0 packed weights, buf1 FP32 x, buf2 FP32 out,
 *                        buf3 FP32 residual; n={n_in,n_out,rows}; u[0]=dtype.
 *                        Same shape rules as LM_HEAD; out = W x + residual, and
 *                        out may alias the residual row.
 *   GEMV_PAIR:           buf0 weights A, buf1 weights B, buf2 FP32 x,
 *                        buf3 FP32 out A, buf4 FP32 out B;
 *                        n={n_in,n_out_a,n_out_b}; u[0]=dtype (both matrices),
 *                        u[2]=epilogue. Both matrices are walked over the same
 *                        activation row in one launch, which is the decode
 *                        counterpart of GEMM_SWIGLU and the projection merge of
 *                        the decode plan. Epilogue 0 writes both rows;
 *                        epilogue 1 requires n_out_a == n_out_b and writes
 *                        silu(A x) * (B x) into out A with buf4 unused. u[1] must
 *                        equal u[0]: a mixed-dtype pair is rejected, because a
 *                        run-time dispatch inside the span loop is measured 60%
 *                        slower than the templated decode.
 *   CONV_L2NORM:         buf0 conv state, buf1 conv weights, buf2 qkv, buf3 out;
 *                        n={channels,kernel,qk_channels,q_channels,state};
 *                        f[0]=eps; f[1]=q scale. out receives the SiLU of the
 *                        convolution, and its first qk_channels channels are
 *                        then L2 normalized per group of state values, with
 *                        q_scale applied to the first q_channels. channels and
 *                        qk_channels must be multiples of 256 and state must be
 *                        128; other shapes fail with YUUKA_ERR_UNSUPPORTED.
 *
 * Paged family. The block table is an int32 buffer with ceil(capacity/block_size)
 * entries mapping logical token positions to pool blocks. The pool holds
 * ceil(capacity/block_size)*block_size rows of kv_dim values in u[2] format
 * (F32, F16, or Q8_0). u[2] defaults to zero (F32). Q8_0 rows use one FP16
 * scale and 32 int8 values per 32-column group, with a padded trailing group.
 *   KV_WRITE_PAGED:         buf0 k, buf1 v, buf2 k_pool, buf3 v_pool,
 *                           buf4 block table; n[0]=kv_dim; u[0]=block_size;
 *                           u[1]=capacity; u[2]=pool dtype
 *   KV_WRITE_BATCH_PAGED:   same buffers; n={kv_dim,rows};
 *                           u[0]=block_size; u[1]=capacity; u[2]=pool dtype
 *   ATTENTION_PAGED:        buf0 q, buf1 k_pool, buf2 v_pool, buf3 out,
 *                           buf4 scores, buf5 block table;
 *                           n={kv_heads,heads,head_dim,kv_dim,score_stride};
 *                           u[0]=block_size; u[1]=capacity; u[2]=pool dtype;
 *                           u[3]=1 selects the dense query layout (no output
 *                           gate), u[3]=0 selects the gated Qwen3.5 layout
 *   ATTENTION_PREFILL_PAGED: buf0 q, buf1 k_pool, buf2 v_pool, buf3 out,
 *                           buf4 block table; flags bit 1 enables experimental
 *                           BLAS F16 KV path with buf5 FP16 scratch;
 *                           optional buf6 holds split-key FP32 scratch
 *                           with four [rows,heads,head_dim+2] records and
 *                           one [rows,heads] prefix maximum array;
 *                           n={kv_heads,heads,head_dim,kv_dim,rows,displacement};
 *                           first query position is execute position plus
 *                           displacement tokens;
 *                           u[0]=block_size; u[1]=capacity; u[2]=pool dtype;
 *                           u[3]=1 selects the dense query layout (no output
 *                           gate), u[3]=0 selects the gated Qwen3.5 layout
 *
 * MoE family. ids and weights are [rows, used]. counts is [experts] and each of
 * the three assignment buffers is [experts, rows*used]. The grouped kernels bin
 * assignments by expert at runtime and are the prefill form; the selected
 * kernels launch one grid over the rows*used routing slots and are the decode
 * form, which must not schedule the grouped experts-by-FFN grid for one token.
 * Expert tensors accept the eight packed dtypes of the general matvec families
 * plus IQ2_XXS, IQ2_S, and IQ3_S, whose 256-value blocks the MoE kernels decode
 * on the device.
 *   MOE_ROUTE:      buf0 logits [rows,experts], buf1 ids, buf2 weights,
 *                   buf3 counts, buf4 assign_rows, buf5 assign_slots,
 *                   buf6 assign_weights; n={rows,experts,used}
 *   MOE_GATE_UP:    buf0 norm, buf1 gate_exps, buf2 up_exps, buf3 counts,
 *                   buf4 assign_rows, buf5 assign_slots, buf6 gate_out,
 *                   buf7 up_out; n={hidden,ffn,rows,experts,used}; u[0]=dtype
 *   MOE_DOWN:       buf0 act, buf1 down_exps, buf2 counts, buf3 assign_rows,
 *                   buf4 assign_slots, buf5 out; n={ffn,hidden,rows,experts,used};
 *                   u[0]=dtype
 *   MOE_REDUCE_ADD: buf0 x, buf1 expert_out, buf2 weights;
 *                   n={hidden,rows,used}
 *   MOE_GATE_UP_SELECTED: buf0 norm [rows,hidden], buf1 ids int32[rows*used],
 *                   buf2 gate_exps, buf3 up_exps, buf4 act [rows*used,ffn]
 *                   holding silu(gate)*up; n={hidden,ffn,rows,used,experts};
 *                   u[0] is the single dtype of both expert matrices
 *   MOE_DOWN_SELECTED:    buf0 act [rows*used,ffn], buf1 ids int32[rows*used],
 *                   buf2 down_exps, buf3 out [rows*used,hidden];
 *                   n={ffn,hidden,rows,used,experts}; u[0]=dtype
 *   MOE_SHARED_ADD: buf0 x [rows,hidden], buf1 src [rows,hidden], buf2 gate
 *                   weight [hidden] F32, buf3 norm [rows,hidden];
 *                   n={hidden,rows}. Computes s = sigmoid(dot(gate, norm[row]))
 *                   and x[row,i] += s * src[row,i].
 *
 * Q8_1 expert family. These are the grouped kernels over an int8 activation
 * tile: QUANT_Q8_1 produces norm_qs/norm_ds from the normalized rows, and
 * SILU_MUL_QUANT_Q8_1 produces act_qs/act_ds from the gate and up outputs. Only
 * IQ2_XXS, IQ2_S, and IQ3_S weights are accepted, hidden and ffn must be
 * multiples of 256, and the K width of each op must be a multiple of 256. The
 * grouped assignment inputs are the MOE_ROUTE outputs.
 *   MOE_GATE_UP_Q8: buf0 norm_qs int8[rows,hidden], buf1 norm_ds
 *                   float2[rows,hidden/32], buf2 gate_exps, buf3 up_exps,
 *                   buf4 counts, buf5 assign_rows, buf6 assign_slots,
 *                   buf7 gate_out [rows*used,ffn], buf8 up_out [rows*used,ffn];
 *                   n={hidden,ffn,rows,experts,used}; u[0]=dtype
 *   MOE_DOWN_Q8: buf0 act_qs int8[rows*used,ffn], buf1 act_ds
 *                float2[rows*used,ffn/32], buf2 down_exps, buf3 counts,
 *                buf4 assign_rows, buf5 assign_slots, buf6 out [rows*used,hidden];
 *                n={ffn,hidden,rows,experts,used}; u[0]=dtype
 *
 * Int8 WMMA expert family. These are the grouped kernels over the same Q8_1
 * activation tile, with the assignment bucket walked in tiles of n[5] rows and
 * the int8 weights dotted by 16x16x16 WMMA instead of dp4a. They accept the
 * K-quant expert dtypes (Q4_K, Q5_K, Q6_K) and the IQ2_XXS, IQ2_S, and IQ3_S
 * codebooks, and require the K width to be a whole number of 256-value blocks
 * and the output width to be a multiple of 128, which covers the K step and the
 * 16-128 channel tile. The assignment inputs are the MOE_ROUTE outputs; n[5] is
 * the assignment tile width, one of 16, 32, 64, 128, and it is a plan-build-time
 * dispatch decision, not a runtime option.
 *   MOE_GATE_UP_MMQ: buf0 norm_qs int8[rows,hidden], buf1 norm_ds
 *                    float2[rows,hidden/32], buf2 gate_exps, buf3 up_exps,
 *                    buf4 counts, buf5 assign_rows, buf6 assign_slots,
 *                    buf7 gate_out [rows*used,ffn], buf8 up_out [rows*used,ffn];
 *                    n={hidden,ffn,rows,experts,used,tile_j}; u[0]=dtype
 *   MOE_DOWN_MMQ: buf0 act_qs int8[rows*used,ffn], buf1 act_ds
 *                 float2[rows*used,ffn/32], buf2 down_exps, buf3 counts,
 *                 buf4 assign_rows, buf5 assign_slots, buf6 out [rows*used,hidden];
 *                 n={ffn,hidden,rows,experts,used,tile_j}; u[0]=dtype
 *
 * Graph capture executes a plan with device-resident dynamic parameters.
 * Position-dependent ops must have a dynamic-parameter kernel; plans containing
 * any other position-dependent op are rejected with YUUKA_ERR_UNSUPPORTED and
 * the caller falls back to normal submission. Ops whose arguments do not depend
 * on the token or position are capturable as-is. */
YUUKA_API yuuka_status yuuka_graph_create(yuuka_plan * plan, yuuka_stream * stream,
                                          yuuka_buffer * dynamic, int run_tail,
                                          yuuka_graph ** out);
YUUKA_API yuuka_status yuuka_graph_replay(yuuka_graph * graph, yuuka_stream * stream,
                                          yuuka_event * completion);
YUUKA_API void yuuka_graph_destroy(yuuka_graph * graph);
YUUKA_API yuuka_status yuuka_plan_create(yuuka_device * dev, yuuka_plan ** out);
YUUKA_API void yuuka_plan_destroy(yuuka_plan * plan);
YUUKA_API yuuka_status yuuka_plan_add(yuuka_plan * plan, const yuuka_op * op);
YUUKA_API yuuka_status yuuka_plan_execute(yuuka_plan * plan, yuuka_stream * stream,
                                          int32_t token, uint32_t position,
                                          int run_tail, yuuka_event * completion);

/* Diagnostic execution path. Records HIP events around every submitted
 * operation, synchronizes once, and returns per-op codes and elapsed
 * nanoseconds. It is intended for offline attribution, not the hot path. */
YUUKA_API yuuka_status yuuka_plan_execute_profiled(
    yuuka_plan * plan, yuuka_stream * stream, int32_t token, uint32_t position,
    int run_tail, uint32_t * op_codes, uint64_t * op_times_ns,
    uint64_t capacity, uint64_t * op_count, yuuka_event * completion);

#ifdef __cplusplus
}
#endif

#endif

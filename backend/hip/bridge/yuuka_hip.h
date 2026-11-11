#ifndef YUUKA_HIP_H
#define YUUKA_HIP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YUUKA_HIP_ABI_VERSION 2

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
YUUKA_API yuuka_status yuuka_memset(yuuka_stream * stream, yuuka_buffer * dst, int value, uint64_t bytes);

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
    YUUKA_OP_CAST_F16 = 25
} yuuka_op_code;

#define YUUKA_OP_FLAG_TAIL 0x1u

/* GEMM implementation selection carried in yuuka_op.u[1]. AUTO uses the
 * compiled default for the dtype. BLAS is the rocBLAS FP16 baseline and
 * requires x to be an FP16 buffer. WMMA is the custom rocWMMA kernel, which
 * accepts FP32 x and every supported weight dtype. */
#define YUUKA_GEMM_PATH_AUTO 0u
#define YUUKA_GEMM_PATH_BLAS 1u
#define YUUKA_GEMM_PATH_WMMA 2u

typedef struct yuuka_op {
    uint32_t code;
    uint32_t flags;
    yuuka_buffer * buf[12];
    uint64_t off[12];
    uint64_t n[6];
    uint32_t u[4];
    float f[4];
} yuuka_op;

typedef struct yuuka_plan yuuka_plan;

typedef enum yuuka_dtype {
    YUUKA_DTYPE_F32 = 0,
    YUUKA_DTYPE_F16 = 1,
    YUUKA_DTYPE_Q8_0 = 8,
    YUUKA_DTYPE_Q4_K = 12,
    YUUKA_DTYPE_Q5_K = 13,
    YUUKA_DTYPE_Q6_K = 14
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
 * w may be any supported dtype, including Q4_K, Q5_K, Q6_K, and Q8_0. This is
 * the prefill path; it never materializes a dequantized weight matrix.
 */
YUUKA_API yuuka_status yuuka_gemm_wmma(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        uint32_t dtype, yuuka_event * completion);

/* rocBLAS FP16 prefill baseline. x must hold batch*n_in FP16 elements. */
YUUKA_API yuuka_status yuuka_gemm_blas(yuuka_stream * stream, yuuka_buffer * out,
                        const yuuka_buffer * x, const yuuka_buffer * w,
                        uint64_t n_in, uint64_t n_out, uint64_t batch,
                        yuuka_event * completion);

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
 * GEMV, GEMM, GEMV_BATCH, and the chunked prefill family are validated when
 * added: RMSNORM_BATCH, RMSNORM_HEADS_BATCH, ROPE_BATCH, KV_WRITE_BATCH,
 * ATTENTION_PREFILL, RECURRENT_CHUNK, EMBEDDING_BATCH, and CAST_F16. Unsupported
 * dtypes and widths that violate the quantization block layout fail with
 * YUUKA_ERR_UNSUPPORTED; invalid dimensions, batch sizes, and undersized
 * buffers fail with YUUKA_ERR_INVALID_ARGUMENT. A rejected operation is not
 * appended, so execution cannot silently skip it.
 *
 * Dynamic positions: ROPE_BATCH, KV_WRITE_BATCH, ATTENTION_PREFILL, and the
 * legacy single-token operators consume the position passed to
 * yuuka_plan_execute. Prefill plans pass the chunk start. ATTENTION_PREFILL and
 * KV_WRITE_BATCH additionally reject position+rows beyond their capacity.
 *
 * Buffer layouts for the prefill family:
 *   RMSNORM_BATCH:       buf0 src, buf1 weight, buf2 dst; n={cols,rows,
 *                        src_stride,dst_stride}
 *   RMSNORM_HEADS_BATCH: buf0 src, buf1 weight, buf2 dst; n={size,heads,
 *                        stride,rows}; rows are contiguous [rows,heads,stride]
 *   ROPE_BATCH:          buf0 data; n={heads,stride,head_dim,rot_dim,rows,
 *                        row_stride}
 *   KV_WRITE_BATCH:      buf0 k, buf1 v, buf2 k_cache, buf3 v_cache;
 *                        n={kv_dim,rows}; u[0]=capacity
 *   ATTENTION_PREFILL:   buf0 q, buf1 k_cache, buf2 v_cache, buf3 out;
 *                        n={kv_heads,heads,head_dim,kv_dim,rows}; u[0]=capacity
 *   RECURRENT_CHUNK:     buf0 conv_state, buf1 matrix_state, buf2 qkv, buf3 z,
 *                        buf4 alpha, buf5 beta, buf6 conv weights, buf7 dt,
 *                        buf8 a, buf9 norm, buf10 out, buf11 conv scratch;
 *                        n={rows,groups,heads,state,conv_dim,inner};
 *                        u[0]=conv kernel; f[0]=eps; f[1]=q scale
 *   EMBEDDING_BATCH:     buf0 weight table, buf1 out, buf2 int32 tokens;
 *                        n={cols,rows,row_bytes,vocab}; u[0]=dtype
 *   CAST_F16:            buf0 FP32 src, buf1 FP16 dst; n[0]=elements
 */
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

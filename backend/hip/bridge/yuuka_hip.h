#ifndef YUUKA_HIP_H
#define YUUKA_HIP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YUUKA_HIP_ABI_VERSION 4

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
    YUUKA_OP_MOE_REDUCE_ADD = 33
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
typedef struct yuuka_graph yuuka_graph;

typedef enum yuuka_dtype {
    YUUKA_DTYPE_F32 = 0,
    YUUKA_DTYPE_F16 = 1,
    YUUKA_DTYPE_Q4_0 = 2,
    YUUKA_DTYPE_Q5_0 = 6,
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
 * GEMV, GEMM, GEMV_BATCH, the legacy single-token family (RMSNORM, L2NORM,
 * GATED_RMSNORM, RMSNORM_HEADS, ROPE, ATTENTION, CONV1D, GDN, SILU_MUL, ADD,
 * COPY, SCALE, and KV_WRITE), and the chunked prefill family are validated when
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
 *                        n={kv_heads,heads,head_dim,kv_dim,rows}; u[0]=capacity;
 *                        u[3] must be zero (gated query layout only)
 *   RECURRENT_CHUNK:     buf0 conv_state, buf1 matrix_state, buf2 qkv, buf3 z,
 *                        buf4 alpha, buf5 beta, buf6 conv weights, buf7 dt,
 *                        buf8 a, buf9 norm, buf10 out, buf11 conv scratch;
 *                        n={rows,groups,heads,state,conv_dim,inner};
 *                        u[0]=conv kernel; f[0]=eps; f[1]=q scale
 *   EMBEDDING_BATCH:     buf0 weight table, buf1 out, buf2 int32 tokens;
 *                        n={cols,rows,row_bytes,vocab}; u[0]=dtype
 *   CAST_F16:            buf0 FP32 src, buf1 FP16 dst; n[0]=elements
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
 *                           buf4 block table;
 *                           n={kv_heads,heads,head_dim,kv_dim,rows};
 *                           u[0]=block_size; u[1]=capacity; u[2]=pool dtype;
 *                           u[3]=1 selects the dense query layout (no output
 *                           gate), u[3]=0 selects the gated Qwen3.5 layout
 *
 * MoE family. ids and weights are [rows, used]. counts is [experts] and each of
 * the three assignment buffers is [experts, rows*used]. The grouped kernels bin
 * assignments by expert at runtime.
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

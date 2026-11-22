#ifndef YUUKA_KERNELS_H
#define YUUKA_KERNELS_H

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace yuuka {

void launch_add_f32(float * dst, const float * a, const float * b, uint64_t n,
                    hipStream_t stream);

void launch_gemv_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q4_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q5_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q8_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q4_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q5_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q6_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);

void launch_gemv_batch_f32(float * out, const float * x, const float * w,
                           uint64_t n_in, uint64_t n_out, uint64_t batch,
                           hipStream_t stream);
void launch_gemv_batch_f16(float * out, const float * x, const void * w,
                           uint64_t n_in, uint64_t n_out, uint64_t batch,
                           hipStream_t stream);
void launch_gemv_batch_q4_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q5_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q8_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q4_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q5_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q6_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);

void launch_gemm_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream);
void launch_gemm_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream);

/* Prefill GEMM family. launch_gemm_wmma_f32out accepts every supported weight
 * dtype and FP32 activations. launch_gemm_blas_f16 is the rocBLAS route: FP16
 * weights, FP16 activations, FP32 compute, and an FP16 output scratch that this
 * call casts to the FP32 result. rocBLAS therefore accumulates in FP32 and
 * rounds only the result. It returns 0 on success or a rocBLAS status. */
void launch_gemm_wmma_f32out(float * out, const float * x, const void * w,
                             uint64_t n_in, uint64_t n_out, uint64_t batch,
                             uint32_t dtype, hipStream_t stream);

/* Fused gate/up WMMA GEMM. Both weights have the same shape and dtype and the
 * kernel computes out = silu(W_gate * x) * (W_up * x) without writing the two
 * GEMM intermediates. */
void launch_gemm_wmma_swiglu_f32out(float * out, const float * x,
                                    const void * w_gate, const void * w_up,
                                    uint64_t n_in, uint64_t n_out, uint64_t batch,
                                    uint32_t dtype, hipStream_t stream);
int launch_gemm_blas_f16(float * out, void * out16, const void * x16, const void * w,
                         uint64_t n_in, uint64_t n_out, uint64_t batch,
                         hipStream_t stream);
void launch_cast_f32_to_f16(void * dst, const float * src, uint64_t n,
                            hipStream_t stream);
void launch_cast_f16_to_f32(void * dst, const void * src, uint64_t n,
                            hipStream_t stream);

/* Int8 MMQ prefill family.
 *
 * launch_quantize_q8_1 converts FP32 activations into Q8_1: qs holds batch*n
 * int8 values in [batch][n] row-major order, ds holds batch*(n/32) float2
 * entries where x is the 32-value block scale and y is the scale times the
 * integer block sum. n must be a multiple of 128.
 *
 * launch_gemm_mmq_f32out computes out[batch, n_out] = W[n_out, n_in] * x with
 * the Q8_1 activations above, int8-decoded Q4_K/Q5_K/Q6_K weights, exact int32
 * WMMA accumulation, and FP32 scale application. n_in must be a multiple of
 * 256, which every K-quant row layout already requires. */
void launch_quantize_q8_1(void * qs, void * ds, const float * x, uint64_t n,
                          uint64_t batch, hipStream_t stream);
void launch_silu_mul_quantize_q8_1(void * qs, void * ds, const float * gate,
                                  const float * up, uint64_t n, uint64_t batch,
                                  hipStream_t stream);
void launch_gemm_mmq_f32out(float * out, const void * qs, const void * ds,
                            const void * w, uint64_t n_in, uint64_t n_out,
                            uint64_t batch, uint32_t dtype, hipStream_t stream);

/* Streaming LM head for the prefill graph.
 *
 * out[rows, n_out] = W[n_out, n_in] * x[rows, n_in], with x and out row major
 * and rows in [1, 16]. Each packed weight row is read exactly once and dotted
 * against every activation row, so the operator runs at the streaming weight
 * bandwidth instead of one GEMV per row. Every dtype in yuuka_dtype is
 * supported. n_in must be a multiple of 256 and rows must be at most 16; an
 * out-of-range shape or an unknown dtype launches nothing. */
// launch_lm_head_residual_f32out is the streaming single-row-block matvec with
// the residual epilogue: out = W x + res, where res may alias out.
// Passing a null res gives the plain form.
void launch_lm_head_residual_f32out(float * out, const float * x, const void * w,
                                    const float * res, uint64_t n_in, uint64_t n_out,
                                    uint64_t rows, uint32_t dtype, hipStream_t stream);
// launch_lm_head_pair_f32out runs two single-row matvecs over one activation row
// in one launch. Both matrices share the packed dtype but may differ in width. A
// runtime dtype dispatch inside the span loop costs 60% of the achieved bandwidth
// on gfx1151, so the pair is templated and the caller must not mix dtypes. With
// swiglu
// the two dots are combined as silu(a) * b into out0 and out1 is unused;
// otherwise both rows are written. It is the decode counterpart of GEMM_SWIGLU
// and the narrow-projection merge of the decode plan.
void launch_lm_head_pair_f32out(float * out0, float * out1, const float * x,
                                const void * w0, const void * w1, uint64_t n_in,
                                uint64_t n_out0, uint64_t n_out1, uint32_t dtype,
                                int swiglu, hipStream_t stream);
void launch_lm_head_f32out(float * out, const float * x, const void * w,
                           uint64_t n_in, uint64_t n_out, uint64_t rows,
                           uint32_t dtype, hipStream_t stream);

/* Int8 decode matvec.
 *
 * out[n_out] = W[n_out, n_in] * x, with x quantized to Q8_1 by
 * launch_quantize_q8_1 (qs holds n_in int8 digits, ds its float2 block scales,
 * which this kernel does not need because it sums the digits per sub-block
 * itself). Q4_K, Q5_K, Q6_K, and Q8_0 weights are decoded straight to int8 and
 * dotted with one v_dot4 per four weights. One row only; n_in must be a multiple
 * of 256. An unknown dtype or a width outside that rule launches nothing, which
 * leaves the caller on the FP32 streaming or per-row route. */
void launch_gemv_i8_f32out(float * out, const void * qs, const void * ds,
                           const void * w, uint64_t n_in, uint64_t n_out,
                           uint32_t dtype, hipStream_t stream);

/* Fused int8 decode matvec: the same product as launch_gemv_i8_f32out, but the
 * activation row is quantized inside the block from the FP32 row instead of
 * reading a separately quantized copy. Q4_K, Q5_K, Q6_K, and Q8_0 weights, one
 * row, n_in a multiple of 256. */
void launch_gemv_i8_fused_f32out(float * out, const float * x, const void * w,
                                 uint64_t n_in, uint64_t n_out, uint32_t dtype,
                                 hipStream_t stream);

/* Dequantizes a packed K-quant weight matrix into FP16.
 *
 * dst holds n_in*n_out FP16 elements. src holds the packed [n_out, n_in] matrix
 * with the dtype's row stride. The decoded values are bit-identical to the
 * values the rocWMMA path stages for the same tile, so only the accumulation
 * differs. Only Q4_K, Q5_K, and Q6_K are supported; an unknown dtype writes
 * nothing. */
void launch_dequant_f16(void * dst, const void * w, uint64_t n_in, uint64_t n_out,
                        uint32_t dtype, hipStream_t stream);

void launch_rmsnorm_f32(float * dst, const float * src, const float * weight,
                        int64_t n, float eps, hipStream_t stream);
void launch_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                              int heads, int64_t n, int64_t stride, float eps,
                              hipStream_t stream);
void launch_l2norm_heads_f32(float * dst, const float * src, int heads, int64_t n,
                             int64_t stride, float eps, float extra_scale,
                             hipStream_t stream);
void launch_gated_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                                    const float * gate, int heads, int64_t n,
                                    int64_t stride, float eps, hipStream_t stream);
void launch_silu_mul_f32(float * out, const float * gate, const float * up,
                         int64_t n, hipStream_t stream);
void launch_add_f32_buffers(float * x, const float * delta, int64_t n, hipStream_t stream);
void launch_copy_f32(float * dst, const float * src, int64_t n, hipStream_t stream);
void launch_scale_f32(float * x, int64_t n, float scale, hipStream_t stream);
void launch_rope_f32(float * data, int heads, int stride, int head_dim, int rot_dim,
                     int pos, float base, hipStream_t stream);
void launch_rope_dynamic_f32(float * data, int heads, int stride, int head_dim,
                             int rot_dim, const int32_t * dynamic, float base,
                             hipStream_t stream);
void launch_attention_f32(float * out, const float * q, const float * k_cache,
                          const float * v_cache, float * scores,
                          int kv_heads, int heads, int head_dim, int kv_dim,
                          int position, int capacity, float scale, hipStream_t stream);
/* launch_attention_paged_f32 runs single-token paged attention. The tiled path
 * (one block per KV head and key tile, split-key online softmax) writes its
 * partials into scores, which must hold
 * heads * ceil(score_stride / tile) * (head_dim + 2) floats for the tile the
 * launcher chooses; the serial fallback needs heads * score_stride floats. */
void launch_attention_paged_f32(float * out, const float * q, const void * k_pool,
                                const void * v_pool, float * scores,
                                const int32_t * block_table, int block_size,
                                int kv_heads, int heads, int head_dim, int kv_dim,
                                int position, int score_stride, float scale,
                                uint32_t dtype, int dense, hipStream_t stream);
void launch_attention_paged_dynamic_f32(float * out, const float * q,
                                        const void * k_pool, const void * v_pool,
                                        float * scores, const int32_t * block_table,
                                        const int32_t * dynamic, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int score_stride, float scale,
                                        uint32_t dtype, int dense, hipStream_t stream);
void launch_conv1d_f32(float * state, const float * weights, const float * qkv,
                       float * out, int channels, int kernel, hipStream_t stream);
void launch_gdn_f32(float * state, const float * q, const float * k, const float * v,
                    const float * alpha, const float * beta, const float * dt,
                    const float * a, float * out, int groups, int heads,
                    int state_size, hipStream_t stream);
// launch_recurrent_token_f32 runs one token of a gated delta-net layer with the
// head's state held in registers: decay, both S^T products, and the rank-1
// update are one pass, and the gated RMSNorm of the head is folded in, so out
// receives the normalized head row rather than the raw delta-net result. The
// state_width is fixed at 128 (kernels/hip/ssm/deltanet.hip).
void launch_recurrent_token_f32(float * state, const float * q, const float * k,
                                const float * v, const float * alpha, const float * beta,
                                const float * dt, const float * a, float * out,
                                const float * norm_weight, const float * gate,
                                int groups, int heads, float eps, hipStream_t stream);
// launch_conv_l2norm_f32 applies the short convolution to every channel and then
// the L2 norm to the first qk_channels channels, q_channels of which use
// q_scale and the rest unity. channels and qk_channels must be multiples of 256
// and the norm group width is fixed at 128.
void launch_conv_l2norm_f32(float * conv_state, const float * weights, const float * qkv,
                            float * out, int channels, int kernel, int qk_channels,
                            int q_channels, float eps, float q_scale, hipStream_t stream);
void launch_embedding_f32(float * out, const void * weights, int32_t token,
                          uint64_t row_bytes, int64_t cols, uint32_t dtype,
                          hipStream_t stream);
void launch_embedding_dynamic_f32(float * out, const void * weights,
                                  const int32_t * dynamic, uint64_t row_bytes,
                                  int64_t cols, uint32_t dtype, hipStream_t stream);

/* Remapped decode matvec: one wave per output row, sixteen K values per lane,
 * no shared-memory staging and no barrier inside K. gemv_row_available reports
 * whether the shape is covered (a 512-value K step and a supported dtype);
 * the launchers are only valid when it returns true and rows is one. */
bool gemv_row_available(uint32_t dtype, uint64_t n_in);
void launch_gemv_row_f32out(float * out, const float * x, const void * w,
                            const float * res, uint64_t n_in, uint64_t n_out,
                            uint32_t dtype, hipStream_t stream);
void launch_gemv_row_pair_f32out(float * out0, float * out1, const float * x,
                                 const void * w0, const void * w1, uint64_t n_in,
                                 uint64_t n_out0, uint64_t n_out1, uint32_t dtype,
                                 int swiglu, hipStream_t stream);
/* The same remapped mapping over a Q8_1 activation row: qs holds one int8 digit
 * per activation value and ds one float2 block scale per 32 values. Only the
 * dtypes whose per-value form is an integer plus an affine block scale are
 * covered. */
void launch_gemv_row_i8_f32out(float * out, const void * qs, const void * ds,
                               const void * w, uint64_t n_in, uint64_t n_out,
                               uint32_t dtype, hipStream_t stream);

/* Chunked prefill operators. Every row-pointer argument is a contiguous
 * [rows, width] buffer in row-major order. position is the token position of
 * row zero. */
void launch_rmsnorm_batch_f32(float * dst, const float * src, const float * weight,
                              int rows, int cols, int src_stride, int dst_stride,
                              float eps, hipStream_t stream);
void launch_rmsnorm_heads_batch_f32(float * dst, const float * src,
                                    const float * weight, int rows, int heads,
                                    int size, int stride, float eps,
                                    hipStream_t stream);
void launch_rope_batch_f32(float * data, int rows, int heads, int stride,
                           int head_dim, int rot_dim, int pos, float base,
                           int row_stride, hipStream_t stream);
void launch_kv_write_batch_f32(float * k_cache, float * v_cache, const float * k,
                               const float * v, int rows, int kv_dim, int position,
                               hipStream_t stream);
void launch_kv_write_paged_f32(void * k_pool, void * v_pool, const float * k,
                               const float * v, int rows, int kv_dim,
                               const int32_t * block_table, int block_size,
                               int position, uint32_t dtype, hipStream_t stream);
void launch_kv_write_paged_dynamic_f32(void * k_pool, void * v_pool,
                                       const float * k, const float * v,
                                       int rows, int kv_dim,
                                       const int32_t * block_table,
                                       const int32_t * dynamic, int block_size,
                                       uint32_t dtype, hipStream_t stream);
void launch_embedding_batch_f32(float * out, const void * weights,
                                const int32_t * tokens, int rows, uint64_t row_bytes,
                                int64_t cols, uint32_t dtype, hipStream_t stream);
void launch_attention_prefill_f32(float * out, const float * q,
                                  const float * k_cache, const float * v_cache,
                                  int kv_heads, int heads, int head_dim, int kv_dim,
                                  int rows, int position, int capacity, float scale,
                                  hipStream_t stream);
// attention_wmma_shared_bytes reports the dynamic shared memory the FP16 MMA
// prefill attention kernel needs for a head width.
size_t attention_wmma_shared_bytes(int head_dim);

// attention_wmma_preferred reports whether the FP16 MMA prefill attention kernel
// beats the tiled kernel at a chunk length and context position.
bool attention_wmma_preferred(int rows, int position);

// attention_wmma_supported reports whether the FP16 MMA prefill attention kernel
// covers a shape and KV format.
bool attention_wmma_supported(int heads, int kv_heads, int head_dim,
                              int block_size, uint32_t dtype);

void launch_attention_prefill_paged_wmma_f32(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int kv_heads, int heads,
    int head_dim, int kv_dim, int rows, int position, float scale,
    uint32_t dtype, int dense, hipStream_t stream);

hipError_t launch_attention_prefill_paged_f32(float * out, const float * q,
                                        const void * k_pool, const void * v_pool,
                                        const int32_t * block_table, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int rows, int position,
                                        float scale, uint32_t dtype, int dense,
                                        bool prefer_wmma, float * partial, int forced_splits,
                                        hipStream_t stream);

hipError_t launch_attention_prefill_paged_split_f16(float * out, const float * q,
        const void * k_pool, const void * v_pool, const int32_t * block_table,
        int block_size, int kv_heads, int heads, int kv_dim, int rows,
        int position, float scale, float * partial, int splits, hipStream_t stream);
// Experimental GEMM-based attention with gathered FP16 KV, batched QK/PV
// matrix multiplication, causal softmax, and gated final FP32 output.
int launch_attention_prefill_paged_blas_f16(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * table, int block_size, int kvheads, int heads, int head_dim,
    int rows, int position, float scale, void * workspace, hipStream_t stream);

/* MoE routing and grouped expert kernels. ids/weights are [rows, used], counts
 * and assignment buckets are [experts], and each bucket holds up to rows*used
 * assignments. */
int launch_moe_route(const float * logits, int rows, int experts, int used,
                     int bucket_capacity, int32_t * ids, float * weights,
                     int32_t * counts, int32_t * assign_rows,
                     int32_t * assign_slots, float * assign_weights,
                     hipStream_t stream);
void launch_moe_gate_up_grouped(const float * norm, const void * gate_exps,
                                const void * up_exps, const int32_t * counts,
                                const int32_t * assign_rows,
                                const int32_t * assign_slots, float * gate_out,
                                float * up_out, int rows, int experts, int used,
                                int hidden, int ffn, uint64_t hidden_row_bytes,
                                uint32_t dtype, hipStream_t stream);
void launch_moe_down_grouped(const float * act, const void * down_exps,
                             const int32_t * counts, const int32_t * assign_rows,
                             const int32_t * assign_slots, float * out, int rows,
                             int experts, int used, int hidden, int ffn,
                             uint64_t ffn_row_bytes, uint32_t dtype,
                             hipStream_t stream);
void launch_moe_reduce_add(float * x, const float * expert_out,
                           const float * weights, int rows, int used, int hidden,
                           hipStream_t stream);

/* Q8_1 expert kernels. The activation tile is quantized to int8 with one FP32
 * scale per 32 values in norm_qs/norm_ds (gate and up) and act_qs/act_ds (down);
 * the packed IQ weight digits are decoded to int8 and dotted with dp4a. Both
 * require the K width and the output width to be multiples of 256 and serve only
 * IQ2_XXS, IQ2_S, and IQ3_S. */
void launch_moe_gate_up_grouped_q8(const int8_t * norm_qs,
                                   const float2 * norm_ds, const void * gate_exps,
                                   const void * up_exps, const int32_t * counts,
                                   const int32_t * assign_rows,
                                   const int32_t * assign_slots, float * gate_out,
                                   float * up_out, int rows, int experts, int used,
                                   int hidden, int ffn, uint64_t hidden_row_bytes,
                                   uint32_t dtype, hipStream_t stream);
void launch_moe_down_grouped_q8(const int8_t * act_qs, const float2 * act_ds,
                                const void * down_exps, const int32_t * counts,
                                const int32_t * assign_rows,
                                const int32_t * assign_slots, float * out, int rows,
                                int experts, int used, int hidden, int ffn,
                                uint64_t ffn_row_bytes, uint32_t dtype,
                                hipStream_t stream);

/* Int8 WMMA (MMQ) expert kernels. Same Q8_1 activation tile and assignment
 * bucket as the dp4a family, but the expert rows are dotted by 16x16x16 int8
 * WMMA over an assignment tile of tile_j rows (16, 32, 64, or 128), so one
 * decoded weight tile serves the whole assignment tile. They serve the K-quant
 * expert dtypes and require hidden/ffn to be multiples of 128. Both return
 * YUUKA_OK or an error; an unsupported dtype/tile combination is an error, not
 * a silent fallback. */
int launch_moe_gate_up_mmq(const int8_t * norm_qs, const float2 * norm_ds,
                           const void * gate_exps, const void * up_exps,
                           const int32_t * counts, const int32_t * assign_rows,
                           const int32_t * assign_slots, float * gate_out,
                           float * up_out, int rows, int experts, int used,
                           int hidden, int ffn, uint64_t hidden_row_bytes,
                           uint32_t dtype, int tile_j, hipStream_t stream);
int launch_moe_down_mmq(const int8_t * act_qs, const float2 * act_ds,
                        const void * down_exps, const int32_t * counts,
                        const int32_t * assign_rows,
                        const int32_t * assign_slots, float * out, int rows,
                        int experts, int used, int hidden, int ffn,
                        uint64_t ffn_row_bytes, uint32_t dtype, int tile_j,
                        hipStream_t stream);

/* Selected-expert MoE kernels. The decode path reads the routing ids and
 * launches one grid over the rows*used selected slots, so an expert row is only
 * scheduled for a token that selected it. */
void launch_moe_gate_up_selected(const float * norm, const int32_t * ids,
                                 const void * gate_exps, const void * up_exps,
                                 float * act, int rows, int used, int hidden,
                                 int ffn, uint64_t hidden_row_bytes,
                                 uint32_t dtype, hipStream_t stream);
void launch_moe_down_selected(const float * act, const int32_t * ids,
                              const void * down_exps, float * out, int rows,
                              int used, int hidden, int ffn,
                              uint64_t ffn_row_bytes, uint32_t dtype,
                              hipStream_t stream);
void launch_moe_shared_add(float * x, const float * src, const float * gate,
                           const float * norm, int rows, int hidden,
                           hipStream_t stream);

// launch_gdn_chunked_f32 runs the chunkwise gated delta rule, normalization,
// and gate activations of one recurrent layer. scratch holds two 32x32
// matrices per 32-token chunk and time-step head.
void launch_gdn_chunked_f32(float * rec, float * conv, float * alpha,
                            float * beta, const float * dt, const float * a,
                            float * out, float * scratch, int rows, int groups,
                            int heads, int state, int conv_dim, int inner,
                            float eps, float q_scale, hipStream_t stream);

// launch_recurrent_chunk_f32 runs the convolution, the gated delta-net
// recurrence, and the gated RMSNorm of one recurrent layer chunk. alpha and
// beta are overwritten only on the triangular path selected by scratch.
// A separate prepared span can hold normalized Q/K and activated gates for
// the register path. State sizes other than 128 use the generic kernel.
void launch_recurrent_chunk_f32(float * conv_state, float * rec_state,
                                const float * qkv, const float * z,
                                float * alpha, float * beta,
                                const float * conv_weights, const float * dt,
                                const float * a, const float * norm, float * conv,
                                float * out, float * scratch, float * prepared, int rows,
                                int groups, int heads, int state_size,
                                int conv_dim, int inner, int conv_kernel,
                                float eps, float q_scale, hipStream_t stream);

} // namespace yuuka

#endif

# GPU Kernels

## 1. Scope

This document is the contract for device-side execution on gfx1151: the prefill
and decode operator families, the supported weight and KV dtypes, the rules that
select between them, the fused operators, and the numerical requirements.

Related contracts: [architecture.md](architecture.md) owns the Go/HIP boundary,
buffer ownership, and plan submission; [gguf.md](gguf.md) owns the packed row
layouts; [serving.md](serving.md) owns scheduling, KV index construction, and the
API.

Measured evidence — per-shape timings, resource figures, and rejected
alternatives — is recorded in [Qwen3.5-4B](benchmarks/qwen3.5_4b.md) and
[Qwen3.6-35B-A3B](benchmarks/qwen3.6_35b_a3b.md). This document states what the
runtime does, not how a kernel is written.

Plan operations are the versioned C ABI declared in
`backend/hip/bridge/yuuka_hip.h`. That header owns the opcode numbers;
renumbering or removing an opcode requires an ABI version bump.

Implementation status: every family in section 4 is implemented on gfx1151 and
covered by the native tests in `tests/hip`, with the Go-side gates in
[testing.md](testing.md).

## 2. Design rules

1. **Prefill and decode are separate operator families.** They share layouts and
   helper primitives but not a single GEMM: decode never routes through a
   large-`M` tile path by default, and prefill never routes through one block per
   output row.
2. **Quantized weights stay packed.** Dequantization is fused into the consuming
   operator, and no stage materializes a dequantized copy of a model matrix. The
   rocBLAS route is the one exception by construction: it reads weights that are
   already stored as `F16`, which is why it serves only `F16` weights and never
   substitutes for a quantized matrix.
3. **Accumulate in FP32.** FP16 and int8 operand products accumulate in FP32,
   and quantized scales and minimums are applied before the final rounding, so no
   output is double-rounded by the unpack step.
4. **Selection happens at plan build time.** Go selectors and the native
   launchers decide the route from dtype, shape, and the KV format; there is no
   runtime autotuner and no per-tensor dispatch inside a token step.
5. **Unsupported work fails explicitly.** A dtype or shape with no kernel is a
   plan-construction error. Silent numerical substitution, silent truncation,
   and silent fallback are forbidden.
6. **Operators are deterministic.** A repeated submission of the same plan
   produces bitwise identical outputs; the native tests assert this for the GEMV,
   LM-head, MoE, attention, and gated delta-net families.
7. **Thresholds are gfx1151 measurements.** Occupancy, bandwidth, and tuning
   facts from other RDNA parts or from CDNA are not accepted as evidence. A new
   device requires re-running the sweeps and updating the selectors and tests.
8. **One request per plan.** Kernel arguments are fixed at plan construction, and
   token and position for a captured graph are read from a device parameter
   buffer, so replay never rewrites arguments.

## 3. Supported dtypes

Weight dtypes are declared in `tensor.DType` and match the GGUF `ggml_type`
values. "Streaming" is the decode matvec family of section 4.

| DType | Decode matvec | Prefill matmul | MoE experts | KV pool | Embedding |
| --- | --- | --- | --- | --- | --- |
| `F32` | per-row, streaming | rocWMMA, batched matvec | grouped | yes | yes |
| `F16` | per-row, streaming | rocWMMA, Direct FP16, rocBLAS, batched matvec | grouped | yes (default) | yes |
| `Q4_0`, `Q5_0` | per-row, streaming | rocWMMA, batched matvec | grouped | no | yes |
| `Q8_0` | per-row, streaming, int8 (opt-in) | rocWMMA, batched matvec | grouped | yes | yes |
| `Q4_K`, `Q5_K`, `Q6_K` | per-row, streaming, int8 (opt-in) | rocWMMA, Direct FP16, int8 MMQ, Direct MMQ, batched matvec, streaming | grouped, Q8_1 `dp4a`, int8 WMMA | no | yes |
| `IQ2_XXS`, `IQ2_S`, `IQ3_S` | no | no | grouped, Q8_1 `dp4a`, int8 WMMA | no | no |

The IQ codebooks are also read by `YUUKA_OP_DEQUANT_F16`
(`yuuka_dequant_f16`), which expands one packed matrix to FP16 for tests and
tooling. Every other family rejects them, so an IQ tensor outside the MoE family
is a planning error rather than a silent zero.

`Q4_1`, `Q5_1`, `Q2_K`, `Q3_K`, `Q8_K`, `IQ1_S`, `IQ2_XS`, `IQ3_XXS`, `IQ4_NL`,
and `IQ4_XS` have no device kernel and are capability failures.

### 3.1 KV formats

The paged KV family accepts the pool formats below. The same block table and pool
geometry are used for all three; only the row size differs, and prefix-cache keys
include the format so blocks are never reused across formats.

| Format | Row layout |
| --- | --- |
| `F32` | `kv_dim` FP32 values |
| `F16` | `kv_dim` FP16 values (default) |
| `Q8_0` | `ceil(kv_dim/32)` groups of one FP16 scale plus 32 int8 values |

## 4. Operator inventory

| Family | Stage | Contract |
| --- | --- | --- |
| Quantized matvec | Decode | `out[n] = W[n,k] · x[k]` over packed weights |
| Streaming matvec | Decode, both output tails | Same product over 1–16 rows; remapped one-row form and channel-parallel form |
| Int8 matvec | Decode, opt-in | The same product over Q8_1 activations and int8 digits; `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` |
| Batched matvec | Prefill, ≤ 16 rows | Reuses each decoded weight element across the row block |
| rocWMMA GEMM | Prefill | FP16 operands, FP32 accumulate, every matvec weight dtype |
| rocBLAS FP16 GEMM | Prefill | `F16` weights, FP16 activation and destination scratch, FP32 compute, one cast to FP32 |
| Int8 MMQ GEMM | Prefill | Q8_1 activations, int8 WMMA with exact int32 accumulation, FP32 weight scales; K-quants |
| Direct MMQ GEMM | Prefill, measured cells | The MMQ arithmetic over a wider channel tile |
| Direct FP16 WMMA GEMM | Prefill, 17–127 rows | Short-chunk form of the rocWMMA route, with a fused SwiGLU pair |
| Narrow `Q8_0` pair | Prefill | Two `Q8_0` projections over one normalized tile |
| Embedding | Both | Row gather of packed weights into FP32 |
| Norm, RoPE, elementwise | Both | RMSNorm, head RMSNorm, L2 norm, gated RMSNorm, RoPE (static and dynamic), SiLU·mul, add, copy, scale |
| Prefill attention, tiled | Prefill | Causal GQA over a chunk with online softmax |
| Prefill attention, FP16 MMA | Prefill | Grouped query heads sharing a KV head |
| Prefill attention, split-key | Prefill | Long prefixes; partial online-softmax records merged with log-sum-exp |
| Prefill attention, rocBLAS F16 | Prefill, opt-in | Gathered K/V, rocBLAS QK and PV, FP32 softmax |
| Decode attention, split-key | Decode | Paged block table, one block per KV head and key range, partials merged per head |
| Decode attention, serial | Decode | One block per query head, sequential positions; fallback and reference |
| KV write | Both | Single row, chunk, paged, and device-position variants |
| Recurrent decode | Decode | Fused convolution + L2 norm, then the per-token state update with gated RMSNorm |
| Recurrent prefill | Prefill | Chunk convolution, the chunk recurrence, and gated RMSNorm |
| MoE routing | Both | Top-k selection with normalized weights; binned form and one-row selected form |
| MoE experts | Prefill | Grouped kernels over the binned assignments |
| MoE experts | Decode | One grid per routing slot of the selected experts |
| MoE combine | Both | Weighted reduce-add into the residual stream, and the sigmoid-gated shared expert |
| Dequantize, cast | Both | `dequant_f16` for K-quants and IQ codebooks; F32↔F16 casts for the rocBLAS route |

## 5. Selection rules

### 5.1 Decode matvec

The single predicate `backend/hip.SelectDecodeGemvPath` (sharing
`SelectLMHeadPath`) decides the family:

1. dtype in the matvec set (section 3), `n_in % 256 == 0`, 1–16 rows, and `n_out`
   at or above the measured crossover — 512 outputs for `Q4_K`/`Q5_K`/`Q6_K`,
   1024 for `F32`/`F16`/`Q4_0`/`Q5_0`/`Q8_0` — takes the **streaming** family.
2. Inside that family a block of exactly one row whose `n_in % 512 == 0` takes the
   **remapped one-row** form; every other row count and K width takes the
   **channel-parallel** form.
3. Anything else takes the **per-row** matvec. The 32-output recurrent `beta` and
   `alpha` projections are the shapes that land here on the dense model; the MoE
   router projection lands here too, which is intended.
4. The **int8** matvec replaces the FP32 route only when `Options.Int8Matvec` is
   set. It is measured slower end to end, so production sessions leave it off.

The output projection uses the same rule, which is what makes the prefill tail
and the decode tail one contract.

### 5.2 Prefill matmul

`backend/hip.SelectPrefillPath` returns the route for a dtype and shape; the
native launcher may then upgrade a rocWMMA or MMQ decision to its Direct form.

| # | Condition (evaluated in this order) | Route |
| --- | --- | --- |
| 1 | K-quant, `K % 256 == 0`, `N ≥ 512`, `1 ≤ M ≤ 16` | streaming matvec |
| 2 | `F16`, `K > N`, `4 ≤ M ≤ 64` | rocBLAS |
| 3 | `F16`, `K > N`, `M ≥ 65` | rocWMMA |
| 4 | `F16`, `K ≤ N`, `M < 8` | batched matvec |
| 5 | `F16`, `K ≤ N`, `8 ≤ M ≤ 127` | rocWMMA |
| 6 | `F16`, `K ≤ N`, `M ≥ 128` | rocBLAS |
| 7 | K-quant, `K % 256 == 0`, `N ≥ 64`, `M ≥ 128` | int8 MMQ |
| 8 | `M < 4`, or `K > N` with `M < 8` | batched matvec |
| 9 | otherwise | rocWMMA |

Native upgrades on gfx1151 only:

- **Direct FP16 WMMA** replaces rocWMMA for `17 ≤ M ≤ 127` when `K % 256 == 0`,
  `N % 128 == 0`, the activation is 16-byte aligned, and the weights are 16-byte
  (`Q4_K`/`Q5_K`) or 2-byte (`Q6_K`) aligned. The fused SwiGLU pair uses the same
  rule.
- **Direct MMQ** replaces the generic MMQ tile at `M ≥ 128` only for the measured
  `(dtype, K, N)` cells enumerated in `measured_direct_shape`, with 16-byte
  aligned packed weights and Q8_1 digits. Every other cell keeps the generic
  tile.

The prefill output tail is an `N >> M` shape with `M ≤ 16`. It uses the streaming
matvec whenever the row count, dtype, `n_in % 256 == 0`, and the output width
predicate hold, and one per-row matvec per row otherwise.

### 5.3 Prefill attention

`launch_attention_prefill_paged_f32` selects the kernel; the Go side mirrors the
choice in `hip.PrefillAttentionMMA`:

1. **Split-key MMA** when the plan binds the split scratch, the KV pool is `F16`,
   `F32`, or `Q8_0`, `head_dim` is 256 with four query heads per KV head (or the
   8:1 `F16` geometry at the same width), and the position falls in a measured
   range:

   | KV format | Query rows | Start positions |
   | --- | --- | --- |
   | `F16` | 512, 1024, 2048 | 6144–14336 |
   | `F16` | 64, 513 | 7680–14336 |
   | `F32` | 512 | 6144–14336 |
   | `Q8_0` | 512 | 8192–14336 |

2. **Unsplit FP16 MMA** when `heads == 4 * kv_heads` for any of the three pool
   formats, or `heads == 8 * kv_heads` for an `F16` pool at width 256;
   `head_dim` is 64, 128, or 256; and either `rows ≥ 64` or
   `position + rows ≥ 256`. A `Q8_0` pool additionally needs position ≥ 512, or
   ≥ 2048 for a 1024-row parent chunk.
3. **Tiled scalar attention** otherwise, including head dimensions 1–256 outside
   the MMA widths. Head dimensions above 256 are rejected as an unsupported
   capability.

The query layout is the gated Qwen3.5 layout: the second head-sized half of each
query vector gates the output through a sigmoid. The ABI also carries a dense
variant without the gate; no supported model selects it. The rocBLAS `F16` route
is opt-in (`Options.LegacyBlasAttention`) and serves only as a comparison point.

### 5.4 Decode attention and KV index

The split-key kernel serves a KV head group of 1, 2, 4, or 8 query heads with
`head_dim ≤ 256` and `head_dim % 4 == 0`; every other shape keeps the serial
per-head kernel. The split tile is 32 positions up to a 1024-position capacity
and 64 above it, and the partials workspace is
`heads * ceil(capacity/tile) * (head_dim + 2)` floats. That size is owned by
`decodeAttentionScratch` in `engine/gpu/model.go` and must agree with the kernel.
A captured graph keeps its recorded tile extent, and positions past the live
length are skipped before any K/V read.

Every position maps through the int32 block table
(`block_table[p / block_size] * block_size + p % block_size`), and the table and
pools are device buffers.

### 5.5 MoE

The plan builds `ROUTE`, `GATE_UP`, `DOWN`, and `REDUCE_ADD` per MoE sub-layer,
plus `SHARED_ADD` for the shared expert. One token with at most
1024 experts and at most 32 routing slots uses the selected-expert form, which
reads only the selected ids and weights; other shapes use the binned form whose
assignment arrays are sized `experts * rows * used`, and a plan that would
overflow a bucket is rejected at construction. A wider-wave variant of the
selected kernels covers the one-token, eight-slot, hidden-2048, FFN-512 IQ cells.

Routed-expert prefill route (`backend/hip.SelectMoEPrefillPath`), for chunks of at
least 16 rows:

| Condition | Route | Assignment tile |
| --- | --- | --- |
| dtype in `Q4_K`, `Q5_K`, `Q6_K`, `IQ2_XXS`, `IQ2_S`, `IQ3_S`, `K % 256 == 0`, `N % 128 == 0` | int8 WMMA | 16 up to 384 rows, 32 at 512–2048, 64 at 4096 |
| `IQ2_XXS`, `IQ2_S`, `IQ3_S`, `K % 256 == 0`, `N % 256 == 0` | Q8_1 `dp4a` | n/a |
| every other dtype and shape | FP32 grouped | n/a |

`Options.ForceMoEQ8` keeps the `dp4a` route for the shapes the WMMA route would
take, so the two can be compared on identical plan inputs.

The router projection is deliberately not on the prefill dispatch table: an `F32`
router runs the FP32 batched GEMM because an FP16-staged activation tile is
enough noise to flip near-tie expert selections. Decode always runs the router as
a per-row matvec.

### 5.6 Gated delta-net

The recurrence is

```text
S_r = gamma_r (I - beta_r k_r k_r^T) S_{r-1} + beta_r k_r v_r^T
o_r = q_r S_r        (then the gated RMSNorm of the head)
```

`beta_r` is the sigmoid-gated write strength and `gamma_r` the per-step decay
from the alpha projection. The state is stored row-major per head and shared by
the decode and prefill paths.

- **Decode** fuses the short convolution and the q/k L2 normalizations into one
  operator and the state update plus gated RMSNorm into a second, when the
  geometry is inside the compiled set (state 128, 256-aligned channel counts).
  Other geometries keep the five separate operators (`CONV1D`, two `L2NORM`,
  `GDN`, `GATED_RMSNORM`). `Options.LegacyDeltaNet` forces the unfused plan.
- **Prefill** runs the preparation pass plus the chunk recurrence when the model
  runs on gfx1151 with state 128, at least two rows, and a preparation span that
  binds in an existing scratch buffer; gated RMSNorm then runs as a separate row
  kernel. Without that span the fallback chunk kernel computes the same
  recurrence, and state sizes other than 128 use the fused per-column kernel.
- The chunkwise WY (triangular-inverse) form of the same recurrence is implemented
  and validated, but it is not the selected path: the op runs it only when the
  caller supplies the chunk scratch slot of the plan operation.

## 6. Fusion

Fused operators exist where they remove a launch or an intermediate pass without
lowering occupancy:

| Stage | Fused form |
| --- | --- |
| Decode projection | Matvec + residual add (`GEMV_RESIDUAL`) |
| Decode pair | Two matvecs over one activation row, either two outputs or the SwiGLU product (`GEMV_PAIR`); the narrow `beta`/`alpha` pair uses the same form |
| Prefill FFN | gate + up + SwiGLU in one kernel (`GEMM_SWIGLU`), including the Direct FP16 variant |
| Prefill FFN, MMQ | SwiGLU plus Q8_1 quantization of the product (`SILU_MUL_QUANT_Q8_1`) |
| Prefill Q8_1 reuse | One activation quantization per projection group; the shared-expert pair reuses the routed experts' tile |
| Prefill norms and RoPE | Row/hand batching of RMSNorm, head RMSNorm, and RoPE over a chunk |
| Prefill KV | Batched KV writes for a chunk |
| Prefill attention | Online softmax, value reduction, and the attention gate |
| Gated delta-net | Convolution state, q/k L2 norm, chunk recurrence, and gated RMSNorm, as described in 5.6 |
| MoE | Expert routing, expert FFN, and the weighted reduce-add; sigmoid-gated shared expert add |

Each fusion must be evaluated on launch count, register and occupancy cost, shared
memory, traffic, and end-to-end latency, never on isolated kernel time alone. The
unfused path stays behind an option so the comparison is reproducible:
`Options.NoResidualFusion`, `NoPairFusion`, `LegacyDeltaNet`, `ForceTiledAttention`,
`LegacyBlasAttention`, `ForceDecodeGemv`, `Int8Matvec`, `PrequantizedInt8`, and
`ForceMoEQ8`.

## 7. Numerical requirements

| Stage | Gate |
| --- | --- |
| Matvec (per-row, streaming, batched) | `2e-3 * (1 + \|expected\|)` against a host dot product over dequantized weights |
| Prefill GEMM (rocWMMA, MMQ) | `2e-2 * (1 + \|expected\|)` against a float reference built from the exact packed weights; `1e-3` when the reference itself is FP16 |
| Prefill attention | `2e-4 * (1 + \|expected\|)` against the FP32 reference, `1e-3` for an `F16` pool |
| Chunked prefill | logits within `2e-3` of the unchunked CPU reference and an identical top-1 token, for chunk sizes 1, 3, 7, 16, 17, 33, and 64 |
| FP16 MMA prefill attention | logits within `0.05` against the forced tiled control on the dense model (`0.3` for the long-prompt variant), and within `2.0` on the 8:1 MoE geometry with an identical top-1 token and continuation |
| Int8 MMQ prefill | inside the recorded `0.0031–0.264` logit band with identical tokens |
| Int8 MoE (`dp4a`, WMMA) | `1e-6` relative against the same Q8_1 inputs; the full Q8_1 pipeline `2e-2` against a host computation over the dequantized weights |
| Int8 decode matvec | kernel digits within `5.7e-4` of a host reference; logits may move up to `0.2656` over eight steps with identical tokens |
| Gated delta-net | `1e-5` against an independent host implementation of the documented order of operations |

Rules that apply to every stage:

- Use stable softmax (max subtraction) and stable normalization.
- Apply quantization scales and minimums before the final rounding; do not double
  round between unpack and accumulate.
- Do not enable fast-math or unsafe FP contraction unless a numerical test shows
  the stage stays inside its gate.
- Handle non-aligned dimensions, zero-length edge cases, and extreme values, and
  check reduction accumulators for NaN/Inf and overflow.
- Compare against the CPU reference and, for critical operators, one independent
  implementation as well.
- Do not tune a kernel against a single prompt or a single shape. Changing a
  documented tolerance requires evidence and review.

## 8. Limitations and open questions

- No fused mixed prefill/decode batch: each step belongs to one request plan, and
  continuous batching interleaves per-request plans at the scheduler level.
- Block-table entries are immutable for a request; all blocks are allocated at
  admission, so replay and kernels never observe a changing table.
- No device sampler: the decode step downloads the FP32 logits row and the Go
  generation loop samples it.
- `Q8_0` KV is supported and reduces cache memory, but it is not faster than
  `F32` at short context.
- Speculative decoding is exact for greedy sampling only, and reuses the chunked
  prefill graph with a multi-row output tail.
- Prefill does not fuse query/key/value projections or RMSNorm into a projection
  kernel.
- The Direct MMQ and Direct FP16 WMMA routes exist only on gfx1151 builds; other
  targets keep the generic tiles.
- `Q8_0`, `Q4_0`, and `Q5_0` weights never take the int8 MMQ prefill route, and
  the IQ codebooks stay on the MoE and dequant paths.
- Attention head dimensions above 256, and the ungated query layout, are outside
  the supported set.

## 9. Acceptance criteria

A kernel is not complete until:

- [x] It matches the CPU reference on representative shapes and buffer-boundary
      cases, including randomized and non-aligned shapes.
- [x] Its numerical tolerance is documented in section 7 and enforced by a test.
- [x] Its shape classes are covered by a benchmark or profile in the benchmark
      records.
- [x] Its resource figures (VGPR, LDS, scratch, occupancy) are recorded for the
      selected device.
- [x] The planner selects it, and each quantized family has a measured fallback.
- [x] It is integrated into an end-to-end model run.
- [x] It does not regress a previously accepted shape class.

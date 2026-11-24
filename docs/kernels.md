# GPU Kernels

## 1. Scope

This document is the contract for device-side execution on gfx1151: the prefill
and decode operator families, the supported weight and KV dtypes, the rules that
select between them, the fused operators, and the numerical requirements.

Related contracts: [architecture.md](architecture.md) owns the Go/HIP boundary,
buffer ownership, and plan submission; [gguf.md](gguf.md) owns the packed row
layouts; [serving.md](serving.md) owns scheduling, KV index construction, and the
API.

Measured evidence—per-shape timings, resource figures, and rejected
alternatives—is recorded in the [Qwen3.5-4B](benchmarks/qwen3.5_4b.md) and
[Qwen3.6-35B-A3B](benchmarks/qwen3.6_35b_a3b.md) reports. This document states
what the runtime does, not how a kernel is written.

Plan operations are a versioned C ABI. Renumbering or removing an opcode
requires an ABI version bump.

## 2. Design rules

1. **Prefill and decode are separate operator families.** They share layouts and
   helper primitives but not a single GEMM: decode never routes through a
   large-`M` tile path by default, and prefill never routes through one block per
   output row.
2. **Quantized weights stay packed.** Dequantization is fused into the consuming
   operator, and no stage materializes a dequantized copy of a model matrix. The
   rocBLAS route is the one exception by construction: it reads weights already
   stored as `F16`, so it never substitutes for a quantized matrix.
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
   produces bitwise identical outputs.
7. **Thresholds are gfx1151 measurements.** Occupancy, bandwidth, and tuning
   facts from other RDNA parts or from CDNA are not accepted as evidence. A new
   device requires re-running the sweeps and updating the selectors and tests.
8. **One request per plan.** Kernel arguments are fixed at plan construction, and
   token and position for a captured graph are read from a device parameter
   buffer, so replay never rewrites arguments.

## 3. Supported dtypes

Weight dtypes match the GGUF `ggml_type` values. "Streaming" is the decode
matvec family of section 4.

| DType | Decode matvec | Prefill matmul | MoE experts | KV pool | Embedding |
| --- | --- | --- | --- | --- | --- |
| `F32` | per-row, streaming | rocWMMA, batched matvec | grouped | yes | yes |
| `F16` | per-row, streaming | rocWMMA, rocBLAS, batched matvec | grouped | yes (default) | yes |
| `Q4_0`, `Q5_0` | per-row, streaming | rocWMMA, batched matvec | grouped | no | yes |
| `Q8_0` | per-row, streaming, int8 (opt-in) | rocWMMA, batched matvec | grouped | yes | yes |
| `Q4_K`, `Q5_K`, `Q6_K` | per-row, streaming, int8 (opt-in) | rocWMMA, int8 MMQ, batched matvec | grouped, Q8_1 `dp4a`, int8 WMMA | no | yes |
| `IQ2_XXS`, `IQ2_S`, `IQ3_S` | no | no | grouped, Q8_1 `dp4a`, int8 WMMA | no | no |

The IQ codebooks are also decoded by the block dequantization operation, which
expands one packed matrix to FP16 for tests and tooling. Every other family
rejects them, so an IQ tensor outside the MoE family is a planning error rather
than a silent zero.

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
| Quantized matvec | Decode | `out[n] = W[n,k] · x[k]` over packed weights; per-row and streaming forms, with an opt-in int8 arithmetic |
| Batched matvec | Prefill, small row blocks | Reuses each decoded weight element across the row block |
| rocWMMA GEMM | Prefill | FP16 operands, FP32 accumulate, every matvec weight dtype |
| rocBLAS GEMM | Prefill, `F16` weights | FP16 activation and destination scratch, FP32 compute, one cast to FP32 |
| Int8 MMQ GEMM | Prefill | Q8_1 activations, int8 matrix instructions with exact int32 accumulation, FP32 weight scales; K-quants |
| Embedding | Both | Row gather of packed weights into FP32 |
| Norm, RoPE, elementwise | Both | RMSNorm, head RMSNorm, L2 norm, gated RMSNorm, RoPE, SiLU·mul, add, copy, scale |
| Prefill attention | Prefill | Causal GQA with online softmax; tiled, FP16 MMA, and long-prefix split-key forms |
| Decode attention | Decode | Paged block table; split-key and serial per-head forms |
| KV write | Both | Single row, chunk, paged, and device-position variants |
| Recurrent decode | Decode | Fused convolution + L2 norm, then the per-token state update with gated RMSNorm |
| Recurrent prefill | Prefill | Chunk convolution, the chunk recurrence, and gated RMSNorm |
| MoE routing | Both | Top-k selection with normalized weights; binned and one-row selected forms |
| MoE experts | Both | Grouped kernels over binned assignments; selected kernels for one token |
| MoE combine | Both | Weighted reduce-add into the residual stream, and the sigmoid-gated shared expert |
| Dequantize, cast | Both | Block dequantization for K-quants and IQ codebooks; F32↔F16 casts for the rocBLAS route |

## 5. Selection rules

### 5.1 Decode matvec

The decode selector decides the family from dtype, row count, and output width:

1. A shape in the matvec dtype set, with a K width aligned to the block layout
   and a row block of at most 16 rows, uses the **streaming** family at or above
   the measured output-width crossover.
2. Inside that family a one-row block with a suitable K width uses the
   **remapped one-row** form; every other row count and K width uses the
   **channel-parallel** form.
3. Anything else uses the **per-row** matvec. The narrow recurrent projections
   (the 32-output `beta` and `alpha` shapes) and the MoE router fall here on
   purpose.
4. The **int8** matvec replaces the FP32 route only when the corresponding
   planner option is set. It stays off by default because the measured route is
   slower end to end; the model report records the comparison.

The output projection uses the same rule, which is what makes the prefill tail
and the decode tail one contract. The measured crossovers are recorded in the
model reports.

### 5.2 Prefill matmul

The prefill selector returns the route for a dtype and shape; the native
launcher may then upgrade a generic decision to a device-specific tile.

| # | Condition | Route |
| --- | --- | --- |
| 1 | K-quant, aligned K, small row block, wide output | streaming matvec |
| 2 | `F16`, tall-skinny (`K > N`), small row block | rocBLAS |
| 3 | `F16`, tall-skinny (`K > N`), larger row block | rocWMMA |
| 4 | `F16`, wide (`K ≤ N`), very small row block | batched matvec |
| 5 | `F16`, wide (`K ≤ N`), medium row block | rocWMMA |
| 6 | `F16`, wide (`K ≤ N`), large row block | rocBLAS |
| 7 | K-quant, aligned K and output, large row block | int8 MMQ |
| 8 | `M < 4`, or `K > N` with a very small row block | batched matvec |
| 9 | otherwise | rocWMMA |

The native launcher may select a wider device-specific tile for aligned K-quant
shapes on gfx1151, including a Direct FP16 form for short chunks and a Direct
MMQ form for measured cells. Every other cell keeps the generic tile. The
measured cell set is recorded in the model reports.

The prefill output tail is an `N >> M` shape with a small row block. It uses the
streaming matvec whenever the row count, dtype, K alignment, and the output
width predicate hold, and one per-row matvec per row otherwise.

### 5.3 Prefill attention

The selector mirrors the native kernel choice:

1. **Split-key MMA** for long prefixes on covered KV formats, head width, and
   query-head geometry, when the plan binds the split scratch and the position
   falls in a measured range.
2. **Unsplit FP16 MMA** when the query heads are a multiple of the KV heads for
   any supported pool format, or the 8:1 grouped-head geometry on an `F16` pool
   at the supported width; head width is 64, 128, or 256, and either the chunk is
   large enough or the absolute position has reached the measured threshold.
3. **Tiled scalar attention** otherwise, including head widths outside the MMA
   set. Head widths above 256 are rejected as an unsupported capability.

The query layout is the gated Qwen3.5 layout: the second head-sized half of each
query vector gates the output through a sigmoid. The ABI also carries a dense
variant without the gate; no supported model selects it. The rocBLAS `F16` route
is opt-in and serves only as a comparison point.

### 5.4 Decode attention and KV index

The split-key decode kernel serves a KV head group of 1, 2, 4, or 8 query heads
with a head width that is a multiple of four; every other shape keeps the serial
per-head kernel. The split tile and the partials workspace are sized at plan
construction, and a captured graph keeps its recorded tile extent. Positions
past the live length are skipped before any K/V read.

Every position maps through the int32 block table
(`block_table[p / block_size] * block_size + p % block_size`), and the table and
pools are device buffers.

### 5.5 MoE

The plan builds routing, gate/up, down, and reduce-add steps per MoE sub-layer,
plus a shared-expert add. One token with a bounded expert count and routing-slot
count uses the selected-expert form, which reads only the selected ids and
weights; other shapes use the binned form whose assignment arrays are sized from
the expert, row, and routing-slot counts. A plan that would overflow a bucket is
rejected at construction.

Routed-expert prefill route for chunks of at least sixteen rows:

| Condition | Route |
| --- | --- |
| K-quant or IQ expert dtype, aligned K and output width | int8 WMMA grouped |
| IQ expert dtype, aligned K and a wider output alignment | Q8_1 `dp4a` grouped |
| every other dtype and shape | FP32 grouped |

The assignment tile is selected per row class from the measured table recorded
in the Qwen3.6 report; the `dp4a` route stays reachable through a planner option
for A/B measurement.

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
  geometry is inside the compiled set. Other geometries keep the five separate
  operators.
- **Prefill** runs a preparation pass plus the chunk recurrence when the model
  runs on gfx1151 with the supported state width, at least two rows, and a
  preparation span that binds in an existing scratch buffer; gated RMSNorm then
  runs as a separate row kernel. Without that span the fallback chunk kernel
  computes the same recurrence, and other state sizes use the fused per-column
  kernel.
- The chunkwise WY (triangular-inverse) form of the same recurrence is
  implemented and validated, but it is not the selected path.

## 6. Fusion

Fused operators exist where they remove a launch or an intermediate pass without
lowering occupancy:

| Stage | Fused form |
| --- | --- |
| Decode projection | Matvec plus residual add |
| Decode pair | Two matvecs over one activation row, either two outputs or the SwiGLU product; the narrow recurrent pair uses the same form |
| Prefill FFN | gate + up + SwiGLU in one kernel, including the short-chunk form |
| Prefill FFN, MMQ | SwiGLU plus Q8_1 quantization of the product |
| Prefill Q8_1 reuse | One activation quantization per projection group; the shared-expert pair reuses the routed experts' tile |
| Prefill norms and RoPE | Row/head batching of RMSNorm, head RMSNorm, and RoPE over a chunk |
| Prefill KV | Batched KV writes for a chunk |
| Prefill attention | Online softmax, value reduction, and the attention gate |
| Gated delta-net | Convolution state, q/k L2 norm, chunk recurrence, and gated RMSNorm |
| MoE | Expert routing, expert FFN, and the weighted reduce-add; sigmoid-gated shared expert add |

Each fusion must be evaluated on launch count, register and occupancy cost,
shared memory, traffic, and end-to-end latency, never on isolated kernel time
alone. The unfused and alternative routes stay reachable through opt-in planner
options so the comparison is reproducible.

## 7. Numerical requirements

| Stage | Comparison |
| --- | --- |
| Matvec | Host dot product over dequantized weights |
| Prefill GEMM | Float reference built from the exact packed weights; a tighter bound when the reference itself is FP16 |
| Prefill attention | FP32 reference; a looser bound for an `F16` pool |
| Chunked prefill | Identical top-1 token and logits inside a documented bound against the unchunked CPU reference |
| MoE experts | Exact-arithmetic comparison on identical Q8_1 inputs; the int8 pipeline against a host computation over the dequantized weights |
| Dequantization and gated delta-net | Independent host implementation of the documented order of operations |

The exact tolerance of each stage is documented and enforced by the test that
covers it; recorded end-to-end deltas are in the model reports.

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
- `Q8_0` KV reduces cache memory at a measured cost recorded in the Qwen3.5
  report.
- Speculative decoding is exact for greedy sampling only, and reuses the chunked
  prefill graph with a multi-row output tail.
- Prefill does not fuse query/key/value projections or RMSNorm into a projection
  kernel.
- Device-specific tiles exist only on gfx1151 builds; other targets keep the
  generic tiles.
- `Q8_0`, `Q4_0`, and `Q5_0` weights never take the int8 MMQ prefill route, and
  the IQ codebooks stay on the MoE and dequantization paths.
- Attention head widths above 256, and the ungated query layout, are outside the
  supported set.

## 9. Acceptance criteria

A kernel is not complete until:

- [ ] It matches the CPU reference on representative shapes and buffer-boundary
      cases, including randomized and non-aligned shapes.
- [ ] Its numerical tolerance is documented in section 7 and enforced by a test.
- [ ] Its shape classes are covered by a benchmark or profile in the benchmark
      records.
- [ ] Its resource figures (VGPR, LDS, scratch, occupancy) are recorded for the
      selected device.
- [ ] The planner selects it, and each quantized family has a measured fallback.
- [ ] It is integrated into an end-to-end model run.
- [ ] It does not regress a previously accepted shape class.

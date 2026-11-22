# Testing and Benchmarking

## 1. Principles

1. **Correctness before performance.** A fast kernel with incorrect logits is a
   regression, not an optimization.
2. **Reproducibility.** Every benchmark records the model file, quantization,
   ROCm and driver versions, sampling settings, warmup, and thermal conditions.
3. **Separate latency from throughput.** Single-request token/s and
   multi-request aggregate throughput are different metrics.
4. **Compare like with like.** Cross-framework comparisons use the same GGUF,
   prompt set, context length, sampling parameters, and hardware conditions.
5. **Test the seams.** The GGUF reader, tokenizer, planner, kernels, scheduler,
   and API each need independent tests in addition to end-to-end tests.

## 2. Correctness strategy

Correctness is established in layers, from format parsing to generated tokens:

```text
GGUF parsing and validation
        |
Tokenizer and chat template (token IDs)
        |
Tensor-level operator comparison vs CPU reference
        |
Logits comparison (tolerance)
        |
Greedy token sequence and stop behavior
        |
End-to-end chat and streaming behavior
```

A failure at any layer blocks the layers above it. Do not compensate for a
tokenizer mismatch by loosening logits tolerances.

### 2.1 Reference implementations

- **CPU reference:** straightforward, readable implementations prioritized for
  clarity, with documented accumulation precision. They are the primary oracle
  for tensor and operator tests.
- **Independent framework:** llama.cpp using the same GGUF is the primary
  external comparison for logits and token sequences. It is a reference, not an
  execution dependency.
- **Mathematical checks:** small hand-computed cases for normalization, RoPE,
  softmax, and quantization decode.

Reference code must not share the optimized kernel's implementation strategy;
otherwise a shared bug can pass both sides.

### 2.2 Tolerance policy

Tolerance is defined per stage, not only for final logits:

| Stage | Comparison | Suggested policy |
| --- | --- | --- |
| Quantized weight decode | Reconstructed values vs exact reference | Exact for integer unpacking; documented tolerance for float scale application |
| GEMV / GEMM | Output tensor vs CPU reference | Absolute and relative tolerance per dtype |
| RMSNorm / RoPE | Output tensor vs CPU reference | Tight tolerance; test extreme values |
| Attention | Output vs CPU reference | Tolerance plus causal-mask and GQA-specific cases |
| Logits | Top-k values and ordering | Tolerance for values; exact top-1 for greedy tests |
| Tokens | Greedy sequence | Exact token IDs on the golden set |

The concrete numeric thresholds are recorded per test and per dtype. They must
be tight enough to catch a wrong quantization scale or a missing mask, and loose
enough to accommodate legitimate floating-point reassociation.

## 3. Test layers

### 3.1 GGUF reader tests

Checked-in rejection tests use synthetic and real fixtures to cover:

- valid single-file and split-file models;
- bad magic, unsupported version, truncated header, unknown metadata type;
- duplicate metadata keys and tensor names, zero dimensions, rank overflow;
- offsets that overlap, exceed the data section, or violate alignment;
- incomplete shards and tensors assigned to multiple shards;
- quantization block sizes that do not divide the tensor dimensions.

The loaders and config builders additionally enforce:

- missing required tensors;
- head/KV-head/head-dimension/context consistency and RoPE section consistency;
- special-token ID range checks and non-Qwen chat-template rejection at render
  time.

Self-contained validation has model-config unit tests; missing-tensor rejection
does not yet have a dedicated end-to-end fixture test. Every
checked-in rejection test asserts the error class and that the failed file is
not left mapped.

### 3.2 Tokenizer tests

Token IDs are the contract. Golden tests (opt-in through `YUUKA_TEST_MODEL`,
`YUUKA_TEST_MOE_MODEL`, and `YUUKA_TEST_SPM_VOCAB`) cover:

- English and Chinese text;
- whitespace, newlines, punctuation, and emoji;
- special tokens and role-marker fragments;
- BOS/EOS insertion and suppression.

Multi-turn chat rendering and stop-string behavior are checked at the rendered
text level. Tab and full multi-turn prompt token IDs are not yet part of the
golden set. Every golden token-ID case compares against hardcoded reference IDs,
and decoded text is checked as a secondary test.

### 3.3 Operator tests

For every operator and quantization type:

- random shapes, including non-aligned `M`, `N`, and `K`;
- small hand-checkable shapes;
- boundary values and extreme magnitudes;
- tensors with a single row or column;
- NaN/Inf behavior where defined;
- deterministic mode comparisons;
- CPU reference, GPU kernel, and library fallback where available.

Tests must cover the actual packed layouts used by GGUF, not only dense FP16
inputs.

Plan submission is also tested: unsupported quantized dtypes, widths that
violate the block layout, batches above the measured limit, null buffers, and
undersized buffers must be rejected when the operation is added, not ignored at
execution time. Every dtype supported by the batched GEMV kernel is checked at
batch sizes 1, 2, 3, 4, 8, and 16.

Prefill operators have their own native tests in `tests/hip/prefill_test.cpp`:

- WMMA GEMM for `F32`, `F16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, and
  `Q6_K` at `M` values 1, 7, 16, and 33 and non-tile-aligned `N`;
- the int8 MMQ route for `Q4_K`, `Q5_K`, and `Q6_K` at `(K, N, M)` 512/64/128,
  512/128/130, 768/70/129, and 2560/128/128. The Q8_1 quantizer is checked digit
  for digit against a host reimplementation of the same rule, including the
  block scale and block sum, and the GEMM is then compared with a CPU dot
  product over those dequantized activations at the 2e-2 per-GEMM bound. Q8_1
  activations are lossy by construction, so the FP32 dot product is not the
  reference for this path: the measured difference reaches 0.04-0.14 on these
  shapes, which is quantization noise, not kernel error;
- the rocBLAS FP16 baseline against a float reference;
- the streaming prefill LM head for `F32`, `F16`, `Q4_0`, `Q5_0`, `Q8_0`,
  `Q4_K`, `Q5_K`, and `Q6_K` at one, three, and sixteen rows, with `N = 400` so
  the last workgroup serves a partial channel tile, and at `K = 512` so the K
  step count differs from the model shape. Each dtype is compared with the
  independent host dot products and a repeated submission must be bitwise
  identical; a plan that asks for more rows than the row cap or for a width
  that is not a multiple of 256 must be rejected when the operation is added;
  `backend/hip/lm_head_test.go` additionally compares Q6_K and Q8_0 against
  dequantized CPU dots, where Q8_0 is bitwise equal;
- row-batched and head-batched RMSNorm, batched RoPE, batch embedding, and
  batch KV writes;
- legacy single-token embedding for Q4_0 and Q5_0, compared against the same
  packed-row reference as the batch path (`tests/hip/embedding_test.cpp`);
- causal GQA Flash Attention with a non-zero chunk start position;
- the fused convolution, L2 normalization, gated delta-net recurrence, and
  gated RMSNorm chunk kernel at the model dimensions (128 rows, 16 groups,
  32 heads, state 128), including updated convolution and matrix state. The
  same operation is run twice from restored state and must be bitwise
  identical; this catches the shared-scratch and state-ordering races that a
  single execution can miss.

Each compares against an independent float implementation and records its
tolerance in the test.

Per-operator tolerances do not bound network-level drift. A prefill matmul that
met the 2e-2 per-GEMM bound while accumulating in FP16 moved the 512-token
logits by 0.305 against a bitwise-identical baseline, six times the 0.05 chunk
gate, because a 2-5e-3 error in one 9216-wide dot compounds across 32 layers.
A route change must therefore clear the chunk gate in
`TestGPUChunkedPrefillLongContext`, not only the per-operator bound. That chunk
gate measures one route against itself and the int8 MMQ route holds it at 0,
because int32 accumulation is exact and the chunked plan reads the same
quantized weights.

The int8 route does move the model output against the FP16 routes it replaced.
On the 2048-token prompt the maximum logit delta is 0.264 (mean 0.045) with the
top-1 token preserved; the cause is the Q8_1 activation grid (~4e-3 relative per
value) against FP16 staging (~5e-4), which compounds across 32 layers. Any
comparison between two non-identical routes therefore sits on that noise floor:
`TestGPUAttentionPathsAgree`, which changes only the prefill attention dispatch
between its arms, measures 0.18-0.25 at 2048 tokens instead of 0.0132 and is
gated at 0.3 with the top-1 token still required to match. This is the accuracy regime that
llama.cpp's default K-quant prefill path runs in. See
[benchmarks.md](benchmarks.md#isolated-kernels).

The qwen35moe 8:1 F16 grouped-head route is a stronger case of the same effect.
Its attention output stays within the native 5e-3 bound, but an FP16-staged
query can flip a near-tie expert selection in this router-sensitive model: on
the fixed 8192-token stream the full-model logits move by at most 1.43 / 1.26 /
0.87 against the forced tiled control at 512 / 2048 / 8192 tokens. The top-1
token and three greedy continuation tokens are identical at every prefix, and
`TestGPUMoEPrefillGreedyContinuity` enforces both that behavior and the
`moeAttentionRouteLogitTolerance = 2.0` envelope recorded in
[moe_real_test.go](../engine/gpu/moe_real_test.go). The dense 4:1 route keeps
the tighter 0.05/0.3 bounds. This gate was approved explicitly for the record
in [qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md).

Paged KV and graph replay have dedicated native coverage in
`tests/hip/paged_test.cpp`:

- KV writes through a permuted block table land in the mapped physical rows;
- paged decode and prefill attention match an independent float reference
  through the same table;
- the prefill attention dispatch matches the same reference at 2e-4 for a
  group of four query heads per KV head: head_dim 4 selects the tiled per-head
  kernel and head_dim 64 at position 2048 selects the FP16 MMA kernel;
- the FP16 MMA prefill attention kernel runs at `head_dim` 256 with four query
  heads per KV head, a permuted block table, and a 64-row chunk, matches the
  FP32 reference within 5e-3 relative, reproduces its output bitwise on a second
  run, and reports its difference against the tiled kernel on the same plan
  input. A second case covers 16 query heads and 2 KV heads at width 256 with
  independent K/V per physical head, a nonzero position, a query tail that is
  not a multiple of the 16-row tile, and a displaced last-row crop. The
  full-model differential test `TestGPUAttentionPathsAgree` compares the default
  FP16 MMA dispatch against the tiled path selected by `ForceTiledAttention`
  (and the opt-in rocBLAS path) and enforces the 0.05 logit gate at a 64-token
  prompt and 0.3 at 2048 tokens, or the recorded 2.0 MoE route envelope;
- device-to-device block copies used for private prefix blocks;
- graph capture and replay with device-resident token/position updates across
  several positions;
- rejection of plans containing position-dependent ops without a
  dynamic-parameter kernel, so the runtime can fall back instead of replaying
  stale arguments.

The fused gated delta-net pair has native coverage in
`tests/hip/deltanet_test.cpp`, which runs the fused plan (`CONV_L2NORM` and
`RECURRENT_TOKEN`) and the unfused plan (`CONV1D`, two `L2NORM`, `GDN`,
`GATED_RMSNORM`) on the same buffers for three tokens at the geometry of one
Qwen3.5-4B recurrent layer:

- both plans are compared against an independent host implementation of the
  documented order of operations, at 1e-5, which is 300x below the 2.6e-4 that a
  missing decay factor produces at the decay of 0.75 the test uses;
- the convolution output, the ring state, the recurrent state, and the
  normalized output are compared between the plans at 1e-5;
- a state width other than 128 and a channel count that is not a multiple of 256
  are rejected with `YUUKA_ERR_UNSUPPORTED`, so a plan builder that selected the
  fused pair for an unsupported geometry fails instead of running.

`tests/hip/plan_validation_test.cpp` covers the two fused matvec forms as well:
an undersized buffer, a mixed-dtype pair, a SwiGLU pair with unequal widths, and
an input width that is not a multiple of 256 are each rejected at plan
construction.

`TestGPUDecodeFusionsMatchUnfused` is the same differential for the matvec
fusions: the default plan against `Options{NoResidualFusion, NoPairFusion}`, with
the operation counts of both asserted (64 `GEMV_RESIDUAL` and 32 `GEMV_PAIR`
against 64 `ADD` and 32 `SILU_MUL`) and eight greedy steps compared at 1e-3. The
fusions move work rather than arithmetic, so the two plans agree bit for bit.

`TestGPUGraphReplayMatchesAndTimes` runs the same greedy loop on a plain session
and on one with `Options.EnableGraph`, requires identical tokens, and reports both
step times. It is the measurement behind the submission-gap question: replay
removes every per-operation host submission, so a ratio near one says there is
no submission gap left to remove. It asserts correctness only, because a timing
assertion on an unpinned clock is not a gate.

`TestGPUFusedDeltaNetMatchesLegacy` is the full-model differential: it builds the
default plan and the `Options.LegacyDeltaNet` plan, checks the operator counts of
both, and compares eight greedy decode steps (identical tokens, 9.5e-6 maximum
logit difference, gated at 0.02).

Serving tests are split by layer:

- `memory`: allocator reference counting, double-release rejection, table
  mapping and rollback, aligned and partial prefix splits with copy-on-write,
  longest-prefix lookup, exact-key comparison, LRU eviction, and allocation
  unblocking. These run with `-race`.
- `scheduler`: queue limits, admission rejection, queue and request timeouts,
  cancellation, build-failure slot release, decode-priority interleaving,
  explicit batch composition, shutdown drain, and latency summaries.
- `serving`: concurrent requests versus isolated runs, full and partial prefix
  reuse versus a fresh run, cancellation and injected decode-failure cleanup
  with balanced allocator counters, and repeated cycles with zero residual
  blocks. GPU session tests also assert that per-request device buffers do not
  grow the model-lifetime allocation list. Both CPU and HIP backends are covered
  when the HIP library is available.

### 3.4 End-to-end tests

- Fixed prompt, greedy sampling, fixed seed where applicable.
- Compare token IDs with the reference for a bounded number of tokens.
- Verify EOS and stop conditions.
- Verify chat template application and multi-turn context.
- Verify streaming deltas reconstruct the non-streaming response.
- Verify cancellation stops generation and releases resources.
- Verify concurrent requests do not affect each other's output.
- Verify prefix-cache reuse produces identical tokens and text, including a
  longer prompt that reuses a shorter cached prefix.
- Verify a fully cached prompt takes the cached-prefix path and matches a fresh
  run's tokens and text.
- Verify graph replay and fallback produce the same token sequence as a plain
  session, including an intentionally invalidated dynamic parameter buffer.
- Verify HTTP API tests remain independent of the real-model environment by
  using the tiny fixture for streaming, error, stop, and cancellation cases.
- Verify chunked prefill at chunk sizes that select batched GEMV and WMMA,
  including partial final chunks, matches unchunked CPU logits and the greedy
  top-1 token.
- Verify one decode step after chunked prefill matches the CPU state produced by
  a single unchunked prefill.
- Verify invalid and oversized chunk sizes are rejected or clamped explicitly.

## 4. Cross-framework comparison

For llama.cpp comparisons:

- use the identical GGUF file and quantization;
- use the same prompt tokens, context length, and sampling parameters;
- compare logits at selected positions with a documented tolerance;
- compare greedy token sequences;
- run on the same device, power profile, and ROCm/driver version;
- record the llama.cpp build/commit and backend (HIP or Vulkan).

Differences in tokenizer or chat template must be resolved before comparing
model output. A text-only comparison is insufficient.

## 5. Test matrix

### 5.1 Correctness matrix

| Dimension | Values |
| --- | --- |
| Model | Qwen3.5-4B hybrid (primary), Qwen3.6-35B-A3B MoE (CPU and HIP), tiny `qwen35` and `qwen35moe` fixtures |
| Quantization | F16, Q8_0, Q6_K, Q5_K, Q4_K, Q5_0, Q4_0, IQ2_XXS, IQ2_S, IQ3_S |
| KV format | F32, F16, Q8_0 |
| Context | short, medium, maximum supported |
| Prompt language | English, Chinese, mixed |
| Batch | 1, small batch, continuous batch |
| Concurrency | 1, 2, 4 |
| Features | no cache, prefix cache, chunked prefill, graph replay, cancellation, speculative decoding including partial-accept rollback, routed and shared MoE experts, selected and grouped MoE kernels, native IQ codebook pinning |
| Backends | CPU reference, HIP GPU, library fallback |

`qwen35moe` runs on the CPU reference and the HIP executor; the HIP planner
rejects an expert dtype the MoE kernels cannot decode and any expert tensor
whose row width violates the block layout. Architectures other than `qwen35` and
`qwen35moe` are rejected at load time and have no test matrix entry.

NPU support is a future task and is not part of Phase 5 acceptance. The
`npu-probe` command remains a diagnostic check only.

### 5.2 Performance matrix

| Dimension | Values |
| --- | --- |
| Prefill length | 128 / 512 / 2048 / 8192 / 32768 tokens |
| Decode length | 128 / 256 tokens |
| Concurrency | 1 / 2 / 4 / 8 / 16 requests |
| Quantization | FP16, Q8_0, Q6_K, Q4_K, Q5_0, Q4_0 |
| Modes | latency and throughput |
| Comparison | llama.cpp HIP, llama.cpp Vulkan (same GGUF) |

Not every combination needs to run on every commit. For the first release, the
release gate is the recorded subset in [benchmarks.md](benchmarks.md):
prompt-length prefill curves at 128-32768 tokens and the 128/256-token decode
matrix at a 128-token prompt. The remaining concurrency and quantization rows
are tracked performance work for the features they cover; they do not block the
single-request GPU-only release. A smaller tracked subset runs regularly.

## 6. Benchmark protocol

### 6.1 Required metadata

Record at minimum:

- model name, file hash, and quantization preset;
- per-tensor dtype summary, if the preset is mixed;
- tokenizer and chat-template version;
- prompt set and token lengths;
- ROCm version, HIP runtime, kernel/driver version, and firmware;
- gfx target and device name;
- host CPU, memory configuration, and power/thermal profile;
- code commit, kernel version, and dispatch-table version;
- sampling parameters and seed;
- warmup iterations and measurement repetitions;
- concurrent request schedule and arrival pattern.

### 6.2 Procedure

1. Start from a clean process and load the model.
2. Record load time and initial memory.
3. Run warmup iterations and discard them from performance results.
4. Measure steady-state prefill and decode separately.
5. Repeat each configuration enough times to report P50/P95/P99, not a single
   run.
6. Interleave configurations where practical to reduce thermal and clock bias.
7. Record clock, power, and throttling indicators if available.
8. Stop on thermal or power throttling and mark the result accordingly.

Do not compare results collected under different power profiles, clock states,
or background load without labeling the difference.

### 6.2.1 Test inputs must expose the failure

Choose test inputs that make the mistake under test visible, and say so in the
test. A kernel that is wrong by a factor of `decay` agrees with a correct kernel
to 1.6e-5 when `decay` is 0.997 and differs by 2.6e-4 when it is 0.75, so a test
written against the first input would have accepted a wrong gated delta-net
kernel; the case is recorded in
[benchmarks.md](benchmarks.md#gated-delta-net). The rules:

- For every decay, gate, scale, or mask the kernel applies, pick at least one
  test value where applying it and not applying it differ by far more than the
  tolerance.
- Compare against an independent implementation, not only against the path being
  replaced: the two device paths can share a mistake, and one of them is usually
  the reference the other is compared against.
- State the input choice in the test so the reason survives the next edit.

### 6.3 What to measure separately

- model load and first-use warmup;
- tokenizer and prompt preparation;
- queue wait and admission;
- pure prefill GPU time;
- pure decode GPU time;
- sampling and detokenization;
- kernel launch and host scheduling overhead;
- KV-cache read cost;
- end-to-end request latency;
- model size and batch-size effects.

Isolated kernel benchmarks are necessary but not sufficient. A kernel that wins
in isolation can lose end-to-end because of launch overhead or occupancy.

Per-operation attribution is a diagnostic step, not a benchmark result.
`TestGPUProfilePrefill` runs a 512-row chunk at positions 0, 2048, and 7680
with HIP events around every plan operation. It is opt-in through
`YUUKA_TEST_MODEL`, and `YUUKA_TEST_KV=f16` selects the F16 KV pool. Use it to
rank work; confirm any accepted change with the protocol above and report it in
[benchmarks.md](benchmarks.md). The current attribution is the prefill
diagnostic described below.

`rocprofv3` cannot profile a Go process. It writes its result database from a
glibc `atexit` handler, and the Go runtime exits without running those
handlers, so a `rocprofv3` session around `yuuka`, `yuuka-server`, or `go test`
leaves no usable database. Collect per-kernel counters from the native binaries
(`build/yuuka_prefill_sweep`, `build/yuuka_gemv_sweep`,
`build/yuuka_gemv_decode_sweep`, `build/yuuka_moe_sweep`) and use the HIP event
timings from `TestGPUProfilePrefill` and `TestGPUProfileDecode` for
per-operation attribution of the engine.

`scripts/prefill_counters.sh` is the reproducible procedure for that
attribution: it verifies the recorded GGUF SHA-256 when a model is supplied,
fails when the device is not `gfx1151`, runs
`build/yuuka_prefill_sweep` once unprofiled, then traces it with
`SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES` and with `LDSBankConflict ALUStalledByLDS
LdsLatency OccupancyPercent`, optionally traces a llama.cpp reference run, and
reduces the databases to a per-MMA counter table. The
reduction lives in `scripts/prefill_counters.py` and pairs `gemm_mmq`
dispatches with sweep cells in submission order, seven submissions per cell;
it fails when the launch geometry no longer matches a cell rather than
aggregating across shapes. The reference column needs one executed MMA count
per matched reference dispatch (`--reference-mma-counts`) because only the
reference's matrix shapes give that denominator; without it the reference side
is reported per wave.

Decode attribution follows the same split. `TestGPUProfileDecode` profiles one
decode step after a real-model forward pass with HIP events around every plan
operation, resolves each operation to the model buffer it reads, and prints the
per-operation, per-tensor, and per-dtype tables plus one line per plan index
sorted by time. It is opt-in through `YUUKA_TEST_MODEL`, and it gates the result
against the recorded attribution of
[benchmarks.md](benchmarks.md#decode-gap-attribution): the plan must hold
the recorded 355 operations, the profiled total must stay inside a wide band
around the recorded 17.100 ms, and the matvec share must stay within five points
of the recorded share. The bands are wide because device clocks are not pinned,
and the recorded constants are re-recorded when a decode kernel lands, because
each change moves the total by design: 97.6 ms in the first record, 89.4 ms
after the streaming LM head, 31.6 ms after the matvec family moved to the
streaming kernel, 24.3 ms after the fused gated delta-net, 23.4 ms after the
residual and gate/up fusions, 17.3 ms after the split-key attention and the
remapped matvec, and 17.1 ms after the norm and narrow-pair rewrite, in 579
operations until the delta-net fusion folded the five recurrent operators of a
layer into two (507), the matvec fusion folded the residual add and the gate/up
pair into the projection (379), and the beta/alpha pair merge removed 24 (355).
The int8 and unfused variants are reported without the operation-count gate,
because they add quantization operations and drop fusions by design.
The operation count is exact because it is structural, so a plan change is
always visible, and a change that moves the attribution outside the bands has
to re-record the constants and the document rather than pass silently. The
per-fusion history is in [benchmarks.md](benchmarks.md#decode-fusion). Event
instrumentation adds a submission gap per operation, so the totals exceed a
release run and are a diagnosis, not a benchmark result.

`YUUKA_TEST_DECODE_UNFUSED=1` builds the same plan without the residual
epilogue and gate/up pair, which is a same-session A/B: the two runs differ only
in the fused operations, so the difference in the report is what the fusions are
worth.

`YUUKA_TEST_DECODE_INT8=fused` or `=prequantized` runs the same attribution with
the opt-in int8 decode matvec instead of the FP32 streaming kernel, which is how
the two activation sources were compared; the default is the selected route.

MoE attribution uses `TestGPUProfileMoE`, which is opt-in through
`YUUKA_TEST_MOE_MODEL` and profiles one 128-row chunk and one decode step of the
`qwen35moe` plan with the same per-operation events, aggregating by operation
code. It also prints the decode routing distribution (how many of the 256
experts the last route selected), because a selected expert re-reads its whole
packed weight set. The recorded tables and the reference comparison are in
[benchmarks/qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md#measured-attribution).
The test has no pass/fail gate: every MoE stage that has a recorded constant is
gated by the real-model tests of the same package
(`TestGPUMoEGoldenGeneration`, `TestGPUMoEChunkedPrefill`,
`TestTinyGPUMoEMatchesCPU`), and the profile is a ranking tool.

`build/yuuka_gemv_decode_sweep --recurrent` is the RECURRENT cell of the decode
matvec table: it builds one recurrent layer from the five separate operators and
from the fused pair, and prints the per-layer and 24-layer submission times with
the state bandwidth each form moves. `build/yuuka_deltanet_sweep` is the design
sweep behind the fused kernel: it times five candidate state-kernel forms (the
five-pass kernel, a read-and-write copy floor, three register-resident thread
counts, and a shared-memory-resident form) with HIP events around one launch and
reports their agreement with the five-pass kernel, so a form is chosen on
measured time and not on the layout argument.

`scripts/decode_profile.sh` is the reproducible procedure for the isolated
decode matvec table. It verifies the recorded GGUF SHA-256 when a model is
supplied, fails when the device is not `gfx1151`, runs
`build/yuuka_gemv_decode_sweep` once unprofiled (five warmups and twenty-five
timed submissions per cell, median host timing), runs the same binary in its
`--threshold`, `--int8`, and `--recurrent` modes, then traces it with
`SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS` and reduces the database with
`scripts/decode_profile.py`. The reduction pairs dispatches with sweep cells in
submission order, thirty per cell, checks that each dispatch matches the form of
its cell, and checks `SQ_WAVES` against the launch geometry before normalizing
instructions per packed weight. It then runs `TestGPUProfileDecode` and prints
the same report `go test -run TestGPUProfileDecode -v` produces. The sweep keeps
every matrix resident, so its GB/s figures are the hot-matrix upper bound of the
real step; the wall time minus the traced kernel time is the per-submission
launch cost, which is the quantity the matvec fusions target.

## 7. Metric definitions

| Metric | Definition |
| --- | --- |
| **TTFT** | Time from request arrival to the first streamed token, including queueing and prefill. Report queue wait separately if possible. |
| **TPOT** | Mean time per output token after the first token, measured over the decode phase. |
| **Inter-token latency** | Distribution of individual gaps between output tokens; report percentiles. |
| **Prefill tok/s** | Prompt tokens divided by prefill duration, excluding queue wait. |
| **Decode tok/s** | Generated tokens divided by decode duration. For batches, report aggregate and per-request values. |
| **Aggregate throughput** | Total output tokens per second across concurrent requests. |
| **Effective weight bandwidth** | Weight bytes read per second implied by decode throughput; label it as an effective estimate, not a hardware counter. |
| **Peak RAM / device memory** | Maximum resident memory during the run, including KV and workspace. |
| **Occupancy / LDS / VGPR** | Profiler-reported kernel resource usage, recorded per tracked kernel. |

## 8. Reporting template

| Field | Value |
| --- | --- |
| Date / commit | |
| Model / hash | |
| Quantization | |
| Backend / ROCm / driver | |
| Device / power profile | |
| Prompt tokens / output tokens | |
| Concurrency | |
| TTFT P50 / P95 / P99 | |
| TPOT P50 / P95 / P99 | |
| Prefill tok/s | |
| Decode tok/s (per request / aggregate) | |
| Effective weight GB/s | |
| Peak memory | |
| Correctness status | |
| Notes / throttling | |

A benchmark result without this metadata is an anecdote, not a result.

## 9. Regression gates

### 9.1 Correctness gates

- Golden token sequences must match exactly for greedy decoding.
- Logits must remain within the documented tolerance.
- Tokenizer golden tests must pass exactly.
- No test may be skipped or loosened to accommodate a kernel change without an
  explicit review decision.

### 9.2 Resource gates

- Repeated request cycles must not grow KV or scratch usage without bound.
- Cancellation and error paths must return all blocks and buffers.
- Graph replay must not retain per-request resources after completion.
- A model reload must not leak device memory.

### 9.3 Performance gates

Performance gates are defined after the first stable baseline, using measured
run-to-run variance. A change is blocked if it causes a statistically
significant regression on a tracked shape outside the accepted noise band.
Thresholds must be recorded with the baseline; do not invent a universal
percentage before measurement.

## 9.4 Current test commands

The repository uses these commands:

```bash
go build ./...
go test ./...
go test -race ./...
go test -tags hip ./...
go test -tags hip ./backend/hip ./engine/gpu ./serving ./cmd/yuuka-server
go test ./memory ./scheduler ./serving
ctest --test-dir build --output-on-failure
scripts/kernel_resources.sh
YUUKA_TEST_SPM_VOCAB=/path/to/ggml-vocab-llama-spm.gguf go test ./tokenizer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

The `hip` build tag is required for the HIP executor and its Go binding. The
default build does not require the native shared library.

Real-model tests are opt-in and skip when the variable is absent:

```bash
YUUKA_TEST_MODEL=/path/to/model.gguf go test -tags hip ./... -count=1
YUUKA_TEST_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  go test ./model/qwen35 -run Qwen35MoE -count=1
YUUKA_TEST_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  go test -tags hip ./engine/gpu -run TestGPUMoE -count=1 -timeout 60m
```

`YUUKA_TEST_MOE_ROUTE=mmq|q8` selects the routed-expert prefill route for
`TestGPUMoEGoldenGeneration`, `TestGPUMoEChunkedPrefill`, `TestGPUProfileMoE`,
`TestGPUMoEMmqMatchesQ8`, and `TestGPUChunkedPrefillLongContext` (with
`YUUKA_TEST_MODEL` pointing at the MoE GGUF). Unset and `mmq` keep the measured
int8 WMMA route; `q8` forces the Q8_1 `dp4a` route through
`Options.ForceMoEQ8`. `TestGPUMoEMmqMatchesQ8` requires both routes to produce
the same first-token logits, a difference of at most 0.05, and the golden
continuation `[11751, 11, 264, 3177]`. The assignment-tile table the dispatch
uses is pinned per row class by `TestSelectMoEPrefillPath`.

The real-model suite covers GGUF loading, tokenizer golden IDs, greedy golden
token sequences, determinism, logits tolerance against a llama.cpp CPU
reference, layer-by-layer hidden-state comparison, the paged HIP executor, KV
formats, and speculative decoding against the plain greedy path. The
`qwen35moe` CPU reference and HIP decoder are both validated on
Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf by greedy token comparison with llama.cpp; IQ
block dequantization has separate reference tests for `IQ2_XXS`, `IQ2_S`, and
`IQ3_S` on both backends. Synthetic coverage includes a tiny F32 `qwen35moe`
fixture on the CPU and HIP paths and a scripted partial-accept speculative
rollback test. Native coverage includes the grouped MoE kernels for F32 and
Q4_K, the selected and grouped pipelines for IQ2_XXS/IQ2_S/IQ3_S expert
matrices, the Q8_1 int8 expert pipeline (compared against a host computation
over the dequantized Q8_1 activations), the int8 WMMA expert pipeline for Q4_K
and Q6_K experts at assignment tiles 16, 32, 64, and 128 with partially filled
buckets (`hip_moe_mmq_correctness`, projections compared against a host
computation over the dequantized weights and the same Q8_1 values), the same
pipeline for the IQ2_XXS, IQ2_S, and IQ3_S codebooks (`hip_iq_moe_mmq_correctness`,
expert rows built from the first block of each codebook of the Qwen3.6 GGUF,
bucket sizes of 1, `J-1`, `J`, `J+1`, and `3J+5` assignments plus an empty
bucket, gate/up and down compared element by element against the dp4a route over
the same Q8_1 tiles at a 1e-5 relative bound, and the pipeline against a host
computation over the `yuuka_dequant_f16` decode at 2e-2; the IQ2_S rows carry
opposite extremes in the two nibbles of each group scale byte, so a per-32
scale application fails), and the shared-expert accumulation. The
real-model HIP tests use an F32 KV pool so the expert arithmetic is compared
without KV storage rounding.

Real-model chunked prefill is gated on token behavior, not on a tight logit
bound. Against the CPU reference the recorded maximum absolute logit difference
grows with the prefix: Qwen3.5-4B reports 0.006 at 8 tokens, 0.022 at 64, and
0.57 at 128, and Qwen3.6-35B-A3B reports 0.009 at 8 tokens, 1.1 at 128, and 0.96
and 1.04 at 64- and 256-row chunks of a 256-token prompt.
Two effects produce that drift, and neither is an MoE plan defect. First, the
device prefill uses the chunkwise delta-net form and FP16-staged GEMM tiles while
the reference runs the per-token scan in FP32, so the two paths separate as the
prefix grows; the dense model shows it without any routing. Second, this MoE
model is routing sensitive: perturbing its router weights by 1e-4 relative moves
its own logits by 0.64, so a hidden-state difference of that size can flip a
near-tie expert selection. `TestGPUMoEChunkedPrefill` therefore asserts the CPU
top-1 token, a 2.0 logit bound, and the four greedy continuation tokens, which
are the behavior the engine promises. Chunk sizes of one and two rows differ
more from the CPU reference (recorded 2.31 and 0.33 on Qwen3.5-4B, unchanged by
the MoE work), so they are not a coverage or tolerance point for either model;
the synthetic tiny fixtures retain chunk sizes 1 and 2 as plan-shape coverage.
HTTP API contract tests use the deterministic tiny fixture so they cover
streaming, errors, stops, and cancellation independently of the optional model
environment. SentencePiece golden IDs additionally accept
`YUUKA_TEST_SPM_VOCAB=/path/to/ggml-vocab-llama-spm.gguf`. Native GPU tests return CTest
skip code 77; Go GPU tests skip with a message when no HIP device is available
or `YUUKA_TEST_MODEL` is unset.

The Phase 0 hardware probe and the current CPU/HIP benchmark are run
explicitly:

```bash
./build/yuuka_hardware_probe
./build/yuuka_gemv_sweep
./build/yuuka_moe_sweep
go run ./cmd/yuuka npu-probe  # diagnostic only; NPU support is a future task
scripts/thermal_probe.sh 15 ./build/yuuka_hardware_probe
/tmp/yuuka-hip bench --model model.gguf --backend hip --prompt-tokens 16 --decode-tokens 16
/tmp/yuuka-hip bench --model model.gguf --backend hip --prompt-tokens 128 \
    --decode-tokens 32 --kv-type q8_0 --speculative
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving
/tmp/yuuka-serving-bench --model model.gguf --backend hip --ctx 256 \
    --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16
/tmp/yuuka-serving-bench --model model.gguf --backend hip --ctx 256 \
    --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16
```

The serving benchmark reports queue wait, TTFT, prefill, decode, and sampling
percentiles plus aggregate request and output-token throughput. Run one
benchmark process at a time; concurrent benchmark processes contend for the GPU
and invalidate both results.

A clean export of the tracked tree can be validated without relying on local
build artifacts:

```bash
scripts/check_clean_build.sh
scripts/check_clean_build.sh --hip
```

The first command checks formatting, vet, build, and race tests. The `--hip`
form also configures and builds the native library, runs CTest, and runs the
HIP-tagged Go tests in the exported tree.

## 10. CI and test execution

| Tier | Environment | Contents |
| --- | --- | --- |
| **Unit** | Any Linux machine | GGUF parser, tokenizer, metadata mapping, CPU reference, paged KV allocator/prefix cache, scheduler and serving state machines |
| **GPU correctness** | gfx1151 runner | Kernel tests, end-to-end generation, logits comparison |
| **GPU performance** | gfx1151 runner, controlled power profile | Tracked benchmark subset, nightly or per-release |
| **Compatibility** | gfx1151 runner | llama.cpp HIP/Vulkan comparison on the same GGUF |

Current status: the CPU tier is checked in as `.github/workflows/ci.yml` and
runs formatting, `go vet`, `go build`, and `go test -race` on pushes to `main`
and on pull requests. The gfx1151 tier is `.github/workflows/hip.yml`; it is manual and
nightly because it requires a self-hosted ROCm runner with rocBLAS and rocWMMA.
The HIP workflow builds the native library, runs `ctest`, builds the `hip` Go
tags, and runs the HIP-tagged tests. Set the `YUUKA_TEST_MODEL` repository
secret to enable the real-model cases on that runner.

Tests that require the real model skip when `YUUKA_TEST_MODEL` or
`YUUKA_TEST_MOE_MODEL` is unset. A
separate subset of tests runs without a model file by generating deterministic
tiny GGUF fixtures from `tests/fixture`, including the tiny `qwen35` GPU/CPU
comparison, the tiny `qwen35moe` CPU comparison, and server request tests.
Fixtures are generated
at test time and are not committed as model artifacts. Real-model coverage
remains opt-in.

Large model files are not committed to the repository. GPU tests must skip with
a clear message when no supported device is present; they must not silently
pass.

## 11. Milestone acceptance

This document defines how phase acceptance is measured: the correctness layers
above, the benchmark protocol, and the reporting metadata. A phase is not
complete because code merged; it is complete when its acceptance evidence is
recorded and reproducible. The phase gates are listed in
[AGENTS.md](../AGENTS.md) section 8.

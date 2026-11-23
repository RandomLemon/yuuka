# Qwen3.5-4B (`qwen35`) benchmark

**Objective.** Record the single-request `qwen35` HIP performance of
Qwen3.5-4B Q4_K_M on gfx1151: prompt-length prefill, decode, KV formats,
serving, isolated kernels, and the comparison against llama.cpp, with the
correctness gates each table was taken with.

**Status.** Every table in this file is from one measurement session on
2026-11-23 at commit `e9b606d62ef595a049f0ac0e083224d70649d1a5`. Earlier
reports are not carried forward; the numbers below replace them. The report
layout and metadata requirements are defined in
[benchmarks.md](../benchmarks.md). Numbers are observations on one machine,
not guarantees; device clock, power, and thermal state are not pinned.

## Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes; 4.21 B parameters reported by llama.cpp |
| Architecture | `qwen35`, 32 layers (8 full attention, 24 gated delta-net) |
| Quantization | Q4_K_M; the GGUF carries mixed per-tensor dtypes and no requantization is applied |
| Tokenizer / chat template | GGUF-embedded Qwen3.5 vocabulary and template |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 compute units, 122,880 MiB reported VRAM |
| ROCm / HIP | ROCm 10.1.0, HIP 7.16.26385, AMD clang 24.0.0git |
| Compiler / rocBLAS / rocWMMA | ROCm 10.1.0 toolchain; rocBLAS 5.7; rocWMMA from the same ROCm tree |
| Host | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads, 120 GiB shared memory |
| Kernel | 7.0.14-28-pve |
| amdgpu driver / firmware | not recorded; the sysfs firmware attribute is empty on this host |
| Power / thermal | GPU DPM `auto`; CPU governor `performance`; clocks not pinned; 41 °C idle at session start |
| Go | go1.24.4, `-tags hip` |
| Commit | `e9b606d62ef595a049f0ac0e083224d70649d1a5` |
| Reference implementation | upstream `llama.cpp` `b1b770d90` (build 12359, version 1.0.0-dev, ROCm backend, `-ngl 99 -b 512 -ub 512`) |
| Sampling | Greedy, temperature 0 |
| KV format | per table: F32, F16, or Q8_0 |
| Prefill chunk | 512 tokens on both engines |
| Warmup / repetitions / statistic | end-to-end: 2 warmups and 5 repetitions for prefill, 2 and 3 for decode; serving: 1 warmup plus 3 measured requests; medians |
| Concurrency | 1 request; 4 in the concurrency-4 serving row |
| Prompt set / token stream | tokenized English corpus truncated to the stated token count; llama-bench uses its own random tokens of the same length, so prompt content is length-matched, not token-matched |

Protocol:

- `yuuka bench` measures end to end through the engine. TTFT is measured from
  submission; decode throughput is `(decode_tokens - 1) / decode p50`, matching
  the CLI report.
- `bench/serving` measures the scheduler path (queue wait, TTFT, prefill,
  decode, sampling) and reports aggregate throughput over its wall interval.
- `llama-bench` measures the reference on the same model and device with
  `-ngl 99 -b 512 -ub 512`. llama.cpp uses its default F16 KV; yuuka rows state
  their KV format, so the reference comparison mixes F32 yuuka rows with F16
  llama.cpp rows. The same-session `scripts/decode_compare.sh` run is the
  like-for-like table for decode because it interleaves both engines.
- `build/yuuka_prefill_sweep` measures isolated GEMM and LM-head cells with two
  warmups and five timed submissions per cell. `build/yuuka_gemv_decode_sweep`
  measures the decode matvec family at the exact Qwen3.5 shapes one token
  dispatches, with every matrix resident, so its GB/s figures are the
  hot-matrix upper bound.
- `TestGPUProfileDecode` attributes one real decode step with HIP events around
  every plan operation; instrumentation adds a submission gap, so its absolute
  total is a diagnosis, not a release number. `TestGPUProfilePrefill` does the
  same for one 512-row F32 prefill chunk.
- `scripts/prefill_counters.sh` traces `build/yuuka_prefill_sweep` with
  `rocprofv3` and reduces the counters per executed MMA. `rocprofv3` cannot
  write a database for a Go process, so engine attribution uses HIP events
  instead.
- Only one benchmark process used the GPU at a time.

## Correctness gates

Same tree and commit as the benchmarks:

| Check | Result |
| --- | --- |
| `ctest --test-dir build --output-on-failure` (ABI, quantized GEMV, operators, GEMM, prefill, paged KV/graph, embedding, MoE, IQ MoE, MoE MMQ, IQ MoE MMQ, deltanet, plan validation) | 13 / 13 pass |
| `TestGPUChunkedPrefillLongContext`, 1024 real-model tokens, chunks 128/512/1024 | top-1 `33075` in all three; maximum logit difference 0 against the chunk-128 reference |
| `TestGPUMatchesCPU` | first-step maximum logit difference `0.000208557`; greedy tokens match |
| `go build ./...` | pass |
| `go vet ./...` | pass |
| `go test ./...` | pass |

The real-model chunked-prefill check runs with an F16 KV pool and 512-token
chunks. Native CTest covers the FP32 and int8 prefill kernels, the LM-head
dtypes, paged KV and graph replay, the delta-net kernels, and plan validation.

## Prefill: prompt length

`yuuka bench --backend hip --chunk-size 512 --decode-tokens 1 --warmup 2
--reps 5 --kv-type <format>`, greedy, one request:

| Prompt | F32 p50 | F32 tok/s | F16 p50 | F16 tok/s | Q8_0 p50 | Q8_0 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.060 s | 2122.2 | 0.061 s | 2099.4 | 0.064 s | 2014.0 |
| 512 | 0.198 s | 2588.4 | 0.197 s | 2593.4 | 0.225 s | 2275.0 |
| 2048 | 0.817 s | 2506.8 | 0.813 s | 2520.2 | 0.871 s | 2352.1 |
| 8192 | 3.898 s | 2101.4 | 3.882 s | 2110.1 | 3.891 s | 2105.1 |

Throughput peaks at 512 tokens (2.59 k tok/s) and falls to about 2.1 k tok/s at
8192 as attention grows. F16 is within 1.1% of F32 at every length. Q8_0 costs
12.1% at 512 tokens and 6.2% at 2048; at 8192 the three formats are within
noise. The longest verified prompt is 8192 tokens at chunk 512.

## Decode: output length

`yuuka bench --chunk-size 512 --prompt-tokens 128 --decode-tokens 128|256
--warmup 2 --reps 3 --kv-type <format>`, greedy:

| KV | Decode tokens | Context | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | 128 | 320 | 0.060 s | 0.060 s | 1.971 s | 64.42 |
| F32 | 256 | 448 | 0.060 s | 0.060 s | 3.962 s | 64.35 |
| F16 | 128 | 320 | 0.060 s | 0.060 s | 1.978 s | 64.21 |
| F16 | 256 | 448 | 0.060 s | 0.060 s | 3.960 s | 64.39 |
| Q8_0 | 128 | 320 | 0.065 s | 0.065 s | 2.005 s | 63.35 |
| Q8_0 | 256 | 448 | 0.065 s | 0.065 s | 3.999 s | 63.77 |

Decode is flat in output length: 128 and 256 tokens differ by at most 0.7% in tok/s.
F16 costs 0.3% at 128 tokens and Q8_0 costs 1.7%. The profiled step reads
2,726.1 MB of packed matmul weights at 179.2 GB/s effective; the matmul family
is 15.214 ms of the 16.941 ms step (89.8%). Decode remains a single-request
weight-bandwidth limit.

## Feature matrix

### Speculative decoding

A 512-token repetitive prompt with 128 generated tokens, F32 KV, prompt-lookup
drafting with `--speculative`:

| Route | Decode p50 | Decode tok/s | TTFT p50 | Drafts | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain greedy | 2.007 s | 63.29 | 0.212 s | - | - |
| Speculative, max draft 4 | 1.216 s | 104.47 | 0.250 s | 128 | 128 / 128 |

Speculative decoding is 1.65x faster on this repetitive prompt with full draft
acceptance. TTFT rises by 38 ms because the first draft is prepared inside the
request. The feature stays opt-in: the gain depends on prompt-lookup acceptance
and this prompt is the best case.

### KV cache formats

F16 is a no-regret replacement for F32 on this model: prefill and decode are
within 1%. Q8_0 is the memory-saving option at a measured cost of 12.1% prefill
at 512 tokens and 1.7% decode at 128 generated tokens. All three formats pass
the real-model chunked-prefill gate.

## Serving

`bench/serving`, HIP backend, context 256, block size 16, 64-token prompt, 16
generated tokens, greedy, F32 KV, p50 of three measured requests (eight for
concurrency 4):

| Configuration | Latency p50 | Queue wait p50 | TTFT p50 | Prefill p50 | Decode p50 | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single request, graph replay | 0.2872 s | 0.000003 s | 0.0494 s | 0.0464 s | 0.2341 s | 55.71 |
| Single request, plain stream | 0.2855 s | 0.000002 s | 0.0495 s | 0.0465 s | 0.2325 s | 56.01 |
| Concurrency 4, graph replay | 1.0844 s | 0.0498 s | 0.1599 s | 0.0469 s | 0.2339 s | 55.62 aggregate |
| Single request, prefix cache | 0.2415 s | 0.000002 s | 0.0044 s | 0 s | 0.2346 s | 61.13 |

Graph replay is within 1% of the plain stream path at this context, so it is
not a default. Prefix cache reports 2 hits and 1 miss over the three requests:
cached TTFT falls from 49.5 ms to 4.4 ms and aggregate output throughput rises
9.1%. Concurrency 4 admits four requests and exposes 49.8 ms of queue wait, but
aggregate throughput is 55.62 tok/s against 56.01 for one request: per-request
decode time is unchanged because each request reads the full quantized weights
on its own. Fused multi-request decode batching is not implemented.

The OpenAI HTTP server path was not re-measured in this session; only the
in-process scheduler is tabulated. This is recorded in Known limits.

## Isolated kernels

### Prefill GEMM

`build/yuuka_prefill_sweep`, M = 512, median of five timed submissions after two
warmups. K-quants select the int8 MMQ path; Q8_0 selects the WMMA path. TFLOP/s
is the achieved product rate:

| dtype | K -> N | Path | ms | TFLOP/s |
| --- | --- | --- | ---: | ---: |
| Q4_K | 2560 -> 9216 | MMQ | 0.972 | 24.8 |
| Q4_K | 2560 -> 8192 | MMQ | 0.865 | 24.8 |
| Q4_K | 9216 -> 2560 | MMQ | 1.084 | 22.3 |
| Q5_K | 2560 -> 9216 | MMQ | 1.476 | 16.4 |
| Q5_K | 2560 -> 8192 | MMQ | 0.901 | 23.8 |
| Q5_K | 9216 -> 2560 | MMQ | 1.359 | 17.8 |
| Q6_K | 2560 -> 9216 | MMQ | 1.719 | 14.1 |
| Q6_K | 2560 -> 8192 | MMQ | 1.532 | 14.0 |
| Q6_K | 9216 -> 2560 | MMQ | 1.378 | 17.5 |
| Q8_0 | 2560 -> 9216 | WMMA | 2.646 | 9.1 |
| Q8_0 | 2560 -> 8192 | WMMA | 2.368 | 9.1 |
| Q8_0 | 9216 -> 2560 | WMMA | 2.870 | 8.4 |

The Q4_K MMQ cell reaches 24.8 TFLOP/s at both wide projection shapes. Q5_K is
close at 23.8 on `2560 -> 8192` but drops to 16.4-17.8 on the other two. Q6_K
and Q8_0 are the slower paths; Q8_0 has no MMQ kernel and runs the FP16 WMMA
form.

Counters for the Q4_K `M = 512, 2560 -> 9216` cell from
`scripts/prefill_counters.sh`: 17.06 VALU and 2.84 LDS instructions per MMA,
0.00% LDS bank conflicts, 0.67% ALU stalls attributed to LDS, 68.02 cycles of
LDS latency, 46.83% occupancy, 8 waves per workgroup, and 24.80 TFLOP/s. The
reference counters could not be normalized per MMA in this run because no
executed-MMA counts were supplied for the llama.cpp dispatches, so only the
yuuka column is reported.

### Decode matvec

`build/yuuka_gemv_decode_sweep`, hot matrix, median of twenty-five timed
submissions after five warmups. `Streaming` is the selected `lm_head` form,
`per-row` is the one-row GEMV, and `batched` reuses weights across four rows:

| Tensor | dtype | K -> N | Streaming ms | GB/s | Per-row ms | Batched ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `ffn_gate` | Q4_K | 2560 -> 9216 | 0.0557 | 238.1 | 0.3505 | 0.5907 |
| `ffn_down` | Q6_K | 9216 -> 2560 | 0.0576 | 336.2 | 0.2523 | 0.5192 |
| `attn_q` | Q4_K | 2560 -> 8192 | 0.0538 | 219.2 | 0.3153 | 0.5255 |
| `attn_qkv` | Q5_K | 2560 -> 8192 | 0.0659 | 218.7 | 0.3547 | 0.5618 |
| `ssm_out` | Q5_K | 4096 -> 2560 | 0.0366 | 196.8 | 0.1848 | 0.2534 |
| `lm_head` | Q6_K | 2560 -> 248320 | 2.2286 | 234.0 | 11.2198 | 17.8643 |

The streaming form is selected for every projection in the decode plan. The
isolated figures are the hot-matrix upper bound; inside the real step the same
matvec family moves 179.2 GB/s because every matmul weight matrix is cold in the
32 MB MALL by the time a token reaches it.

### Int8 decode matvec

`build/yuuka_gemv_decode_sweep --int8`, same shapes. `FP32` is the selected
streaming kernel; `prequantized` submits one Q8_1 pass per activation group;
`fused` quantizes inside the matvec block:

| Tensor | FP32 ms | Prequantized ms | Fused ms |
| --- | ---: | ---: | ---: |
| `ffn_gate` Q4_K 2560 -> 9216 | 0.0584 | 0.0591 | 0.0701 |
| `ffn_down` Q6_K 9216 -> 2560 | 0.0591 | 0.0703 | 0.1134 |
| `attn_q` Q4_K 2560 -> 8192 | 0.0546 | 0.0586 | 0.0687 |
| `lm_head` Q6_K 2560 -> 248320 | 2.2300 | 2.4456 | 2.8490 |

The int8 routes are slower on these cells, and the engine attribution confirms
it end to end: the prequantized route runs 636 operations and 18.081 ms per
decode step against 355 operations and 16.941 ms for FP32. The int8 matvec stays
opt-in and is not the default.

### Decode fusion

`TestGPUProfileDecode` with and without `YUUKA_TEST_DECODE_UNFUSED=1`, one
decode step after the same real forward pass:

| Plan | Operations | Matmul family | Delta-net state and convolution | Total |
| --- | ---: | ---: | ---: | ---: |
| Fusions off | 507 | 14.931 ms | 0.647 ms | 17.124 ms |
| Fusions on | 355 | 15.214 ms | 0.655 ms | 16.941 ms |

The residual epilogue and the gate/up pair remove 152 operations. The fused
matmul family is 0.283 ms slower in isolation, but the total step is 0.183 ms
(1.1%) faster because the removed launches and the smaller sampling of the
delta-net state dominate. The fused plan is the selected one.

### Gated delta-net

`build/yuuka_gemv_decode_sweep --recurrent`, one recurrent layer, 32 heads, 16
groups, 128-wide state, 24 layers:

| Path | Operations | ms/layer | 24 layers | State GB/s |
| --- | ---: | ---: | ---: | ---: |
| Five separate operators | 5 | 0.3328 | 8.0 ms | 31.5 |
| Fused pair | 2 | 0.0169 | 0.4 ms | 247.7 |

`build/yuuka_deltanet_sweep` confirms the register-resident form on the same
shape: 0.0111 ms per layer and 377.8 GB/s for `reg4`, against 0.0335 ms and
125.3 GB/s for the shared-memory form and 0.3022 ms for the legacy five-pass
kernel. The fused pair is the selected path; the chunkwise prefill form is not
selected.

### Prefill attention

`TestGPUProfilePrefill`, one 512-row F32 chunk on Qwen3.5 geometry, one
`ATTENTION_PREFILL_PAGED` op per full-attention layer, HIP events:

| Position | Attention, 8 layers | Share of chunk | Chunk wall |
| ---: | ---: | ---: | ---: |
| 0 | 5.070 ms | 2.4% | 209.854 ms |
| 2048 | 30.492 ms | 13.1% | 231.566 ms |
| 7680 | 85.496 ms | 29.8% | 286.809 ms |

Attention grows 16.9x from the empty prefix to position 7680 and becomes the
second-largest component of a long-context chunk after the matmuls. This run
records the selected route in step; it does not repeat the isolated tiled-vs-MMA
crossover.

## llama.cpp comparison

Token-rate cells from one `scripts/decode_compare.sh` session, both engines
interleaved on the same model and device, three measured repetitions after two
warmups. yuuka uses F32 KV; llama.cpp uses its default F16 KV:

| Cell | yuuka prefill tok/s | llama.cpp prefill tok/s | yuuka decode tok/s | llama.cpp decode tok/s | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp128 / tg128 | 2031.6 | 1775.9 | 63.96 | 61.84 | 1.03 |
| pp128 / tg32 | 2042.7 | 1773.6 | 63.97 | 61.41 | 1.04 |
| pp128 / tg512 | 2010.4 | 1772.6 | 63.75 | 62.01 | 1.03 |
| pp512 / tg128 | 2554.5 | 2147.5 | 64.00 | 62.08 | 1.03 |
| pp2048 / tg32 | 2505.6 | 2142.4 | 62.05 | 61.66 | 1.01 |

The standalone prefill matrix above reaches 2.12-2.59 k tok/s; the same-session
interleaved cells are up to 4% lower because each cell runs a full process. yuuka
decode is 1-4% ahead of the reference across these cells. The reference
comparison is length-matched, not token-matched: yuuka tokenizes an English
corpus to the requested length, while `llama-bench` generates its own random
tokens. The prefill table is therefore a protocol comparison, not an
identical-token comparison.

## Decode gap attribution

One decode step at a 64-token context, yuuka per-tensor attribution joined with
a `rocprofv3` trace of the reference decode loop by `(dtype, K, N)`. `blocks`
counts the matmul weight blocks each engine reads per step:

| Tensor | dtype | K -> N | yuuka GB/s | reference GB/s | Reference / yuuka |
| --- | --- | --- | ---: | ---: | ---: |
| `ffn_gate.weight` | Q4_K | 2560 -> 9216 x2 | 190.9 | 218.7 | 1.15 |
| `token_embd.weight` | Q6_K | 2560 -> 248320 | 233.0 | 233.4 | 1.00 |
| `attn_qkv.weight` | Q5_K | 2560 -> 8192 | 182.1 | 219.1 | 1.20 |
| `ffn_down.weight` | Q6_K | 9216 -> 2560 | 194.8 | 191.4 | 0.98 |
| `ffn_down.weight` | Q4_K | 9216 -> 2560 | 177.1 | 178.6 | 1.01 |
| `ssm_out.weight` | Q5_K | 4096 -> 2560 | 169.9 | 208.3 | 1.23 |
| `attn_gate.weight` | Q4_K | 2560 -> 4096 | 154.1 | 201.4 | 1.31 |
| `attn_q.weight` | Q4_K | 2560 -> 8192 | 170.1 | 211.5 | 1.24 |
| `attn_output.weight` | Q4_K | 4096 -> 2560 | 154.8 | 208.3 | 1.35 |
| `ssm_beta.weight` | Q8_0 | 2560 -> 32 x2 | 18.1 | 18.8 | 1.04 |
| `attn_k.weight`, `attn_v.weight` | Q4_K | 2560 -> 1024 | 109.1 | 173.2 | 1.59 |
| `attn_v.weight` | Q6_K | 2560 -> 1024 | 137.2 | 185.9 | 1.35 |

The gap is concentrated in the small attention projections: the reference
reads `attn_k`/`attn_v` at 173.2 GB/s against 109.1, and `attn_output` at 208.3
against 154.8. The large FFN and embedding matrices are at parity (0.98-1.15x).
The engine step moves 2,837.0 MB at 167.5 GB/s overall; the matmul family is
15.214 ms (89.8%), delta-net state and convolution 0.655 ms (3.9%), and decode
attention 0.242 ms (1.4%). At 512 tokens attention is 0.0345 ms per layer; at
2048 tokens it is 0.0694 ms with F16 KV and 0.0599 ms with Q8_0.

The reference trace is a diagnostic, not a release measurement: it uses
`llama-bench -p 0 -n 32 -r 2`, and the event instrumentation in the yuuka
profile adds a submission gap per operation. The per-shape ratios are the
durable part, not the absolute milliseconds.

## Decisions and rejected routes

Each decision below is supported by the tables in this report:

- **The FP32 streaming matvec is the decode default.** It beats the one-row
  GEMV and the batched form on every projection, and both int8 forms are slower
  in isolation and in the engine step (18.081 ms and 636 operations against
  16.941 ms and 355). The int8 route stays opt-in.
- **The streaming LM head is selected for the vocabulary projection.** At
  K=2560, N=248320 it takes 2.229 ms against 11.220 ms for the per-row GEMV and
  17.864 ms for the four-row batch.
- **The fused gate/up and residual epilogues stay.** They remove 152 operations
  and 0.183 ms per step; the fused matmul family is 0.283 ms slower in
  isolation, so the gain comes from the removed launches and the smaller state
  pass, not from the matmul itself.
- **The fused delta-net pair stays.** It is 19.7x faster per layer than the five
  separate operators (0.0169 against 0.3328 ms) and moves state at 247.7 GB/s
  against 31.5.
- **Graph replay is not a default.** It is within 1% of the plain stream path
  at the measured context (55.71 against 56.01 output tok/s). It remains
  available for contexts where it is re-measured to help.
- **Concurrency 4 does not raise aggregate throughput.** Four requests reach
  55.62 tok/s against 56.01 for one, because decode is weight-bandwidth bound
  and fused mixed-shape batching does not exist. Admission and queueing are
  correct, but the throughput claim stops at one request.
- **Speculative decoding is opt-in.** It reaches 104.47 tok/s against 63.29 on a
  repetitive prompt with full acceptance; that prompt is the best case, not the
  average.
- **F16 KV is the no-regret format; Q8_0 is the memory-saving one.** F16 is
  within 1% of F32 in prefill and decode. Q8_0 costs 12.1% prefill at 512
  tokens and 1.7% decode at 128 output tokens.
- **Prefill attention grows with the prefix.** It is 5.07 ms of a 512-row chunk
  at position 0 and 85.50 ms at position 7680, where it is 29.8% of the chunk.
  Long-context prefill work remains attention-bound after the matmuls.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving
MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf
sha256sum "$MODEL"   # must match 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4

ctest --test-dir build --output-on-failure
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run 'TestGPUChunkedPrefillLongContext|TestGPUMatchesCPU' -count=1 -v

# Prefill and decode matrices.
for kv in f32 f16 q8_0; do
  for p in 128 512 2048 8192; do
    /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
      --prompt-tokens "$p" --decode-tokens 1 --warmup 2 --reps 5 --kv-type "$kv"
  done
  for d in 128 256; do
    /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
      --prompt-tokens 128 --decode-tokens "$d" --warmup 2 --reps 3 --kv-type "$kv"
  done
done
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 512 --decode-tokens 128 --warmup 1 --reps 3 --kv-type f32 --speculative

# Serving and prefix cache.
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 64 --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 0 --graph --prefix-cache

# Reference and same-session decode comparison.
scripts/decode_compare.sh --model "$MODEL" \
  --llama-bench /path/to/llama.cpp/build/bin/llama-bench \
  --cells "128:128,128:32,128:512,512:128,2048:32" --reps 3 --warmup 2

# Isolated sweeps and attribution.
./build/yuuka_prefill_sweep
./build/yuuka_gemv_decode_sweep
./build/yuuka_gemv_decode_sweep --int8
./build/yuuka_gemv_decode_sweep --recurrent
./build/yuuka_deltanet_sweep
scripts/prefill_counters.sh --model "$MODEL"
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfileDecode -count=1 -v
YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_DECODE_UNFUSED=1 \
  go test -tags hip ./engine/gpu -run TestGPUProfileDecode -count=1 -v
YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_DECODE_INT8=prequantized \
  go test -tags hip ./engine/gpu -run TestGPUProfileDecode -count=1 -v
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfilePrefill -count=1 -v
```

`scripts/decode_profile.sh` is not part of the table above: its traced
dispatch/cell pairing failed in this run because the `lm_head` cell dispatched
`gemv_row_kernel` while the reducer expected the streaming form. The unprofiled
sweep and `TestGPUProfileDecode` cover the same shapes without that pairing.

## Known limits

- One model, one quantization, one device. Qwen3.5-4B at Q4_K_M on gfx1151;
  other GGUFs, dtypes, and devices are not measured.
- The longest verified prompt is 8192 tokens at chunk 512. No 32768-token row
  was run.
- The llama.cpp comparison is length-matched, not token-matched, and mixes F32
  yuuka KV with llama.cpp's default F16 KV. Both are labeled in the tables; do
  not read the ratios as identical-input results.
- The OpenAI HTTP server path was not re-measured; the serving table is the
  in-process scheduler only.
- The prefill attention table is in-step attribution, not an isolated
  tiled-vs-MMA crossover. The isolated crossover from earlier runs is not
  carried forward.
- Kernel resource usage (VGPR, LDS, occupancy) was compiled but not tabulated in
  this report.
- Device clocks and thermal state are not pinned. Numbers are medians of one
  session, not guarantees.
- No fused mixed-shape multi-request decode batching exists; concurrency-4
  aggregate throughput is therefore a scheduler result, not a batching result.

# Benchmark Report: Qwen3.5-4B Q4_K_M

This file is the current performance record for the repository: every table
below was measured on the code in the tree at the commit listed under
[Environment and protocol](#environment-and-protocol), with the stated
protocol. Superseded phase-by-phase reports are kept in git history and are not
repeated here. Numbers are observations on one machine, not guarantees; device
clock, power, and thermal state are not pinned.

## Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-18 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes |
| Architecture | `qwen35`, 32 layers (8 full attention, 24 gated delta-net) |
| Quantization | Q4_K_M (the GGUF carries mixed per-tensor dtypes; no requantization is applied) |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs, 122,880 MiB reported VRAM |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads, 120 GiB shared memory |
| Kernel | 7.0.14-28-pve |
| Power performance level | `auto` (amdgpu sysfs) |
| Go | go1.24.4, Release `cmake --build build` |
| Commit | `d56ad88` |
| Tokenizer / chat template | GGUF-embedded Qwen3.5 vocabulary and template |
| Sampling | Greedy, temperature 0 |
| Prefill chunk | 512 tokens (`--chunk-size 512`), except where a row states otherwise |
| Warmup / repetitions | 2 / 5 for prefill rows, 2 / 3 for decode and serving rows, median reported |

Tables re-measured at the commit above are the prompt-length prefill rows, the
correctness gates, the prefill LM head, and the kernel resource usage. The decode
matvec table was measured at `bfd36c0`, which differs from the commit above only
in documentation. The
decode, KV-format, attention, isolated GEMM, batched GEMV, MoE, gated delta-net,
serving, server, and llama.cpp comparison tables were measured at `161fbba` and
were not repeated: the LM head change is confined to the prefill tail and does
not touch the decode dispatch. `scripts/prefill_counters.sh` was rerun and still
reproduces the recorded MMQ GEMM instruction counters.

Protocol:

- `build/yuuka_prefill_sweep` measures isolated kernels (GEMM, GEMV) per dtype,
  shape, and row count; each cell is the median of five timed submissions after
  two warmups, and the K-quant `mmq` cell includes the Q8_1 activation pass.
- `yuuka bench` measures end to end through the engine. TTFT is measured from
  submission; decode throughput is `(decode_tokens - 1) / decode p50`, matching
  the CLI report.
- `bench/serving` measures the scheduler path (queue wait, TTFT, prefill,
  decode, sampling) and reports aggregate throughput over its wall interval.
- The server comparison drives the OpenAI-compatible HTTP surface of
  `yuuka-server` and `llama-server` with a stock streaming client on the same
  host, so TTFT includes HTTP and SSE overhead. Prompt caching is disabled on
  both sides (`--prefix-cache=false`, `"cache_prompt": false`).
- `scripts/prefill_counters.sh` reproduces the per-MMA prefill counters
  (instruction mix, bank conflicts, occupancy) recorded in
  [kernels.md](kernels.md#63-int8-mmq-gemm). It profiles the
  native sweep binary only: `rocprofv3` cannot write its database for a Go
  process, as recorded in [testing.md](testing.md#63-what-to-measure-separately).
- `scripts/decode_profile.sh` reproduces the [decode matvec](#decode-matvec)
  table in one command: it runs `build/yuuka_gemv_decode_sweep` unprofiled and
  under `rocprofv3`, runs its `--threshold` and `--int8` modes for the output
  projection crossover and the [int8 comparison](#int8-decode-matvec), then
  runs `TestGPUProfileDecode` for the per-operation attribution of one real
  decode step. The sweep keeps every matrix resident, so its GB/s figures are the
  hot-matrix upper bound of the real step.
- `scripts/decode_compare.sh` reproduces the [llama.cpp comparison](#llamacpp-comparison)
  and the [decode gap attribution](#decode-gap-attribution) against the
  reference in one command: `yuuka bench` and `llama-bench` on the same GGUF at
  the recorded prompt/decode cells, then the yuuka per-tensor attribution of
  one decode step joined by `(dtype, K, N)` with a `rocprofv3` trace of the
  reference decode loop (`scripts/decode_reference_shapes.py`). It fails when
  the device is not `gfx1151` or the model hash differs from the recorded one.
  Raw traces stay in a temporary directory and never enter the repository.
- Only one benchmark process may use the GPU at a time; two processes sharing
  the device invalidate both results.
- The llama.cpp reference is `strix-llama.cpp` commit `c8912d7f5` (build 12216,
  version 0.6.0-dev), ROCm backend, `-ngl 99 -b 512 -ub 512`. `llama-bench`
  rejects an `f32` KV cache type, so its rows use llama.cpp's default `f16` KV;
  yuuka rows state their KV format.

## Correctness gates

Current tree, same commit:

| Check | Result |
| --- | --- |
| `ctest --test-dir build` (native ABI, quantized GEMV, operators, GEMM, prefill, paged KV/graph, embedding, MoE) | 9 / 9 pass |
| `TestGPUChunkedPrefillLongContext`, 1024 real-model tokens, chunks 128/512/1024 | top-1 `33075` in all three; maximum logit difference 0 against the chunk-128 reference |
| `TestGPUMatchesCPU` | first-step maximum logit difference `0.00309944`, greedy tokens match |
| `tests/hip/prefill_test.cpp` MMQ checks | Q8_1 digits identical to the host quantizer; GEMM maximum absolute difference `6.9e-5` (Q4_K) and `1.5e-4` (Q5_K) against the CPU product |
| `tests/hip/prefill_test.cpp` LM head checks | every dtype within `1.3e-4` of the host dot products for 1/3/16 rows, bitwise reproducible |
| `TestPlanLMHeadMatchesDequant` | Q8_0 bitwise equal to the dequantized CPU dot; Q6_K within `2e-6` |
| `yuuka run` at 101 / 501 / 2001 prompt tokens | generated tokens identical before and after the LM head change |
| `go build ./...`, `go vet ./...`, `go test ./...`, `go test -tags hip ./...` | pass |

The Q8_1 prefill path moves logits against an FP16-activation reference
(measured earlier at 0.0031 maximum for the tiny prompt and 0.264 for a
2048-token prompt, top-1 preserved). Chunk sizes stay inside the 0.05 chunk
gate; the quantization grid itself is the source of that delta, and it is the
regime llama.cpp's K-quant prefill also runs in.

## Prefill: prompt length

`yuuka bench --backend hip --decode-tokens 1 --chunk-size 512 --kv-type f32`,
F32 KV unless stated, median of five repetitions:

| Prompt tokens | F32 KV p50 | F32 tok/s | F16 KV p50 | F16 tok/s | Q8_0 KV p50 | Q8_0 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.085 s | 1511.8 | 0.095 s | 1344.4 | 0.096 s | 1326.9 |
| 512 | 0.292 s | 1756.2 | 0.313 s | 1634.8 | 0.330 s | 1553.6 |
| 2048 | 1.201 s | 1705.6 | 1.248 s | 1641.5 | 1.524 s | 1343.7 |
| 8192 | 5.697 s | 1437.9 | 5.696 s | 1438.2 | 9.480 s | 864.1 |

Measured 2026-11-18 after the streaming prefill LM head
([kernels.md](kernels.md#65-streaming-lm-head)). The F32 column is a same-session
comparison: back-to-back runs at `aec0570` with the same warmup and repetitions
gave 1345.1 / 1626.2 / 1673.9 / 1409.2 tok/s at the four lengths, so the stage is
+12.4% / +8.0% / +1.9% / +2.0%. The gain follows the 8.4 ms the tail saves per
chunk, which is a larger share of the shorter prompts. F32 and F16 KV are within
4% of each other at every length, because both run the same attention kernel;
Q8_0 keeps the tiled kernel and diverges from 2048 tokens, where attention
starts to dominate. The F16 and Q8_0 rows are the pre-LM-head measurements and
were not repeated.

Before this stage the same table read 1380.1 / 1704.8 / 1715.2 / 1441.7 tok/s for
F32 KV at `161fbba`. That earlier row was measured after the FP16 MMA prefill
attention kernel ([kernels.md](kernels.md#75-fp16-mma-paged-prefill-attention)),
which had moved the same F32 KV lengths from 1609.1 / 1415.4 / 877.7 to
1704.8 / 1715.2 / 1441.7 at 512 / 2048 / 8192 tokens. Its values differ from the
current ones by 0.5-1.6% at the longer prompts, which is the run-to-run noise of
an unpinned clock on this device, so the same-session A/B above is the number to
compare the LM head stage against.

The [Q4_K FFN specialization acceptance](prefill-optimization/acceptance/02-q4-ffn.md)
records the 2026-11-19 gfx1151 prefill comparison using F16 KV,
including isolated MMQ, complete FFN and prompt-length medians.
The [final FFN acceptance](prefill-optimization/acceptance/03-final-ffn.md)
records the next F16 KV comparison, with one FFN row in the ordinary
prefill output tail.
The [MMQ shape expansion](prefill-optimization/acceptance/04a-mmq-shapes.md)
records projection shape medians and the short- and long-prompt A/B.
The [prepared GDN comparison](prefill-optimization/acceptance/04b-gdn.md)
records the recurrence timings and the model-state comparison by lane mapping.
The [FFN fusion comparison](prefill-optimization/acceptance/05-ffn-fusion.md)
records the isolated quantizer check, complete FFN time and prompt rates.
The [final query crop comparison](prefill-optimization/acceptance/06-final-query.md)
records K/V equivalence, output work per chunk and the resulting prompt rates.
The [long attention comparison](prefill-optimization/acceptance/07-long-attention.md)
records split-key numerics, merge costs, scratch and the 8K chunk sweep.
The [official v1.0.0 matched-input comparison](prefill-optimization/acceptance/08-upstream-comparison.md)
records host logit latency and the per-length gate. At measured chunk settings
the prefill leads were +0.61%, +6.61%, +9.04% and +6.69% for 128, 512,
2048 and 8192 tokens respectively.

## Decode: output length

`yuuka bench --backend hip --chunk-size 512 --prompt-tokens 128
--decode-tokens 128|256 --kv-type f32`, F32 KV, greedy, warmup 2 and median of 3
repetitions, 2026-11-19 on gfx1151 with `Qwen3.5-4B-Q4_K_M`:

| Decode tokens | Context | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 320 | 0.068 s | 0.068 s | 2.083 s | 60.96 |
| 256 | 448 | 0.072 s | 0.072 s | 4.206 s | 60.62 |

Decode is weight-bandwidth bound and flat in output length: the two rows differ
by 0.6%. The 2,726 MB of packed weights move at 165 GB/s effective, and the
profiled step puts the matvec family at 15.49 ms of 17.16 ms, 176 GB/s of the
same bytes; the rest is the norm, delta-net, attention, RoPE, and KV tail
([decode gap attribution](#decode-gap-attribution)). The decode rows measure the
selected route, which is the FP32 remapped matvec (the int8 route is measured
slower and stays opt-in; see [int8 decode matvec](#int8-decode-matvec)).

The table was re-measured seven times, each time a decode kernel landed: the
streaming LM head, the rest of the matvec family, the fused gated delta-net, the
residual and gate/up epilogues, the split-key attention, the remapped matvec, and
the norm/tail rewrite. Same flags, 128 decode tokens: 13.241 s, 12.079 s,
4.245 s, 3.341 s, 3.280 s, 2.111 s, 2.083 s. The profiled step went 97.4, 89.4,
31.6, 24.3, 23.4, 17.3, 17.1 ms with the matvec family at 87.6, 79.2, 21.6,
21.5, 21.1, 15.4, 15.5 ms. Prefill is unaffected: 0.068-0.088 s across all
seven.

### Decode fusion

`TestGPUProfileDecode` with and without `YUUKA_TEST_DECODE_UNFUSED=1`, same
session, same plan input, one decode step after a real forward pass. This A/B is
recorded at the plan shape of its session, before the split-key attention and the
matvec remap; the test now reports the current counts (355 with fusions, 379
without). The saved operations are structural, so the comparison still holds,
while the absolute times are historical:

| Plan | Operations | Matvec | Everything else | Step |
| --- | ---: | ---: | ---: | ---: |
| fusions off | 507 | 21.452 ms | 3.01 ms | 24.461 ms |
| fusions on | 379 | 21.104 ms | 2.39 ms | **23.492 ms** |

The residual epilogue replaces 64 ADD operations, and the gate/up pair replaces
64 operations (two matvecs and a SILU_MUL) with 32. Both plans produce
bit-identical logits over eight greedy steps
(`TestGPUDecodeFusionsMatchUnfused`, maximum difference 0).

`TestGPUGraphReplayMatchesAndTimes`, 32 decode steps: 22.398 ms per step on the
plain submission path against 22.385 ms with the plan captured into a HIP Graph,
a ratio of 0.999 with identical tokens. Replay removes every per-operation host
submission, so the inter-operation gap is below the resolution of that comparison,
which is what the matvec fusion targeted.

Variants measured and rejected for the same kernel:

| Variant | Step | Matvec | Verdict |
| --- | ---: | ---: | --- |
| fused delta-net tree (before the fusion) | 24.26 ms | 21.45 ms | - |
| eight output channels per wave | 25.9 ms | 21.8 ms | 28% slower, spills |
| `kStepK` 512 | 24.2 ms | 21.8 ms | 7% slower, staging outweighs barriers |
| pair with a run-time dtype switch | 26.2 ms | 23.9 ms | 40% slower than the templated pair |

### Gated delta-net

`build/yuuka_deltanet_sweep`: one recurrent layer at the decode geometry (32
heads, 16 groups, 128-wide state), each candidate kernel timed with HIP events
around a single launch, five warmups and twenty-five submissions. The state is
resident, so this is the hot upper bound; the in-step figures below are lower. The
five-pass form reads the state three times and writes it twice, the fused forms
once each, which is the traffic the GB/s column uses.

| Form | ms/layer | State GB/s | x five-pass |
| --- | ---: | ---: | ---: |
| five-pass kernel (legacy) | 0.3023 | 34.7 | 1.00 |
| copy floor (read and write once) | 0.0177 | 236.8 | 17.1 |
| register, 128 threads/head | 0.0133 | 316.2 | 22.8 |
| register, 256 threads/head | 0.0117 | 357.2 | 25.8 |
| register, 512 threads/head (selected) | 0.0112 | 375.1 | 27.0 |
| register, 512 threads/head, no gated RMSNorm | 0.0108 | 387.6 | 27.9 |
| shared memory, 128 threads/head | 0.0334 | 125.5 | 9.0 |

The register form wins because the 64 KB state cannot share the workgroup budget
with the normalization scratch the shared form would need, and because 512
threads give the memory pipeline more independent loads per lane than the copy
kernel itself: the selected form is faster than a read-and-write copy of the same
bytes, which puts the limiter at memory-level parallelism rather than bandwidth.

The same layer inside a decode step (`TestGPUProfileDecode`), per token:

| Operation | Ops | ms | Share |
| --- | ---: | ---: | ---: |
| `recurrent_token` | 24 | 0.467 | 1.9% |
| `conv_l2norm` | 24 | 0.328 | 1.3% |
| delta-net unit | 48 | 0.790 | 3.3% |
| the five operators it replaced | 120 | 8.177 | 25.9% |

The recurrent state alone moves 100.7 MB per token at 217 GB/s inside the step,
against 34.7 GB/s for the five-pass form. The operation count of the whole decode
step fell from 579 to 507.

`build/yuuka_gemv_decode_sweep --recurrent` is the reproducible RECURRENT cell: it
builds the same layer from both plans and reports 0.3371 ms per layer for the five
operators against 0.0168 ms for the fused pair, that is 8.1 ms against 0.40 ms for
24 layers, at 249.5 GB/s of state traffic.

Numerics: `tests/hip/deltanet_test.cpp` bounds both the fused pair and the five
separate operators against an independent host reference over three tokens with a
decay of about 0.75; the worst difference is 3e-8, and the two device paths agree
with each other to 7e-9. On the real model the two plans produce identical greedy
tokens over eight steps with a 9.5e-6 maximum logit difference.

### Int8 decode matvec

`build/yuuka_gemv_decode_sweep --int8`, the same protocol as the decode matvec
table, one row per submission with the matrix resident. The int8 forms are the
int8 route: the pre-quantized form reads the Q8_1 digits of the activation
(`YUUKA_OP_QUANT_Q8_1`), the fused form quantizes the row inside the matvec block.

| Shape | dtype | FP32 stream | int8 pre-quantized | int8 fused |
| --- | --- | ---: | ---: | ---: |
| `ffn_gate` 2560 -> 9216 | Q4_K | 198.8 | 210.6 | 194.2 |
| `ffn_up` 2560 -> 9216 | Q4_K | 197.4 | 209.9 | 184.2 |
| `ffn_down` 9216 -> 2560 | Q6_K | 204.5 | 196.1 | 175.2 |
| `ffn_down` 9216 -> 2560 | Q4_K | 141.6 | 159.2 | 133.6 |
| `attn_q` 2560 -> 8192 | Q4_K | 177.7 | 190.3 | 168.7 |
| `attn_k` 2560 -> 1024 | Q4_K | 64.3 | 66.1 | 57.3 |
| `attn_v` 2560 -> 1024 | Q6_K | 95.8 | 97.2 | 86.8 |
| `attn_output` 4096 -> 2560 | Q4_K | 109.6 | 146.7 | 108.0 |
| `attn_gate` 2560 -> 4096 | Q4_K | 165.3 | 166.8 | 162.8 |
| `attn_qkv` 2560 -> 8192 | Q5_K | 183.5 | 224.2 | 199.0 |
| `ssm_out` 4096 -> 2560 | Q5_K | 126.8 | 133.8 | 130.3 |
| `lm_head` 2560 -> 248320 | Q6_K | 182.6 | 194.9 | 183.4 |

In the decode step (`TestGPUProfileDecode` with
`YUUKA_TEST_DECODE_INT8=fused|prequantized`), and end to end on the 128-token
prompt:

| Route | Matvec | Quantization | Profiled step | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP32 streaming, selected | 21.39 ms | - | 31.57 ms | 29.91 |
| int8 pre-quantized | 20.07 ms | 1.09 ms | 31.78 ms | 30.64 |
| int8 fused | 21.9 ms | - | 32.27 ms | 29.75 |

The int8 arithmetic is 6% faster in step, the quantization pass costs 5% back,
and the fused form costs more than the launch it removes. Decode stays on the
FP32 streaming kernel: the step reads 2,726 MB of weights at 127 GB/s against a
244 GB/s read ceiling, so it is not instruction bound, and the 31 tokens/s
target is missed. Numerics: identical tokens over eight greedy steps with
0.2656 maximum logit difference against the FP32 route, and 5.7e-4 relative
against the quantized host reference in `tests/hip/gemv_test.cpp`.

Re-evaluated on the remapped one-row mapping, 2026-11-19, same device and model:

| Route | Matvec in step | Profiled step | Step over 32 decode steps |
| --- | ---: | ---: | ---: |
| FP32 remapped, selected | 15.44 ms, 176.6 GB/s | 17.31 ms | 16.56 ms |
| int8 Q8_1, pre-quantized, remapped | 15.55 ms, 175.3 GB/s | 18.55 ms | 17.23 ms |

In-step the int8 route is faster on `ffn_down` Q4_K (173.2 against 158.4 GB/s)
and level on `attn_qkv` (179.0) and `ffn_down` Q6_K (186.5 against 189.7), but it
loses the gate/up pair fusion: the paired Q4_K projection drops from 193.6 GB/s
as one launch to 184.4 GB/s as two, and the 217 activation quantizations add
operations of their own. The route is therefore 0.67 ms per step slower end to
end and stays opt-in (`Options.Int8Matvec` with `PrequantizedInt8`), which is the
recorded negative result. `TestGPUInt8MatvecMatchesFp32` now
pins both the fused and the pre-quantized variants to the same tokens.

## Speculative decoding

Exact greedy prompt-lookup speculation, 512-token prompt, 128 generated tokens,
F32 KV, median of three repetitions:

| Path | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain greedy | 0.396 s | 0.396 s | 14.050 s | 9.04 | 1.00x |
| Speculative (`--speculative`) | 0.396 s | 0.509 s | 3.500 s | 36.28 | 4.01x |

The generated tokens are identical to the plain greedy path. TTFT rises because
the first step drafts and verifies a block before emitting a token. Acceptance
is prompt-dependent: the repeated-prose benchmark prompt accepts draft blocks
almost fully, while a prompt with no repeated n-grams falls back to normal
decode and gains nothing.

## KV cache formats

Qwen3.5-4B has 8 full-attention layers, 4 KV heads, and head size 256, so one
KV row is 1024 values per layer:

| Format | Bytes per token (K+V) | Relative to F32 | 4096-token request |
| --- | ---: | ---: | ---: |
| F32 | 65,536 | 1.00x | 256 MiB |
| F16 | 32,768 | 0.50x | 128 MiB |
| Q8_0 | 17,408 | 0.266x | 68 MiB |

At a 128-token prompt and 32 generated tokens, median of three repetitions:

| KV format | Prefill p50 | Decode p50 | Decode tok/s |
| --- | ---: | ---: | ---: |
| F32 | 0.123 s | 3.203 s | 9.68 |
| F16 | 0.123 s | 3.189 s | 9.72 |
| Q8_0 | 0.123 s | 3.182 s | 9.74 |

The differences are inside run-to-run noise: KV format is a memory-capacity
feature at this context length, and the prefill effect appears only through the
attention kernel the format selects (see [prefill attention](#prefill-attention)).
The real-model
correctness test produced the same greedy tokens for all three formats, with
maximum logit difference from F32 of `1.12e-3` (F16) and `4.53e-2` (Q8_0).

## Prefill attention

Isolated paged prefill attention, one layer, F32 pool, Qwen3.5 geometry (16
query heads, 4 KV heads, head width 256), median of 20 launches after a warmup.
`tiled` is `attention_prefill_paged_f32_kernel`, `mma` is
`attention_prefill_paged_wmma_kernel` from
[kernels.md](kernels.md#75-fp16-mma-paged-prefill-attention). Op-level harness:
one `ATTENTION_PREFILL_PAGED` op, no other work in the plan.

| Rows | Position | Tiled | MMA | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0.016 ms | 0.039 ms | 0.41x |
| 32 | 0 | 0.027 ms | 0.041 ms | 0.66x |
| 48 | 0 | 0.044 ms | 0.052 ms | 0.85x |
| 64 | 0 | 0.075 ms | 0.057 ms | 1.32x |
| 128 | 0 | 0.195 ms | 0.083 ms | 2.35x |
| 512 | 0 | 2.468 ms | 0.507 ms | 4.87x |
| 16 | 256 | 0.209 ms | 0.119 ms | 1.76x |
| 32 | 1024 | 0.844 ms | 0.311 ms | 2.71x |
| 16 | 2048 | 3.518 ms | 0.609 ms | 5.78x |
| 32 | 7680 | 14.651 ms | 3.266 ms | 4.49x |
| 512 | 2048 | 18.809 ms | 3.409 ms | 5.52x |
| 512 | 7680 | 70.243 ms | 14.815 ms | 4.74x |

The dispatch rule follows the table: the MMA kernel is selected from 64 rows or
from a 256-token prefix. Over the eight full-attention layers of a 512-token
chunk that is 4.1 ms of attention at position 0 and 117.6 ms at position 7680,
against 19.7 ms and 562 ms for the tiled kernel (the design targets are 6 ms and
150 ms).

Numerics, same harness: the MMA output differs from the FP32 reference by
9.3e-5 maximum absolute value at head width 256 and differs from the tiled
kernel by the same 9.3e-5, which is entirely the FP16 staging step. Model level,
`TestGPUAttentionPathsAgree` records 0.015 maximum logit difference at a 64-token
prompt (top-1 `33075` in both arms), against 0.011 for the F32-to-F16 KV storage
change on the tiled kernel over the same prompt. At 2048 tokens the same
comparison records 0.220 (MMA) and 0.220 (rocBLAS) against 0.235 for the
KV-format change, because the int8 prefill route amplifies any ~1e-4 hidden-state
change onto different Q8_1 grid points.

## Isolated kernels

### Prefill GEMM

`build/yuuka_prefill_sweep`, TFLOP/s per cell, five timed repetitions after two
warmups. `mmq` is the int8 Q8_1 route and includes its activation-quantization
pass; `wmma` is the custom rocWMMA kernel. K-quant dispatch selects `mmq` for
`M >= 128` and `N >= 64`.

| dtype | shape (K -> N) | M | wmma | mmq |
| --- | --- | ---: | ---: | ---: |
| Q4_K | 2560 -> 9216 | 128 | 8.6 | 17.2 |
| Q4_K | 2560 -> 9216 | 512 | 8.4 | 17.3 |
| Q4_K | 9216 -> 2560 | 128 | 9.8 | 18.4 |
| Q4_K | 9216 -> 2560 | 512 | 8.0 | 18.3 |
| Q4_K | 2560 -> 8192 | 128 | 7.8 | 15.6 |
| Q4_K | 2560 -> 8192 | 512 | 8.3 | 17.9 |
| Q5_K | 2560 -> 9216 | 128 | 8.5 | 17.2 |
| Q5_K | 2560 -> 9216 | 512 | 8.3 | 17.6 |
| Q5_K | 9216 -> 2560 | 128 | 8.6 | 17.4 |
| Q5_K | 9216 -> 2560 | 512 | 7.5 | 18.1 |
| Q5_K | 2560 -> 8192 | 128 | 7.9 | 14.8 |
| Q5_K | 2560 -> 8192 | 512 | 8.5 | 17.5 |
| Q6_K | 2560 -> 9216 | 128 | 7.3 | 14.0 |
| Q6_K | 2560 -> 9216 | 512 | 7.6 | 14.4 |
| Q6_K | 9216 -> 2560 | 128 | 7.1 | 14.0 |
| Q6_K | 9216 -> 2560 | 512 | 6.4 | 12.4 |
| Q6_K | 2560 -> 8192 | 128 | 6.8 | 12.9 |
| Q6_K | 2560 -> 8192 | 512 | 7.2 | 13.1 |

F16 and Q8_0 rows are not routed through `mmq`; for reference, `wmma` reaches
8.0-9.2 TFLOP/s on Q8_0 and 3.4-4.7 on F16, while dequantize-plus-rocBLAS
reaches 16.0-19.0 on F16 for the wide shapes (8.1 on the tall-`K` row) plus the
dequantize pass, and is rejected for K-quants on numeric grounds.

The int8 route profiles at 22.2 VALU and 4.8 LDS instructions per
4096-MAC WMMA on Q4_K `2560 -> 9216` at M=512, with `LDSBankConflict` at 0.0%
and two blocks resident per CU (`OccupancyPercent` 45.6%), against an MMA-only
ceiling of 48-52 TFLOP/s for the same tile geometry. The loop is VALU-issue
bound: 16 of the 22.2 VALU are the scale application, which is four operations
per accumulator element per 32-value block by construction, and 2.5 of the 4.8
LDS are the `ds_swizzle_b32` that rocWMMA uses to duplicate gfx11 wave32
operands. See [decisions and rejected routes](#decisions-and-rejected-routes)
and [kernels.md](kernels.md#63-int8-mmq-gemm).

### Batched GEMV

`build/yuuka_gemv_sweep` at `n_in = n_out = 2560`, two warmups and ten
repetitions per cell. Values are the batched kernel's speedup over one GEMV
submission per row:

| dtype | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 1.03 | 1.59 | 2.35 | 3.17 | 3.91 |
| Q5_K | 1.15 | 1.68 | 2.51 | 3.51 | 4.32 |
| Q6_K | 1.08 | 1.28 | 1.87 | 2.52 | 3.14 |
| Q8_0 | 0.74 | 0.95 | 1.28 | 1.64 | 1.73 |
| Q4_0 | 1.03 | 0.96 | 1.33 | 1.59 | 1.97 |
| Q5_0 | 0.73 | 0.96 | 1.25 | 1.81 | 2.07 |
| F16 | 0.71 | 0.89 | 1.17 | 1.31 | 1.49 |

K-quants win from batch 2, the other types from batch 4; the dispatch table uses
those thresholds. Single-request decode always runs batch 1, so this table is
prefill and batching evidence, not decode evidence.

### Prefill LM head

`LMHEAD` cells of `build/yuuka_prefill_sweep`: the streaming LM head against one
GEMV submission per row, `n_in = 2560`, median of five timed submissions after
two warmups, per-row GEMV inside one plan:

| dtype | N | M | per-row GEMV | GB/s | streaming LM head | GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 512 | 1 | 0.039 ms | 19 | 0.024 ms | 31 |
| Q4_K | 1024 | 1 | 0.056 ms | 26 | 0.024 ms | 62 |
| Q4_K | 2048 | 1 | 0.098 ms | 30 | 0.035 ms | 83 |
| Q4_K | 4096 | 1 | 0.173 ms | 34 | 0.035 ms | 167 |
| Q4_K | 32768 | 1 | 1.876 ms | 25 | 0.319 ms | 148 |
| Q4_K | 248320 | 1 | 15.008 ms | 24 | 2.356 ms | 152 |
| Q4_K | 248320 | 16 | 240.817 ms | 1.5 | 6.585 ms | 54 |
| Q6_K | 512 | 1 | 0.076 ms | 14 | 0.022 ms | 48 |
| Q6_K | 1024 | 1 | 0.047 ms | 46 | 0.023 ms | 93 |
| Q6_K | 2048 | 1 | 0.076 ms | 57 | 0.036 ms | 121 |
| Q6_K | 4096 | 1 | 0.136 ms | 63 | 0.039 ms | 220 |
| Q6_K | 32768 | 1 | 1.519 ms | 45 | 0.393 ms | 175 |
| Q6_K | 248320 | 1 | 11.937 ms | 44 | 2.896 ms | 180 |
| Q6_K | 248320 | 16 | 191.002 ms | 2.7 | 7.667 ms | 68 |
| Q8_0 | 512 | 1 | 0.072 ms | 20 | 0.092 ms | 15 |
| Q8_0 | 1024 | 1 | 0.025 ms | 111 | 0.025 ms | 110 |
| Q8_0 | 2048 | 1 | 0.045 ms | 125 | 0.040 ms | 140 |
| Q8_0 | 4096 | 1 | 0.089 ms | 125 | 0.036 ms | 310 |
| Q8_0 | 32768 | 1 | 0.925 ms | 96 | 0.442 ms | 202 |
| Q8_0 | 248320 | 1 | 7.215 ms | 94 | 3.113 ms | 217 |
| Q8_0 | 248320 | 16 | 115.507 ms | 5.8 | 7.135 ms | 95 |

The streaming form wins from 512 outputs for the K-quants and from 2048 for
Q8_0; the dispatch rule therefore stops at 1024 outputs, where Q8_0 is level.
The 521 MB Q6_K cell at 180 GB/s is 74% of the 244 GB/s a pure streaming read
reaches on this device with the same matrix size, and the row count is nearly
free: sixteen rows cost 2.6x one row instead of 16x. See
[kernels.md](kernels.md#65-streaming-lm-head) for the kernel contract.

### Decode matvec

`build/yuuka_gemv_decode_sweep`, the exact shapes one `qwen35` decode token
dispatches at the dtypes the GGUF carries, measured 2026-11-18 at commit
`bfd36c0` on the reference device with the matrix resident. Each cell runs five
warmups and twenty-five timed submissions of one plan operation; the value is
the median submission, timed on the host around the submit and the wait. The
traced run of the same binary supplies the kernel time and the instruction
counts. `scripts/decode_profile.sh` reproduces this table, and
[testing.md](testing.md#63-what-to-measure-separately) describes the
per-operation attribution it feeds.

`GEMV` is the per-row kernel the decode plan uses today, `batch 4` the batched
form, and `LM head` the streaming kernel that the prefill tail already uses.
`attn_k` and `attn_v` share the `2560 -> 1024` shape, so the `Q4_K` row covers
both; `ffn_gate` and `ffn_up` share `2560 -> 9216`.

| tensor | dtype | shape (K -> N) | GEMV ms | GB/s | batch 4 ms | GB/s | LM head ms | GB/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate`, `ffn_up` | Q4_K | 2560 -> 9216 | 0.351 | 38 | 0.553 | 24 | 0.067 | 197 |
| `ffn_down` | Q6_K | 9216 -> 2560 | 0.255 | 76 | 0.494 | 39 | 0.097 | 199 |
| `ffn_down` | Q4_K | 9216 -> 2560 | 0.316 | 42 | 0.517 | 26 | 0.096 | 139 |
| `attn_q` | Q4_K | 2560 -> 8192 | 0.316 | 37 | 0.501 | 24 | 0.069 | 172 |
| `attn_k`, `attn_v` | Q4_K | 2560 -> 1024 | 0.059 | 25 | 0.094 | 16 | 0.023 | 64 |
| `attn_v` | Q6_K | 2560 -> 1024 | 0.055 | 39 | 0.094 | 23 | 0.022 | 97 |
| `attn_output` | Q4_K | 4096 -> 2560 | 0.146 | 40 | 0.230 | 26 | 0.054 | 109 |
| `attn_gate` | Q4_K | 2560 -> 4096 | 0.171 | 34 | 0.264 | 22 | 0.036 | 165 |
| `attn_qkv` | Q5_K | 2560 -> 8192 | 0.355 | 41 | 0.533 | 27 | 0.083 | 173 |
| `ssm_out` | Q5_K | 4096 -> 2560 | 0.185 | 39 | 0.249 | 29 | 0.058 | 124 |
| `ssm_beta` | Q8_0 | 2560 -> 32 | 0.009 | 10 | 0.016 | 6 | 0.024 | 4 |
| `lm_head` | Q6_K | 2560 -> 248320 | 11.218 | 46 | 17.138 | 30 | 2.780 | 188 |

Per-cell instruction counts from the traced run, as wave-level instructions
times 32 lanes per packed weight. `lds/w` counts shared-memory instructions, and
the per-row GEMV's scale staging is what puts it two orders of magnitude above
the streaming kernel:

| cell | form | waves | VALU per weight | LDS per weight |
| --- | --- | ---: | ---: | ---: |
| `ffn_gate` Q4_K 2560 -> 9216 | GEMV | 73,728 | 18.0 | 2.66 |
| `ffn_gate` Q4_K 2560 -> 9216 | batch 4 | 73,728 | 62.6 | 1.65 |
| `ffn_gate` Q4_K 2560 -> 9216 | LM head | 2,304 | 8.5 | 0.16 |
| `ffn_down` Q6_K 9216 -> 2560 | GEMV | 20,480 | 26.9 | 0.11 |
| `ffn_down` Q6_K 9216 -> 2560 | LM head | 640 | 10.9 | 0.11 |
| `attn_qkv` Q5_K 2560 -> 8192 | GEMV | 65,536 | 20.2 | 2.66 |
| `attn_qkv` Q5_K 2560 -> 8192 | LM head | 2,048 | 13.8 | 0.16 |
| `lm_head` Q6_K 2560 -> 248320 | GEMV | 1,986,560 | 29.1 | 0.41 |
| `lm_head` Q6_K 2560 -> 248320 | LM head | 62,080 | 11.3 | 0.16 |

Findings:

- The per-row GEMV is 4-6x slower than the streaming kernel on every shape the
  model uses except `ssm_beta` at 32 outputs, where it is 2.6x faster. That
  crossover is why the streaming route stays behind a shape rule rather than
  becoming the default.
- The instruction counts explain the shape of the loss, but not all of it: the
  per-row GEMV executes 18 to 29 VALU per weight against 8.5 to 13.8 for the
  streaming kernel, a factor of two, while the time differs by four to six. The
  rest is the per-row GEMV's block structure: one 256-thread block per output
  row, a block-wide tree reduction, and two barriers per 256-value K step.
- Batching four rows into one submission does not amortize the weight read. The
  batched kernel executes 3.5x the instructions of the single-row form because
  it repeats the K loop per row, so its per-row cost is 1.5-2x better and its
  per-submission cost is worse. Decode at batch 1 gains nothing from it.
- Wall time minus kernel time is the per-submission launch cost: 20-30
  microseconds for the small cells. The instrumented decode step pays that 579
  times, which is the 6.2% submission gap of the profiled decode step
  ([decode fusion](#decode-fusion)).
- Two consecutive runs of `scripts/decode_profile.sh` agree within 9% on every
  cell and within 0.3% on the profiled decode step, so the table is stable
  enough to judge a kernel change against.

`build/yuuka_gemv_decode_sweep --threshold` measures the rows = 1 crossover that
`lmHeadMinOutputs` encodes, at the decode width with five warmups and
twenty-five timed submissions per cell. "From" is the smallest output count at
which the streaming kernel wins and keeps winning:

| dtype | GEMV at 256 | GEMV at 1024 | streaming at 1024 | streaming wins from |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 20.2 GB/s | 24.8 GB/s | 65 GB/s | 512 outputs |
| Q5_K | 23.2 GB/s | 27.9 GB/s | 72 GB/s | 512 outputs |
| Q6_K | 34.4 GB/s | 39.7 GB/s | 99 GB/s | 512 outputs |
| Q8_0 | 58.1 GB/s | 116.8 GB/s | 115 GB/s | 1536 outputs |
| F16 | 30.5 GB/s | 233.6 GB/s | 150 GB/s | 1536 outputs |

The threshold stays at 1024: it is the smallest single width that does not
regress Q8_0 or F16, where the vectorized GEMV is still ahead at that width, and
no output projection sits between 512 and 1536 outputs. See
[kernels.md](kernels.md#65-streaming-lm-head) for the rule these cells encode.

The hot-matrix sweep and the real decode step do not measure the same thing,
because a single resident matrix can stay in the 32 MB MALL cache while a token
rotates through 2.7 GB of weights. Per-layer comparison, same run, sweep value
against the in-step value of the profiled cell:

| matrix | size | sweep GB/s | step GB/s | ratio |
| --- | ---: | ---: | ---: | ---: |
| `attn_gate` Q4_K 2560 -> 4096 | 5.9 MB | 34 | 26.3 | 1.29 |
| `ssm_out` Q5_K 4096 -> 2560 | 7.2 MB | 39 | 28.6 | 1.36 |
| `attn_q` Q4_K 2560 -> 8192 | 11.8 MB | 37 | 26.1 | 1.42 |
| `ffn_gate` Q4_K 2560 -> 9216 | 13.3 MB | 38 | 26.3 | 1.44 |
| `ffn_down` Q4_K 9216 -> 2560 | 13.3 MB | 42 | 29.7 | 1.41 |
| `attn_qkv` Q5_K 2560 -> 8192 | 14.4 MB | 41 | 27.1 | 1.51 |
| `ffn_down` Q6_K 9216 -> 2560 | 19.4 MB | 76 | 47.9 | 1.59 |
| `lm_head` Q6_K 2560 -> 248320 | 521.5 MB | 46 | 46.4 | 0.99 |

Every matrix below the cache size reads 1.3-1.6x faster in the sweep, and the
one matrix far above it agrees within 1%. Use the sweep for the kernel's own
ceiling and the step column for what a token costs.

### Grouped MoE pipeline

Synthetic single-layer shape: 128 experts, top-8, hidden 2048, expert FFN 768,
Q4_K weights. One route plus grouped gate/up, SwiGLU, grouped down, and weighted
reduce on one stream. `build/yuuka_moe_sweep`, median of five timed submissions
after two warmups, 2026-11-19 after the
[MoE optimization pass](benchmarks/qwen3.6_35b_a3b.md); the
gate/up and down columns run the general-dtype codebook instance, which decodes
one value per scalar step:

| Rows | Route ms | Gate/up ms | Down ms | Reduce ms | Total ms | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.010 | 0.482 | 0.225 | 0.016 | 0.728 | 1374 |
| 2 | 0.011 | 1.080 | 0.465 | 0.017 | 1.581 | 1265 |
| 4 | 0.012 | 2.051 | 0.827 | 0.018 | 2.886 | 1386 |
| 8 | 0.012 | 3.364 | 1.342 | 0.023 | 4.817 | 1661 |
| 16 | 0.012 | 5.457 | 2.169 | 0.027 | 7.593 | 2107 |
| 32 | 0.012 | 8.037 | 3.091 | 0.036 | 11.122 | 2877 |

Before the pass the same cells read 0.220-0.231 ms for the route, 0.389-12.631
for gate/up, 0.401-8.838 for down, and 1.023-21.607 ms in total. The route
kernel now costs 20-40x less and no longer appears in the total; the expert
columns carry the whole time.

The same harness selects the int8 WMMA expert route with `--route mmq
--tile-j 16|32|64|128`, which adds the Q8_1 activation pass and runs the expert
projections through `moe_gate_up_mmq` and `moe_down_mmq`. Measured 2026-11-22 on
the same shape and protocol as the table above, three runs per cell and the
median reported (the stage columns are one profiled submission after three
warmups; the total column is the mean of twenty timed submissions):

| Rows | Route | Gate/up ms | Down ms | Total ms | Tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | FP32 grouped | 5.356 | 2.141 | 7.449 | 2148 |
| 16 | MMQ, tile 16 | 1.787 | 0.912 | 2.832 | 5649 |
| 16 | MMQ, tile 32 | 1.562 | 0.885 | 2.542 | 6291 |
| 32 | FP32 grouped | 7.399 | 2.903 | 10.326 | 3097 |
| 32 | MMQ, tile 16 | 2.360 | 1.194 | 3.652 | 8763 |
| 32 | MMQ, tile 32 | 2.021 | 1.135 | 3.229 | 9910 |
| 64 | FP32 grouped | 12.296 | 4.890 | 17.218 | 3715 |
| 64 | MMQ, tile 16 | 2.941 | 1.457 | 4.481 | 14284 |
| 64 | MMQ, tile 32 | 2.515 | 1.428 | 3.993 | 16027 |
| 128 | FP32 grouped | 19.711 | 8.203 | 28.035 | 4557 |
| 128 | MMQ, tile 16 | 3.008 | 1.511 | 4.681 | 27343 |
| 128 | MMQ, tile 32 | 2.539 | 1.440 | 4.170 | 30699 |

The WMMA route is 2.6-6.9x faster on the two expert stages at this shape, and it
stops growing with the assignment count the way the FP32 route does: gate/up
goes from 1.56 to 2.54 ms while the FP32 route goes from 5.36 to 19.71 ms. The
widest tiles lose on this shape because a bucket averages `128*8/128 = 8`
assignments, so a tile of 64 or 128 leaves most columns padded (tile 128:
4.816 / 6.131 / 7.588 / 7.944 ms total). The assignment tile is therefore a
dispatch decision, not a constant; the tile table the dispatch uses was measured
per row class afterwards and is recorded with the target model in
[benchmarks/qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md#assignment-tile-ac7c697).

`--route q8` runs the same shape through the Q8_1 `dp4a` pipeline, and
`--dtype q4_k|q5_k|q6_k|iq2xxs|iq2s|iq3s` selects the synthetic expert codebook
for either quantized route, so one binary covers the two quantized forms per
codebook. Measured 2026-11-22 with the IQ staging of
[qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md#earlier-128-row-attribution-ac7c697),
`--dtype iq2xxs --experts 256 --used 8 --hidden 2048 --ffn 512`, three runs per
cell with the median reported:

| Rows | FP32 grouped ms | Q8_1 `dp4a` ms | int8 WMMA ms | WMMA tile |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.593 | 1.405 | 0.985 | 16 |
| 32 | 2.742 | 2.400 | 1.634 | 16 |
| 64 | 4.073 | 3.507 | 2.100 | 16 |
| 128 | 5.591 | 4.434 | 2.491 | 16 |
| 512 | 17.675 | 10.086 | 4.337 | 32 |

The columns are the total of the route, gate/up, down, and reduce stages. The
WMMA route is 1.4x the Q8_1 route at 16 rows and 2.3x at 512 rows; the
assignment tile in the last column is the table `SelectMoEPrefillPath` uses for
that row class.

These are kernel-pipeline numbers for a synthetic shape. End-to-end
`qwen35moe` throughput on Qwen3.6-35B-A3B-UD-IQ2_XXS at 512, 2048, and 8192
prefill tokens, next to llama.cpp on the same protocol and fixed 512 chunk, is
recorded in
[benchmarks/qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md#matched-release-comparison).
The matched run leads by 1.065x, 1.076x, and 1.011x at those lengths. Its
per-operation attribution shows the 8:1 F16 grouped-head WMMA attention saving
4679 ms of the 8192-token run, the F32 router tile and Q5_K direct cells saving
772 ms of dense GEMM, and the direct IQ2_XXS gate/up layout saving 319 ms of the
expert stage. The earlier 128-row record, where the expert kernels held 42% of
a chunk and the dense GEMM and gated delta-net carried most of the gap, remains
in the same document for the stage history.

## Kernel resource usage

`scripts/kernel_resources.sh` compiles the dispatched families with
`-Rpass-analysis=kernel-resource-usage`. Every listed kernel reports zero
scratch, and every one reports zero VGPR spills except the generic MoE gate/up
instance noted under the table. Dynamic LDS is set by the launcher and does not
appear in the compiler's static figure.

| Kernel | VGPR | SGPR | Static LDS |
| --- | ---: | ---: | ---: |
| `gemm_mmq_kernel<Q4_K>` / `<Q5_K>` / `<Q6_K>` | 176 / 177 / 162 | 48 / 48 / 52 | 0 (64512 B dynamic) |
| `lm_head_kernel<*,1>` | 45-67 | 26-32 | 0 (dynamic `rows * 1024 B`) |
| `lm_head_kernel<*,16>` | 97-127 | 42-44 | 0 (dynamic `rows * 1024 B`) |
| `quantize_q8_1_kernel` | 20 | 27 | 0 |
| `gemm_wmma_kernel<*,64,2>` | 84-89 | 28-48 | 16,384 B |
| `gemm_wmma_k32_kernel<*>` | 42-58 | 25-46 | 14,336 B |
| `gemm_wmma_swiglu_kernel<*,64,2>` | 85-138 | 28-50 | 24,576 B |
| `gemm_wmma_swiglu_kernel<*,32,1>` | 85-90 | 30-49 | 20,480 B |
| `gemv_f32_kernel` / `gemv_f16_kernel` | 11 / 11 | 17 / 17 | 1,024 B |
| `gemv_q8_0_kernel` / `gemv_q6_k_kernel` | 12 / 24 | 17 / 20 | 1,024 B |
| `gemv_packed_kernel<Q4_0>` / `<Q5_0>` | 13 / 15 | 17 / 17 | 1,024 B |
| `gemv_k_kernel<Q4_K/Q5_K>` / `<Q6_K>` | 20 / 24 | 18 / 18 | 1,088 B |
| `gemv_batch_kernel<*>` | 26-34 | 37 | 1,024 B |
| `attention_prefill_f32_kernel` | 49 | 48 | 0 (36,864 B dynamic) |
| `attention_prefill_paged_f32_kernel<*>` | 51-53 | 61-63 | 0 |
| `attention_prefill_paged_wmma_kernel<*,256,*>` | 256 | 30-34 | 0 (50,176 B dynamic) |
| `gdn_chunk_f32_kernel` / `gdn_chunk_splitreg_f32_kernel<128>` | 37 / 93 | 70 / 34 | 8 B / 1,064 B |
| `gdn_norm_gate_rows_f32_kernel` | 17 | 18 | 0 |
| `rmsnorm_batch_f32_kernel` / `rmsnorm_heads_batch_f32_kernel` | 9 / 9 | 18 / 20 | 1,024 B |
| `rope_batch_f32_kernel` | 19 | 18 | 0 |
| `kv_write_batch_f32_kernel` / `kv_write_paged_kv_kernel` | 14 / 30 | 34 / 36 | 0 |
| `embedding_batch_f32_kernel` | 21 | 19 | 0 |
| `conv1d_chunk_f32_kernel` / `conv1d_state_f32_kernel` | 20 / 7 | 40 / 17 | 0 |
| `cast_f32_to_f16_kernel` / `cast_f16_to_f32_kernel` | 7 / 7 | 18 / 18 | 0 |
| `moe_route_kernel<1..32>` | 17-79 | 22 | 0 |
| `moe_gate_up_grouped_kernel<*,8>` / `<Generic,4>` | 125-141 / 192 | 88-93 | 0 |
| `moe_down_grouped_kernel<*,8>` / `<Generic,4>` | 132-137 | 62-88 | 0 |
| `moe_gate_up_selected_kernel<*>` / `moe_down_selected_kernel<*>` | 52-78 / 44-67 | 26-60 | 0 |
| `moe_reduce_add_kernel` / `moe_shared_add_kernel` | 9 / 8 | 21 / 20 | 0 / 1,024 B |
| `moe_gemm_mmq_kernel<*, 16, 128>` (six codebooks, both activation layouts) | 93-106 | 46-50 | 0 (22.5 KiB dynamic) |
| `moe_gemm_mmq_kernel<*, 32, 128>` | 111-159 | 46-48 | 0 (25 KiB dynamic) |
| `moe_gemm_mmq_kernel<*, 64, 128>` | 138-149 | 46-49 | 0 (30 KiB dynamic) |
| `moe_gemm_mmq_kernel<*, 128, 64>` | 128-172 | 48-51 | 0 (15 KiB dynamic) |

`moe_gate_up_grouped_kernel<Generic,4>` is the one listed kernel that spills:
two SGPR slots and thirty VGPR slots. It is the scalar decoder for the eight
general dtypes, which measures better at the smaller tile than at the IQ tile,
and it is not on the path of the model this document records.

## Serving and server mode

### In-process scheduler

`bench/serving`, HIP backend, context 256, block size 16, 64-token prompt and 16
generated tokens, greedy, p50 of three measured requests:

| Configuration | Latency | Queue wait | TTFT | Prefill | Decode | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single request, graph replay | 1.5605 s | 2 µs | 0.1064 s | 0.0935 s | 1.4505 s | 10.21 |
| Single request, plain stream | 1.5595 s | 8 µs | 0.1044 s | 0.0925 s | 1.4517 s | 10.21 |
| Concurrency 4, graph replay | 6.3596 s | 0.1051 s | 0.6142 s | 0.0934 s | 1.4896 s | 9.90 aggregate |

Graph replay is inside noise at this context, so the server default stays the
plain stream path. Concurrency 4 admits four requests and exposes queue wait and
scheduler overhead, but aggregate output throughput stays at the single-request
rate: per-request decode time moves by under 3% from the single-user row because
each request's decode step reads the full quantized weights on its own. Fused
mixed-shape multi-request decode batching is not implemented.

Prefix cache, three identical 64-token prompts, no warmup: the first request
paid 0.109 s TTFT and 0.0956 s prefill; the two subsequent requests reported
4.286 ms TTFT p50 with zero prefill time, and aggregate throughput over the run
was 10.70 output tokens/s against 10.21 without the cache.

### OpenAI server

`yuuka-server` with `--backend hip --ctx 4096 --kv-type f32 --max-concurrency 1`
(4 for the concurrency row) and `llama-server` on the same model and device,
driven by an identically configured streaming client on localhost, prompt cache
off, greedy:

| Request | Metric | llama.cpp | yuuka |
| --- | --- | ---: | ---: |
| 140-token prompt, ≤128 out | Prefill tok/s | 1028 | 765 |
| | TTFT | 119 ms | 193 ms |
| | Decode tok/s | 59.07 | 9.49 |
| 530-token prompt, ≤128 out | Prefill tok/s | 1630 | 1157 |
| | TTFT | 287 ms | 465 ms |
| | Decode tok/s | 56.59 | 8.95 |
| 2051-token prompt, ≤64 out | Prefill tok/s | 2148 | 1115 |
| | TTFT | 946 ms | 1848 ms |
| | Decode tok/s | 50.53 | 7.35 |
| 4 concurrent, 140-token prompt, ≤64 out | Aggregate decode tok/s | 149.5 | 9.29 |
| | Per-request decode tok/s | 39.17 | 2.41 |
| | TTFT median | 371 ms | 690 ms |

The yuuka decode column of this table was measured before the streaming matvec
and gated delta-net work and is kept as the recorded server comparison; the CLI rows above are the current decode
rate, and the server path has not been re-measured against llama.cpp since.

Both servers tokenize the benchmark prompt to the same length (140, 530, 2051),
so the prefill comparison is like for like. yuuka-server prefill throughput sits
5-28% below the CLI rows at comparable prompt lengths (28% at ~140 tokens, 13%
at ~530, 5% at ~2050); that span covers the chat template, the HTTP path, and
server scheduling, so CLI and server numbers are not interchangeable.

### Gated delta-net chunk

Per recurrent layer of a 512-row chunk, one `RECURRENT_CHUNK` op in a plan,
20 repetitions after 3 warmups, 24 layers reported as the total. The per-token
path is the selected one; the chunkwise form is measured for
[kernels.md](kernels.md#91-chunkwise-form).

| Path / kernel | Per layer | 24 layers |
| --- | ---: | ---: |
| Per-token `gdn_chunk_splitreg_f32_kernel`, whole op | 2.01 ms | 48.2 ms |
| Chunkwise, whole op | 3.10 ms | 74.5 ms |
| Chunkwise, `gdn_scan_f32_kernel` | 1.73 ms | 41.5 ms |
| Chunkwise, `gdn_prep_f32_kernel` | 0.85 ms | 20.4 ms |
| Chunkwise, convolution + gated RMSNorm + preparation | 0.36 ms | 8.5 ms |

The convolution, gated RMSNorm, and preparation kernels are shared by both
paths. The chunkwise scan moves 2x the arithmetic of the per-token kernel
through the FP32 FMA pipe and is LDS-latency bound, so it does not pay for the
extra work; see [kernels.md](kernels.md#91-chunkwise-form).

## llama.cpp comparison

`llama-bench -ngl 99 -b 512 -ub 512 -r 3`, F16 KV (llama.cpp's default; its
`llama-bench` rejects an `f32` KV type), against the yuuka rows from
[prefill](#prefill-prompt-length) and [decode](#decode-output-length):

| Test | llama.cpp t/s | yuuka t/s | yuuka/llama |
| --- | ---: | ---: | ---: |
| pp128 | 1743.28 ± 233.98 | 1344.4 (F16 KV) | 0.77 |
| pp512 | 2346.17 ± 80.87 | 1634.8 (F16 KV) | 0.70 |
| pp2048 | 2409.70 ± 5.39 | 1641.5 (F16 KV) | 0.68 |
| pp8192 | 2170.79 ± 4.19 | 1438.2 (F16 KV) | 0.66 |
| tg128 (prompt 128) | 62.29 ± 0.19 | 60.70 (F32 KV) | 0.98 |

The remaining gaps, in order of size:

1. Decode, 3-5% at every measured context (0.95-0.98 in
   [decode gap attribution](#decode-gap-attribution)). The matvec family reads
   the 2.726 GB of weights at 176 GB/s where llama.cpp reads them at 209, and
   the tail (norm launches, delta-net, conv/L2) is 0.5-1 ms of the 17.2 ms step
   ([decode tail](#decode-tail)). Two structural causes were closed:
   decode attention is split over the key range and the matvec is one wave per
   output row.
2. Prefill at short and mid prompt lengths, 1.3-1.7x: the quantized prefill GEMM
   runs at 17-18 TFLOP/s against llama.cpp's int8 MMQ kernel, measured at
   24.9 TFLOP/s on the same device in the 2026-11-16 attribution (see
   [isolated kernels](#isolated-kernels)); the gated delta-net chunk adds
   further time yuuka does not measure separately here.
3. Long-context prefill, 1.5x at 8192: the FP16 MMA attention kernel closed
   most of this gap, so what remains is the quantized GEMM plus the delta-net
   chunk, not attention. Q8_0 KV keeps the tiled kernel and stays at 0.40.
4. Multi-request decode, 16x aggregate: llama.cpp batches the resident requests
   into one weight pass; yuuka re-reads the weights per request because fused
   mixed-shape decode batching is not implemented.

### Decode gap attribution

Re-recorded 2026-11-19 against `../llama.cpp` build `b1b770d90` (12359,
version 1.0.0), same session, same model: `yuuka bench --kv-type f32
--chunk-size 512`, `llama-bench -ngl 99 -b 512 -ub 512` (F16 KV), greedy,
warmup then medians; yuuka per-operation numbers from `TestGPUProfileDecode`
(one warmup decode step first), llama.cpp numbers from `rocprofv3
--kernel-trace` around `llama-bench -p 0 -n 32 -r 2`, 65 decode steps reduced by
kernel name and grid.

Token rate by context, both engines flat:

| Test | yuuka t/s | llama.cpp t/s | ratio |
| --- | ---: | ---: | ---: |
| tg32 after pp128 | 60.7 | 61.93 | 0.98 |
| tg128 after pp128 | 60.70 | 62.08 | 0.98 |
| tg512 after pp128 | 60.4 | 62.03 | 0.97 |
| tg128 after pp512 | 60.41 | 61.88 | 0.98 |
| tg32 after pp2048 | 58.60 | 61.60 | 0.95 |

Per operation, one decode step: yuuka is 17.16 ms profiled (16.5 ms
unprofiled), and the reference is 14.75 ms of kernels in a 16.13 ms loop. The
matvec family is 15.49 ms of the yuuka step, 176 GB/s over the same 2,726 MB
that the reference moves at 209 GB/s; it is no longer the whole step and no
longer the whole gap. Decode attention is 0.17 ms for all eight layers at a
32-token context and 0.27 ms at 512, against the reference's 0.067-0.084 ms.

| Group | ops | ms |
| --- | ---: | ---: |
| `lm_head` (single-row projections) | 73 | 5.955 |
| `gemv_pair` (gate/up and beta/alpha) | 56 | 4.832 |
| `gemv_residual` (output projections) | 64 | 4.704 |
| RMSNorm (row and head) | 81 | 0.692 |
| gated delta-net (`recurrent_token`) | 24 | 0.465 |
| conv + L2 norm (`conv_l2norm`) | 24 | 0.180 |
| decode attention (`attention_paged`) | 8 | 0.170 |
| RoPE, KV write, embedding | 25 | 0.166 |
| **profiled total** | 355 | **17.16** |

Per-shape weight bandwidth of the same projections inside a step, yuuka from
`TestGPUProfileDecode` and the reference from the trace joined by
`(dtype, K, N)` with `scripts/decode_reference_shapes.py`:

| Projection | dtype | K -> N | yuuka GB/s | ref GB/s | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| `ffn_gate`, `ffn_up` | Q4_K | 2560 -> 9216 x2 | 188.8 | 219.0 | 0.86 |
| `lm_head` | Q6_K | 2560 -> 248320 | 230.2 | 233.5 | 0.99 |
| `attn_qkv` | Q5_K | 2560 -> 8192 | 178.2 | 218.9 | 0.81 |
| `ffn_down` | Q6_K | 9216 -> 2560 | 186.6 | 189.9 | 0.98 |
| `ffn_down` | Q4_K | 9216 -> 2560 | 157.6 | 177.2 | 0.89 |
| `ssm_out` | Q5_K | 4096 -> 2560 | 130.9 | 208.3 | 0.63 |
| `attn_gate` | Q4_K | 2560 -> 4096 | 156.3 | 201.3 | 0.78 |
| `attn_q` | Q4_K | 2560 -> 8192 | 160.8 | 212.1 | 0.76 |
| `attn_output` | Q4_K | 4096 -> 2560 | 125.4 | 207.7 | 0.60 |
| `ssm_alpha`, `ssm_beta` | Q8_0 | 2560 -> 32 x2 | 12.5 | 18.9 | 0.66 |
| `attn_k`, `attn_v` | Q4_K | 2560 -> 1024 | 92.3 | 171.1 | 0.54 |
| `attn_v` | Q6_K | 2560 -> 1024 | 134.4 | 186.1 | 0.72 |

What remains, in order of size:

1. The narrow and short-K projections, 0.54-0.72 of the reference: the one-row
   remap removed the grid and load-width defects on the wide shapes, but a
   1024- or 4096-input row still pays a per-output-row cost of roughly 27 ns on
   top of its weight bytes. Closing that needs the int8 route (measured slower
   end to end, [int8 decode matvec](#int8-decode-matvec)) or a different
   reduction, not another block-shape change.
2. The tail, 1.67 ms, of which 0.69 ms is norm launches at 9 us each. The
   reference pays 5.7 us for the same dispatch, so the remaining difference is
   launch count, not arithmetic.
3. Prefill at short and mid prompt lengths, 1.3-1.7x: the quantized prefill GEMM
   runs at 17-18 TFLOP/s against llama.cpp's int8 MMQ kernel, measured at
   24.9 TFLOP/s on the same device in the 2026-11-16 attribution (see
   [isolated kernels](#isolated-kernels)); the gated delta-net chunk adds
   further time yuuka does not measure separately here.
4. Long-context prefill, 1.5x at 8192: the FP16 MMA attention kernel closed
   most of this gap, so what remains is the quantized GEMM plus the delta-net
   chunk, not attention. Q8_0 KV keeps the tiled kernel and stays at 0.40.
5. Multi-request decode, 16x aggregate: llama.cpp batches the resident requests
   into one weight pass; yuuka re-reads the weights per request because fused
   mixed-shape decode batching is not implemented.

### Split-key decode attention

Split-key decode attention landed on 2026-11-19. Measured on the same device
and model (SHA-256 `00fe7986...`), `yuuka bench --kv-type f32
--chunk-size 512`, greedy, warmup 2 and 3 repetitions, against the same
llama.cpp build `b1b770d90`:

| Test | before (`7b7aa4c`) | after | llama.cpp |
| --- | ---: | ---: | ---: |
| pp128/tg32 | 41.12 | 44.74 | 61.93 |
| pp128/tg128 | 38.81 | 44.02 | 62.08 |
| pp512/tg128 | 30.97 | 43.75 | 61.88 |
| pp2048/tg32 | 17.42 | 42.90 | 61.60 |

The decode rate is now flat within 4% from a 128-token to a 2048-token context;
the remaining 5% against the 128-token row is the KV traffic itself. In
`TestGPUProfileDecode`, which adds one submission gap per operation, decode
attention costs 0.021 ms per layer at a 7-token context and 0.033 ms at 512
(0.017 and 1.04 ms with the serial kernel). Unprofiled, an attention-only plan
over the eight full-attention layers costs 21-24 us per layer at a 512-token
context, 55/49/37 us at 1024, and 93/67/57 us at 2048 for F32/F16/Q8_0 KV.

The 2048-token F32 figure moves 16 MB of KV per layer, which is 68 us at the
235 GB/s read ceiling measured on this device, so
what remains there is the pool bytes rather than the kernel; F16 and Q8_0 read
half and a quarter of that. The same profile with all three pools
(`YUUKA_TEST_DECODE_KV`) measures 0.096/0.071/0.061 ms per layer at 2048 tokens
and 0.0335/0.0353/0.0318 ms at 512 for F32/F16/Q8_0, and the graph replay
reproduces each format's plain token stream (ratio 1.003-1.006).

## Decisions and rejected routes

Each of these is a measured decision that still holds in the current code:

- **FP32 accumulation is required.** FP16 compute with an FP16 destination
  moves the 512-token prefill logits by 0.305 against the 0.05 chunk gate, so
  every prefill matmul keeps an FP32 accumulator.
- **The dequantize-plus-rocBLAS prefill route is removed** for K-quants. It was
  1.6-3.3x faster per GEMM from 256 rows upward but numerically rejected on
  tall-K shapes and needed an FP16 copy of every weight matrix; the int8 MMQ
  route replaced it.
- **The generic MMQ path uses a 128-value K step.** It stages
  both operands in WMMA fragment order (0.0% bank conflicts from 35.9%),
  expands Q4_K/Q5_K digits with word-wide masks, and uses a 128-value K step so
  two blocks stay resident per CU: 13.8 -> 17.3-18.5 TFLOP/s on Q4_K
  `2560 -> 9216` at M=512. Rejected against it (see
  [decisions and rejected routes](#decisions-and-rejected-routes)):
  a 60 KiB double-buffered 256-value stage (one block per CU, 14.6 TFLOP/s),
  register prefetch of the next step's weights (192 VGPRs, 161-195 spills,
  6.4 TFLOP/s), a 64-value K step (16.4 TFLOP/s), and rebuilding the rocWMMA
  gfx11 operand duplication by storing duplicate operands in LDS. The
  [gfx1151 specialization](kernels.md#gfx1151-direct-mmq-specialization) uses
  LDS broadcast and vectorized activation staging on the measured Q4_K cell.
- **The chunkwise gated delta-net form is implemented but not selected.** It
  matches the per-token recurrence to `1.6e-7` and removes the per-token barrier
  chain, but takes 3.10 ms per recurrent layer against 2.01 ms for the
  per-token kernel, because its dense products run on the FP32 FMA pipe and the
  scan is LDS-latency bound (0.4 instructions per cycle per CU). One state
  column per thread (2.23 ms for the scan alone) and four columns per thread
  (192 VGPRs, 909 spill slots) were measured and rejected. Moving the dense
  products to the FP16 WMMA path is the recorded follow-up.
- **The MMQ scale epilogue is not reducible to two FMA per element.** The
  activation scale differs per accumulator tile, so the 32 multiplies of a
  32-value block cannot be hoisted, and `v_fma_mix_f32` does not convert an
  integer operand on gfx1151. The loop keeps four VALU per element per block,
  which is 16 of its 22.2 VALU per WMMA.
- **128-row WMMA tiles and a 64-wide K step are rejected** for the custom GEMM.
  The 64x64/K32 and 32x64 variants remain the dispatch choice for the paths they
  serve, and the wide K-quant shapes in this report come from the int8 MMQ route
  instead.
- **Prefill attention is settled on the FP16 MMA kernel** for F32 and F16 pools
  from 64 rows or a 256-token prefix; the tiled kernel stays for Q8_0, for
  shapes the MMA kernel does not cover, and below that crossover (measured
  1.2-2.4x faster there). Staging K transposed per key tile instead of Q once
  per workgroup, and keeping the accumulator at 192 registers instead of
  `__launch_bounds__(128, 2)`, were measured and rejected: the first costs 128
  scalar shared-memory loads per key tile, the second spilled 21 slots into the
  key loop for 1.5x. The earlier group-of-four tiled kernel and the rocBLAS
  path are no longer selected by default; rocBLAS stays reachable through
  `LegacyBlasAttention` for A/B measurement.
- **Disabled by default:** the FFN FP16 weight cache (removed with the
  dequantize route) and HIP Graph replay (no gain at the serving contexts
  measured). Speculative decoding is opt-in: exact and 4x faster on repetitive
  prompts, no gain where draft acceptance is low.
- **The decode matvec remap stops at sixteen K values per lane.** Thirty-two
  values per lane measured 131 GB/s against 176 because the wider accumulator
  and activation register set cuts occupancy; four rows per wave, four waves per
  block with four rows per wave, and shared-memory staging of the activation
  step measured 156, 156, and 163 GB/s. The activation is read once per wave
  step and reused by the wave's two rows, which is what the 176 GB/s figure
  reflects.
- **The int8 route does not beat the FP32 remap.** On the remapped mapping the
  Q8_1 form measures 175.3 GB/s in step against 176.6 and 0.67 ms slower per
  step, because it loses the gate/up pair fusion; it stays opt-in.
- **The decode attention tile is 32 positions below a 1024-position capacity
  and 64 above.** 128 measured 77 us per layer at 512 tokens against 23, and 16
  measured 33 against 30; more blocks win at short context and fewer partials
  win at long context, hence the split.
- **Known limits:** decode throughput is a single-request weight-bandwidth
  number until fused multi-request decode batching exists; NPU execution is a
  diagnostic probe only; the `qwen35moe` HIP path has one measured model
  configuration in [benchmarks/qwen3.6_35b_a3b.md](benchmarks/qwen3.6_35b_a3b.md);
  the longest measured prompt is 8192 tokens at chunk 512.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving
MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf

ctest --test-dir build --output-on-failure
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run 'TestGPUChunkedPrefillLongContext|TestGPUMatchesCPU' -count=1 -v

./build/yuuka_prefill_sweep
./build/yuuka_gemv_sweep
./build/yuuka_moe_sweep
./build/yuuka_gemv_decode_sweep
scripts/kernel_resources.sh
YUUKA_TEST_MODEL="$MODEL" scripts/decode_profile.sh   # decode matvec table
YUUKA_TEST_MODEL="$MODEL" scripts/decode_compare.sh    # llama.cpp decode comparison
YUUKA_TEST_MODEL="$MODEL" scripts/prefill_counters.sh # prefill counter table

# One decode step per route and context. The profile test warms one step first.
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfileDecode -count=1 -v
YUUKA_TEST_DECODE_CONTEXT=512 YUUKA_TEST_MODEL="$MODEL" go test -tags hip \
  ./engine/gpu -run TestGPUProfileDecode -count=1 -v
YUUKA_TEST_DECODE_CONTEXT=2048 YUUKA_TEST_DECODE_KV=f16 YUUKA_TEST_MODEL="$MODEL" \
  go test -tags hip ./engine/gpu -run TestGPUProfileDecode -count=1 -v
YUUKA_TEST_DECODE_INT8=prequantized YUUKA_TEST_MODEL="$MODEL" go test -tags hip \
  ./engine/gpu -run TestGPUProfileDecode -count=1 -v

for p in 128 512 1024 2048 4096 8192; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
    --prompt-tokens "$p" --decode-tokens 1 --warmup 2 --reps 5 --kv-type f32
done
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 8192 --decode-tokens 1 --warmup 2 --reps 5 --kv-type f16
for d in 128 256; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
    --prompt-tokens 128 --decode-tokens "$d" --warmup 2 --reps 3 --kv-type f32
done
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 512 --decode-tokens 128 --warmup 2 --reps 3 --kv-type f32
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 512 --decode-tokens 128 --warmup 2 --reps 3 --kv-type f32 --speculative

/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 64 --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 0 --graph --prefix-cache

# llama.cpp reference, same model and device.
llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 \
  -p 128,512,2048,8192 -n 0 -r 3 -o md
llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 -p 128 -n 128 -r 3 -o md
```

Server-mode comparison: start `yuuka-server --model "$MODEL" --backend hip
--ctx 4096 --kv-type f32 --prefill-chunk 512 --max-concurrency 1
--prefix-cache=false` and `llama-server -m "$MODEL" -ngl 99 -c 4096 -ctk f16
-ctv f16 -b 512 -ub 512 -np 1 --jinja`, then post greedy streaming requests to
`/v1/chat/completions` with `cache_prompt: false` on the llama.cpp side. One
client thread per request, requests alternating between servers, three
repetitions per cell.

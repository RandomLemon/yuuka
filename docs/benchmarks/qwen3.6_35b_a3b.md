# Qwen3.6-35B-A3B (`qwen35moe`) benchmark

**Objective.** Record the `qwen35moe` HIP prefill throughput on gfx1151 at 512,
2048, and 8192 prompt tokens against llama.cpp on the same model, device, KV
format, chunk size, and single-request protocol, with the per-operation
attribution of what changed and the correctness gates the run was taken with.
The same document records the single-token decode release comparison: 128
teacher-forced one-token steps after 512, 2048, and 8192 populated prompt
tokens, against the same local llama.cpp build.

**Status.** Measured on 2026-11-22 at `d0253fde` against the frozen
`1bc54d18` baseline and upstream `b1b770d90`. The matched fixed-512 ABBA
comparison now leads at every requested length: 1718.6 against 1613.8 tokens/s
at 512 (1.065x), 1682.4 against 1564.0 at 2048 (1.076x), and 1493.4 against
1477.1 at 8192 (1.011x), using 20 samples per engine per length. The three
prefill speed gates and the behavior gates pass; the MoE attention route's
numerical envelope is recorded in the matched comparison and in
[moe_real_test.go](../../engine/gpu/moe_real_test.go). The 2026-11-23 decode
comparison below passes all four ordered ratios and the pooled ratio at 512,
2048, and 8192 tokens (1.0736x, 1.0662x, 1.0430x against llama.cpp). The earlier
`ac7c697` dense-stage record and 128-row attribution below are historical and
are kept for the stage comparison.

## Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-22 |
| Model | `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` |
| Model SHA-256 | `2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef` |
| Model size | 10,756,586,464 bytes (10.01 GiB, 2.0625 bpw, 34.66 B parameters) |
| Architecture | `qwen35moe`, 40 layers (10 full attention, 30 gated delta-net), 256 routed experts, top-8, expert FFN 512, shared FFN 512 |
| Quantization | Router and shared-expert gate F32; shared expert Q5_K/Q6_K; expert gate/up IQ2_XXS; expert down IQ2_S (37 layers) and IQ3_S (3 layers) |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, Wave32, 122,880 MiB reported VRAM |
| ROCm / HIP | ROCm 10.1.0, HIP 7.16.26385, clang 24.0.0git |
| Host | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads |
| Kernel | 7.0.14-28-pve |
| Power profile | GPU DPM `auto` (amdgpu sysfs `power_dpm_force_performance_level`), CPU governor `performance` (`amd-pstate-epp`) |
| Go | go1.24.4, `-tags hip` |
| Optimized commit | `d0253fde1e9172143b9923a838b1ac393cd3d639` |
| Frozen baseline commit | `1bc54d18b8911723498df6f2e76a72b8492d9e6f` |
| llama.cpp | upstream `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, ROCm Release build, `-ngl 99`, `-fa auto` |
| Tokenizer / chat template | GGUF-embedded Qwen3.6 vocabulary and template |
| Sampling | None; both drivers compare the final host logit row and greedy top-1 |
| KV format | F16 on both sides |
| Prefill chunk | 512 tokens on both sides (1, 4, and 16 chunks for the three lengths) |
| Warmup / repetitions | 2 / 5 per process, five-sample medians, two reversed-order processes per engine per length |
| Token stream | One saved 8192-token little-endian `int32` file, SHA-256 `e613216db780a72fdbe796b31c63b4afabf273199dbc67f799b5c082d18e3e30` |
| Concurrency | 1 request; no prefix cache, profiling, or parallel GPU job |
| Thermal state | Idle 47-48 °C before each accepted block; sensor edge 52-61 °C during the runs; sclk 2.17-2.40 GHz |

Each engine runs `y1, l1, l2, y2` at every length. `ycompare` times
`Session.Prefill(tokens, true)` through the final host row, and the llama driver
times the same saved tokens through `llama_batch` with `n_batch = n_ubatch =
512` and full GPU offload. The score pools all ten samples per engine and
requires both reversed-order pairs and the pooled median to exceed 1.00x.
A first confirm block started at 56 °C and its 512-token first process was
discarded as thermally contaminated; the accepted confirmation block started at
47 °C. The GPU ran with DPM `auto`, so the clock snapshots in the raw log span
2.17-2.40 GHz. Numbers are observations on one machine, not guarantees.

The decode gate uses the same saved stream and also resets for every repetition.
`ycompare -decode 128` prefills `[0,P)` and fetches the final prompt host row
before starting the timer, then times 128 sequential `Session.Decode` calls over
`[P,P+128)`. The llama driver takes the optional trailing `DECODE` argument and
times 128 one-token `llama_decode` calls at absolute positions `P..P+127`,
calling `llama_get_logits_ith(-1)` after each step; that call synchronizes the
host row inside the interval. Both drivers append `decode` and
`decode_samples_ns` to the JSON and leave `samples_ns` as the prefill series.
The Yuuka driver rejects a short or misaligned token file, an out-of-vocabulary
id, or a missing final host row instead of shortening the sample. Neither driver
samples, checks intermediate rows, or fetches logits outside the timed loop.

## Matched release comparison

| Prompt | Frozen baseline | Optimized | llama.cpp | Optimized / llama |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 416.9 ms, 1228.1 tok/s | 297.9 ms, 1718.6 tok/s | 317.3 ms, 1613.8 tok/s | **1.065x** |
| 2048 | 1886.2 ms, 1085.8 tok/s | 1217.3 ms, 1682.4 tok/s | 1309.4 ms, 1564.0 tok/s | **1.076x** |
| 8192 | 11705.2 ms, 699.9 tok/s | 5485.6 ms, 1493.4 tok/s | 5546.2 ms, 1477.1 tok/s | **1.011x** |

The medians pool 20 samples per engine per length from two complete ABBA
blocks. The accepted release block (`release3`) and its confirmation block
(`confirm2`) give these reversed-order pairs:

| Prompt | Release y1/l1 | Release y2/l2 | Confirmation y1/l1 | Confirmation y2/l2 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1.068x | 1.083x | 1.034x | 1.044x |
| 2048 | 1.055x | 1.052x | 1.097x | 1.073x |
| 8192 | 1.019x | 0.995x | 1.047x | 1.012x |

The 8192 release pair `y2/l2` is 0.5% below parity; the fresh confirm block
passes both pairs at every length, and the pooled 8192 ratio is 1.011x. The
pooled 512 and 2048 ratios are 1.065x and 1.076x. Individual process medians
(ms, `y1/y2` and `l1/l2`) were 299.7/297.2 and 320.2/321.9 at 512,
1196.8/1248.3 and 1263.2/1313.3 at 2048, and 5415.4/5507.4 and 5516.8/5478.8
at 8192 in the release block; the confirmation block was 300.2/297.7 and
310.6/310.9, 1204.9/1221.9 and 1321.4/1310.9, and 5404.1/5511.1 and
5657.4/5577.8.

### Attribution at the release chunk

`TestGPUProfileMoEPrefillLong` executes every 512-row chunk at its populated
absolute position for the same 8192-token stream, so this table charges attention
and recurrence the context they carry in the release run. It is a diagnostic:
events add submission overhead and clocks are not pinned. Both columns run the
current plan code against the frozen and optimized shared libraries, so the
table isolates kernel changes. The plan-level changes (the prepared recurrent
span and the shared-expert quantizer reuse) are present in both columns; their
effect appears in the release baseline totals, which are 416.9 ms at 512 and
11705.2 ms at 8192 against the profiled 386.7 and 11106.0 ms.

| Stage | Base 512 | New 512 | Base 2048 | New 2048 | Base 8192 | New 8192 | 8192 saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill attention | 22.2 | 5.9 | 352.1 | 61.1 | 5505.6 | 826.2 | 4679.4 |
| dense GEMM | 160.5 | 115.0 | 649.6 | 456.6 | 2555.2 | 1783.0 | 772.2 |
| expert gate/up | 95.6 | 76.6 | 367.1 | 286.7 | 1405.6 | 1086.3 | 319.3 |
| expert down | 49.5 | 47.9 | 189.9 | 182.6 | 751.7 | 698.4 | 53.3 |
| recurrent chunk | 26.0 | 26.4 | 103.6 | 105.5 | 424.3 | 418.5 | 5.8 |
| expert reduce | 15.2 | 16.1 | 60.7 | 65.3 | 244.3 | 263.4 | -19.1 |
| other | 17.7 | 17.6 | 55.4 | 55.5 | 219.3 | 212.8 | 6.5 |
| **total profiled** | **386.7** | **305.5** | **1778.4** | **1213.3** | **11106.0** | **5288.5** | **5817.5** |

The dense GEMM column at 8192 decomposes from 2555.2 to 1783.0 ms as: F32 router
610.6 to 104.2 (the new 64x64 tile), Q5_K Q/QKV 469.7 to 366.5, recurrent gate
263.9 to 196.1, attention Q 141.7 to 110.2, attention output 80.8 to 55.7,
shared gate/up 124.9 to 89.6, K/V 31.6 to 22.8, and the unchanged F32 recurrent
alpha/beta and Q6_K SSM-out/shared-down cells. The expert gate/up column is the
direct IQ2_XXS layout; 1086.3 ms is 67.9 ms per 512-row chunk against 88.9 ms
before. The 19.1 ms increase in expert reduce is within the run-to-run stage
spread and was not optimized.

The recurrent row is neutral in this table because both columns already bind
the prepared four-lane core from the current plan; the frozen release baseline
ran the inline recurrence, so the plan-level saving is in the release baseline
totals above rather than in the row. When attention is forced to the tiled
kernel, the optimized build produces the same logits and `Session.Snapshot`
digest as the frozen build, so the recurrence change is state-identical. The
Q6_K dense trial and contiguous expert-store trials from the plan were not
needed once the three speed gates passed.

### Stage sweeps

`build/yuuka_prefill_sweep --moe-dense` ran in base, candidate, candidate, base
library order at M=512, two warmups and five samples per cell. Medians in ms:

| Cell | Base | Candidate | Base | Change |
| --- | ---: | ---: | ---: | ---: |
| F32 router 2048->256 | 0.802 | 0.139 | 0.813 | 5.8x faster |
| Q5_K 2048->8192 | 0.981 | 0.727 | 0.970 | 25% faster |
| Q5_K 2048->4096 | 0.507 | 0.368 | 0.500 | 27% faster |
| Q5_K 2048->512 | 0.116 | 0.087 | 0.120 | 26% faster |
| Q5_K 4096->2048 | 0.532 | 0.392 | 0.520 | 26% faster |
| Q6_K 4096->2048 | 0.624 | 0.641 | 0.636 | unchanged |
| Q6_K 512->2048 | 0.141 | 0.132 | 0.138 | unchanged |

`build/yuuka_moe_sweep --route mmq --dtype iq2xxs --experts 256 --used 8
--hidden 2048 --ffn 512 --rows 512 --tile-j 32` measured gate/up 2.499 ms
before and 1.898 ms after (24% less); the IQ2_S and IQ3_S gate/up and all three
down projections are unchanged. The real-prompt attribution above confirms the
same direction at 19-20 ms per 512-row chunk.

### Numerical and behavioral gates

- Optimized, baseline, and llama.cpp top-1 agree at every length: 13477 at 512,
  5388 at 2048, and 13477 at 8192.
- The optimized final logits differ from the frozen baseline by at most 1.4246,
  1.1723, and 0.7767 at 512, 2048, and 8192 (mean 0.213, 0.168, 0.122), and
  from llama.cpp by 2.0572, 1.3264, and 1.0292. The difference is the approved
  8:1 F16 grouped-head WMMA route; `TestGPUMoEPrefillGreedyContinuity` records
  1.43/1.26/0.87 against the forced tiled control, keeps the greedy tokens
  `[13477 37550 33075 888]` at 512, `[5388 13 561 3841]` at 2048, and
  `[13477 37550 33075 888]` at 8192, and enforces
  `moeAttentionRouteLogitTolerance = 2.0`. The route gate change was approved
  explicitly for this record; the dense 4:1 route keeps 0.05/0.3.
- `TestGPUMoEMmqMatchesQ8` reports 0 maximum logit difference and the golden
  continuation `[11751 11 264 3177]`; `TestGPUMoEGoldenGeneration` keeps a
  0.00286746 first-token difference from the CPU reference;
  `TestGPUMoEChunkedPrefill` keeps the CPU top-1 and continuation
  `[279 15217 5388 13]` at 1.07/1.14 against the 2.0 bound; and
  `TestGPUChunkedPrefillLongContext` passes with the same top-1.
- Native coverage passes: the router tile matches the scalar 513-row control
  byte for byte; the Q5_K direct cells match the generic M=513 control byte for
  byte; the direct IQ2_XXS gate/up matches dp4a at 2.4e-7 relative with IQ2_S
  and IQ3_S down at 1.7e-8; and the 16-head/2-KV-head F16 attention case is
  within 5e-3 of the tiled and host references. The selected CTest suite is
  8/8, and `kernel_resources.sh` reports no spills for the grouped-head WMMA
  instantiation.
- `TestGPULongSplitAttentionMatchesDirect` skips because the 8:1 geometry does
  not allocate the split workspace; split attention was not shipped for this
  model.

The independent driver build and the `y1, l1, l2, y2` process order follow
[the matched upstream comparison](../prefill-optimization/acceptance/08-upstream-comparison.md);
the only changes are the verified model path, the saved token file, and the
fixed 512 chunk. The Go gates use `YUUKA_TEST_MODEL` and `YUUKA_TEST_MOE_MODEL`
set to that GGUF, `YUUKA_TEST_KV=f16`, and
`YUUKA_TEST_ATTENTION_LONG=1` for `TestGPUAttentionPathsAgree`. The native
gates are the selected CTest suite and the `--moe-dense` sweep in base,
candidate, candidate, base order.

## Decode release comparison (2026-11-23)

The decode target is the shipping single-request HIP path at 512, 2048, and
8192 populated prompt tokens, followed by 128 one-token decode steps. All four
ordered `(y1,l1)` and `(y2,l2)` ratios and the pooled 20-sample ratio exceed
1.00 at every length, and the same final build improves on the frozen Yuuka
baseline in both reversed pairings at every length.

| Prompt | Frozen Yuuka | Final Yuuka | llama.cpp | Final / llama | Frozen / final |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 15.572 ms, 64.2 tok/s | 13.968 ms, 71.6 tok/s | 14.996 ms, 66.7 tok/s | **1.0736x** | 1.1148x |
| 2048 | 15.913 ms, 62.8 tok/s | 14.297 ms, 69.9 tok/s | 15.243 ms, 65.6 tok/s | **1.0662x** | 1.1130x |
| 8192 | 16.897 ms, 59.2 tok/s | 15.214 ms, 65.7 tok/s | 15.868 ms, 63.0 tok/s | **1.0430x** | 1.1106x |

`ms/token` is `median(decode_samples_ns) / (128e6)`; the final and llama
medians pool the 20 samples of the two independent blocks, and the frozen
median pools the 20 samples of the `b1/f1/f2/b2` recheck. The four ordered
ratios are:

| Prompt | Release y1/l1 | Release y2/l2 | Confirm y1/l1 | Confirm y2/l2 | Pooled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.0783x | 1.0684x | 1.0697x | 1.0774x | 1.0736x |
| 2048 | 1.0572x | 1.0588x | 1.0638x | 1.0703x | 1.0662x |
| 8192 | 1.0427x | 1.0432x | 1.0483x | 1.0417x | 1.0430x |

The frozen comparison uses `base,final,final,base` with the same flags, tokens,
2 warmups, and 5 repetitions per process:

| Prompt | Recheck b1/f1 | Recheck b2/f2 | Pooled base/final |
| ---: | ---: | ---: | ---: |
| 512 | 1.1224x | 1.1079x | 1.1162x |
| 2048 | 1.1147x | 1.1015x | 1.1106x |
| 8192 | 1.1086x | 1.1133x | 1.1112x |

### Decode attribution

`TestGPUProfileMoEDecodeContexts` profiles steps 0, 63, and 127 after the
populated prefix; the average of those three steps at 8192 is 16.813 ms with
events enabled, against the 15.214 ms release median without them. The plan went
from 713 to 673 operations. The stages that changed:

| Stage (per decode step, 8192) | Frozen | Final |
| --- | ---: | ---: |
| Router top-k | 0.521 ms | 0.377 ms |
| Routed reduce | 0.676 ms | 0.202 ms |
| Selected gate/up (IQ2_XXS) | 2.048 ms | 1.929 ms |
| Selected down (IQ2_S/IQ3_S) | 1.620 ms | 1.494 ms |
| Shared expert | 1.336 ms | 1.045 ms |
| Paged decode attention | 1.603 ms | 1.466 ms |
| Dense matvecs, recurrence, norms, and head | 11.20 ms | 10.30 ms |
| **Profiled step** | **19.00 ms** | **16.81 ms** |

The router runs one wave without the count memset or assignment atomics, and the
one-row/eight-slot/2048 reduce covers the device with 64 Wave32 blocks instead
of one block. The selected IQ kernels load the eight-value activation span once
per 256-value K step and use four waves per block, one output row per wave. The
one-row matvec remap uses one row per wave instead of two, which doubled the
block count on the latency-bound projections; the K/V and F32 beta/alpha pairs
remove two launches without changing the per-row reduction. Decode attention
merges its tile weights once per 64-dimension block in bounded shared memory and
launches only live tiles on plain submission. The measured stage reductions
cover the baseline gap at each length; no single microbenchmark is used as the
gate.

Correctness for the gate: the recorded frozen controls compare bitwise over the
prefill row and all 128 teacher-forced decode rows at every length, the
`The capital of France is` golden continuation is `[11751, 11, 264, 3177]`, and
the fixed-prefix greedy continuations are `[13477, 37550, 33075, 888]`,
`[5388, 13, 561, 3841]`, and `[13477, 37550, 33075, 888]`. The native tests pin
both router forms on tied and negative logits, the reduce against the generic
block reduce, the 2048/512 selected kernels against the host dequantization, and
the attention merge and live-tile replay against the reference.

Environment: same model and SHA-256 as above; token stream SHA-256
`2367fa295341e579a00189d5a5f52df5b742c5e732f16c6b8f5b15f6709acf81`, first 8192
ids `e613216db780a72fdbe796b31c63b4afabf273199dbc67f799b5c082d18e3e30`;
Yuuka source at `3575b28cf35261848a73b0518c8110f6f076be9f` with this decode
change; llama.cpp `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`; ROCm/HIP and
device as above. F16 KV, chunk 512, full GPU offload, one request, 2 warmups and
5 repetitions per process, one process at a time. The final library sha256 was
`03957aa2a338bad3a71aee73f953b99682426ff8b70b71a85850c7d18ba7fff2`. The three
gate blocks started at 56-58 °C and stayed at 57-65 °C with 16-100 W board
power; no thermal throttle was logged. The earlier prefill blocks and the
decode blocks agree across repeated orders, which is why no block was discarded.

## Earlier dense-stage record (ac7c697)

| Measurement | yuuka run 1 | yuuka run 2 | llama.cpp | ratio (run 1) |
| --- | ---: | ---: | ---: | ---: |
| prefill 128 tokens | 954.11 | 933.13 | 1092.23 ± 30.22 | 0.87x |
| prefill 512 tokens | 1249.15 | 1225.54 | 1561.85 ± 27.04 | 0.80x |
| prefill 2048 tokens | 1070.01 | 1084.60 | 1512.87 ± 6.66 | 0.71x |
| decode, 128-token prompt | 63.48 | 63.81 | 67.84 ± 0.79 | 0.94x |
| decode, 512-token prompt | 63.02 | 63.16 | 67.74 ± 0.74 | 0.93x |
| decode, 2048-token prompt | 61.94 | 61.82 | 67.16 ± 0.08 | 0.92x |

The table is tokens/s; the yuuka decode rows are the 128 generated tokens after
the given prompt length, and the llama.cpp `tg` rows are its 128-token
continuations after the same prompts.

Detail of run 1:

- yuuka prefill p50/p95 was 0.134/0.136 s, 0.410/0.415 s, and 1.914/1.918 s at
  128, 512, and 2048 tokens. TTFT equals the prefill time because the
  measurement is single request, and there is no prefix cache.
- yuuka decode p50 was 2.001 s, 2.015 s, and 2.051 s at the three prompt
  lengths, so the per-token rate falls 2.4% from the shortest to the longest
  prompt as attention reads a longer KV context.
- Prefill rate is 954 tokens/s at 128 rows and 1249 at 512 rows: a 512-row
  chunk fills the 32-assignment tile of the grouped expert route, while a
  128-row chunk runs the 16-wide tile at a quarter of its assignment rows. It
  falls back to 1070 tokens/s at 2048 tokens because that run's four 512-row
  chunks pay the attention cost of a context that grows to 2048 positions.
- llama.cpp's prefill rate is flat at 1513-1562 tokens/s from 512 to 2048
  tokens, so the prefill gap widens from 1.25x to 1.41x over that range.

## Earlier 128-row attribution (ac7c697)

One 128-row prefill chunk and one decode step, per operation. The yuuka column
is `TestGPUProfileMoE`, which attributes the planned operations of one chunk and
one step with HIP events. The reference column is a `rocprofv3 --kernel-trace`
of `llama-bench -p 128 -n 0 -r 1` for the chunk and `-p 0 -n 16 -r 1` for the
decode, reduced per kernel name and grouped into the same stages; the IQ types
are resolved from `ggml_type`: 16 is IQ2_XXS, 22 IQ2_S, 21 IQ3_S, 13 Q5_K, and
14 Q6_K.

Neither column is a release measurement. The instrumented yuuka chunk is
142.5 ms against the 134 ms release prefill (128 tokens at 954 tokens/s), and
its instrumented decode step is 19.2 ms against the 15.8 ms release step, so the
yuuka rows are per-operation kernel time and the ratio column compares that.
The traced llama.cpp chunk sums to 112.9 ms of kernel time against a 117.2 ms
release prefill, and its traced decode step sums to 12.7 ms against a 14.7 ms
release step. Both sides are therefore compared as kernel time, with the same
bias of a few percent on each.

### Prefill, one 128-row chunk

| Stage | yuuka ms | share | llama.cpp ms | share | yuuka / llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert gate/up (IQ2_XXS) | 39.37 | 28.3% | 31.15 | 27.6% | 1.26x |
| expert down (IQ2_S, IQ3_S) | 19.27 | 13.8% | 20.43 | 18.1% | 0.94x |
| dense GEMM (router, attention and shared-expert projections) | 53.19 | 38.2% | 40.09 | 35.5% | 1.33x |
| gated delta-net chunk | 14.90 | 10.7% | 4.41 | 3.9% | 3.38x |
| prefill attention | 1.96 | 1.4% | 1.73 | 1.5% | 1.13x |
| MoE router top-k | 0.65 | 0.5% | 2.25 | 2.0% | 0.29x |
| MoE expert reduce | 3.16 | 2.3% | 1.98 | 1.8% | 1.60x |
| activation quantization (Q8_1) | 2.45 | 1.8% | 2.09 | 1.9% | 1.17x |
| norms, elementwise, LM head, copies | 4.39 | 3.1% | 8.77 | 7.8% | 0.50x |
| **total** | **139.33** | 100% | **112.90** | 100% | **1.23x** |

The yuuka rows are `moe_gate_up_mmq` 39.37, `moe_down_mmq` 19.27, `gemm` 53.19,
`recurrent_chunk` 14.90, `attention_prefill_paged` 1.96, `moe_route` 0.65,
`moe_reduce_add` 3.16, `quant_q8_1` plus `silu_mul_quant_q8_1` 2.45, and the
remaining norms, elementwise, LM head, and copy operations 4.39 ms. The
reference's dense GEMM row is `mul_mat_q<13>`/`<14>` 28.86 plus the rocBLAS
kernels 11.04 and 0.19 of F32 matvecs, and its last row includes 1.32 ms of
Q5_K/Q6_K single-row matvecs.

The routed expert stage is 58.6 ms of the 139.3 ms chunk against the
reference's 51.6 ms: gate/up is 26% slower and down is 6% faster, so the two
expert matrices together are within 1.14x of the reference's int8 MMQ kernels at
this chunk size. The 26.4 ms the chunk trails by is elsewhere: the dense GEMM is
13.1 ms behind, the gated delta-net chunk 10.5, the expert gate/up 8.2, the
expert-output reduction 1.2, quantization 0.4, and attention 0.2, offset by
yuuka's norms and elementwise stages (4.4 ahead), its router (1.6 ahead), and
its expert down projection (1.2 ahead). The dense GEMM and the gated delta-net
chunk are the next targets; this record does not attribute them further.

### Decode, one token

| Stage | yuuka ms | share | llama.cpp ms | share | yuuka / llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected expert gate/up (IQ2_XXS) | 2.02 | 11.8% | 1.29 | 10.1% | 1.56x |
| selected expert down (IQ2_S, IQ3_S) | 1.57 | 9.2% | 1.19 | 9.4% | 1.32x |
| LM head and dense matvec family | 10.13 | 59.3% | 7.49 | 59.0% | 1.35x |
| gated delta-net | 0.81 | 4.7% | 0.38 | 3.0% | 2.13x |
| MoE router top-k | 0.48 | 2.8% | 0.26 | 2.0% | 1.85x |
| MoE expert reduce | 0.63 | 3.7% | 0.08 | 0.6% | 7.9x |
| attention | 0.13 | 0.7% | 0.11 | 0.9% | 1.18x |
| activation quantization (Q8_1) | — | — | 0.38 | 3.0% | — |
| norms, elementwise, copies | 1.31 | 7.7% | 1.51 | 11.9% | 0.87x |
| **total** | **17.07** | 100% | **12.69** | 100% | **1.35x** |

The yuuka rows are `moe_gate_up_selected` 2.02, `moe_down_selected` 1.57,
`lm_head` with `gemv`, `gemv_pair`, and `gemv_residual` 10.13,
`recurrent_token` with `conv_l2norm` 0.81, `moe_route` 0.48, `moe_reduce_add`
0.63, `attention_paged` 0.13, and the remaining norms, rope, KV writes, and
copies 1.31 ms. The reference's dense matvec row is its
`mul_mat_vec_q<12..14>` (Q4_K/Q5_K/Q6_K) and `mul_mat_vec_f` kernels; its expert
matvecs (`mul_mat_vec_q<16>`, `<22>`, `<21>`) are the two rows above. yuuka's
decode runs the FP32 streaming matvec and decodes the expert digits per value,
so it has no Q8_1 pass in this step, while the reference spends 0.38 ms
quantizing the activations its int8 matvecs read.

The experts are 3.6 ms of the 17.1 ms step against the reference's 2.5 ms, and
the largest single difference is the LM head and dense matvec family: 10.13 ms
across 311 operations against 7.49 ms.

## Assignment tile (ac7c697)

The grouped int8 WMMA route walks an expert's assignment bucket in tiles of
`n[5]` rows, and the tile width is a dispatch table measured with
`build/yuuka_moe_sweep --route mmq --dtype iq2xxs --experts 256 --used 8
--hidden 2048 --ffn 512`, three runs per cell, median reported. The stage
columns are one profiled submission after three warmups:

| Chunk rows | Tile | Gate/up ms | Down ms | Total ms |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 1.611 | 0.677 | 2.485 |
| 128 | 32 | 2.254 | 0.966 | 3.433 |
| 128 | 64 | 3.604 | 1.579 | 5.363 |
| 128 | 128 | 6.359 | 3.130 | 9.719 |
| 512 | 16 | 2.425 | 1.318 | 4.284 |
| 512 | 32 | 2.416 | 1.334 | 4.351 |
| 512 | 64 | 3.633 | 1.808 | 5.966 |
| 512 | 128 | 6.744 | 3.644 | 10.945 |
| 1024 | 32 | 3.590 | 2.485 | 7.176 |
| 2048 | 32 | 6.135 | 5.487 | 13.365 |
| 4096 | 64 | 10.580 | 12.673 | 26.351 |

`SelectMoEPrefillPath` uses this table: 16 up to 384 rows, 32 from 512 to 2048,
and 64 from 4096. The synthetic sweep ties between 16 and 32 at 512 rows, and
`TestGPUProfileMoE` decides that boundary: with 16 the 512-row expert stage is
157.6 ms and with 32 it is 142.1 ms, because routing in the real model
concentrates more assignments per expert than a random draw and a 16-wide tile
then needs a second weight pass. The 128-wide tile loses at every row count
because it halves the channel tile and doubles the block count.

## Correctness gates at ac7c697

Every number above was taken at a commit whose MoE path was verified on the same
host and model:

- GPU decode keeps the CPU top-1 token, stays within `0.00286746` maximum
  absolute logit difference of the CPU reference on the first prompt token, and
  emits the four llama.cpp golden tokens `[11751, 11, 264, 3177]`
  (` Paris, a city`) (`TestGPUMoEGoldenGeneration`).
- The int8 WMMA expert route and the Q8_1 `dp4a` route produce identical
  first-token logits (maximum difference 0) and the same golden continuation on
  the same prompt (`TestGPUMoEMmqMatchesQ8`), so the route the dispatch selects
  does not change model behavior.
- A 1024-token prompt prefilled at chunk sizes 128, 512, and 1024 keeps top-1
  `33075` with a maximum logit difference of 0, so the grouped expert route is
  chunk invariant (`TestGPUChunkedPrefillLongContext`).
- A 256-token prompt at 64- and 256-row chunks keeps the CPU top-1 token and
  reproduces the CPU greedy continuation `[279, 15217, 5388, 13]` with logit
  differences of 1.07 and 1.14 against the 2.0 gate
  (`TestGPUMoEChunkedPrefill`; [testing.md](../testing.md) explains why prefill
  is gated on tokens rather than on a tight logit bound).
- `ctest --test-dir build` passes 13/13, including `hip_iq_moe_correctness`
  (the three IQ codebooks against the CPU reference values and the grouped and
  selected pipelines within `1.8e-7` of a host computation),
  `hip_iq_moe_mmq_correctness` (the int8 WMMA IQ route against the `dp4a` route
  at a `1e-5` relative bound and against the dequantized decode at `2e-2`, for
  all three codebooks at assignment tiles 16/32/64/128), and
  `hip_moe_mmq_correctness` (the K-quant form of the same route).
- `go build ./...`, `go vet ./...`, `go test ./...`, `go test -race ./...`, and
  `go test -tags hip ./...` pass.

## Reproduction

```bash
cd /path/to/yuuka
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
MODEL=/path/to/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf

# Throughput. F16 KV and a 512-token chunk are the defaults.
for p in 128 512 2048; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip \
    --prompt-tokens "$p" --decode-tokens 128 --warmup 1 --reps 3
done

# Reference, same protocol.
cd /path/to/llama.cpp/build/bin
for p in 128 512 2048; do
  LD_LIBRARY_PATH=. ./llama-bench -m "$MODEL" -p "$p" -n 128 -r 3 -ngl 99
done

# yuuka attribution: one 128-row chunk plus one decode step per operation.
cd /path/to/yuuka
YUUKA_TEST_MOE_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfileMoE -count=1 -v

# Reference attribution. rocprofv3 cannot write its database from a Go process,
# so the reference is traced as a native binary and reduced by kernel name.
mkdir -p /tmp/ref && cd /tmp/ref
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin rocprofv3 --kernel-trace -f csv \
  -o pp -d /tmp/ref -- \
  /path/to/llama.cpp/build/bin/llama-bench -m "$MODEL" -p 128 -n 0 -r 1 -ngl 99
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin rocprofv3 --kernel-trace -f csv \
  -o tg -d /tmp/ref -- \
  /path/to/llama.cpp/build/bin/llama-bench -m "$MODEL" -p 0 -n 16 -r 1 -ngl 99

# Assignment tile sweep.
cd /path/to/yuuka
for j in 16 32 64 128; do
  build/yuuka_moe_sweep --route mmq --dtype iq2xxs --experts 256 --used 8 \
    --hidden 2048 --ffn 512 --rows 128,512,1024,2048,4096 --tile-j "$j"
done
```

For the reference trace, sum `End_Timestamp - Start_Timestamp` per `Kernel_Name`
and split the dispatch sequence on gaps wider than 3 ms: the prefill chunk is
the single large group, and the decode group holds one `topk_moe_cuda` dispatch
per layer per step, which gives the step count.

## Known limits

- Single request only; fused multi-request MoE decode does not exist, so this is
  not a serving-throughput number.
- The attribution covers one chunk size (128 rows) and one decode step at a
  short context. The throughput rows are direct measurements; the attribution
  rows are kernel time from an instrumented run, as described above.
- Prompt lengths above 2048 tokens, other KV formats, and other `qwen35moe`
  GGUFs are not measured for this model.
- The routed expert stage streams 9.06 GB of unique weight bytes at 155 GB/s for
  a 128-row chunk and 64 GB/s for a 512-row chunk, against the 244 GB/s read
  ceiling of [benchmarks.md](../benchmarks.md). The 512-row loss is the work
  around the weight pass (two token tiles of MMA, four times the gathered
  activation bytes, and four times the expert-output writes), not the read
  itself; the fused gate/up/SwiGLU/quantize epilogue that would remove the
  intermediate gate/up tiles is not implemented.
- The prefill protocol of the repository is owned by
  [benchmarks.md](../benchmarks.md); this file adds one model to it and does not
  redefine it.

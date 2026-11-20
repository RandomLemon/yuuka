# Qwen3.6-35B-A3B (`qwen35moe`) benchmark

**Objective.** Record the end-to-end throughput of the `qwen35moe` HIP path on
gfx1151 for prompt lengths 128, 512, and 2048 and a 128-token decode, next to
llama.cpp on the same model, device, KV format, and single-request protocol.
This is a first measured record for the MoE path, not an acceptance target: the
`qwen35moe` HIP kernels were added for correctness and have no throughput
optimization yet.

**Status.** Measured on 2026-11-19 at `dd5dbf4`. Prefill reaches 51.8-61.7
tokens/s and decode 19.4-19.5 tokens/s; llama.cpp reaches 1111-1571 tokens/s
prefill and 67.9 tokens/s decode. The gap is measured and attributed under
[Measured attribution](#measured-attribution) and summarized under
[Why the paths differ](#why-the-paths-differ); correctness at this commit is
recorded under [Correctness gates](#correctness-gates).

## Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-19 |
| Model | `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` |
| Model SHA-256 | `2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef` |
| Model size | 10,756,586,464 bytes (10.01 GiB, 2.0625 bpw, 34.66 B parameters) |
| Architecture | `qwen35moe`, 40 layers (10 full attention, 30 gated delta-net), 256 routed experts, top-8, expert FFN 512, shared FFN 512 |
| Quantization | Router and shared-expert gate F32; shared expert Q5_K/Q6_K; expert gate/up IQ2_XXS; expert down IQ2_S (37 layers) and IQ3_S (3 layers) |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, Wave32, 122,880 MiB reported VRAM |
| ROCm / HIP | ROCm 10.1.0, HIP 7.16.26385, clang 24.0.0git |
| Host | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads, 120 GiB shared memory |
| Kernel | 7.0.14-28-pve |
| Power profile | GPU DPM `auto` (amdgpu sysfs), CPU governor `performance` |
| Go | go1.24.4, `-tags hip` |
| Commit | `dd5dbf4` |
| llama.cpp | upstream v1.0.0, `b1b770d90`, ROCm build, `-ngl 99`, flash attention `auto` |
| Tokenizer / chat template | GGUF-embedded Qwen3.6 vocabulary and template |
| Sampling | Greedy; decode throughput does not depend on which token is drawn |
| KV format | F16 on both sides (the engine default) |
| Prefill chunk | 512 tokens on both sides (`--chunk-size 512`, llama.cpp batch 2048 / ubatch 512) |
| Warmup / repetitions | 1 / 3, median (`yuuka bench`) and mean ± standard deviation (`llama-bench`), no prefix cache |
| Concurrency | 1 request |

`yuuka bench` reports TTFT from submission and `decode = (decode_tokens - 1) /
decode p50`, matching the prefill and decode split of `llama-bench`. Numbers are
observations on one machine, not guarantees; device clock, power, and thermal
state are not pinned, and the two sides ran sequentially, not interleaved.

## Results

| Measurement | yuuka (tokens/s) | yuuka wall clock p50 | llama.cpp (tokens/s) | upstream / yuuka |
| --- | ---: | ---: | ---: | ---: |
| prefill 128 tokens | 61.67 | 2.076 s | 1111.04 ± 21.79 | 18.0x |
| prefill 512 tokens | 57.39 | 8.921 s | 1544.60 ± 16.88 | 26.9x |
| prefill 2048 tokens | 51.79 | 39.543 s | 1570.72 ± 16.22 | 30.3x |
| decode 128 tokens | 19.49 | 6.515 s | 67.86 ± 0.79 | 3.5x |

Detail:

- yuuka prefill p95 was 2.187 s, 8.962 s, and 40.031 s at 128, 512, and 2048
  tokens; TTFT equals the prefill time because the measurement is single
  request.
- yuuka decode measured 6.504 s (19.53 tokens/s), 6.515 s (19.49), and 6.549 s
  (19.39) at the three prompt lengths, so the decode rate is independent of
  prompt length within 0.7%.
- llama.cpp decode measured 67.86 ± 0.79 tokens/s in the 512-token run and
  66.95 ± 0.03 tokens/s in the 128/2048-token run.
- yuuka prefill rate falls 16% from 128 to 2048 tokens; llama.cpp rises 41%
  over the same range, so the gap widens with prompt length.

## Measured attribution

Measured 2026-11-19 in one session on the same device and GGUF as the table
above. The yuuka column is `TestGPUProfileMoE`, which attributes one 128-row
prefill chunk and one decode step per planned operation with HIP events. The
reference column is a `rocprofv3 --kernel-trace` of
`llama-bench -p 128 -n 8 -r 1 -ngl 99`, reduced by kernel name, with the
IQ types resolved from `ggml_type`: 16 is IQ2_XXS, 22 IQ2_S, 21 IQ3_S.

`rocprofv3` inflates a traced run per dispatch. The reference's own traced run
reported 18.8 ms per decode token against a 25.3 ms kernel sum, so the decode
reference rows below subtract the implied 2.2 µs per dispatch; the prefill rows
need no correction because the kernels dominate (110.5 traced against a
116 ms wall clock).

### Prefill, one 128-token chunk

| Stage | yuuka ms | share | llama.cpp ms | share | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert gate/up (IQ2_XXS) | 1046.6 | 51.0% | 30.1 | 27.3% | 34.7x |
| expert down (IQ2_S, IQ3_S) | 903.0 | 44.0% | 19.6 | 17.7% | 46.0x |
| MoE router top-k | 20.8 | 1.0% | 0.3 | 0.3% | 70x |
| dense GEMM, attention, delta-net, tail | 80.6 | 3.9% | 60.5 | 54.7% | 1.33x |
| **total** | **2051.0** | 100% | **110.5** | 100% | **18.6x** |

Both columns are per-operation or per-dispatch timings, so they exceed the
release wall clocks in the table above (2.076 s for yuuka, 0.116 s for the
reference) by the instrumentation overhead.

98% of the 1.94 s prefill gap is the two expert kernels; every other stage
is inside 1.33x of the reference. The reference's split is the inverse:
`mul_mat_q<16>` (expert gate/up) 30.1 ms, `mul_mat_q<22>` and `<21>` (expert
down) 19.6 ms, `topk_moe_cuda` 0.3 ms, and 60.5 ms for the rest
(`mul_mat_q<13>` 22.1, rocBLAS 10.8, `mul_mat_q<14>` 6.8,
`gated_delta_net_cuda` 3.7, `quantize_mmq_q8_1` 2.1, `mm_ids_helper` 1.8,
`moe_weighted_reduction_f32` 1.9, attention and elementwise kernels).

### Decode, one token

| Stage | yuuka ms | share | llama.cpp ms | share | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE router top-k | 17.6 | 33.9% | 0.34 | 1.8% | 51x |
| selected expert gate/up + down | 19.2 | 36.8% | 4.40 | 23.4% | 4.4x |
| matvec family, LM head, attention, norms | 15.2 | 29.3% | 14.06 | 74.8% | 1.1x |
| **total** | **52.0** | 100% | **14.8** | 100% | **3.4x** |

The yuuka total is its profiled step against a 51.0 ms `yuuka bench` wall
clock; the reference total is its release wall clock, because its trace has to
be corrected first. Both columns describe one token at a short context.

The dense and attention tail is at parity, which is the same conclusion
[benchmarks.md](../benchmarks.md#llamacpp-comparison) records for `qwen35`. The
decode gap is the router kernel (17.3 ms) plus the expert kernels (14.8 ms).

### Root cause 1: one scalar decode per weight value

Both expert stages are governed by the same quantity: how fast one packed
weight value is decoded and consumed. `iq2xxs_value`, `iq2s_value`, and
`iq3s_value` recompute the block offset and fetch the codebook entry, the sign
byte, the aux words, and the block scale per element. The compiler lowers those
`__constant__` tables to per-lane loads because the index is divergent, so one
value costs about five memory operations and thirty integer operations.
llama.cpp's `vec_dot_iq2_xxs_q8_1` instead fetches one 8-value codebook entry
per eight values and accumulates them with two `dp4a` instructions against Q8_1
activations; its MMQ path additionally stages a decoded int8 tile in shared
memory once and reuses it across the rows of the tile.

| Stage | yuuka decoded values | yuuka | llama.cpp |
| --- | ---: | ---: | ---: |
| prefill 128 rows, expert FFN | 128.8 G | 66 G/s (1949.6 ms) | 648 G/s (49.7 ms) |
| decode 1 token, expert FFN | 1.007 G | 52.6 G/s (19.2 ms) | 229 G/s (4.40 ms) |

yuuka decodes 128.8 G values in the chunk because of the next cause; the
reference decodes the same weight set once, 32.2 G values. The prefill stage
gap is then the product of the two factors: 648/66 = 9.8x per decoded value,
times the 4.0x extra values, which is the measured 39x.

### Root cause 2: no cross-row weight reuse (prefill only)

The grouped kernels loop over an expert's binned assignments inside the block
and re-read and re-decode the packed row for each one
([kernels.md](../kernels.md#81-moe-expert-pipeline)), so a chunk decodes
`rows * used` weight rows instead of one row per expert: at 128 rows that is 4x
more decode work than the reference, and the factor grows with the prompt length
because `rows * used / experts` grows with it. The measured consequence is that
the per-row expert cost equals the per-token decode cost:

| Prefill rows | yuuka ms/row | llama.cpp ms/row |
| ---: | ---: | ---: |
| 128 | 15.8 | 0.91 |
| 512 | 18.1 | 0.63 |
| 2048 | 19.0 | 0.63 |

yuuka's expert stage costs 19 ms per row against the 19.2 ms of a one-token
decode step, so a chunk is charged as N independent decode steps and the
attainable prefill rate cannot rise above about 1/19 ms. The reference amortizes
one weight read over the whole tile and falls to 0.63 ms per row as the prompt
lengthens.

### Root cause 3: a serial router

The decode router kernel is serialization, not arithmetic: it launches one
block per row and lets thread 0 run an `O(used x experts x used)` scan with
dependent loads, which is 0.44 ms per layer for a single row. The reference
holds the 256 logits in warp registers and reduces within the warp, 13 µs per
layer. Removing this kernel's cost alone would take the decode step from
52.0 ms to about 34.4 ms (19.6 to about 29 tokens/s) without touching the
experts.

### Bounds

- The expert weights are 9.06 GB (expert gate/up 5.54 GB of IQ2_XXS, expert down
  3.53 GB of IQ2_S and IQ3_S). Every expert receives assignments at 128 rows
  (1024 assignments over 256 experts), so one chunk must stream that whole set
  at least once: 37 ms at the 244 GB/s read ceiling
  ([benchmarks.md](../benchmarks.md#isolated-kernels)). The reference measures
  49.7 ms, that is 182 GB/s, and is therefore memory bound at this length. yuuka
  moves the same 9.06 GB of unique bytes in 1949.6 ms, 4.6 GB/s, 40x off the
  roofline, which is what the per-value decode rate and the 4x re-decode add up
  to.
- Decode reads 8 of 256 experts, 283 MB per token. The reference moves it at
  64 GB/s in the matvec path; yuuka at 15 GB/s.
- The decode router kernel serializes a whole layer's routing into one thread
  (see [root cause 3](#root-cause-3-a-serial-router)).

### What the gap is not

- Not quantization, KV format, output selection, or sampling: one greedy
  request, F16 KV on both sides, no prefix cache, no speculation.
- Not the router projection: it is 2048x256 per row and costs 54.5 ms of the
  2051 ms chunk together with every other dense GEMM. The FP32 router GEMM is a
  numerical choice, not a throughput one
  ([kernels.md](../kernels.md#81-moe-expert-pipeline)).
- Not attention: 2.0 ms of the chunk at this length.
- Not the GPU being busy elsewhere: the kernels above account for 95% of the
  chunk and 99% of the step.

## Why the paths differ

In one sentence: the expert FFN, 95% of the chunk and 37% of the token, decodes
every IQ value with scalar code and re-decodes each weight row once per assigned
token, while the reference stages decoded tiles in shared memory, consumes them
with `dp4a` against Q8_1 activations, and amortizes each staged tile over the
rows of the tile. The dense and attention stages that the earlier `qwen35` work
optimized are already within 1.1-1.3x of the reference on this model.

## Correctness gates

The numbers above were taken at a commit whose MoE path was verified on the same
host and model:

- GPU decode keeps the CPU top-1 token, stays within `0.0027` maximum absolute
  logit difference of the CPU reference on the first prompt token, and emits the
  four llama.cpp golden tokens `[11751, 11, 264, 3177]` (` Paris, a city`).
- A 256-token prompt at 64- and 256-row chunks keeps the CPU top-1 token and
  reproduces the CPU greedy continuation; the recorded logit differences are
  0.96 and 1.04 ([testing.md](../testing.md) explains why prefill is gated on
  tokens rather than on a tight logit bound).
- `ctest --test-dir build` passes 11/11, including `hip_iq_moe_correctness`,
  which pins the three IQ codebooks against the CPU reference values and
  requires the grouped and selected expert pipelines to agree with a host
  reference within `1.7e-7`.
- `go test ./...`, `go test -race ./...`, and `go test -tags hip ./...` pass.
- `yuuka run --backend hip` generates ` Paris, a city` with F16 and F32 KV.

## Reproduction

```bash
cd /path/to/yuuka
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
MODEL=/path/to/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf

/tmp/yuuka-hip bench --model "$MODEL" --backend hip \
  --prompt-tokens 512 --decode-tokens 128 --reps 3 --warmup 1

# Measured attribution: one 128-row chunk plus one decode step per operation.
YUUKA_TEST_MOE_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfileMoE -count=1 -v

# Reference attribution. rocprofv3 cannot write its database from a Go process,
# so the reference is traced as a native binary and reduced by kernel name.
cd /path/to/llama.cpp/build/bin
LD_LIBRARY_PATH=. ./llama-bench -m "$MODEL" -p 128,512,2048 -n 128 -r 3 -ngl 99
mkdir -p /tmp/ref && cd /tmp/ref && LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin \
  rocprofv3 --kernel-trace -f csv -o ref -d /tmp/ref -- \
  /path/to/llama.cpp/build/bin/llama-bench -m "$MODEL" -p 128 -n 8 -r 1 -ngl 99
```

Reduce the reference trace by summing `End_Timestamp - Start_Timestamp` per
`Kernel_Name`, splitting the dispatch sequence on gaps wider than 3 ms into the
prefill chunk and the decode steps.

## Known limits

- Single request only; fused multi-request MoE decode does not exist, so this is
  not a serving-throughput number.
- No IQ kernel specialization, weight-row reuse across assignments, or int8
  prefill route for the expert matrices; each of those is a candidate for a
  later optimization pass measured against this table.
- The attribution covers one chunk size (128 rows) and one decode step at a
  short context. The prefill rows are direct measurements; the reference decode
  rows carry the per-dispatch correction described above, so they are estimates
  within a few percent, not traced values.
- Prompt lengths above 2048 tokens, other KV formats, and other IQ2_XXS
  `qwen35moe` GGUFs are not measured.
- The prefill protocol of the repository is owned by
  [benchmarks.md](../benchmarks.md); this file adds one model to it and does not
  redefine it.

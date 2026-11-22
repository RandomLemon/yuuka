# Qwen3.6-35B-A3B (`qwen35moe`) benchmark

**Objective.** Record the `qwen35moe` HIP throughput on gfx1151 for prompt
lengths 128, 512, and 2048 and for a 128-token decode, next to llama.cpp on the
same model, device, KV format, and single-request protocol, with the
per-operation attribution of the difference and the correctness gates the run
was taken with.

**Status.** Measured on 2026-11-22 at `ac7c697`. yuuka reaches 954, 1249, and
1070 tokens/s prefill at 128, 512, and 2048 prompt tokens and 63.5, 63.0, and
61.9 tokens/s decode, that is 0.87, 0.80, and 0.71 of the reference's prefill and
0.94, 0.93, and 0.92 of its decode. The grouped int8 WMMA route
([kernels.md](../kernels.md#81-moe-expert-pipeline)) runs the routed experts; its
expert stage is within 1.14x of the reference's at a 128-row chunk, and the
remaining gap is in the dense stages (table below).

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
| Power profile | GPU DPM `performance` (amdgpu sysfs), CPU governor `performance` (`amd-pstate-epp`) |
| Go | go1.24.4, `-tags hip` |
| Commit | `ac7c697` |
| llama.cpp | upstream `b1b770d90` (build 12359), ROCm build, `-ngl 99`, `-fa auto` |
| Tokenizer / chat template | GGUF-embedded Qwen3.6 vocabulary and template |
| Sampling | Greedy; decode throughput does not depend on which token is drawn |
| KV format | F16 on both sides (the engine default) |
| Prefill chunk | 512 tokens on both sides (`--chunk-size 512`, llama.cpp batch 2048 / ubatch 512) |
| Warmup / repetitions | 1 / 3 (median for `yuuka bench`, mean ± standard deviation for `llama-bench`) |
| Concurrency | 1 request |

`yuuka bench` reports TTFT from submission and `decode = (decode_tokens - 1) /
decode p50`, matching the prefill and decode split of `llama-bench` (`pp` and
`tg`). Numbers are observations on one machine, not guarantees; device clocks,
power, and thermal state are not pinned beyond the settings above, and the two
sides ran sequentially, not interleaved. Both were run twice; the two yuuka runs
differ by at most 2.2%.

## Results

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

## Measured attribution

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

## Assignment tile

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

## Correctness gates

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

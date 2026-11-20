# Qwen3.6-35B-A3B (`qwen35moe`) benchmark

**Objective.** Record the end-to-end throughput of the `qwen35moe` HIP path on
gfx1151 for prompt lengths 128, 512, and 2048 and a 128-token decode, next to
llama.cpp on the same model, device, KV format, and single-request protocol.
The record has two parts: the measured gap as first implemented, and the
optimization pass that followed the attribution.

**Status.** Measured on 2026-11-19. The first record at `dd5dbf4` reached
51.8-61.7 tokens/s prefill and 19.4-19.5 tokens/s decode against llama.cpp's
1111-1571 and 67.9. [Measured attribution](#measured-attribution) then located
98% of the prefill gap in the two grouped expert kernels and a third of the
decode step in the router kernel; the [optimization record](#optimization-record)
lists the three kernels written against that attribution and their effect.
At `HEAD` prefill reaches 448-534 tokens/s and decode 55.0-57.4 tokens/s, that
is 0.29-0.47 of the reference's prefill and 0.83-0.87 of its decode, with
correctness recorded under [Correctness gates](#correctness-gates).

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

The first column pair is the record of the correctness implementation at
`dd5dbf4`; the second pair is the same measurement protocol after the
[optimization record](#optimization-record). Both use 128 decode tokens for the
decode row, F16 KV, and one request.

| Measurement | yuuka `dd5dbf4` | llama.cpp | ratio | yuuka `HEAD` | llama.cpp | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill 128 tokens | 61.67 | 1111.04 ± 21.79 | 18.0x | 509.50 | 1092.24 ± 35.11 | 2.14x |
| prefill 512 tokens | 57.39 | 1544.60 ± 16.88 | 26.9x | 533.80 | 1600.79 ± 4.76 | 3.00x |
| prefill 2048 tokens | 51.79 | 1570.72 ± 16.22 | 30.3x | 448.38 | 1558.81 ± 11.67 | 3.48x |
| decode, 16 tokens | 19.49 | 67.86 ± 0.79 | 3.5x | 57.38 | 65.98 ± 0.17 | 1.15x |

The `HEAD` rows are `yuuka bench --warmup 1 --reps 3` and
`llama-bench -r 3`, measured back to back in one session on 2026-11-19; the
decode ratio uses the 128-token prompt row. Device clocks are not pinned, and
the 2048-token prefill row moved between 448 and 481 tokens/s across repeated
runs of the same binary.

Detail of the first record:

- yuuka prefill p95 was 2.187 s, 8.962 s, and 40.031 s at 128, 512, and 2048
  tokens; TTFT equals the prefill time because the measurement is single
  request.
- yuuka decode measured 6.504 s (19.53 tokens/s), 6.515 s (19.49), and 6.549 s
  (19.39) at the three prompt lengths, so the decode rate was independent of
  prompt length within 0.7%.
- llama.cpp decode measured 67.86 ± 0.79 tokens/s in the 512-token run and
  66.95 ± 0.03 tokens/s in the 128/2048-token run.
- yuuka prefill rate fell 16% from 128 to 2048 tokens; llama.cpp rose 41% over
  the same range, so the gap widened with prompt length.

Detail of the `HEAD` rows:

- yuuka prefill p50 was 0.251 s, 0.959 s, and 4.568 s at 128, 512, and 2048
  tokens, 0.259 s of it previously.
- The prefill rate still falls with prompt length (509.5 to 448.4 tokens/s over
  a 16x longer prompt), because the assignment tile of the grouped kernels
  bounds the weight decode to one pass per eight assignments while the tile
  covers only four assignments at 128 rows: a longer chunk re-decodes each
  expert row ceil(rows * used / experts / 8) times.
- yuuka decode measured 57.38, 57.34, and 55.01 tokens/s at the three prompt
  lengths, so it stays flat in prompt length within 4%.
- Decode is now behind the reference by 15% and prefill by 2.1-3.5x; the
  remaining prefill gap is the per-value arithmetic of the FP32 expert dot, see
  [what remains](#what-remains).

## Measured attribution

Measured 2026-11-19 in one session on the same device and GGUF as the table
above. The yuuka column is `TestGPUProfileMoE`, which attributes one 128-row
prefill chunk and one decode step per planned operation with HIP events. The
reference column is a `rocprofv3 --kernel-trace` of
`llama-bench -p 128 -n 8 -r 1 -ngl 99`, reduced by kernel name, with the
IQ types resolved from `ggml_type`: 16 is IQ2_XXS, 22 IQ2_S, 21 IQ3_S.

`rocprofv3` inflates a traced run per dispatch. The reference's own traced run
reported 18.8 ms per decode token against a 25.3 ms kernel sum, so the decode
reference rows below are scaled to the reference's release step; the prefill
rows need no correction because the kernels dominate (110.5 traced against a
116 ms wall clock).

Two sets of yuuka numbers appear below: the correctness implementation of
`dd5dbf4`, which is the record the [optimization record](#optimization-record)
answers, and `HEAD`.

### Prefill, one 128-token chunk

| Stage | `dd5dbf4` ms | `HEAD` ms | share | llama.cpp ms | `HEAD` ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert down (IQ2_S, IQ3_S) | 903.0 | 89.7 | 35.2% | 19.6 | 4.6x |
| expert gate/up (IQ2_XXS) | 1046.6 | 77.4 | 30.4% | 30.1 | 2.6x |
| dense GEMM (router, attention, shared expert) | 54.5 | 59.5 | 23.3% | 39.7 | 1.5x |
| gated delta-net chunk | 15.1 | 16.7 | 6.5% | 3.7 | 4.5x |
| prefill attention | 2.0 | 2.2 | 0.9% | 1.7 | 1.3x |
| MoE router top-k | 20.8 | 0.7 | 0.3% | 0.3 | 2.3x |
| norms, elementwise, LM head, activation quantization | 8.7 | 8.7 | 3.4% | 15.4 | 0.6x |
| **total** | **2051.0** | **254.9** | 100% | **110.5** | **2.3x** |

The reference's 110.5 ms is `mul_mat_q<16>` (expert gate/up) 30.1,
`mul_mat_q<22>` and `<21>` (expert down) 19.6, `mul_mat_q<13>` 22.1,
rocBLAS 10.8, `mul_mat_q<14>` 6.8, `gated_delta_net_cuda` 3.7,
`quantize_mmq_q8_1` 2.1, `mm_ids_helper` 1.8,
`moe_weighted_reduction_f32` 1.9, `topk_moe_cuda` 0.3, attention 1.7, and the
remaining elementwise kernels 15.4.

The expert stage is now 167.1 ms of the 254.9 ms chunk instead of 1949.6 ms of
2051.0 ms, and the router kernel is 0.7 ms instead of 20.8 ms. The dense GEMM,
the gated delta-net chunk, and the attention kernel are the stages the earlier
`qwen35` pass already optimized, and they are the largest remaining non-expert
items.

### Decode, one token

The yuuka shares come from the profiled step (19.2 ms against a 17.4 ms release
step); the reference shares come from its trace and are scaled to its release
step (15.2 ms per token).

| Stage | `dd5dbf4` ms | `HEAD` ms | share | llama.cpp ms | `HEAD` ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected expert gate/up + down | 19.2 | 3.2 | 18.5% | 2.9 | 1.1x |
| MoE router top-k | 17.6 | 0.4 | 2.4% | 0.3 | 1.4x |
| LM head and dense matvec family | 12.1 | 11.0 | 63.2% | 9.0 | 1.2x |
| norms, delta-net, attention, KV, reduce | 3.0 | 2.8 | 15.9% | 3.0 | 0.9x |
| **total** | **52.0** | **17.4** | 100% | **15.2** | **1.15x** |

The 52.0 ms of `dd5dbf4` is the profiled step; the release step was 51.0 ms.
Decode is now within 15% of the reference, and every stage is inside 1.6x of it.
The remaining difference is the number of small operations: 793 planned
operations per step against the reference's 2979 dispatches, whose per-dispatch
cost is lower.

### Root cause 1: one scalar decode per weight value

Both expert stages were governed by the same quantity: how fast one packed
weight value is decoded and consumed. `iq2xxs_value`, `iq2s_value`, and
`iq3s_value` recompute the block offset and fetch the codebook entry, the sign
byte, the aux words, and the block scale per element. The compiler lowers those
`__constant__` tables to per-lane loads because the index is divergent, so one
value cost about five memory operations and thirty integer operations.
llama.cpp's `vec_dot_iq2_xxs_q8_1` instead fetches one 8-value codebook entry
per eight values and accumulates them with two `dp4a` instructions against Q8_1
activations; its MMQ path additionally stages a decoded int8 tile in shared
memory once and reuses it across the rows of the tile.

`dd5dbf4` decoded 66 G weight values/s in prefill and 52 G/s in decode; the
reference's MMQ tile path decodes 648 G values/s in prefill and its `dp4a`
matvec 229 G values/s in decode. The [optimization record](#optimization-record)
closed the codebook-fetch half of that difference for decode and most of the
reuse half for prefill; the per-value arithmetic is what remains.

### Root cause 2: no cross-row weight reuse (prefill only)

The grouped kernels of `dd5dbf4` looped over an expert's binned assignments
inside the block and re-read and re-decoded the packed row for each one, so a
chunk decoded `rows * used` weight rows instead of one row per expert: 4x more
decode work at 128 rows, and the factor grew with the prompt length because
`rows * used / experts` grows with it. The measured consequence was that the
per-row expert cost equalled the per-token decode cost:

| Prefill rows | `dd5dbf4` ms/row | `HEAD` ms/row | llama.cpp ms/row |
| ---: | ---: | ---: | ---: |
| 128 | 15.8 | 1.99 | 0.91 |
| 512 | 18.1 | 1.87 | 0.63 |
| 2048 | 19.0 | 1.03 | 0.63 |

`HEAD` costs 1.99 ms per row at 128 rows because the assignment tile covers
four assignments while the eight-value tile has to be filled by one pass over
ceil(count / 8) tiles; the per-row cost falls to 1.03 ms at 2048 rows, where the
tile pays for itself.

### Root cause 3: a serial router

The decode router kernel was serialization, not arithmetic: it launched one
block per row and let thread 0 run an `O(used x experts x used)` scan with
dependent loads, which is 0.44 ms per layer for a single row. The reference
holds the 256 logits in warp registers and reduces within the warp, 13 us per
layer. At `HEAD` the router costs 12 us per layer, which is 2.4% of the decode
step and 0.3% of the chunk.

### Bounds

- The expert weights are 9.06 GB (expert gate/up 5.54 GB of IQ2_XXS, expert down
  3.53 GB of IQ2_S and IQ3_S). Every expert receives assignments at 128 rows
  (1024 assignments over 256 experts), so one chunk must stream that whole set
  at least once: 37 ms at the 244 GB/s read ceiling
  ([benchmarks.md](../benchmarks.md#isolated-kernels)). The reference measures
  49.7 ms, that is 182 GB/s, and is therefore memory bound at this length. At
  `HEAD` the expert stage is 167.1 ms, 54 GB/s of unique weight bytes: it is
  still decode bound, not bandwidth bound, and the FP32 per-value arithmetic is
  the reason.
- Decode reads 8 of 256 experts, 283 MB per token. The reference moves it at
  64 GB/s in the matvec path; `HEAD` at 78 GB/s.
- The chunk's non-expert stages add up to 87.8 ms at `HEAD` against the
  reference's 60.8 ms; the dense GEMM and the gated delta-net chunk are 76.2 ms
  of that.

### What remains

1. **Per-value arithmetic of the expert dot (prefill, 2.6-4.6x on that stage).**
   One codebook fetch per eight values is already in place, but each value still
   costs an int8-to-float conversion, a sign select, and a scale multiply before
   its FP32 product. The reference keeps the values as int8 digits and consumes
   them with `dp4a` against Q8_1 activations, four products per instruction. That
   is the next step, and it changes the arithmetic of the expert path, so it
   needs its own correctness pass and a re-recorded logit gate.
2. **The assignment tile on short chunks.** Four assignments fill a tile of
   eight at 128 rows, so the tile cannot pay for itself; a chunk of 2048 rows
   already runs at 1.03 ms per row.
3. **The decode tail.** 793 planned operations per token at 2.7 ms of small
   kernels; the reference dispatches more operations in less time.
4. **The gated delta-net chunk**, 16.7 ms of the chunk against the reference's
   3.7 ms, which [kernels.md](../kernels.md#91-chunkwise-form) already records
   as an open item.

## Optimization record

Three kernel changes were written against the attribution above, and each was
measured with the protocol of this file. The stage rows are direct
measurements; the decode and prefill rates are the release numbers before and
after the whole pass, because the intermediate steps were not re-run end to
end.

**Router: one warp per row.** `moe_route_kernel` was rewritten from a
single-thread scan to a warp-per-row top-k that keeps the row's experts in lane
registers, masks the winner out of the register set, and reduces the slots with
a butterfly max. The routing result is unchanged: the same lowest-index
tie-break, the same normalization, and the same bucket order.

| Measurement | before | after |
| --- | ---: | ---: |
| `moe_route`, one row, per layer | 0.44 ms | 12 us |
| `moe_route`, 128-row chunk, 40 layers | 20.8 ms | 0.7 ms |

**Expert decode: eight values per codebook fetch.** `iq2xxs_span8`,
`iq2s_span8`, and `iq3s_span8` decode eight consecutive values from one codebook
entry (two for IQ3_S) and hoist the block offset, the sign word, and the scale,
instead of decoding one value per table fetch. The selected-expert kernels were
restructured to one warp per output row with a warp reduction, so the block-wide
tree reduction and its barriers are gone. The decoded values are the same
expressions the per-value decoders evaluate.

| Measurement | before | after |
| --- | ---: | ---: |
| `moe_gate_up_selected` + `moe_down_selected`, per token | 19.2 ms | 3.6 ms |

**Expert prefill: decode once per assignment tile.** The grouped kernels now
walk an expert's assignment list in tiles and reuse one decoded weight span
across the tile, which removes the per-assignment re-decode of
`dd5dbf4`. With the span decoder in place they also issue one codebook fetch per
eight values. `kAssignTileIQ` is eight for the IQ codebooks and four for the
general dtypes, whose scalar decoder needs the registers; the tile is a template
parameter, so the two forms are separate instantiations.

| Measurement | before | after |
| --- | ---: | ---: |
| `moe_gate_up_grouped` + `moe_down_grouped`, 128-row chunk | 1949.6 ms | 167.1 ms |
| prefill 128 tokens | 61.7 tokens/s | 509.5 tokens/s |
| prefill 512 tokens | 57.4 tokens/s | 533.8 tokens/s |
| prefill 2048 tokens | 51.8 tokens/s | 448.4 tokens/s |
| synthetic grouped pipeline, 128 experts, 32 rows | 21.6 ms | 11.1 ms |
| decode, 16 tokens | 19.5 tokens/s | 57.4 tokens/s |

The synthetic row is `build/yuuka_moe_sweep`
([benchmarks.md](../benchmarks.md#grouped-moe-pipeline)), which uses Q4_K expert
weights and therefore the generic codebook instance. The routing kernel is 20x
faster there, and the generic gate/up instance still spills 30 VGPRs, which is
why its gain is smaller than the IQ path's.

## Correctness gates

Every record above was taken at a commit whose MoE path was verified on the same
host and model:

- GPU decode keeps the CPU top-1 token, stays within `0.0029` maximum absolute
  logit difference of the CPU reference on the first prompt token (0.0027 before
  the optimization), and emits the four llama.cpp golden tokens
  `[11751, 11, 264, 3177]` (` Paris, a city`).
- A 256-token prompt at 64- and 256-row chunks keeps the CPU top-1 token and
  reproduces the CPU greedy continuation; the recorded logit differences are
  0.65 and 1.25 after the optimization, against 0.96 and 1.04 before it ([testing.md](../testing.md) explains why prefill is gated on
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

# Qwen3.6-35B-A3B (`qwen35moe`) benchmark

**Objective.** Record the `qwen35moe` HIP prefill and decode performance on
gfx1151 against llama.cpp on the same model and device, with the per-operation
attribution that explains the result, the assignment-tile table the dispatch
uses, and the correctness gates the run was taken with.

**Status.** Every table in this file is from one measurement session on
2026-11-23. The matched comparison uses one `llama.cpp` build and one protocol;
the attribution and tile sweep use the same commit. Earlier reports are not
carried forward; the numbers below replace them. The report layout and metadata
requirements are defined in [benchmarks.md](../benchmarks.md). Numbers are
observations on one machine, not guarantees; device clock, power, and thermal
state are not pinned.

## Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-23 |
| Model | `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` |
| Model SHA-256 | `2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef` |
| Model size | 10,756,586,464 bytes (10.01 GiB, 2.0625 bpw); 34.66 B parameters reported by llama.cpp |
| Architecture | `qwen35moe`, 40 layers (10 full attention, 30 gated delta-net), 256 routed experts, top-8, expert FFN 512, shared FFN 512 |
| Quantization | Router and shared-expert gate F32; shared expert Q5_K/Q6_K; expert gate/up IQ2_XXS; expert down IQ2_S (37 layers) and IQ3_S (3 layers) |
| Tokenizer / chat template | GGUF-embedded Qwen3.6 vocabulary and template |
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
| KV format | F16 on both engines |
| Prefill chunk | 512 tokens on both engines |
| Warmup / repetitions / statistic | 1 warmup and 3 measured repetitions for yuuka; 3 measured repetitions for llama-bench; medians |
| Concurrency | 1 request; no prefix cache, profiling, or parallel GPU job during the timed rows |
| Prompt set / token stream | yuuka tokenizes an English corpus to the stated length; llama-bench generates its own random tokens of the same length. Prompt content is length-matched, not token-matched, and this model is routing sensitive |

Protocol:

- `yuuka bench` measures end to end through the engine. TTFT is measured from
  submission; decode throughput is `(decode_tokens - 1) / decode p50`.
- `llama-bench -p <prompt> -n 128 -r 3` measures the reference with full GPU
  offload, `-b 512 -ub 512`, three repetitions per cell. The prefill and decode
  cells are taken from the same per-length runs so each row pairs one yuuka
  process with one reference process.
- `TestGPUProfileMoE` attributes one 128-row and one 512-row chunk and one
  decode step with HIP events around every plan operation. Event
  instrumentation adds a submission gap, so the profiled wall is a diagnosis,
  not a release number.
- `build/yuuka_moe_sweep` measures the synthetic grouped-expert pipeline with
  the release shape (256 experts, top-8, hidden 2048, FFN 512) and reports the
  median of five timed submissions after two warmups per cell.
- Only one benchmark process used the GPU at a time.

## Correctness gates

Same tree and commit as the benchmarks:

| Check | Result |
| --- | --- |
| `ctest --test-dir build --output-on-failure` (13 native tests, including MoE, IQ MoE, MoE MMQ, IQ MoE MMQ, deltanet, plan validation) | 13 / 13 pass |
| `TestGPUMoEGoldenGeneration` | maximum first-logit difference `0.00150061`; golden tokens `[11751, 11, 264, 3177]` |
| `TestGPUMoEChunkedPrefill` | chunk 64 maximum logit difference `1.06693`; chunk 256 `1.13872`; continuation `[279, 15217, 5388, 13]` |
| `go build ./...` | pass |
| `go vet ./...` | pass |
| `go test ./...` | pass |

The chunked-prefill gate is behavior-based: it requires the CPU top-1 token,
the recorded continuation, and a 2.0 logit bound, because the MoE router is
sensitive to hidden-state drift. A 1e-4 relative perturbation of the router
weights moves this model's own logits by about 0.64, so a tight bitwise logit
bound would not be a meaningful release gate.

## Matched release comparison

One request, F16 KV, 512-token chunk, greedy. The prefill rate is tokens/s of
the prompt; the decode rate is the 128 generated tokens after that prompt:

| Prompt | yuuka prefill tok/s | llama.cpp prefill tok/s | Prefill ratio | yuuka TTFT | yuuka decode tok/s | llama.cpp decode tok/s | Decode ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 996.5 | 1092.8 | 0.912 | 0.128 s | 71.66 | 68.17 | 1.051 |
| 512 | 1703.7 | 1559.0 | 1.093 | 0.301 s | 69.99 | 67.98 | 1.030 |
| 2048 | 1714.9 | 1521.7 | 1.127 | 1.194 s | 68.96 | 67.03 | 1.029 |
| 8192 | 1558.4 | 1425.9 | 1.093 | 5.257 s | 65.03 | 67.08 | 0.969 |

The result is not a uniform lead. At 128 prompt tokens yuuka is 8.8% behind the
reference; from 512 to 8192 it leads prefill by 9.3-12.7%. Decode is ahead by
2.9-5.1% at 128-2048 prompt tokens and 3.1% behind at 8192. The 128-token cell
is launch- and weight-limited for this model: the 10.01 GiB of weights
cannot be amortized over a short prompt, and the reference's 512-token default
batch has more work to hide the same weight pass.

The table is length-matched, not token-matched, because the two drivers build
their own prompt token sequences. This model routes tokens to 8 of 256 experts,
so different token content can move throughput by more than it would for a
dense model. The per-cell ratios are the durable part; a different prompt set
is a different measurement.

## Prefill attribution

`TestGPUProfileMoE`, one chunk at positions 0 and 512, HIP events per operation.
Times are profiled and shares are of the profiled chunk:

| Stage | 128 rows, ms | 128 rows, share | 512 rows, ms | 512 rows, share |
| --- | ---: | ---: | ---: | ---: |
| Dense GEMM | 53.884 | 40.9% | 107.727 | 35.7% |
| `moe_gate_up_mmq` | 40.660 | 30.9% | 80.012 | 26.5% |
| `moe_down_mmq` | 19.971 | 15.2% | 50.063 | 16.6% |
| `recurrent_chunk` | 5.884 | 4.5% | 27.046 | 9.0% |
| `moe_reduce_add` | 3.119 | 2.4% | 15.962 | 5.3% |
| `lm_head` | 1.454 | 1.1% | 1.453 | 0.5% |
| `attention_prefill_paged` | 0.915 | 0.7% | 5.557 | 1.8% |
| Chunk wall / profiled | 135.895 / 131.775 | 100% | 299.351 / 301.858 | 100% |

The expert stages are 46.1% of the 128-row chunk and 43.1% of the 512-row
chunk; the dense GEMM is the single largest stage. The recurrent chunk grows
from 4.5% to 9.0% as rows increase, and attention remains under 2% at both
lengths. There are 965 plan operations per chunk.

## Decode attribution

`TestGPUProfileMoE`, one token after a 64-token prompt, 673 plan operations.
The wall is 17.088 ms and the profiled total is 15.072 ms; shares are of the
profiled total:

| Stage | Operations | ms | Share |
| --- | ---: | ---: | ---: |
| `lm_head` | 111 | 5.449 | 36.1% |
| `moe_gate_up_selected` | 40 | 1.918 | 12.7% |
| `gemv_residual` | 40 | 1.550 | 10.3% |
| `moe_down_selected` | 40 | 1.447 | 9.6% |
| `gemv_pair` | 80 | 1.099 | 7.3% |
| `gemv` | 40 | 0.912 | 6.1% |
| `recurrent_token` | 30 | 0.597 | 4.0% |
| `rmsnorm` | 81 | 0.524 | 3.5% |
| `moe_route_selected` | 40 | 0.364 | 2.4% |
| `moe_shared_add` | 40 | 0.360 | 2.4% |
| `conv_l2norm` | 30 | 0.219 | 1.5% |
| `moe_reduce_add` | 40 | 0.195 | 1.3% |
| `attention_paged` | 10 | 0.132 | 0.9% |

Decode is dominated by the vocabulary projection (36.1%) and the selected
expert stages (22.3% for gate/up plus down). The dense GEMV family is 23.7%,
the router and shared-expert combine are 4.8%, and the recurrent state pass is
4.0%. The last decode route selected experts `[154, 39, 156, 104, 233, 124, 35,
38]`, which is the shape the selected-expert kernels read.

## Assignment tile

`build/yuuka_moe_sweep --route mmq --dtype iq2xxs --experts 256 --used 8
--hidden 2048 --ffn 512 --rows 128,512,1024,2048,4096 --tile-j <J>`, total
milliseconds per row count:

| Rows | J=16 | J=32 | J=64 | J=128 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | **2.555** | 3.499 | 5.559 | 9.953 |
| 512 | 4.342 | **3.910** | 6.131 | 11.136 |
| 1024 | 7.703 | **7.224** | 7.538 | 13.343 |
| 2048 | 14.749 | 13.388 | **13.364** | 20.974 |
| 4096 | 29.669 | 26.773 | **26.298** | 47.107 |

The sweep winner shifts with the row count: 16 at 128 rows, 32 at 512 and
1024, and 64 at 2048 and 4096. At 2048 rows J=32 is within 0.2% of J=64, so
the selected row-class rule keeps 32 through 2048 and switches to 64 at 4096;
the difference is inside run-to-run noise at 2048 and 1.8% at 4096. The table
supports a row-class rule rather than one global tile. The synthetic shape has
256 experts, top-8, hidden 2048, and expert FFN 512; it is the release shape,
not a different model.

## Decisions and rejected routes

- **The IQ2_XXS expert route runs the int8 WMMA MMQ kernels.** Both the gates
  and the profile resolve `mmq`, and the tile table above is the dispatch input.
- **Assignment tile is selected per row class**, not globally: 16 at 128 rows,
  32 at 512-2048, 64 at 4096. The 2048-row cell is within 0.2% of the J=64
  winner and the rule keeps 32 there; at 4096 the switch to 64 wins by 1.8%.
  The widest tile (128) is slower at every measured row count.
- **F16 KV and a 512-token chunk stay the defaults.** They are the protocol of
  the comparison and the gates; no other KV format was measured for this model.
- **Decode stays on the selected-expert path.** Gate/up and down are 22.3% of
  the step; the dense GEMV family is 23.7%, and the vocabulary projection alone
  is 36.1%. The step is weight-bandwidth bound, so fusing more work into the
  existing launches matters more than adding parallel expert work.
- **Prefill is limited by the dense GEMM and the expert stages, not attention.**
  Attention is under 2% of both measured chunks; the expert stages are 43-46%
  and the dense GEMM 36-41%. Long-context attention becomes material only after
  the expert work.
- **The 128-token prompt is not a win against llama.cpp.** The reference's
  default 512-token batch hides more of the weight pass at that length; this is
  recorded as a measured limitation rather than hidden in an aggregate.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
MODEL=/path/to/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
sha256sum "$MODEL"   # must match 2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef

ctest --test-dir build --output-on-failure
YUUKA_TEST_MOE_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run 'TestGPUMoEGoldenGeneration|TestGPUMoEChunkedPrefill' -count=1 -v

# Throughput cells: prefill and 128 decode tokens at each prompt length.
for p in 128 512 2048 8192; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
    --prompt-tokens "$p" --decode-tokens 128 --warmup 1 --reps 3 --kv-type f16
done

# Reference, same cells and protocol.
for p in 128 512 2048 8192; do
  /path/to/llama.cpp/build/bin/llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 \
    -p "$p" -n 128 -r 3 -o md
done

# Attribution and tile sweep.
YUUKA_TEST_MOE_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfileMoE -count=1 -v
for j in 16 32 64 128; do
  ./build/yuuka_moe_sweep --route mmq --dtype iq2xxs --experts 256 --used 8 \
    --hidden 2048 --ffn 512 --rows 128,512,1024,2048,4096 --tile-j "$j"
done
```

## Known limits

- One model configuration and one device: Qwen3.6-35B-A3B UD-IQ2_XXS on
  gfx1151. Other quantizations, expert mixes, and devices are not measured.
- The comparison is length-matched, not token-matched. The two engines build
  different prompt token sequences, and this model is routing sensitive; a
  different prompt set is a different measurement.
- Prompt lengths above 8192, other KV formats, and other GGUFs are not measured
  for this model.
- The 128-token prefill cell is 8.8% behind llama.cpp, and decode at 8192 is
  3.1% behind. The model does not lead at every length.
- The OpenAI HTTP server path was not measured; the serving result in this
  repository is the in-process scheduler benchmark on Qwen3.5.
- Attribution is a diagnostic. Event instrumentation adds a submission gap,
  and the clocks are not pinned; use the stage shares, not the absolute
  milliseconds, as the durable result.
- No fused mixed-shape multi-request MoE decode exists, so this is not a
  serving-throughput number.

# Qwen3.6-35B-A3B (`qwen35moe`) benchmark

**Objective.** Record the end-to-end throughput of the `qwen35moe` HIP path on
gfx1151 for prompt lengths 128, 512, and 2048 and a 128-token decode, next to
llama.cpp on the same model, device, KV format, and single-request protocol.
This is a first measured record for the MoE path, not an acceptance target: the
`qwen35moe` HIP kernels were added for correctness and have no throughput
optimization yet.

**Status.** Measured on 2026-11-19 at `dd5dbf4`. Prefill reaches 51.8-61.7
tokens/s and decode 19.4-19.5 tokens/s; llama.cpp reaches 1111-1571 tokens/s
prefill and 67.9 tokens/s decode. The gap is expected and explained under
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

## Why the paths differ

- The MoE expert kernels decode IQ2_XXS, IQ2_S, and IQ3_S one value at a time
  from the device codebooks (`moe_packed_value` and the `iq*_value` helpers in
  `kernels/hip/iq_decode.cuh`), and the grouped prefill kernel re-reads a packed
  weight row for every binned assignment. [kernels.md](../kernels.md#81-moe-expert-pipeline)
  records this as the correctness implementation with no throughput claim.
  llama.cpp dispatches vectorized IQ k-quant kernels.
- Decode uses the selected-expert kernels, which decode the same codebooks per
  value; llama.cpp uses its tuned matrix-vector path.
- Everything else is matched: one request, F16 KV, every layer resident on the
  device, no prefix cache, no speculative decoding, greedy sampling.
- The prefill router projection runs as an FP32 GEMM rather than a WMMA tile so
  the expert selection does not see FP16 activation staging
  ([kernels.md](../kernels.md#81-moe-expert-pipeline)); it is 2048x256 per row
  and is not the throughput limit at these lengths.

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

cd /path/to/llama.cpp/build/bin
LD_LIBRARY_PATH=. ./llama-bench -m "$MODEL" -p 128,512,2048 -n 128 -r 3 -ngl 99
```

## Known limits

- Single request only; fused multi-request MoE decode does not exist, so this is
  not a serving-throughput number.
- No IQ kernel specialization, weight-row reuse across assignments, or int8
  prefill route for the expert matrices; each of those is a candidate for a
  later optimization pass measured against this table.
- Prompt lengths above 2048 tokens, other KV formats, and other IQ2_XXS
  `qwen35moe` GGUFs are not measured.
- The prefill protocol of the repository is owned by
  [benchmarks.md](../benchmarks.md); this file adds one model to it and does not
  redefine it.

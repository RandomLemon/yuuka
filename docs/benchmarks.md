# Benchmark reporting

Measured results live in one report per model under `docs/benchmarks/`. This
file owns the report format, the model-report index, and the metadata and
reproduction requirements. The benchmark protocol is owned by
[testing.md §6](testing.md#6-benchmark-protocol); the correctness strategy,
metric definitions, and regression gates are owned by the rest of
[testing.md](testing.md).

## Model reports

Use one report per model file and quantization preset. Do not merge two models,
architectures, or quantization presets in one file: a report that mixes them
cannot state one model hash, one environment, or one set of gates. File names
are `docs/benchmarks/<family>_<size>[_<variant>].md`, lowercase ASCII, with dots
kept in version numbers and fields separated by `_`. Examples are
`qwen3.5_4b.md` and `qwen3.6_35b_a3b.md`. A second quantization of the same
model gets a quantization suffix, for example `qwen3.5_4b_q8_0.md`.

Measured results belong only in these model reports. Design documents link to
the owning report section instead of restating numbers. A model report may
include isolated kernel or resource tables measured on synthetic shapes when
they support that report's dispatch decisions; label the shape and name the
target model in the section.

| Report | Model file | Architecture | Quantization | Device | Contents |
| --- | --- | --- | --- | --- | --- |
| [Qwen3.5-4B](benchmarks/qwen3.5_4b.md) | `Qwen3.5-4B-Q4_K_M.gguf` | `qwen35` | Q4_K_M | `gfx1151` | Single-request prefill, decode, KV formats, isolated kernels, the synthetic grouped-MoE pipeline sweep, serving, llama.cpp comparison, and measured dispatch decisions |
| [Qwen3.6-35B-A3B](benchmarks/qwen3.6_35b_a3b.md) | `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` | `qwen35moe` | UD-IQ2_XXS | `gfx1151` | Matched prefill and decode comparisons against llama.cpp, per-operation attribution, assignment-tile table, and gates |

## Adding a model report

1. Create `docs/benchmarks/<name>.md` and copy the section order from
   [Report structure](#report-structure).
2. Fill the identity and environment table from the actual run, including the
   full model file name and its SHA-256.
3. Run the correctness gates and record their exact commands and observed
   results before any performance table is written.
4. Measure every coverage row that applies and list the rest in Known limits.
5. Add a Reproduction section that runs from the recorded commit with variables
   and relative paths only.
6. Add the report to the index above and to the module index in
   [README.md](README.md).
7. Update the report in the same change as a dispatch, default, or behavior
   change that invalidates a table.

## Report structure

A model report uses the following sections in this order:

1. Title `# <model> (<architecture>) benchmark`.
2. Objective and status paragraph with the date, full commit, gates, and a
   statement about historical sections.
3. `## Environment and protocol`.
4. Correctness gates, as a top-level section or a subsection before the
   performance tables.
5. Result sections for prefill, decode, serving, feature matrices, isolated
   kernels, and reference comparisons.
6. `## Decisions and rejected routes` when a measurement selected or rejected a
   route, dtype, tile, or default.
7. `## Reproduction`.
8. `## Known limits`.

Historical sections are allowed when they are labeled with their date and
commit. Existing reports keep their current section order; apply this order to
new reports and to a report that is substantially rewritten.

## Model identity and environment

Each report carries one table with these rows:

| Field | Required content |
| --- | --- |
| Date | ISO 8601 date of the measurement |
| Model | Full GGUF file name, not a short model name |
| Model SHA-256 | Lowercase 64-hex digest |
| Model size | Bytes, and parameter count when the report records one |
| Architecture | `general.architecture`, layer counts, attention/recurrent mix |
| Quantization | Preset and per-tensor dtype summary when mixed |
| Tokenizer / chat template | `GGUF-embedded`, or the override and its source |
| Device | Marketing name, gfx target, CU count, reported VRAM |
| ROCm / HIP | ROCm version, HIP runtime, compiler, rocBLAS/rocWMMA versions |
| Host | CPU model, threads, memory |
| Kernel / driver / firmware | Kernel version and amdgpu firmware/driver |
| Power / thermal | GPU DPM level, CPU governor, thermal state or `not pinned` |
| Go | Go version and build tags |
| Commit | Full 40-hex commit of the measured tree |
| Reference implementation | Project, exact commit, build flags, and version whenever a comparison table exists |
| Sampling | Greedy or parameters, temperature, seed |
| KV format | Format per table, or the default when all tables share one |
| Prefill chunk | Chunk size per table, or the default |
| Warmup / repetitions / statistic | Warmup count, repetition count, median/percentile |
| Concurrency | Request count and arrival schedule; `1` for single-request rows |
| Prompt set / token stream | Prompt token lengths per table; integer prompt lengths or a saved token-stream hash |

Use the full model file name in the table and in the first prose mention. A
different quantization preset or per-tensor mix is a different report. Write
`not recorded` with the reason for a missing historical value and never infer
it. Write `n/a` with the reason when a field does not apply. A new or
re-measured report fills every field. Verify the model SHA-256 before the run
and record the command in Reproduction.

## Coverage

Each model report covers the rows its architecture and backends support. The
full test matrix is in
[testing.md §5](testing.md#5-test-matrix).

- **Correctness gates before performance.** Record native `ctest` and `go test`
  results, plus the real-model tests, with exact commands and observed results.
  No performance table is published without a gate.
- **Prefill.** Measure 128, 512, 2048, and 8192 prompt tokens, and 32768 when
  the model supports it. Report p50 latency and tokens/s per row, the chunk
  size, and the KV format. Record the longest verified prompt in Known limits.
- **Decode.** Measure 128 and 256 generated tokens after a 128-token prompt.
  Report prefill p50, TTFT p50, decode p50, decode tokens/s, context, and KV
  format.
- **Serving.** Measure the in-process scheduler and the OpenAI server at
  concurrency 1 and at least one higher concurrency when the feature exists.
  Report queue wait, TTFT, prefill, decode, per-request throughput, and
  aggregate throughput.
- **Feature matrices.** Measure every opt-in feature that changes cost or
  memory (KV formats, speculative decoding, graph replay, MoE routing) against
  its default. An opt-in feature without a measured row is not accepted.
- **Isolated kernels.** Record dtype, shape, row count, tile, median,
  repetitions, and whether the cell is hot-matrix or in-step. Label
  per-operation attribution as a diagnostic, not a release result.
- **Reference comparison.** Use the same model file, device, chunk size,
  context, sampling, and KV format where the reference supports it. Record the
  reference commit and build flags. Label every mismatch.
- **Known limits.** List unmeasured lengths, unsupported dtypes and shapes,
  single-request scope, and anything the tables do not establish.

Use medians with warmup and repetition counts. Put units in every table header.
Do not copy a table from another report; link to the owning report instead.

## Reproduction

Every report has a `## Reproduction` section. It starts from a clean checkout at
the recorded commit and builds the HIP library and the Go binaries with `cmake`
and `go build -tags hip`. It sets `MODEL` to the full GGUF path and verifies
its SHA-256 against the report, runs the correctness gates, and runs one command
per table with the flags and environment variables that select the row. When a
reference comparison exists, it builds or invokes the reference implementation
at the recorded commit with the recorded flags. It uses variables and relative
paths, never the author's absolute paths and never a checked-in model file.

The minimum preamble is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
MODEL=/path/to/<full-model-file>.gguf
sha256sum "$MODEL"   # must match the report's Model SHA-256
```

A table without a reproduction command is not accepted. Update the section when
a flag, script, or binary name changes.

## Cross-model rules

- Only one benchmark process may use the GPU at a time; record exclusivity.
- Do not compare numbers from different commits, power profiles, or devices
  without labeling the difference.
- Every performance claim in another document links to the model report
  section, not to this file.
- A dispatch, default, or kernel change that invalidates a table updates the
  table and its decision in the same change.
- Numbers are observations on one machine, not guarantees; state that in the
  report.

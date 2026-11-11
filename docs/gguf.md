# GGUF and Model Compatibility

## 1. Role and boundaries

The GGUF layer is a **pure Go** parser and model-configuration builder. It is
responsible for reading the file format correctly and producing validated,
typed descriptors for the runtime. It is not responsible for HIP allocation,
kernel selection, or request scheduling.

```text
GGUF file(s)
    |
Reader: header, metadata, tensor directory, shards
    |
Model adapter: architecture metadata -> ModelConfig
    |
WeightStore: tensor descriptors + mapped weight bytes
    |
Tokenizer + chat template
    |
Planner / Backend (device upload and kernels)
```

The reader must not leak packed quantization details into model code. Model
builders bind named tensors; kernels interpret the quantization layout.

## 2. Reader pipeline

| Step | Work | Failure behavior |
| --- | --- | --- |
| 1. Open | Open the GGUF file and any split shards; establish read-only access | Return a path-qualified error |
| 2. Header | Validate magic, version, tensor count, and metadata count | Reject unsupported version or implausible counts |
| 3. Metadata | Parse typed key/value entries, including architecture and tokenizer metadata | Reject malformed types, duplicate keys, or oversized values |
| 4. Tensor directory | Read each tensor name, rank, dimensions, `ggml_type`, and data offset | Reject duplicate names, invalid ranks, zero dimensions, or overflow |
| 5. Layout | Resolve alignment, tensor byte ranges, shard assignment, and data-section bounds | Reject overlaps, out-of-range offsets, or incomplete shards |
| 6. Model config | Map architecture metadata through an architecture adapter | Reject missing required fields or inconsistent head dimensions |
| 7. Weight store | Build tensor descriptors and map weight bytes on demand | Reject missing required weights; warn on unexpected extras |
| 8. Tokenizer | Build vocabulary, merges/scores, special tokens, and chat template | Reject invalid token IDs or unparsable templates |
| 9. Handoff | Expose `ModelConfig`, `Tokenizer`, and `WeightStore` to the planner | No device allocation before validation completes |

The reader should use `mmap` or on-demand loading to avoid copying the entire
file into Go heap memory. Tensor data is read-only for the model lifetime.

## 3. On-disk details the reader must handle

- **Magic and version.** Validate the GGUF magic and supported version range.
- **Metadata value types.** Integers, floats, booleans, strings, and arrays,
  including nested arrays where the format allows them.
- **Alignment.** Respect the file alignment when computing tensor data offsets;
  do not assume a hard-coded alignment for every file.
- **Tensor descriptors.** Rank, dimensions, `ggml_type`, and offset relative to
  the tensor data section.
- **Byte ranges.** Compute element counts and byte sizes with overflow checks;
  validate `offset + bytes` against the shard or file size.
- **Split files.** Handle multi-shard models, including split metadata, shard
  ordering, and tensor-to-shard assignment. A model is usable only when every
  required tensor is present exactly once.
- **Duplicate and missing entries.** Duplicate metadata keys or tensor names are
  errors. Missing required tensors are errors. Extra tensors are reported and
  ignored unless the model adapter requires them.
- **Read-only mapping.** The mapping must not be mutated. Device upload reads
  from the mapping; it does not modify the source file.

## 4. Quantization semantics

The reader records the **actual per-tensor** `ggml_type`; it does not infer a
single dtype from the file name.

Common file labels such as `Q4_K_M` describe a quantization preset for the
model. They do not mean that every tensor uses `Q4_K`. Embeddings, output heads,
normalization weights, and some layer tensors may use different types. The
planner and kernels must consult each `TensorDesc.DType`.

Initial native kernel coverage targets:

| Type | Layout notes | Initial use |
| --- | --- | --- |
| `F16` / `F32` | Unquantized baseline | Reference and fallback |
| `Q8_0` | Simple block quantization, high bandwidth | Decode baseline |
| `Q4_0` | 32-value block, one scale, nibble pairs | Decode, embedding, prefill |
| `Q5_0` | Q4_0 plus a high-bit plane | Decode, embedding, prefill |
| `Q6_K` | K-quant superblock layout | Decode and prefill |
| `Q4_K` | 256-weight superblock with scale/min encoding | Primary decode target |
| `Q5_K` | K-quant layout related to Q4_K/Q6_K | Decode and prefill |

The reader validates block-size divisibility where the format requires it, but
must not reject a tensor merely because a particular kernel does not support its
shape. Unsupported combinations are reported by the planner as capability
errors.

> Do not pre-expand all quantized weights into a full FP16 matrix for decode.
> Dequantization, multiplication, and reduction are fused in the kernel. See
> [kernels.md](kernels.md).

## 5. Metadata to runtime configuration

GGUF keys are namespaced by architecture. The exact key set varies by model
family, so each architecture adapter maps its metadata into a common
`ModelConfig`. Typical mappings:

| Runtime field | Typical GGUF metadata |
| --- | --- |
| Architecture | `general.architecture` |
| Layer count | `<arch>.block_count` |
| Hidden size | `<arch>.embedding_length` |
| Feed-forward length | `<arch>.feed_forward_length` |
| Attention heads | `<arch>.attention.head_count` |
| KV heads | `<arch>.attention.head_count_kv` |
| Head dimension | `<arch>.attention.key_length` / `value_length`, or derived from hidden size and head count |
| RoPE dimension | `<arch>.rope.dimension_count` |
| RoPE frequency base | `<arch>.rope.freq_base` |
| RMS norm epsilon | `<arch>.attention.layer_norm_rms_epsilon` |
| Context length | `<arch>.context_length` |
| Vocabulary size | Token array length, or `<arch>.vocab_size` when present |
| BOS / EOS token IDs | `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id` |
| Tokenizer model | `tokenizer.ggml.model` |
| Token list / merges / scores | `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.scores` |
| Chat template | `tokenizer.chat_template` |
| Quantization version | `general.quantization_version` |
| File type | `general.file_type` |

Validation must cover consistency between fields: head count times head
dimension must agree with hidden size where the architecture requires it; KV
head count must divide query head count for GQA; context length must be
compatible with RoPE scaling; special token IDs must be within the vocabulary.

## 6. Weight store and mapping

`WeightStore` exposes tensor descriptors and read access, not raw device
pointers. It is responsible for:

- mapping the tensor data section read-only;
- resolving tensor names to shapes, dtypes, offsets, and byte sizes;
- grouping tensors for device upload without changing their quantization;
- tracking which tensors are required by the selected architecture;
- reporting missing, duplicated, or unexpectedly shaped tensors before
  execution.

The device upload policy belongs to the memory planner. On a UMA APU, mapped
host pages may be directly visible to the GPU, but that does not guarantee the
best read bandwidth. The planner decides whether weights are copied into a
device pool or accessed through a mapped path based on measurements.

## 7. Tokenizer and chat template

Tokenizer correctness is a first-class requirement. The implementation must
support the tokenizer families used by the target models:

- byte-level BPE (GPT-2-style tokenizer metadata);
- SentencePiece-style vocabularies and scores, as used by Llama-family models;
- special tokens and added tokens;
- chat template rendering from GGUF metadata;
- BOS/EOS insertion rules and stop conditions.

Current implementation: Qwen2/Qwen3.5 GPT-2 byte-level BPE and Llama-family
SentencePiece vocabularies are implemented, including special-token matching,
byte fallback, SentencePiece scores, dummy-prefix handling, and Qwen chat
rendering. Non-Qwen chat templates are rejected explicitly and are not yet
implemented.

Correctness checks must compare **token IDs**, not only decoded text, against a
trusted reference implementation. The comparison set must include:

- English and Chinese prompts;
- whitespace, punctuation, and Unicode edge cases;
- special tokens and role markers;
- multi-turn chat templates;
- BOS/EOS behavior and stop-string handling;
- empty and very short prompts.

A text-level match is insufficient: different tokenizations can decode to the
same string while producing different model behavior.

## 8. Architecture support plan

| Priority | Architecture | Key features |
| --- | --- | --- |
| P0 | Qwen3.5 Dense hybrid (`qwen35`) | RMSNorm, MRoPE, GQA with output gate, gated delta-net linear attention, convolution state, SwiGLU |
| P0 | Qwen3 Dense | RMSNorm, RoPE, GQA, Q/K norm, SwiGLU |
| P0 | Llama Dense | RMSNorm, RoPE, GQA, SwiGLU |
| P1 | Gemma | Architecture-specific norm, attention, and logit details |
| P1 | Qwen3 MoE (`qwen3moe`) | Routing, experts, grouped expert kernels |
| P2 | DeepSeek-like | MLA, MoE, specialized KV representation |
| P2 | Hybrid attention/SSM | Separate state cache and state-space operators |

The current implementation target is **Qwen3.5-4B with a Q4_K_M GGUF**
(`general.architecture = qwen35`). It uses full attention every fourth layer and
gated delta-net linear attention in the remaining layers. The CPU reference path
and the native HIP execution path are both implemented and tested for this
architecture. The HIP path keeps GGUF weights packed and executes the full
decoder.

Dense Llama and Qwen3 CPU reference paths are implemented with generated
fixtures. The Llama fixture is cross-checked against llama.cpp greedy tokens;
Qwen3 fixture coverage is structural and determinism-based because no real
Qwen3 dense weight file is part of the test matrix. HIP execution for dense
Llama/Qwen3 is not implemented.

`qwen3moe` is implemented as a CPU reference and a HIP decode/prefill path.
Expert tensors are rank-3 `ffn_{gate,up,down}_exps` tensors with the GGUF layout
`[columns, rows_per_expert, experts]` and a `ffn_gate_inp` router. The current
adapter requires a scalar `expert_feed_forward_length` or a layer array whose
entries are all equal; per-layer expert widths are rejected explicitly. Top-k routing
and grouped expert kernels are shared by decode and chunked prefill. The HIP
path is validated on a synthetic four-expert fixture because no standard
`qwen3moe` GGUF is available on the target host. Other architectures in the
table remain planned.

## 9. Quantization support by phase

| Phase | Reader | Execution |
| --- | --- | --- |
| Phase 1 | Header, metadata, tensor directory, shards, per-tensor dtypes | F16/F32 baseline path; quantized files validated but not the performance path |
| Phase 2 | Q4_K, Q5_K, Q6_K, Q8_0 metadata and block validation | Fused quantized GEMV for decode; Q8_0 baseline |
| Phase 3 | Same, plus shape classes used by WMMA tiles | Quantized GEMM for prefill |
| Phase 5 | Additional K-quants and architecture-specific types as required | Case-by-case, driven by model coverage |

Support for a quantization type is not complete until it has reader validation,
a correct reference path, a GPU kernel, and a benchmark entry.

Current status: fused GEMV decode for `F16`, `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`,
`Q5_K`, and `Q6_K` and quantized WMMA prefill GEMM for `F16`, `F32`, `Q4_0`,
`Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` are implemented. The prefill dispatch
table and benchmark evidence are in [kernels.md](kernels.md) and
[benchmarks.md](benchmarks.md). Q4_1/Q5_1, Q2_K/Q3_K, Q8_K, and the IQ families
remain capability failures.

## 10. Validation checklist

A model load must fail early, with a specific diagnostic, if any of the
following checks fail:

- [ ] Magic and version are supported.
- [ ] Metadata counts and tensor counts are plausible for the file size.
- [ ] Every metadata value has a known type and a valid length.
- [ ] Every tensor name is unique and every required tensor is present.
- [ ] Tensor rank is valid and every dimension is greater than zero.
- [ ] Element count and byte size do not overflow.
- [ ] Tensor offsets respect alignment and lie within the data section.
- [ ] Split shards are complete and no tensor is assigned twice.
- [ ] Quantization block sizes divide the tensor dimensions they require.
- [ ] Head counts, KV head counts, and head dimensions are mutually consistent.
- [ ] Vocabulary size and special token IDs are consistent.
- [ ] The tokenizer and chat template can be constructed.
- [ ] The architecture adapter accepts every required metadata field.
- [ ] The selected backend reports support for every required dtype and shape
      class, or the planner selects a documented fallback.

## 11. Error taxonomy

| Class | Examples | Runtime response |
| --- | --- | --- |
| **Malformed file** | Bad magic, truncated header, invalid offsets, duplicate names | Reject the model; no allocation |
| **Unsupported format** | Unknown GGUF version, unknown metadata type, unsupported architecture | Reject with a capability message |
| **Unsupported execution** | Dtype or shape not implemented by the backend | Planner fallback or explicit rejection |
| **Resource failure** | mmap failure, out of memory, device allocation failure | Release partial state and return an error |
| **Tokenizer failure** | Missing merges, invalid special token ID, template parse error | Reject the model or the request, depending on scope |

Errors must include the file, tensor or metadata key, and failing check where
applicable. The loader must not leave partially mapped files or allocated
buffers behind.

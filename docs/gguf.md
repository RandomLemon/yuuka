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
| 2. Header | Validate magic, version, tensor count, and metadata count | Reject unsupported version or counts above the fixed reader caps; truncation is caught while parsing |
| 3. Metadata | Parse typed key/value entries, including architecture and tokenizer metadata | Reject malformed types, duplicate keys, or oversized values |
| 4. Tensor directory | Read each tensor name, rank, dimensions, `ggml_type`, and data offset | Reject duplicate names, invalid ranks, zero dimensions, or overflow |
| 5. Layout | Resolve alignment, tensor byte ranges, shard assignment, and data-section bounds | Reject overlaps, out-of-range offsets, or incomplete shards |
| 6. Model config | Map architecture metadata through an architecture adapter | Reject missing required fields or inconsistent head dimensions |
| 7. Weight store | Build tensor descriptors and map weight bytes on demand | Reject missing required weights; extra tensors are permitted |
| 8. Tokenizer | Build vocabulary, merges/scores, special tokens, and read the chat-template metadata | Reject invalid token IDs or malformed metadata types; non-Qwen templates are rejected when chat rendering is requested |
| 9. Handoff | Expose `ModelConfig`, `Tokenizer`, and `WeightStore` to the planner | No device allocation before validation completes |

The reader must use `mmap` or equivalent on-demand loading so the file is not
copied into Go heap memory. Tensor data is read-only for the model lifetime.

## 3. On-disk details the reader must handle

- **Magic and version.** Validate the GGUF magic and supported version range.
- **Metadata value types.** Integers, floats, booleans, strings, and arrays,
  including nested arrays within the reader's bounded depth limit.
- **Alignment.** Respect the file alignment when computing tensor data offsets;
  do not assume a hard-coded alignment for every file.
- **Tensor descriptors.** Rank, dimensions, `ggml_type`, and offset relative to
  the tensor data section.
- **Byte ranges.** Compute element counts and byte sizes with overflow checks;
  validate `offset + bytes` against the shard or file size.
- **Split files.** Handle multi-shard models, including split metadata, shard
  ordering, and tensor-to-shard assignment. Every shard must repeat the split
  count declared by shard 0 and carry the expected shard index. A model is
  usable only when every required tensor is present exactly once.
- **Duplicate and missing entries.** Duplicate metadata keys or tensor names are
  errors. Missing required tensors are errors. Extra tensors are permitted and
  do not change the model view.
- **Read-only mapping.** The mapping must not be mutated. Device upload reads
  from the mapping; it does not modify the source file.

## 4. Quantization semantics

The reader records the **actual per-tensor** `ggml_type`; it does not infer a
single dtype from the file name.

Common file labels such as `Q4_K_M` describe a quantization preset for the
model. They do not mean that every tensor uses `Q4_K`. Embeddings, output heads,
normalization weights, and some layer tensors may use different types. The
planner and kernels must consult each `TensorDesc.DType`.

The format must be readable for the following types. Layout notes belong
here; execution support per backend is owned by
[kernels.md](kernels.md#3-supported-dtypes).

| Type | Layout notes |
| --- | --- |
| `F16` / `F32` | Unquantized baseline |
| `Q8_0` | 32-value block, one FP16 scale plus int8 values |
| `Q4_0` | 32-value block, one scale, nibble pairs |
| `Q5_0` | `Q4_0` plus a high-bit plane |
| `Q6_K` | 256-value K-quant superblock |
| `Q4_K` | 256-value superblock with scale/min encoding |
| `Q5_K` | 256-value K-quant superblock |
| `IQ2_XXS` | 256-value codebook block, 66 bytes |
| `IQ2_S` | 256-value codebook block with 8-bit scales, 82 bytes |
| `IQ3_S` | 256-value codebook block, 110 bytes |

The reader validates block-size divisibility where the format requires it, but
must not reject a tensor merely because a particular kernel does not support its
shape. Unsupported execution combinations are reported by the planner as
capability errors.

`Q4_1`, `Q5_1`, `Q2_K`, `Q3_K`, `Q8_K`, `IQ1_S`, `IQ2_XS`, `IQ3_XXS`,
`IQ4_NL`, and `IQ4_XS` have no execution path and are capability failures.

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
| Vocabulary size | Token array length; `<arch>.vocab_size` only when the token array is absent |
| Expert count | `<arch>.expert_count` |
| Experts per token | `<arch>.expert_used_count` |
| Routed expert width | `<arch>.expert_feed_forward_length` |
| Shared expert width | `<arch>.expert_shared_feed_forward_length` |
| BOS / EOS token IDs | `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id` |
| Tokenizer model | `tokenizer.ggml.model` |
| Token list / merges / scores | `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.scores` |
| Chat template | `tokenizer.chat_template` |
| Quantization version | `general.quantization_version` |
| File type | `general.file_type` |

Validation must cover consistency between fields: explicit key and value head
dimensions must agree with each other and with the tensor shapes; a missing key
dimension is derived from hidden size and head count. KV head count must divide
query head count for GQA, and special token IDs must be within the vocabulary.
RoPE scaling metadata is not executed in this release: validated models use the
base RoPE table, and a file that declares linear or YaRN scaling is rejected
during configuration loading rather than executed with the wrong frequency
table.

## 6. Weight store and mapping

`WeightStore` exposes tensor descriptors and read access, not raw device
pointers. It is responsible for:

- mapping the tensor data section read-only;
- resolving tensor names to shapes, dtypes, offsets, and byte sizes;
- grouping tensors for device upload without changing their quantization;
- tracking which tensors are required by the selected architecture;
- reporting missing or duplicated tensors before execution.

Unexpected extra tensors are permitted. A backend may upload them or ignore
them; they never become part of the model view.

The device upload policy belongs to the memory planner. On a UMA APU, mapped
host pages may be directly visible to the GPU, but that does not guarantee the
best read bandwidth. The planner decides whether weights are copied into a
device pool or accessed through a mapped path based on measurements.

## 7. Tokenizer and chat template

Tokenizer correctness is a first-class requirement. The implementation must
support the tokenizer families used by the target models:

- byte-level BPE (GPT-2-style tokenizer metadata);
- SentencePiece-style vocabularies and scores;
- special tokens and added tokens;
- Qwen chat rendering gated on the GGUF chat-template metadata; arbitrary
  templates are validated for the Qwen marker, not executed;
- BOS/EOS insertion rules and stop conditions.

The tokenizer contract is:

- tokenization is language-agnostic and includes byte fallback where the
  vocabulary requires it;
- BOS and EOS insertion is explicit and configurable, and an already present
  token is not duplicated;
- raw encoding without implicit insertion stays available for tokenizer and
  cross-framework comparison;
- non-Qwen chat templates are rejected explicitly when chat rendering is
  requested;
- arbitrary chat templates are never executed.

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

## 8. Architecture support

| Architecture | Key features | Execution paths |
| --- | --- | --- |
| Qwen3.5 dense hybrid (`qwen35`) | RMSNorm, RoPE over validated MRoPE sections (text-only path), GQA with output gate, gated delta-net linear attention, convolution state, SwiGLU | CPU reference, HIP |
| Qwen3.6 MoE hybrid (`qwen35moe`) | The `qwen35` hybrid attention and recurrent state with top-k routed experts and a sigmoid-gated shared expert | CPU reference, HIP |

`qwen35` and `qwen35moe` are the supported architectures. An architecture must
not be treated as supported merely because an adapter exists or a synthetic
fixture passes. Adding an architecture requires adapting its metadata, tensor
layouts, forward pass, and backend execution, then validating it on a real GGUF
model.

All other architectures—including Llama, dense Qwen3, Qwen3 MoE, Gemma, and
DeepSeek-style models—have no adapter. They fail at load time with an explicit
unsupported-architecture error and have no CPU or HIP execution path.

`qwen35` uses full attention every fourth layer and gated delta-net linear
attention in the remaining layers. `qwen35.rope.dimension_sections` is validated
for the text-only configuration; because every position component is equal, the
forward pass applies standard RoPE over the full RoPE dimension.

`qwen35moe` reuses that hybrid attention and recurrent state and replaces the
dense FFN with routed experts, top-k routing, and one shared expert. Expert
tensors are rank-3 `ffn_{gate,up,down}_exps` tensors with the GGUF layout
`[columns, rows_per_expert, experts]`, plus a `ffn_gate_inp` router. The adapter
requires a scalar `expert_feed_forward_length` or a layer array whose entries
are all equal; per-layer expert widths are rejected. The planner accepts an
expert tensor only when the selected backend decodes its dtype; every other
dtype is rejected during planning instead of falling back silently.

## 9. Quantization support by phase

| Phase | Reader | Execution |
| --- | --- | --- |
| Phase 1 | Header, metadata, tensor directory, shards, per-tensor dtypes | F16/F32 baseline path; quantized files validated but not the performance path |
| Phase 2 | Q4_K, Q5_K, Q6_K, Q8_0 metadata and block validation | Fused quantized GEMV for decode; Q8_0 baseline |
| Phase 3 | Same, plus shape classes used by WMMA tiles | Quantized GEMM for prefill |
| Phase 5 | Additional K-quants and architecture-specific types as required | Case-by-case, driven by model coverage |

Support for a quantization type is not complete until it has reader validation,
a correct reference path, a GPU kernel, and a benchmark entry. The supported
set and its execution routes are defined in
[kernels.md](kernels.md#3-supported-dtypes); the evidence is recorded in the
model reports.

## 10. Validation checklist

A model load must fail early, with a specific diagnostic, if any of the
following checks fail:

- [ ] Magic and version are supported.
- [ ] Metadata counts and tensor counts are bounded and parse within the file size.
- [ ] Every metadata value has a known type and a valid length.
- [ ] Every tensor name is unique and every required tensor is present.
- [ ] Tensor rank is valid and every dimension is greater than zero.
- [ ] Element count and byte size do not overflow.
- [ ] Tensor offsets respect alignment and lie within the data section.
- [ ] Split shards are complete and no tensor is assigned twice.
- [ ] Quantization block sizes divide the tensor dimensions they require.
- [ ] Head counts, KV head counts, and head dimensions are mutually consistent.
- [ ] Special token IDs are within the token vocabulary.
- [ ] The tokenizer can be constructed; non-Qwen chat templates are rejected at
      render time, not load time.
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

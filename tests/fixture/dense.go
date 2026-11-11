package fixture

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/yuuka-ai/yuuka/tensor"
)

// WriteTinyDense writes a deterministic two-layer dense decoder with either
// Llama or Qwen3 tensor layout. The model is small enough for unit tests and
// exercises tied and untied output heads as well as optional Q/K norm.
func WriteTinyDense(path, architecture string) error {
	if architecture != "llama" && architecture != "qwen3" {
		return fmt.Errorf("unsupported dense fixture architecture %q", architecture)
	}
	const (
		hidden     = 32
		ffn        = 64
		heads      = 2
		kvHeads    = 1
		headDim    = 8
		ropeDim    = 8
		layers     = 2
		context    = 128
		blockCount = 2
	)
	tokens, tokenTypes, scores, mer, model, pre, bos, eos, chat := denseTokenizer(architecture)
	qDim := heads * headDim
	kvDim := kvHeads * headDim

	b := newBuilder()
	b.addMetadata("general.architecture", stringValue(architecture))
	b.addMetadata("general.name", stringValue("yuuka-tiny-"+architecture))
	b.addMetadata("general.alignment", uintValue(alignment))
	b.addMetadata("general.quantization_version", uintValue(2))
	b.addMetadata("general.file_type", uintValue(0))
	b.addMetadata(architecture+".block_count", uintValue(blockCount))
	b.addMetadata(architecture+".context_length", uintValue(context))
	b.addMetadata(architecture+".embedding_length", uintValue(hidden))
	b.addMetadata(architecture+".feed_forward_length", uintValue(ffn))
	b.addMetadata(architecture+".attention.head_count", uintValue(heads))
	b.addMetadata(architecture+".attention.head_count_kv", uintValue(kvHeads))
	b.addMetadata(architecture+".attention.key_length", uintValue(headDim))
	b.addMetadata(architecture+".attention.value_length", uintValue(headDim))
	b.addMetadata(architecture+".attention.layer_norm_rms_epsilon", floatValue(1e-6))
	b.addMetadata(architecture+".rope.dimension_count", uintValue(ropeDim))
	b.addMetadata(architecture+".rope.freq_base", floatValue(10000))
	b.addMetadata("tokenizer.ggml.model", stringValue(model))
	b.addMetadata("tokenizer.ggml.pre", stringValue(pre))
	b.addMetadata("tokenizer.ggml.tokens", stringArrayValue(tokens))
	b.addMetadata("tokenizer.ggml.token_type", intArrayValue(tokenTypes))
	b.addMetadata("tokenizer.ggml.bos_token_id", uintValue(uint64(bos)))
	b.addMetadata("tokenizer.ggml.eos_token_id", uintValue(uint64(eos)))
	if architecture == "llama" {
		b.addMetadata("tokenizer.ggml.scores", floatArrayValue(scores))
		b.addMetadata("tokenizer.ggml.add_bos_token", boolValue(true))
		b.addMetadata("tokenizer.ggml.add_space_prefix", boolValue(true))
	} else {
		b.addMetadata("tokenizer.ggml.merges", stringArrayValue(mer))
		b.addMetadata("tokenizer.ggml.add_bos_token", boolValue(false))
		b.addMetadata("tokenizer.chat_template", stringValue(chat))
	}

	rng := rand.New(rand.NewPCG(11, 13))
	b.addTensor(f32Matrix(rng, "token_embd.weight", len(tokens), hidden, 0.08))
	b.addTensor(repeatVector("output_norm.weight", hidden, 1))
	if architecture == "llama" {
		b.addTensor(f32Matrix(rng, "output.weight", len(tokens), hidden, 0.05))
	}
	for layer := range layers {
		prefix := fmt.Sprintf("blk.%d.", layer)
		b.addTensor(repeatVector(prefix+"attn_norm.weight", hidden, 1))
		b.addTensor(repeatVector(prefix+"ffn_norm.weight", hidden, 1))
		b.addTensor(f32Matrix(rng, prefix+"attn_q.weight", qDim, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"attn_k.weight", kvDim, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"attn_v.weight", kvDim, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"attn_output.weight", hidden, qDim, 0.03))
		if architecture == "qwen3" {
			b.addTensor(repeatVector(prefix+"attn_q_norm.weight", headDim, 1))
			b.addTensor(repeatVector(prefix+"attn_k_norm.weight", headDim, 1))
		}
		b.addTensor(f32Matrix(rng, prefix+"ffn_gate.weight", ffn, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"ffn_up.weight", ffn, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"ffn_down.weight", hidden, ffn, 0.03))
	}
	_ = layers
	return b.write(path)
}

// f32ExpertTensor writes a packed rank-3 expert weight [cols, rows, experts].
func f32ExpertTensor(rng *rand.Rand, name string, rows, cols, experts int, scale float32) tensorSpec {
	data := make([]byte, rows*cols*experts*4)
	for i := range rows * cols * experts {
		value := (rng.Float32()*2 - 1) * scale
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return tensorSpec{
		name:  name,
		dims:  []uint64{uint64(cols), uint64(rows), uint64(experts)},
		dtype: tensor.F32,
		data:  data,
	}
}

// WriteTinyMoE writes a deterministic two-layer qwen3moe model with four
// experts and top-2 routing. The router and expert matrices differ per layer so
// routing cannot pass by symmetry.
func WriteTinyMoE(path string) error {
	return writeTinyMoE(path, false)
}

// WriteTinyMoEZeroExperts writes a qwen3moe fixture whose expert weights are
// zero, isolating the shared attention path for comparison tests.
func WriteTinyMoEZeroExperts(path string) error {
	return writeTinyMoE(path, true)
}

func writeTinyMoE(path string, zeroExperts bool) error {
	const (
		hidden    = 32
		experts   = 4
		used      = 2
		expertFFN = 16
		heads     = 2
		kvHeads   = 1
		headDim   = 8
		ropeDim   = 8
		layers    = 2
		context   = 128
	)
	tokens, tokenTypes, _, merges, model, pre, _, eos, chat := denseTokenizer(Qwen3MoEArchitecture)
	qDim := heads * headDim
	kvDim := kvHeads * headDim

	b := newBuilder()
	b.addMetadata("general.architecture", stringValue(Qwen3MoEArchitecture))
	b.addMetadata("general.name", stringValue("yuuka-tiny-qwen3moe"))
	b.addMetadata("general.alignment", uintValue(alignment))
	b.addMetadata("general.quantization_version", uintValue(2))
	b.addMetadata("general.file_type", uintValue(0))
	b.addMetadata(Qwen3MoEArchitecture+".block_count", uintValue(layers))
	b.addMetadata(Qwen3MoEArchitecture+".context_length", uintValue(context))
	b.addMetadata(Qwen3MoEArchitecture+".embedding_length", uintValue(hidden))
	b.addMetadata(Qwen3MoEArchitecture+".feed_forward_length", uintValue(expertFFN*used))
	b.addMetadata(Qwen3MoEArchitecture+".expert_count", uintValue(experts))
	b.addMetadata(Qwen3MoEArchitecture+".expert_used_count", uintValue(used))
	b.addMetadata(Qwen3MoEArchitecture+".expert_feed_forward_length", uintValue(expertFFN))
	b.addMetadata(Qwen3MoEArchitecture+".attention.head_count", uintValue(heads))
	b.addMetadata(Qwen3MoEArchitecture+".attention.head_count_kv", uintValue(kvHeads))
	b.addMetadata(Qwen3MoEArchitecture+".attention.key_length", uintValue(headDim))
	b.addMetadata(Qwen3MoEArchitecture+".attention.value_length", uintValue(headDim))
	b.addMetadata(Qwen3MoEArchitecture+".attention.layer_norm_rms_epsilon", floatValue(1e-6))
	b.addMetadata(Qwen3MoEArchitecture+".rope.dimension_count", uintValue(ropeDim))
	b.addMetadata(Qwen3MoEArchitecture+".rope.freq_base", floatValue(10000))
	b.addMetadata("tokenizer.ggml.model", stringValue(model))
	b.addMetadata("tokenizer.ggml.pre", stringValue(pre))
	b.addMetadata("tokenizer.ggml.tokens", stringArrayValue(tokens))
	b.addMetadata("tokenizer.ggml.merges", stringArrayValue(merges))
	b.addMetadata("tokenizer.ggml.token_type", intArrayValue(tokenTypes))
	b.addMetadata("tokenizer.ggml.add_bos_token", boolValue(false))
	b.addMetadata("tokenizer.ggml.eos_token_id", uintValue(uint64(eos)))
	b.addMetadata("tokenizer.chat_template", stringValue(chat))

	rng := rand.New(rand.NewPCG(21, 23))
	b.addTensor(f32Matrix(rng, "token_embd.weight", len(tokens), hidden, 0.08))
	b.addTensor(repeatVector("output_norm.weight", hidden, 1))
	for layer := range layers {
		prefix := fmt.Sprintf("blk.%d.", layer)
		b.addTensor(repeatVector(prefix+"attn_norm.weight", hidden, 1))
		b.addTensor(repeatVector(prefix+"ffn_norm.weight", hidden, 1))
		b.addTensor(f32Matrix(rng, prefix+"attn_q.weight", qDim, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"attn_k.weight", kvDim, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"attn_v.weight", kvDim, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"attn_output.weight", hidden, qDim, 0.03))
		b.addTensor(repeatVector(prefix+"attn_q_norm.weight", headDim, 1))
		b.addTensor(repeatVector(prefix+"attn_k_norm.weight", headDim, 1))
		expertScale := float32(0.04)
		if zeroExperts {
			expertScale = 0
		}
		b.addTensor(f32Matrix(rng, prefix+"ffn_gate_inp.weight", experts, hidden, 0.1))
		b.addTensor(f32ExpertTensor(rng, prefix+"ffn_gate_exps.weight", expertFFN, hidden, experts, expertScale))
		b.addTensor(f32ExpertTensor(rng, prefix+"ffn_up_exps.weight", expertFFN, hidden, experts, expertScale))
		b.addTensor(f32ExpertTensor(rng, prefix+"ffn_down_exps.weight", hidden, expertFFN, experts, expertScale))
	}
	return b.write(path)
}

// Qwen3MoEArchitecture is the GGUF architecture name for Qwen3 sparse models.
const Qwen3MoEArchitecture = "qwen3moe"

func denseTokenizer(architecture string) (tokens []string, types []int64, scores []float64, merges []string, model, pre string, bos, eos int32, chat string) {
	if architecture == "llama" {
		tokens, types, scores = tinySPMVocab()
		return tokens, types, scores, nil, "llama", "default", 256, 257, ""
	}
	tokens = byteTokens()
	tokens = append(tokens, "<|im_start|>", "<|im_end|>", "<|endoftext|>")
	types = make([]int64, len(tokens))
	for i := range types {
		types[i] = 1
	}
	for i := 256; i < len(tokens); i++ {
		types[i] = 3
	}
	return tokens, types, nil, nil, "gpt2", "qwen2", 258, 258, "<|im_start|>"
}

func tinySPMVocab() (tokens []string, types []int64, scores []float64) {
	tokens = make([]string, 0, 256+8+10)
	types = make([]int64, 0, 256+8+10)
	scores = make([]float64, 0, 256+8+10)
	for b := range 256 {
		tokens = append(tokens, byteToken(b))
		types = append(types, 6)
		scores = append(scores, 0)
	}
	for _, token := range []string{"<s>", "</s>", "<unk>"} {
		tokens = append(tokens, token)
		types = append(types, 3)
		scores = append(scores, 0)
	}
	normal := []struct {
		text  string
		score float64
	}{
		{"▁H", 1}, {"▁He", 2}, {"▁Hel", 3}, {"▁Hell", 4}, {"▁Hello", 5},
		{"▁w", 1}, {"▁wo", 2}, {"▁wor", 3}, {"▁worl", 4}, {"▁world", 5},
	}
	for _, token := range normal {
		tokens = append(tokens, token.text)
		types = append(types, 1)
		scores = append(scores, token.score)
	}
	return tokens, types, scores
}

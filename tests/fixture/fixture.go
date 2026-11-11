// Package fixture builds deterministic synthetic GGUF files for tests.
//
// The fixtures are generated at test time instead of being committed as model
// weights. They are intentionally tiny but exercise the same metadata, tensor,
// tokenizer, and architecture paths as real models.
package fixture

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/yuuka-ai/yuuka/tensor"
)

const alignment = 32

type metadataValue struct {
	typ     uint32
	elem    uint32
	uint    uint64
	int     int64
	float   float64
	str     string
	boolean bool
	array   []metadataValue
}

type tensorSpec struct {
	name  string
	dims  []uint64
	dtype tensor.DType
	data  []byte
}

type builder struct {
	metadata []struct {
		key   string
		value metadataValue
	}
	tensors []tensorSpec
}

func newBuilder() *builder { return &builder{} }

func (b *builder) addMetadata(key string, value metadataValue) {
	b.metadata = append(b.metadata, struct {
		key   string
		value metadataValue
	}{key, value})
}

func (b *builder) addTensor(spec tensorSpec) { b.tensors = append(b.tensors, spec) }

func (b *builder) write(path string) error {
	var header bytes.Buffer
	header.WriteString("GGUF")
	writeU32(&header, 3)
	writeU64(&header, uint64(len(b.tensors)))
	writeU64(&header, uint64(len(b.metadata)))
	for _, entry := range b.metadata {
		writeString(&header, entry.key)
		writeU32(&header, entry.value.typ)
		writeValue(&header, entry.value)
	}

	type writtenTensor struct {
		spec   tensorSpec
		offset uint64
	}
	written := make([]writtenTensor, 0, len(b.tensors))
	data := &bytes.Buffer{}
	for _, spec := range b.tensors {
		for data.Len()%alignment != 0 {
			data.WriteByte(0)
		}
		offset := uint64(data.Len())
		if _, err := data.Write(spec.data); err != nil {
			return err
		}
		written = append(written, writtenTensor{spec: spec, offset: offset})
	}
	for _, item := range written {
		writeString(&header, item.spec.name)
		writeU32(&header, uint32(len(item.spec.dims)))
		for _, dim := range item.spec.dims {
			writeU64(&header, dim)
		}
		writeU32(&header, uint32(item.spec.dtype))
		writeU64(&header, item.offset)
	}
	for header.Len()%alignment != 0 {
		header.WriteByte(0)
	}
	for data.Len()%alignment != 0 {
		data.WriteByte(0)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	out := make([]byte, 0, header.Len()+data.Len())
	out = append(out, header.Bytes()...)
	out = append(out, data.Bytes()...)
	return os.WriteFile(path, out, 0o644)
}

func writeU32(w *bytes.Buffer, value uint32) {
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], value)
	w.Write(raw[:])
}

func writeU64(w *bytes.Buffer, value uint64) {
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], value)
	w.Write(raw[:])
}

func writeString(w *bytes.Buffer, value string) {
	writeU64(w, uint64(len(value)))
	w.WriteString(value)
}

func writeValue(w *bytes.Buffer, value metadataValue) {
	switch value.typ {
	case 0: // uint8
		w.WriteByte(byte(value.uint))
	case 1: // int8
		w.WriteByte(byte(int8(value.int)))
	case 2: // uint16
		var raw [2]byte
		binary.LittleEndian.PutUint16(raw[:], uint16(value.uint))
		w.Write(raw[:])
	case 3: // int16
		var raw [2]byte
		binary.LittleEndian.PutUint16(raw[:], uint16(int16(value.int)))
		w.Write(raw[:])
	case 4: // uint32
		writeU32(w, uint32(value.uint))
	case 5: // int32
		writeU32(w, uint32(int32(value.int)))
	case 6: // float32
		writeU32(w, math.Float32bits(float32(value.float)))
	case 7: // bool
		if value.boolean {
			w.WriteByte(1)
		} else {
			w.WriteByte(0)
		}
	case 8: // string
		writeString(w, value.str)
	case 9: // array
		writeU32(w, value.elem)
		writeU64(w, uint64(len(value.array)))
		for _, item := range value.array {
			writeValue(w, item)
		}
	case 10: // uint64
		writeU64(w, value.uint)
	case 11: // int64
		writeU64(w, uint64(value.int))
	case 12: // float64
		writeU64(w, math.Float64bits(value.float))
	default:
		panic("fixture: unsupported metadata type")
	}
}

func stringValue(value string) metadataValue {
	return metadataValue{typ: 8, str: value}
}

func uintValue(value uint64) metadataValue {
	return metadataValue{typ: 4, uint: value}
}

func floatValue(value float64) metadataValue {
	return metadataValue{typ: 12, float: value}
}

func boolValue(value bool) metadataValue {
	return metadataValue{typ: 7, boolean: value}
}

func stringArrayValue(values []string) metadataValue {
	items := make([]metadataValue, len(values))
	for i, value := range values {
		items[i] = stringValue(value)
	}
	return metadataValue{typ: 9, elem: 8, array: items}
}

func intArrayValue(values []int64) metadataValue {
	items := make([]metadataValue, len(values))
	for i, value := range values {
		items[i] = metadataValue{typ: 5, int: value}
	}
	return metadataValue{typ: 9, elem: 5, array: items}
}

func f32Matrix(rng *rand.Rand, name string, rows, cols int, scale float32) tensorSpec {
	data := make([]byte, rows*cols*4)
	for i := range rows * cols {
		value := (rng.Float32()*2 - 1) * scale
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return tensorSpec{
		name:  name,
		dims:  []uint64{uint64(cols), uint64(rows)},
		dtype: tensor.F32,
		data:  data,
	}
}

func f32Vector(name string, values []float32) tensorSpec {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return tensorSpec{
		name:  name,
		dims:  []uint64{uint64(len(values))},
		dtype: tensor.F32,
		data:  data,
	}
}

func repeatVector(name string, count int, value float32) tensorSpec {
	values := make([]float32, count)
	for i := range values {
		values[i] = value
	}
	return f32Vector(name, values)
}

// WriteTinyQwen35 writes a deterministic two-layer qwen35 hybrid model. Layer
// zero uses gated delta-net recurrence and layer one uses full GQA attention,
// so both decoder paths are covered. All tensors are F32 and the LM head is
// tied to the embedding.
func WriteTinyQwen35(path string) error {
	const (
		hidden      = 32
		ffn         = 64
		heads       = 2
		kvHeads     = 1
		headDim     = 8
		ropeDim     = 8
		fullEvery   = 2
		context     = 256
		ssmState    = 8
		ssmRank     = 2
		ssmGroups   = 1
		ssmInner    = ssmRank * ssmState
		ssmConv     = 2
		layerCount  = 2
		specialBase = 256
	)
	qDim := heads * headDim * 2
	oDim := heads * headDim
	kvDim := kvHeads * headDim
	convDim := 2*ssmGroups*ssmState + ssmInner

	tokens := byteTokens()
	tokens = append(tokens, "<|im_start|>", "<|im_end|>", "<|endoftext|>")
	tokenTypes := make([]int64, len(tokens))
	for i := range tokenTypes {
		tokenTypes[i] = 1
	}
	for i := specialBase; i < len(tokens); i++ {
		tokenTypes[i] = 3
	}

	b := newBuilder()
	b.addMetadata("general.architecture", stringValue("qwen35"))
	b.addMetadata("general.name", stringValue("yuuka-tiny-qwen35"))
	b.addMetadata("general.alignment", uintValue(alignment))
	b.addMetadata("general.quantization_version", uintValue(2))
	b.addMetadata("general.file_type", uintValue(0))
	b.addMetadata("qwen35.block_count", uintValue(layerCount))
	b.addMetadata("qwen35.context_length", uintValue(context))
	b.addMetadata("qwen35.embedding_length", uintValue(hidden))
	b.addMetadata("qwen35.feed_forward_length", uintValue(ffn))
	b.addMetadata("qwen35.attention.head_count", uintValue(heads))
	b.addMetadata("qwen35.attention.head_count_kv", uintValue(kvHeads))
	b.addMetadata("qwen35.attention.key_length", uintValue(headDim))
	b.addMetadata("qwen35.attention.value_length", uintValue(headDim))
	b.addMetadata("qwen35.attention.layer_norm_rms_epsilon", floatValue(1e-6))
	b.addMetadata("qwen35.rope.dimension_count", uintValue(ropeDim))
	b.addMetadata("qwen35.rope.dimension_sections", intArrayValue([]int64{2, 2, 0, 0}))
	b.addMetadata("qwen35.rope.freq_base", floatValue(10000))
	b.addMetadata("qwen35.ssm.conv_kernel", uintValue(ssmConv))
	b.addMetadata("qwen35.ssm.state_size", uintValue(ssmState))
	b.addMetadata("qwen35.ssm.group_count", uintValue(ssmGroups))
	b.addMetadata("qwen35.ssm.time_step_rank", uintValue(ssmRank))
	b.addMetadata("qwen35.ssm.inner_size", uintValue(ssmInner))
	b.addMetadata("qwen35.full_attention_interval", uintValue(fullEvery))
	b.addMetadata("tokenizer.ggml.model", stringValue("gpt2"))
	b.addMetadata("tokenizer.ggml.pre", stringValue("qwen35"))
	b.addMetadata("tokenizer.ggml.tokens", stringArrayValue(tokens))
	b.addMetadata("tokenizer.ggml.merges", stringArrayValue(nil))
	b.addMetadata("tokenizer.ggml.token_type", intArrayValue(tokenTypes))
	b.addMetadata("tokenizer.ggml.bos_token_id", uintValue(258))
	b.addMetadata("tokenizer.ggml.eos_token_id", uintValue(258))
	b.addMetadata("tokenizer.ggml.padding_token_id", uintValue(258))
	b.addMetadata("tokenizer.ggml.add_bos_token", boolValue(false))
	b.addMetadata("tokenizer.chat_template", stringValue("<|im_start|>"))

	rng := rand.New(rand.NewPCG(1, 2))
	b.addTensor(f32Matrix(rng, "token_embd.weight", len(tokens), hidden, 0.08))
	b.addTensor(repeatVector("output_norm.weight", hidden, 1))

	for layer := 0; layer < layerCount; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		b.addTensor(repeatVector(prefix+"attn_norm.weight", hidden, 1))
		b.addTensor(repeatVector(prefix+"post_attention_norm.weight", hidden, 1))
		if (layer+1)%fullEvery != 0 {
			b.addTensor(f32Matrix(rng, prefix+"attn_qkv.weight", convDim, hidden, 0.03))
			b.addTensor(f32Matrix(rng, prefix+"attn_gate.weight", ssmInner, hidden, 0.03))
			b.addTensor(f32Matrix(rng, prefix+"ssm_out.weight", hidden, ssmInner, 0.03))
			b.addTensor(f32Matrix(rng, prefix+"ssm_beta.weight", ssmRank, hidden, 0.02))
			b.addTensor(f32Matrix(rng, prefix+"ssm_alpha.weight", ssmRank, hidden, 0.02))
			b.addTensor(f32Matrix(rng, prefix+"ssm_conv1d.weight", ssmConv, convDim, 0.03))
			b.addTensor(repeatVector(prefix+"ssm_dt.bias", ssmRank, 0.1))
			b.addTensor(repeatVector(prefix+"ssm_a", ssmRank, -1))
			b.addTensor(repeatVector(prefix+"ssm_norm.weight", ssmState, 1))
		} else {
			b.addTensor(f32Matrix(rng, prefix+"attn_q.weight", qDim, hidden, 0.03))
			b.addTensor(f32Matrix(rng, prefix+"attn_k.weight", kvDim, hidden, 0.03))
			b.addTensor(f32Matrix(rng, prefix+"attn_v.weight", kvDim, hidden, 0.03))
			b.addTensor(f32Matrix(rng, prefix+"attn_output.weight", hidden, oDim, 0.03))
			b.addTensor(repeatVector(prefix+"attn_q_norm.weight", headDim, 1))
			b.addTensor(repeatVector(prefix+"attn_k_norm.weight", headDim, 1))
		}
		b.addTensor(f32Matrix(rng, prefix+"ffn_gate.weight", ffn, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"ffn_up.weight", ffn, hidden, 0.03))
		b.addTensor(f32Matrix(rng, prefix+"ffn_down.weight", hidden, ffn, 0.03))
	}
	return b.write(path)
}

func byteTokens() []string {
	tokens := make([]string, 256)
	next := rune(256)
	for b := range 256 {
		printable := (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
		if printable {
			tokens[b] = string(rune(b))
		} else {
			tokens[b] = string(next)
			next++
		}
	}
	return tokens
}

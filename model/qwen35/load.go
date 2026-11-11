package qwen35

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tensor"
)

// FullAttention holds one full GQA attention layer.
type FullAttention struct {
	Q     tensor.Matrix
	K     tensor.Matrix
	V     tensor.Matrix
	O     tensor.Matrix
	QNorm []float32
	KNorm []float32
}

// Recurrent holds one gated delta-net linear-attention layer.
type Recurrent struct {
	QKV   tensor.Matrix
	Gate  tensor.Matrix
	Out   tensor.Matrix
	Beta  tensor.Matrix
	Alpha tensor.Matrix
	Conv  []float32
	DT    []float32
	A     []float32
	Norm  []float32
}

// FFN is the dense SwiGLU feed-forward block.
type FFN struct {
	Gate tensor.Matrix
	Up   tensor.Matrix
	Down tensor.Matrix
}

// Layer is one decoder block.
type Layer struct {
	AttnNorm []float32
	PostNorm []float32
	Full     *FullAttention
	Rec      *Recurrent
	FFN      FFN
}

// Model owns tensor descriptors that borrow the GGUF reader's mappings. The
// reader must stay open for the model's lifetime.
type Model struct {
	Config        *Config
	Embedding     tensor.Matrix
	Output        tensor.Matrix
	OutputNorm    []float32
	Layers        []Layer
	outputIsTied  bool
	embeddingRowB uint64
}

// Load validates required tensors and builds the model view. It does not copy
// weight bytes; all matrices reference the reader's read-only mapping.
func Load(r *gguf.Reader) (*Model, error) {
	cfg, err := LoadConfig(r)
	if err != nil {
		return nil, err
	}
	m := &Model{Config: cfg}
	m.Embedding, err = loadMatrix(r, "token_embd.weight", cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	m.embeddingRowB, _ = m.Embedding.DType.RowBytes(m.Embedding.Cols)
	if _, ok := r.Tensor("output.weight"); ok {
		m.Output, err = loadMatrix(r, "output.weight", cfg.VocabSize, cfg.HiddenSize)
		if err != nil {
			return nil, err
		}
	} else {
		m.Output = m.Embedding
		m.outputIsTied = true
	}
	m.OutputNorm, err = loadVector(r, "output_norm.weight", cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	m.Layers = make([]Layer, cfg.BlockCount)
	qDim := cfg.HeadCount * cfg.HeadDim * 2
	kvDim := cfg.KVHeadCount * cfg.HeadDim
	oDim := cfg.HeadCount * cfg.HeadDim
	convDim := 2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize
	for i := range m.Layers {
		layer := &m.Layers[i]
		if layer.AttnNorm, err = loadVector(r, fmt.Sprintf("blk.%d.attn_norm.weight", i), cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.PostNorm, err = loadVector(r, fmt.Sprintf("blk.%d.post_attention_norm.weight", i), cfg.HiddenSize); err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(i) {
			rec := &Recurrent{}
			prefix := fmt.Sprintf("blk.%d.", i)
			if rec.QKV, err = loadMatrix(r, prefix+"attn_qkv.weight", convDim, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if rec.Gate, err = loadMatrix(r, prefix+"attn_gate.weight", cfg.SSMInnerSize, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if rec.Out, err = loadMatrix(r, prefix+"ssm_out.weight", cfg.HiddenSize, cfg.SSMInnerSize); err != nil {
				return nil, err
			}
			if rec.Beta, err = loadMatrix(r, prefix+"ssm_beta.weight", cfg.SSMTimeStepRank, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if rec.Alpha, err = loadMatrix(r, prefix+"ssm_alpha.weight", cfg.SSMTimeStepRank, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if rec.Conv, err = loadF32(r, prefix+"ssm_conv1d.weight", cfg.SSMConvKernel*convDim); err != nil {
				return nil, err
			}
			if rec.DT, err = loadVector(r, prefix+"ssm_dt.bias", cfg.SSMTimeStepRank); err != nil {
				return nil, err
			}
			if rec.A, err = loadVector(r, prefix+"ssm_a", cfg.SSMTimeStepRank); err != nil {
				return nil, err
			}
			if rec.Norm, err = loadVector(r, prefix+"ssm_norm.weight", cfg.SSMStateSize); err != nil {
				return nil, err
			}
			layer.Rec = rec
		} else {
			full := &FullAttention{}
			prefix := fmt.Sprintf("blk.%d.", i)
			if full.Q, err = loadMatrix(r, prefix+"attn_q.weight", qDim, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if full.K, err = loadMatrix(r, prefix+"attn_k.weight", kvDim, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if full.V, err = loadMatrix(r, prefix+"attn_v.weight", kvDim, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if full.O, err = loadMatrix(r, prefix+"attn_output.weight", cfg.HiddenSize, oDim); err != nil {
				return nil, err
			}
			if full.QNorm, err = loadVector(r, prefix+"attn_q_norm.weight", cfg.HeadDim); err != nil {
				return nil, err
			}
			if full.KNorm, err = loadVector(r, prefix+"attn_k_norm.weight", cfg.HeadDim); err != nil {
				return nil, err
			}
			layer.Full = full
		}
		prefix := fmt.Sprintf("blk.%d.", i)
		if layer.FFN.Gate, err = loadMatrix(r, prefix+"ffn_gate.weight", cfg.FFNSize, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.FFN.Up, err = loadMatrix(r, prefix+"ffn_up.weight", cfg.FFNSize, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.FFN.Down, err = loadMatrix(r, prefix+"ffn_down.weight", cfg.HiddenSize, cfg.FFNSize); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// OutputTied reports whether the LM head shares the token embedding matrix.
func (m *Model) OutputTied() bool { return m.outputIsTied }

func loadMatrix(r *gguf.Reader, name string, rows, cols int) (tensor.Matrix, error) {
	info, ok := r.Tensor(name)
	if !ok {
		return tensor.Matrix{}, fmt.Errorf("missing required tensor %q", name)
	}
	if len(info.Dimensions) != 2 {
		return tensor.Matrix{}, fmt.Errorf("tensor %q has rank %d, want 2", name, len(info.Dimensions))
	}
	if info.Dimensions[0] != uint64(cols) || info.Dimensions[1] != uint64(rows) {
		return tensor.Matrix{}, fmt.Errorf("tensor %q has shape [%d,%d], want [%d,%d]",
			name, info.Dimensions[0], info.Dimensions[1], cols, rows)
	}
	if !info.DType.Supported() {
		return tensor.Matrix{}, fmt.Errorf("tensor %q uses unsupported dtype %s", name, info.DType)
	}
	matrix, err := info.Matrix()
	if err != nil {
		return tensor.Matrix{}, err
	}
	if err := matrix.Validate(); err != nil {
		return tensor.Matrix{}, fmt.Errorf("tensor %q: %w", name, err)
	}
	return matrix, nil
}

func loadVector(r *gguf.Reader, name string, length int) ([]float32, error) {
	info, ok := r.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("missing required tensor %q", name)
	}
	if len(info.Dimensions) != 1 || info.Dimensions[0] != uint64(length) {
		return nil, fmt.Errorf("tensor %q has shape %s, want [%d]", name, info.ShapeString(), length)
	}
	data, err := info.Data()
	if err != nil {
		return nil, err
	}
	if info.DType != tensor.F32 {
		return nil, fmt.Errorf("tensor %q uses dtype %s, want F32", name, info.DType)
	}
	out := make([]float32, length)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

func loadF32(r *gguf.Reader, name string, count int) ([]float32, error) {
	info, ok := r.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("missing required tensor %q", name)
	}
	if info.DType != tensor.F32 {
		return nil, fmt.Errorf("tensor %q uses dtype %s, want F32", name, info.DType)
	}
	data, err := info.Data()
	if err != nil {
		return nil, err
	}
	if len(data) != count*4 {
		return nil, fmt.Errorf("tensor %q has %d bytes, want %d", name, len(data), count*4)
	}
	out := make([]float32, count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

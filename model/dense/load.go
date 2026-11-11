package dense

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tensor"
)

// Layer holds one pre-norm decoder block.
type Layer struct {
	AttnNorm []float32
	FFNNorm  []float32
	Q        tensor.Matrix
	K        tensor.Matrix
	V        tensor.Matrix
	O        tensor.Matrix
	QNorm    []float32
	KNorm    []float32
	Gate     tensor.Matrix
	Up       tensor.Matrix
	Down     tensor.Matrix
	// MoE fields are populated instead of Gate/Up/Down when Config.IsMoE.
	GateInp  tensor.Matrix
	GateExps tensor.Matrix
	UpExps   tensor.Matrix
	DownExps tensor.Matrix
}

// Model owns tensor descriptors that borrow the GGUF reader's mapping. The
// reader must stay open for the model's lifetime.
type Model struct {
	Config        *Config
	Embedding     tensor.Matrix
	Output        tensor.Matrix
	OutputNorm    []float32
	Layers        []Layer
	outputTied    bool
	embeddingRowB uint64
}

// Load validates required tensors and builds the model view.
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
		m.outputTied = true
	}
	m.OutputNorm, err = loadVector(r, "output_norm.weight", cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	qDim := cfg.HeadCount * cfg.HeadDim
	kvDim := cfg.KVHeadCount * cfg.HeadDim
	m.Layers = make([]Layer, cfg.BlockCount)
	for i := range m.Layers {
		layer := &m.Layers[i]
		prefix := fmt.Sprintf("blk.%d.", i)
		for _, suffix := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias",
			"attn_output.bias", "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"} {
			if _, ok := r.Tensor(prefix + suffix); ok {
				return nil, fmt.Errorf("tensor %q is not supported: dense builder has no bias path", prefix+suffix)
			}
		}
		if layer.AttnNorm, err = loadVector(r, prefix+"attn_norm.weight", cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.FFNNorm, err = loadVector(r, prefix+"ffn_norm.weight", cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.Q, err = loadMatrix(r, prefix+"attn_q.weight", qDim, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.K, err = loadMatrix(r, prefix+"attn_k.weight", kvDim, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.V, err = loadMatrix(r, prefix+"attn_v.weight", kvDim, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.O, err = loadMatrix(r, prefix+"attn_output.weight", cfg.HiddenSize, qDim); err != nil {
			return nil, err
		}
		if cfg.HasQKNorm {
			if layer.QNorm, err = loadVector(r, prefix+"attn_q_norm.weight", cfg.HeadDim); err != nil {
				return nil, err
			}
			if layer.KNorm, err = loadVector(r, prefix+"attn_k_norm.weight", cfg.HeadDim); err != nil {
				return nil, err
			}
		}
		if cfg.IsMoE() {
			if layer.GateInp, err = loadMatrix(r, prefix+"ffn_gate_inp.weight",
				cfg.ExpertCount, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if layer.GateExps, err = loadExperts(r, prefix+"ffn_gate_exps.weight",
				cfg.ExpertFFNSize, cfg.HiddenSize, cfg.ExpertCount); err != nil {
				return nil, err
			}
			if layer.UpExps, err = loadExperts(r, prefix+"ffn_up_exps.weight",
				cfg.ExpertFFNSize, cfg.HiddenSize, cfg.ExpertCount); err != nil {
				return nil, err
			}
			if layer.DownExps, err = loadExperts(r, prefix+"ffn_down_exps.weight",
				cfg.HiddenSize, cfg.ExpertFFNSize, cfg.ExpertCount); err != nil {
				return nil, err
			}
		} else {
			if layer.Gate, err = loadMatrix(r, prefix+"ffn_gate.weight", cfg.FFNSize, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if layer.Up, err = loadMatrix(r, prefix+"ffn_up.weight", cfg.FFNSize, cfg.HiddenSize); err != nil {
				return nil, err
			}
			if layer.Down, err = loadMatrix(r, prefix+"ffn_down.weight", cfg.HiddenSize, cfg.FFNSize); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// OutputTied reports whether the LM head shares the token embedding matrix.
func (m *Model) OutputTied() bool { return m.outputTied }

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

// loadExperts validates a rank-3 expert tensor and exposes it as a packed
// matrix with one expert's rows per contiguous expert slice.
func loadExperts(r *gguf.Reader, name string, rows, cols, experts int) (tensor.Matrix, error) {
	info, ok := r.Tensor(name)
	if !ok {
		return tensor.Matrix{}, fmt.Errorf("missing required tensor %q", name)
	}
	if len(info.Dimensions) != 3 ||
		info.Dimensions[0] != uint64(cols) ||
		info.Dimensions[1] != uint64(rows) ||
		info.Dimensions[2] != uint64(experts) {
		return tensor.Matrix{}, fmt.Errorf("tensor %q has shape %s, want [%d,%d,%d]",
			name, info.ShapeString(), cols, rows, experts)
	}
	if !info.DType.Supported() {
		return tensor.Matrix{}, fmt.Errorf("tensor %q uses unsupported dtype %s", name, info.DType)
	}
	block, err := info.DType.BlockLayout()
	if err != nil {
		return tensor.Matrix{}, fmt.Errorf("tensor %q dtype %s: %w", name, info.DType, err)
	}
	if block.Elements > 1 && cols%int(block.Elements) != 0 {
		return tensor.Matrix{}, fmt.Errorf("tensor %q column count %d does not match dtype %s",
			name, cols, info.DType)
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

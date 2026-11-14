// Package qwen35 implements the Qwen3.5 dense hybrid decoder: gated delta-net
// linear-attention layers interleaved with full GQA attention layers.
package qwen35

import (
	"fmt"
	"strings"

	"github.com/yuuka-ai/yuuka/gguf"
)

const Architecture = "qwen35"

// Config is the validated runtime configuration for one Qwen3.5 model.
type Config struct {
	Architecture          string
	BlockCount            int
	ContextLength         int
	HiddenSize            int
	FFNSize               int
	HeadCount             int
	KVHeadCount           int
	HeadDim               int
	ValueDim              int
	RoPEDim               int
	RoPESections          [4]int
	RoPEFreqBase          float64
	RMSNormEps            float32
	VocabSize             int
	SSMConvKernel         int
	SSMStateSize          int
	SSMGroupCount         int
	SSMTimeStepRank       int
	SSMInnerSize          int
	FullAttentionInterval int
}

// LoadConfig reads and validates Qwen3.5 hyperparameters from GGUF metadata.
func LoadConfig(r *gguf.Reader) (*Config, error) {
	arch, err := r.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != Architecture {
		return nil, fmt.Errorf("model architecture %q is not %q", arch, Architecture)
	}
	if value, ok := r.MetadataValue(arch + ".rope.scaling.type"); ok {
		if err := validateRoPEScaling(arch+".rope.scaling.type", value); err != nil {
			return nil, err
		}
	}
	cfg := &Config{Architecture: arch}
	intKey := func(key string) (int, error) {
		value, err := r.Int(key)
		if err != nil {
			return 0, err
		}
		if value <= 0 || value > 1<<31 {
			return 0, fmt.Errorf("metadata %s has invalid value %d", key, value)
		}
		return int(value), nil
	}
	floatKey := func(key string) (float64, error) {
		value, err := r.Float(key)
		if err != nil {
			return 0, err
		}
		return value, nil
	}
	if cfg.BlockCount, err = intKey("qwen35.block_count"); err != nil {
		return nil, err
	}
	if cfg.ContextLength, err = intKey("qwen35.context_length"); err != nil {
		return nil, err
	}
	if cfg.HiddenSize, err = intKey("qwen35.embedding_length"); err != nil {
		return nil, err
	}
	if cfg.FFNSize, err = intKey("qwen35.feed_forward_length"); err != nil {
		return nil, err
	}
	if cfg.HeadCount, err = intKey("qwen35.attention.head_count"); err != nil {
		return nil, err
	}
	if cfg.KVHeadCount, err = intKey("qwen35.attention.head_count_kv"); err != nil {
		return nil, err
	}
	if cfg.HeadDim, err = intKey("qwen35.attention.key_length"); err != nil {
		return nil, err
	}
	if cfg.ValueDim, err = intKey("qwen35.attention.value_length"); err != nil {
		return nil, err
	}
	if cfg.RoPEDim, err = intKey("qwen35.rope.dimension_count"); err != nil {
		return nil, err
	}
	if cfg.RoPEFreqBase, err = floatKey("qwen35.rope.freq_base"); err != nil {
		return nil, err
	}
	eps, err := floatKey("qwen35.attention.layer_norm_rms_epsilon")
	if err != nil {
		return nil, err
	}
	cfg.RMSNormEps = float32(eps)
	if cfg.SSMConvKernel, err = intKey("qwen35.ssm.conv_kernel"); err != nil {
		return nil, err
	}
	if cfg.SSMStateSize, err = intKey("qwen35.ssm.state_size"); err != nil {
		return nil, err
	}
	if cfg.SSMGroupCount, err = intKey("qwen35.ssm.group_count"); err != nil {
		return nil, err
	}
	if cfg.SSMTimeStepRank, err = intKey("qwen35.ssm.time_step_rank"); err != nil {
		return nil, err
	}
	if cfg.SSMInnerSize, err = intKey("qwen35.ssm.inner_size"); err != nil {
		return nil, err
	}
	if cfg.FullAttentionInterval, err = intKey("qwen35.full_attention_interval"); err != nil {
		return nil, err
	}
	if value, ok := r.MetadataValue("qwen35.rope.dimension_sections"); ok {
		sections, err := value.AsInts()
		if err != nil {
			return nil, fmt.Errorf("qwen35.rope.dimension_sections: %w", err)
		}
		if len(sections) != 4 {
			return nil, fmt.Errorf("qwen35.rope.dimension_sections has %d entries, want 4", len(sections))
		}
		for i, s := range sections {
			if s < 0 || s > 1<<20 {
				return nil, fmt.Errorf("invalid rope section %d", s)
			}
			cfg.RoPESections[i] = int(s)
		}
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.tokens"); ok {
		tokens, err := value.AsStrings()
		if err != nil {
			return nil, fmt.Errorf("tokenizer.ggml.tokens: %w", err)
		}
		cfg.VocabSize = len(tokens)
	} else if cfg.VocabSize, err = intKey("qwen35.vocab_size"); err != nil {
		return nil, fmt.Errorf("vocabulary size is unavailable")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// validateRoPEScaling rejects scaling types the runtime does not execute. The
// default and explicit none/default values use the base RoPE table.
func validateRoPEScaling(name string, value gguf.Value) error {
	text, err := value.AsString()
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	switch strings.ToLower(strings.TrimSpace(text)) {
	case "", "none", "default":
		return nil
	default:
		return fmt.Errorf("%s %q is not supported", name, text)
	}
}

func (c *Config) validate() error {
	if c.HeadCount%c.KVHeadCount != 0 {
		return fmt.Errorf("head count %d is not a multiple of KV head count %d",
			c.HeadCount, c.KVHeadCount)
	}
	if c.HeadDim != c.ValueDim {
		return fmt.Errorf("key head dim %d differs from value head dim %d",
			c.HeadDim, c.ValueDim)
	}
	if c.RoPEDim%2 != 0 || c.RoPEDim > c.HeadDim {
		return fmt.Errorf("invalid RoPE dimension %d for head dim %d", c.RoPEDim, c.HeadDim)
	}
	if sum := c.RoPESections[0] + c.RoPESections[1] + c.RoPESections[2] + c.RoPESections[3]; sum*2 != c.RoPEDim {
		return fmt.Errorf("RoPE sections sum to %d pairs, want %d", sum, c.RoPEDim/2)
	}
	if c.FullAttentionInterval <= 0 {
		return fmt.Errorf("full attention interval must be positive")
	}
	if c.SSMInnerSize != c.SSMTimeStepRank*c.SSMStateSize {
		return fmt.Errorf("ssm inner size %d != time_step_rank %d * state_size %d",
			c.SSMInnerSize, c.SSMTimeStepRank, c.SSMStateSize)
	}
	if c.SSMConvKernel < 2 {
		return fmt.Errorf("invalid ssm conv kernel %d", c.SSMConvKernel)
	}
	if c.ContextLength <= 0 || c.VocabSize <= 0 {
		return fmt.Errorf("invalid context or vocabulary size")
	}
	return nil
}

// IsRecurrent reports whether layer uses gated delta-net linear attention.
func (c *Config) IsRecurrent(layer int) bool {
	return (layer+1)%c.FullAttentionInterval != 0
}

// FullAttentionLayers returns layer indices that use full GQA attention.
func (c *Config) FullAttentionLayers() []int {
	var layers []int
	for i := 0; i < c.BlockCount; i++ {
		if !c.IsRecurrent(i) {
			layers = append(layers, i)
		}
	}
	return layers
}

// KVHeadsPerGroup returns query heads sharing one KV head.
func (c *Config) KVHeadsPerGroup() int {
	return c.HeadCount / c.KVHeadCount
}

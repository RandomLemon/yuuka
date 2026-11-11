// Package dense implements CPU reference inference for dense decoder
// Transformers with Llama or Qwen3 tensor layouts.
package dense

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/gguf"
)

const (
	Llama = "llama"
	Qwen3 = "qwen3"
)

// Config is the validated runtime configuration for one dense model.
type Config struct {
	Architecture string
	BlockCount   int
	Context      int
	HiddenSize   int
	FFNSize      int
	HeadCount    int
	KVHeadCount  int
	HeadDim      int
	ValueDim     int
	RoPEDim      int
	RoPEFreqBase float64
	RMSNormEps   float32
	VocabSize    int
	HasQKNorm    bool
}

// LoadConfig reads and validates dense Transformer hyperparameters.
func LoadConfig(r *gguf.Reader) (*Config, error) {
	arch, err := r.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != Llama && arch != Qwen3 {
		return nil, fmt.Errorf("model architecture %q is not %q or %q", arch, Llama, Qwen3)
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
	optionalInt := func(key string, fallback int) (int, error) {
		if _, ok := r.MetadataValue(key); !ok {
			return fallback, nil
		}
		return intKey(key)
	}
	if cfg.BlockCount, err = intKey(arch + ".block_count"); err != nil {
		return nil, err
	}
	if cfg.Context, err = intKey(arch + ".context_length"); err != nil {
		return nil, err
	}
	if cfg.HiddenSize, err = intKey(arch + ".embedding_length"); err != nil {
		return nil, err
	}
	if cfg.FFNSize, err = intKey(arch + ".feed_forward_length"); err != nil {
		return nil, err
	}
	if cfg.HeadCount, err = intKey(arch + ".attention.head_count"); err != nil {
		return nil, err
	}
	if cfg.KVHeadCount, err = intKey(arch + ".attention.head_count_kv"); err != nil {
		return nil, err
	}
	if cfg.HeadDim, err = optionalInt(arch+".attention.key_length", 0); err != nil {
		return nil, err
	}
	if cfg.HeadDim == 0 {
		if cfg.HiddenSize%cfg.HeadCount != 0 {
			return nil, fmt.Errorf("hidden size %d is not divisible by head count %d",
				cfg.HiddenSize, cfg.HeadCount)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.HeadCount
	}
	if cfg.ValueDim, err = optionalInt(arch+".attention.value_length", cfg.HeadDim); err != nil {
		return nil, err
	}
	if cfg.RoPEDim, err = optionalInt(arch+".rope.dimension_count", cfg.HeadDim); err != nil {
		return nil, err
	}
	if cfg.RoPEFreqBase, err = r.Float(arch + ".rope.freq_base"); err != nil {
		cfg.RoPEFreqBase = 10000
	}
	eps, err := r.Float(arch + ".attention.layer_norm_rms_epsilon")
	if err != nil {
		return nil, err
	}
	cfg.RMSNormEps = float32(eps)
	if value, ok := r.MetadataValue("tokenizer.ggml.tokens"); ok {
		tokens, err := value.AsStrings()
		if err != nil {
			return nil, fmt.Errorf("tokenizer.ggml.tokens: %w", err)
		}
		cfg.VocabSize = len(tokens)
	} else if cfg.VocabSize, err = intKey(arch + ".vocab_size"); err != nil {
		return nil, fmt.Errorf("vocabulary size is unavailable")
	}
	cfg.HasQKNorm = arch == Qwen3
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	if c.HeadCount%c.KVHeadCount != 0 {
		return fmt.Errorf("head count %d is not a multiple of KV head count %d",
			c.HeadCount, c.KVHeadCount)
	}
	if c.HeadDim != c.ValueDim {
		return fmt.Errorf("key head dim %d differs from value head dim %d", c.HeadDim, c.ValueDim)
	}
	if c.RoPEDim%2 != 0 || c.RoPEDim > c.HeadDim {
		return fmt.Errorf("invalid RoPE dimension %d for head dim %d", c.RoPEDim, c.HeadDim)
	}
	if c.Context <= 0 || c.VocabSize <= 0 {
		return fmt.Errorf("invalid context or vocabulary size")
	}
	return nil
}

// KVHeadsPerGroup returns query heads sharing one KV head.
func (c *Config) KVHeadsPerGroup() int { return c.HeadCount / c.KVHeadCount }

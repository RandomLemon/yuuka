package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/serving"
)

func newBenchBackend(reader *gguf.Reader, model *qwen35.Model, cfg benchBackendConfig) (serving.Backend, error) {
	switch cfg.Backend {
	case "cpu":
		return serving.NewQwen35Backend(model, cfg.ModelKey, cfg.Blocks, cfg.BlockSize)
	case "hip":
		return newBenchHIPBackend(reader, model, cfg)
	default:
		return nil, fmt.Errorf("unknown backend %q", cfg.Backend)
	}
}

//go:build hip && linux

package main

import (
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/serving"
)

func newHIPBackend(reader *gguf.Reader, model *qwen35.Model, cfg backendConfig) (serving.Backend, error) {
	return serving.NewQwen35GPUBackendWithKV(reader, model, cfg.ModelKey, cfg.Blocks,
		cfg.BlockSize, cfg.PrefillChunk, cfg.Graph, cfg.KVFormat.DType())
}

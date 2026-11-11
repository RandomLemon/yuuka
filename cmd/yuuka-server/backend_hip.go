//go:build hip && linux

package main

import (
	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/engine/gpu"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
)

func buildRuntime(reader *gguf.Reader, model *qwen35.Model, capacity int, backend string) (engine.Runtime, error) {
	return gpu.New(reader, model, capacity)
}

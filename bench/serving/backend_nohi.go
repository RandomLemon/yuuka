//go:build !hip || !linux

package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/serving"
)

func newBenchHIPBackend(*gguf.Reader, *qwen35.Model, benchBackendConfig) (serving.Backend, error) {
	return nil, fmt.Errorf("HIP backend is not built; use 'go build -tags hip'")
}

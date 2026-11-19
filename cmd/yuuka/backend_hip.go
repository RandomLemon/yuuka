//go:build hip && linux

package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/engine/gpu"
	"github.com/yuuka-ai/yuuka/model/qwen35"
)

func newHIPRuntime(l *loadedModel, capacity int, format qwen35.KVFormat) (engine.Runtime, error) {
	if l.qwen == nil {
		return nil, fmt.Errorf("HIP backend does not implement %q yet; use --backend cpu", l.arch)
	}
	runtime, err := gpu.NewWithKVType(l.reader, l.qwen, capacity, format.DType())
	if err != nil {
		return nil, fmt.Errorf("HIP runtime: %w", err)
	}
	return runtime, nil
}

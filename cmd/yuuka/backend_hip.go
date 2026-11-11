//go:build hip && linux

package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/engine/gpu"
)

func newHIPRuntime(l *loadedModel, capacity int) (engine.Runtime, error) {
	if l.qwen == nil {
		return nil, fmt.Errorf("HIP backend does not implement %q yet; use --backend cpu", l.arch)
	}
	runtime, err := gpu.New(l.reader, l.qwen, capacity)
	if err != nil {
		return nil, fmt.Errorf("HIP runtime: %w", err)
	}
	return runtime, nil
}

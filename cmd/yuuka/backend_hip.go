//go:build hip && linux

package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/engine/gpu"
)

func newHIPRuntime(l *loadedModel, capacity int) (engine.Runtime, error) {
	runtime, err := gpu.New(l.reader, l.model, capacity)
	if err != nil {
		return nil, fmt.Errorf("HIP runtime: %w", err)
	}
	return runtime, nil
}

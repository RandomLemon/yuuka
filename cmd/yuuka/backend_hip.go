//go:build hip && linux

package main

import (
	"errors"
	"fmt"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/engine/gpu"
	"github.com/yuuka-ai/yuuka/model/qwen35"
)

func newHIPRuntime(l *loadedModel, capacity int, format qwen35.KVFormat, chunk chunkSizeOption) (engine.Runtime, error) {
	if l.qwen == nil {
		return nil, fmt.Errorf("HIP backend does not implement %q yet; use --backend cpu", l.arch)
	}
	runtime, err := gpu.NewWithKVType(l.reader, l.qwen, capacity, format.DType())
	if err != nil {
		return nil, fmt.Errorf("HIP runtime: %w", err)
	}
	if chunk.rows <= 0 {
		return nil, errors.Join(fmt.Errorf("chunk size must be positive"), runtime.Close())
	}
	rows := chunk.resolve(runtime.DeviceArchitecture(), l.qwen.Config, l.quantizationLabel(), format)
	if err := runtime.SetPrefillChunk(rows); err != nil {
		return nil, errors.Join(err, runtime.Close())
	}
	return runtime, nil
}

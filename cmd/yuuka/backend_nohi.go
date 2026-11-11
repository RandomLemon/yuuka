//go:build !hip || !linux

package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/model/qwen35"
)

func newHIPRuntime(l *loadedModel, capacity int, format qwen35.KVFormat) (engine.Runtime, error) {
	return nil, fmt.Errorf("HIP backend is not built; use 'go build -tags hip'")
}

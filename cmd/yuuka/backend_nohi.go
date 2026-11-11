//go:build !hip || !linux

package main

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/engine"
)

func newHIPRuntime(l *loadedModel, capacity int) (engine.Runtime, error) {
	return nil, fmt.Errorf("HIP backend is not built; use 'go build -tags hip'")
}

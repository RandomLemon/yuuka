package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
)

func TestCPUEngineRejectsKVFormatForDense(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-llama.gguf")
	if err := fixture.WriteTinyDense(path, "llama"); err != nil {
		t.Fatal(err)
	}
	loaded, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if _, err := loaded.cpuEngine(16, nil, qwen35.KVQ8_0); err == nil {
		t.Fatal("dense CPU runtime accepted --kv-type q8_0")
	} else if !strings.Contains(err.Error(), "only f32 is implemented") {
		t.Fatalf("unexpected error: %v", err)
	}
	runner, err := loaded.cpuEngine(16, nil, qwen35.KVF32)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCPUEngineAcceptsKVFormatForQwen35(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	runner, err := loaded.cpuEngine(16, nil, qwen35.KVQ8_0)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}

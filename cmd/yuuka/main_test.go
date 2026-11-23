package main

import (
	"flag"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
)

func TestCLIChunkSizeSelection(t *testing.T) {
	reference := qwen35.Config{
		Architecture: qwen35.Architecture,
		BlockCount:   32,
		HiddenSize:   2560,
		FFNSize:      9216,
		HeadCount:    16,
		KVHeadCount:  4,
		HeadDim:      256,
	}
	t.Run("prompt and explicit flags", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			prompt int
			args   []string
			want   int
		}{
			{name: "below band", prompt: 1023, want: 512},
			{name: "start of band", prompt: 1024, want: 1024},
			{name: "boundary tail", prompt: 1025, want: 1024},
			{name: "end of band", prompt: 8192, want: 1024},
			{name: "above band", prompt: 8193, want: 512},
			{name: "explicit baseline", prompt: 2049, args: []string{"--chunk-size", "512"}, want: 512},
			{name: "explicit candidate", prompt: 530, args: []string{"--chunk-size", "1024"}, want: 1024},
			{name: "explicit larger", prompt: 2049, args: []string{"--chunk-size", "2048"}, want: 2048},
			{name: "explicit invalid", prompt: 2049, args: []string{"--chunk-size", "0"}, want: 0},
		} {
			t.Run(tc.name, func(t *testing.T) {
				fs := flag.NewFlagSet(tc.name, flag.ContinueOnError)
				chunkSize := fs.Int("chunk-size", defaultChunkSize, "")
				if err := fs.Parse(tc.args); err != nil {
					t.Fatal(err)
				}
				option := prefillChunkOption(fs, *chunkSize, tc.prompt)
				if got := option.resolve("gfx1151:sramecc-:xnack-", &reference, "Q4_K_M", qwen35.KVF16); got != tc.want {
					t.Fatalf("selected chunk %d, want %d", got, tc.want)
				}
			})
		}
	})
	other := reference
	other.HiddenSize = 4096
	option := chunkSizeOption{rows: defaultChunkSize, promptTokens: 2049}
	for _, tc := range []struct {
		name, arch, quantization string
		cfg                      *qwen35.Config
		kv                       qwen35.KVFormat
	}{
		{"F32 KV", "gfx1151", "Q4_K_M", &reference, qwen35.KVF32},
		{"Q8_0 KV", "gfx1151", "Q4_K_M", &reference, qwen35.KVQ8_0},
		{"other GPU", "gfx1100", "Q4_K_M", &reference, qwen35.KVF16},
		{"other quantization", "gfx1151", "Q6_K", &reference, qwen35.KVF16},
		{"other model geometry", "gfx1151", "Q4_K_M", &other, qwen35.KVF16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := option.resolve(tc.arch, tc.cfg, tc.quantization, tc.kv); got != defaultChunkSize {
				t.Fatalf("selected chunk %d, want %d", got, defaultChunkSize)
			}
		})
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

func TestCPUEngineLoadsQwen35MoE(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35moe.gguf")
	if err := fixture.WriteTinyQwen35MoE(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if loaded.qwen == nil || !loaded.qwen.Config.IsMoE() {
		t.Fatal("qwen35moe fixture did not load as a MoE model")
	}
	runner, err := loaded.cpuEngine(16, nil, qwen35.KVQ8_0)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}

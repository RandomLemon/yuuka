//go:build hip && linux

// prefill-compare measures HIP prefill and an optional sequential one-token
// decode through the final host logit rows.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/yuuka-ai/yuuka/engine/gpu"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

type config struct {
	model, tokens, logits               string
	prompt, chunk, warmups, repetitions int
	decode                              int
	writeTokens                         bool
}

func main() {
	var cfg config
	flag.StringVar(&cfg.model, "model", "", "GGUF model path")
	flag.StringVar(&cfg.tokens, "tokens", "", "little-endian int32 token file")
	flag.StringVar(&cfg.logits, "logits", "", "write final logits as little-endian float32")
	flag.IntVar(&cfg.prompt, "prompt", 8192, "number of input tokens")
	flag.IntVar(&cfg.chunk, "chunk", 512, "maximum prefill rows")
	flag.IntVar(&cfg.warmups, "warmup", 2, "untimed repetitions")
	flag.IntVar(&cfg.repetitions, "reps", 5, "timed repetitions")
	flag.IntVar(&cfg.decode, "decode", 0, "continuation tokens decoded one at a time after prefill")
	flag.BoolVar(&cfg.writeTokens, "write-tokens", false, "tokenize the fixed benchmark prompt into the token file")
	flag.Parse()
	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(cfg config) (err error) {
	if cfg.model == "" || cfg.tokens == "" || cfg.prompt <= 0 || cfg.prompt > math.MaxInt32-64 ||
		cfg.decode < 0 || cfg.decode > math.MaxInt32-cfg.prompt ||
		cfg.chunk <= 0 || cfg.chunk > 4096 || cfg.warmups < 0 || cfg.repetitions <= 0 {
		return errors.New("invalid benchmark arguments")
	}
	reader, err := gguf.Open(cfg.model)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, reader.Close()) }()
	if cfg.writeTokens {
		tok, err := tokenizer.NewFromGGUF(reader)
		if err != nil {
			return err
		}
		tokens, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", cfg.prompt))
		if err != nil {
			return err
		}
		if len(tokens) < cfg.prompt {
			return fmt.Errorf("fixed prompt has %d tokens, need %d", len(tokens), cfg.prompt)
		}
		return writeBinary(cfg.tokens, tokens[:cfg.prompt])
	}
	data, err := os.ReadFile(cfg.tokens)
	if err != nil {
		return err
	}
	needTokens := cfg.prompt + cfg.decode
	if len(data)%4 != 0 || len(data)/4 < needTokens {
		return fmt.Errorf("token file has %d bytes, need at least %d aligned bytes", len(data), 4*needTokens)
	}
	tokens := make([]int32, needTokens)
	if err := binary.Read(bytes.NewReader(data[:4*needTokens]), binary.LittleEndian, tokens); err != nil {
		return err
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		return err
	}
	for i, token := range tokens {
		if token < 0 || int(token) >= model.Config.VocabSize {
			return fmt.Errorf("token %d at index %d is outside model vocabulary", token, i)
		}
	}
	const blockSize = 16
	capacity := cfg.prompt + 64
	if cfg.decode > 0 {
		capacity = cfg.prompt + cfg.decode
	}
	chunk := min(cfg.chunk, cfg.prompt)
	device, err := gpu.Open(reader, model, gpu.Options{
		Blocks:       (capacity + blockSize - 1) / blockSize,
		BlockSize:    blockSize,
		PrefillChunk: chunk,
		KVType:       tensor.F16,
		KVTypeSet:    true,
	})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, device.Close()) }()
	table := memory.NewTable(device.Allocator())
	defer table.Release()
	if err := table.Cover(capacity); err != nil {
		return err
	}
	session, err := device.NewSession(table, 0, nil, gpu.SessionOptions{Capacity: capacity})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, session.Close()) }()

	var logits []float32
	samples := make([]int64, 0, cfg.repetitions)
	decodeSamples := make([]int64, 0, cfg.repetitions)
	for i := -cfg.warmups; i < cfg.repetitions; i++ {
		if err := session.Reset(); err != nil {
			return err
		}
		start := time.Now()
		logits, err = session.Prefill(tokens[:cfg.prompt], true)
		elapsed := time.Since(start)
		if err != nil {
			return err
		}
		if i >= 0 {
			samples = append(samples, elapsed.Nanoseconds())
		}
		if cfg.decode > 0 {
			decodeStart := time.Now()
			for j := range cfg.decode {
				logits, err = session.Decode(tokens[cfg.prompt+j])
				if err != nil {
					return fmt.Errorf("decode token %d: %w", j, err)
				}
			}
			decodeElapsed := time.Since(decodeStart)
			if logits == nil {
				return fmt.Errorf("decode token %d: missing host logits", cfg.decode-1)
			}
			if i >= 0 {
				decodeSamples = append(decodeSamples, decodeElapsed.Nanoseconds())
			}
		}
	}
	if logits == nil || len(logits) != model.Config.VocabSize {
		return fmt.Errorf("final host logits row has %d values, want %d", len(logits), model.Config.VocabSize)
	}
	top := 0
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("nonfinite logit at position %d", i)
		}
		if value > logits[top] {
			top = i
		}
	}
	if cfg.logits != "" {
		if err := writeBinary(cfg.logits, logits); err != nil {
			return err
		}
	}
	return json.NewEncoder(os.Stdout).Encode(struct {
		Prompt          int     `json:"prompt"`
		Chunk           int     `json:"chunk"`
		Warmups         int     `json:"warmup"`
		Top             int     `json:"top1"`
		SamplesNS       []int64 `json:"samples_ns"`
		Decode          int     `json:"decode,omitempty"`
		DecodeSamplesNS []int64 `json:"decode_samples_ns,omitempty"`
	}{cfg.prompt, chunk, cfg.warmups, top, samples, cfg.decode, decodeSamples})
}

func writeBinary(path string, data any) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	return errors.Join(binary.Write(file, binary.LittleEndian, data), file.Close())
}

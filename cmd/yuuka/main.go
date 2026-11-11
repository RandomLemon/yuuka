// Command yuuka loads a GGUF model for inspection, generation, and
// single-request benchmarking.
package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "inspect":
		err = runInspect(os.Args[2:])
	case "run":
		err = runGenerate(os.Args[2:])
	case "bench":
		err = runBench(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "yuuka:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage: yuuka <command> [options]

commands:
  inspect   print model metadata, configuration, and tensor types
  run       generate text from a prompt
  bench     measure prefill and decode throughput

`)
}

type loadedModel struct {
	reader    *gguf.Reader
	tokenizer *tokenizer.Tokenizer
	model     *qwen35.Model
}

func load(path string) (*loadedModel, error) {
	if path == "" {
		return nil, errors.New("--model is required")
	}
	r, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.NewFromGGUF(r)
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	model, err := qwen35.Load(r)
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	return &loadedModel{reader: r, tokenizer: tok, model: model}, nil
}

func (l *loadedModel) Close() error { return l.reader.Close() }

func runInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	modelPath := fs.String("model", "", "GGUF model path")
	tensors := fs.Bool("tensors", false, "list every tensor")
	hash := fs.Bool("hash", false, "print the model SHA-256")
	if err := fs.Parse(args); err != nil {
		return err
	}
	l, err := load(*modelPath)
	if err != nil {
		return err
	}
	defer l.Close()
	cfg := l.model.Config
	fmt.Printf("model: %s\n", l.reader.Path())
	fmt.Printf("gguf_version: %d\n", l.reader.Version())
	fmt.Printf("alignment: %d\n", l.reader.Alignment())
	fmt.Printf("architecture: %s\n", cfg.Architecture)
	fmt.Printf("tensors: %d\n", len(l.reader.Tensors()))
	fmt.Printf("layers: %d (full_attention=%d, recurrent=%d)\n",
		cfg.BlockCount, len(cfg.FullAttentionLayers()), cfg.BlockCount-len(cfg.FullAttentionLayers()))
	fmt.Printf("hidden: %d  ffn: %d  vocab: %d\n", cfg.HiddenSize, cfg.FFNSize, cfg.VocabSize)
	fmt.Printf("attention: heads=%d kv_heads=%d head_dim=%d rope_dim=%d\n",
		cfg.HeadCount, cfg.KVHeadCount, cfg.HeadDim, cfg.RoPEDim)
	fmt.Printf("ssm: conv=%d state=%d groups=%d time_step_rank=%d inner=%d\n",
		cfg.SSMConvKernel, cfg.SSMStateSize, cfg.SSMGroupCount, cfg.SSMTimeStepRank, cfg.SSMInnerSize)
	fmt.Printf("context: %d  rms_eps: %g  rope_base: %g\n",
		cfg.ContextLength, cfg.RMSNormEps, cfg.RoPEFreqBase)
	fmt.Printf("tokenizer: model=%s pre=%s vocab=%d eos=%d\n",
		l.tokenizerModel(), l.tokenizerPre(), l.tokenizer.VocabSize(), l.tokenizer.EOSToken())
	summary := l.reader.DTypeSummary()
	types := make([]tensor.DType, 0, len(summary))
	for dtype := range summary {
		types = append(types, dtype)
	}
	sort.Slice(types, func(i, j int) bool { return types[i] < types[j] })
	fmt.Print("dtypes:")
	for _, dtype := range types {
		fmt.Printf(" %s=%d", dtype, summary[dtype])
	}
	fmt.Println()
	if *tensors {
		for _, info := range l.reader.Tensors() {
			fmt.Printf("%-40s %-10s %s\n", info.Name, info.DType, info.ShapeString())
		}
	}
	if *hash {
		sum, err := fileSHA256(l.reader.Path())
		if err != nil {
			return err
		}
		fmt.Printf("sha256: %s\n", sum)
	}
	return nil
}

// Small helpers avoid exporting display-only metadata from the reader.
func (l *loadedModel) tokenizerModel() string {
	value, _ := l.reader.MetadataValue("tokenizer.ggml.model")
	s, _ := value.AsString()
	return s
}

func (l *loadedModel) tokenizerPre() string {
	value, _ := l.reader.MetadataValue("tokenizer.ggml.pre")
	s, _ := value.AsString()
	return s
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

func runGenerate(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	modelPath := fs.String("model", "", "GGUF model path")
	prompt := fs.String("prompt", "", "prompt text (reads stdin when empty)")
	system := fs.String("system", "", "system message for chat mode")
	chat := fs.Bool("chat", false, "render the model chat template")
	thinking := fs.Bool("thinking", false, "enable an open reasoning block in chat mode")
	contextSize := fs.Int("ctx", 4096, "context capacity")
	backend := fs.String("backend", "cpu", "execution backend: cpu or hip")
	maxTokens := fs.Int("max-tokens", 128, "maximum generated tokens")
	temperature := fs.Float64("temperature", 0, "sampling temperature (0 = greedy)")
	topK := fs.Int("top-k", 40, "top-k sampling cutoff")
	topP := fs.Float64("top-p", 0.95, "nucleus sampling cutoff")
	seed := fs.Int64("seed", 0, "sampling seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	l, err := load(*modelPath)
	if err != nil {
		return err
	}
	defer l.Close()
	text := *prompt
	if text == "" {
		raw, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		text = string(raw)
	}
	if text == "" {
		return errors.New("prompt is empty")
	}
	if *chat {
		text, err = l.tokenizer.RenderPrompt(*system, text, tokenizer.ChatOptions{
			AddGenerationPrompt: true,
			EnableThinking:      *thinking,
		})
		if err != nil {
			return err
		}
	}
	ids, err := l.tokenizer.Encode(text)
	if err != nil {
		return err
	}
	if *contextSize > l.model.Config.ContextLength {
		return fmt.Errorf("--ctx %d exceeds model context %d", *contextSize, l.model.Config.ContextLength)
	}
	sampler, err := engine.NewSampler(engine.SamplerConfig{
		Temperature: float32(*temperature),
		TopK:        *topK,
		TopP:        float32(*topP),
		Seed:        *seed,
	}, l.model.Config.VocabSize)
	if err != nil {
		return err
	}
	runner, err := buildEngine(l, *contextSize, sampler, *backend)
	if err != nil {
		return err
	}
	defer runner.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	started := time.Now()
	result, err := runner.Generate(ctx, ids, engine.GenerateOptions{
		MaxTokens: *maxTokens,
		OnToken: func(_ int32, delta string) error {
			_, err := io.WriteString(os.Stdout, delta)
			return err
		},
	})
	if err != nil {
		return err
	}
	if !strings.HasSuffix(result.Text, "\n") {
		fmt.Println()
	}
	fmt.Fprintf(os.Stderr, "prompt_tokens=%d output_tokens=%d finish=%s\n",
		result.PromptTokens, result.OutputTokens, result.FinishReason)
	fmt.Fprintf(os.Stderr, "prefill=%.3fs (%.2f tok/s) ttft=%.3fs decode=%.3fs (%.2f tok/s) total=%.3fs\n",
		result.PrefillTime.Seconds(), result.PromptTokensPS,
		result.TimeToFirst.Seconds(), result.DecodeTime.Seconds(),
		result.DecodeTokensPS, time.Since(started).Seconds())
	return nil
}

func runBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	modelPath := fs.String("model", "", "GGUF model path")
	promptTokens := fs.Int("prompt-tokens", 128, "prompt length in tokens")
	decodeTokens := fs.Int("decode-tokens", 32, "generated tokens per repetition")
	contextSize := fs.Int("ctx", 0, "context capacity (default: prompt + decode + 64)")
	backend := fs.String("backend", "cpu", "execution backend: cpu or hip")
	warmup := fs.Int("warmup", 1, "warmup repetitions")
	reps := fs.Int("reps", 3, "measured repetitions")
	hash := fs.Bool("hash", false, "print the model SHA-256")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *promptTokens <= 0 || *decodeTokens <= 0 {
		return errors.New("prompt and decode token counts must be positive")
	}
	l, err := load(*modelPath)
	if err != nil {
		return err
	}
	defer l.Close()
	corpus := strings.Repeat("The quick brown fox jumps over the lazy dog. ", (*promptTokens+8)/8+2)
	ids, err := l.tokenizer.Encode(corpus)
	if err != nil {
		return err
	}
	if len(ids) < *promptTokens {
		return fmt.Errorf("corpus produced %d tokens, need %d", len(ids), *promptTokens)
	}
	ids = ids[:*promptTokens]
	capacity := *contextSize
	if capacity == 0 {
		capacity = *promptTokens + *decodeTokens + 64
	}
	if capacity > l.model.Config.ContextLength {
		return fmt.Errorf("--ctx %d exceeds model context %d", capacity, l.model.Config.ContextLength)
	}
	sampler, err := engine.NewSampler(engine.SamplerConfig{}, l.model.Config.VocabSize)
	if err != nil {
		return err
	}
	runner, err := buildEngine(l, capacity, sampler, *backend)
	if err != nil {
		return err
	}
	defer runner.Close()
	run := func() (*engine.GenerateResult, error) {
		return runner.Generate(context.Background(), ids, engine.GenerateOptions{
			MaxTokens: *decodeTokens,
			IgnoreEOS: true,
		})
	}
	for range *warmup {
		if _, err := run(); err != nil {
			return err
		}
	}
	var prefill, decode, ttft float64
	var prefillPS, decodePS float64
	for i := 0; i < *reps; i++ {
		result, err := run()
		if err != nil {
			return err
		}
		prefill += result.PrefillTime.Seconds()
		decode += result.DecodeTime.Seconds()
		ttft += result.TimeToFirst.Seconds()
		prefillPS += result.PromptTokensPS
		decodePS += result.DecodeTokensPS
	}
	n := float64(*reps)
	fmt.Printf("model: %s\n", l.reader.Path())
	fmt.Printf("architecture: %s  quantization: %s\n", l.model.Config.Architecture, l.quantizationLabel())
	if *hash {
		sum, err := fileSHA256(l.reader.Path())
		if err != nil {
			return err
		}
		fmt.Printf("sha256: %s\n", sum)
	}
	fmt.Printf("prompt_tokens: %d  decode_tokens: %d  reps: %d  warmup: %d\n",
		*promptTokens, *decodeTokens, *reps, *warmup)
	fmt.Printf("prefill: %.3fs (%.2f tok/s)\n", prefill/n, prefillPS/n)
	fmt.Printf("ttft: %.3fs  decode: %.3fs (%.2f tok/s)\n", ttft/n, decode/n, decodePS/n)
	return nil
}

func buildEngine(l *loadedModel, capacity int, sampler *engine.Sampler, backend string) (*engine.Engine, error) {
	switch backend {
	case "cpu":
		return engine.New(l.model, l.tokenizer, capacity, sampler)
	case "hip":
		runtime, err := newHIPRuntime(l, capacity)
		if err != nil {
			return nil, err
		}
		runner, err := engine.NewWithRuntime(runtime, l.tokenizer, sampler)
		if err != nil {
			_ = runtime.Close()
			return nil, err
		}
		return runner, nil
	default:
		return nil, fmt.Errorf("unknown backend %q (want cpu or hip)", backend)
	}
}

func (l *loadedModel) quantizationLabel() string {
	value, ok := l.reader.MetadataValue("general.file_type")
	if !ok {
		return "unknown"
	}
	raw, err := value.AsUint()
	if err != nil {
		return "unknown"
	}
	switch raw {
	case 15:
		return "Q4_K_M"
	case 14:
		return "Q5_K_M"
	case 17:
		return "Q6_K"
	default:
		return fmt.Sprintf("file_type=%d", raw)
	}
}

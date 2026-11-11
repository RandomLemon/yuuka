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
	"math"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/yuuka-ai/yuuka/backend/npu"
	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/dense"
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
	case "npu-probe":
		err = runNPUProbe()
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
  npu-probe report NPU device, driver, and userspace runtime availability

`)
}

type loadedModel struct {
	reader    *gguf.Reader
	tokenizer *tokenizer.Tokenizer
	qwen      *qwen35.Model
	dense     *dense.Model
	arch      string
}

func runNPUProbe() error {
	caps := npu.Probe()
	fmt.Printf("state: %s\n", caps.State)
	fmt.Printf("device: %s\n", caps.DevicePath)
	fmt.Printf("driver: %t\n", caps.DriverLoaded)
	fmt.Printf("userspace: %t\n", caps.Userspace)
	if caps.RuntimePath != "" {
		fmt.Printf("runtime: %s\n", caps.RuntimePath)
	}
	if caps.Reason != "" {
		fmt.Printf("reason: %s\n", caps.Reason)
	}
	device, reason := npu.Request(caps)
	fmt.Printf("fallback: %s (%s)\n", device, reason)
	return nil
}

func requestNPU(enabled bool) {
	if !enabled {
		return
	}
	caps := npu.Probe()
	device, reason := npu.Request(caps)
	fmt.Fprintf(os.Stderr, "npu: %s; execution device %s\n", reason, device)
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
	arch, err := r.String("general.architecture")
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	loaded := &loadedModel{reader: r, tokenizer: tok, arch: arch}
	switch arch {
	case qwen35.Architecture:
		loaded.qwen, err = qwen35.Load(r)
	case dense.Llama, dense.Qwen3, dense.Qwen3MoE:
		loaded.dense, err = dense.Load(r)
	default:
		err = fmt.Errorf("model architecture %q is not supported", arch)
	}
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	return loaded, nil
}

func (l *loadedModel) Close() error { return l.reader.Close() }

func (l *loadedModel) contextLength() int {
	if l.qwen != nil {
		return l.qwen.Config.ContextLength
	}
	return l.dense.Config.Context
}

func (l *loadedModel) vocabSize() int {
	if l.qwen != nil {
		return l.qwen.Config.VocabSize
	}
	return l.dense.Config.VocabSize
}

func (l *loadedModel) cpuEngine(capacity int, sampler *engine.Sampler, format qwen35.KVFormat) (*engine.Engine, error) {
	if l.qwen != nil {
		return engine.NewWithKVFormat(l.qwen, l.tokenizer, capacity, sampler, format)
	}
	runtime, err := dense.NewRuntime(l.dense, capacity)
	if err != nil {
		return nil, err
	}
	runner, err := engine.NewWithRuntime(runtime, l.tokenizer, sampler)
	if err != nil {
		_ = runtime.Close()
		return nil, err
	}
	return runner, nil
}

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
	fmt.Printf("model: %s\n", l.reader.Path())
	fmt.Printf("gguf_version: %d\n", l.reader.Version())
	fmt.Printf("alignment: %d\n", l.reader.Alignment())
	fmt.Printf("architecture: %s\n", l.arch)
	fmt.Printf("tensors: %d\n", len(l.reader.Tensors()))
	if l.qwen != nil {
		cfg := l.qwen.Config
		fmt.Printf("layers: %d (full_attention=%d, recurrent=%d)\n",
			cfg.BlockCount, len(cfg.FullAttentionLayers()), cfg.BlockCount-len(cfg.FullAttentionLayers()))
		fmt.Printf("hidden: %d  ffn: %d  vocab: %d\n", cfg.HiddenSize, cfg.FFNSize, cfg.VocabSize)
		fmt.Printf("attention: heads=%d kv_heads=%d head_dim=%d rope_dim=%d\n",
			cfg.HeadCount, cfg.KVHeadCount, cfg.HeadDim, cfg.RoPEDim)
		fmt.Printf("ssm: conv=%d state=%d groups=%d time_step_rank=%d inner=%d\n",
			cfg.SSMConvKernel, cfg.SSMStateSize, cfg.SSMGroupCount, cfg.SSMTimeStepRank, cfg.SSMInnerSize)
		fmt.Printf("context: %d  rms_eps: %g  rope_base: %g\n",
			cfg.ContextLength, cfg.RMSNormEps, cfg.RoPEFreqBase)
	} else {
		cfg := l.dense.Config
		fmt.Printf("layers: %d (dense full attention)\n", cfg.BlockCount)
		fmt.Printf("hidden: %d  ffn: %d  vocab: %d\n", cfg.HiddenSize, cfg.FFNSize, cfg.VocabSize)
		fmt.Printf("attention: heads=%d kv_heads=%d head_dim=%d rope_dim=%d qk_norm=%t\n",
			cfg.HeadCount, cfg.KVHeadCount, cfg.HeadDim, cfg.RoPEDim, cfg.HasQKNorm)
		fmt.Printf("context: %d  rms_eps: %g  rope_base: %g tied_output=%t\n",
			cfg.Context, cfg.RMSNormEps, cfg.RoPEFreqBase, l.dense.OutputTied())
		if cfg.IsMoE() {
			fmt.Printf("moe: experts=%d used=%d expert_ffn=%d\n",
				cfg.ExpertCount, cfg.ExpertUsedCount, cfg.ExpertFFNSize)
		}
	}
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
	chunkSize := fs.Int("chunk-size", 512, "HIP prompt chunk size")
	maxTokens := fs.Int("max-tokens", 128, "maximum generated tokens")
	temperature := fs.Float64("temperature", 0, "sampling temperature (0 = greedy)")
	topK := fs.Int("top-k", 40, "top-k sampling cutoff")
	topP := fs.Float64("top-p", 0.95, "nucleus sampling cutoff")
	seed := fs.Int64("seed", 0, "sampling seed")
	kvType := fs.String("kv-type", "f32", "KV cache format: f32, f16, or q8_0")
	speculative := fs.Bool("speculative", false, "enable greedy prompt-lookup speculative decoding")
	specDraft := fs.Int("spec-max-draft", 4, "maximum speculative draft tokens")
	specMin := fs.Int("spec-min-match", 2, "minimum prompt-lookup suffix length")
	npuOpt := fs.Bool("npu", false, "request NPU execution and fall back to the selected backend")
	if err := fs.Parse(args); err != nil {
		return err
	}
	requestNPU(*npuOpt)
	kvFormat, err := qwen35.ParseKVFormat(*kvType)
	if err != nil {
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
	if *contextSize > l.contextLength() {
		return fmt.Errorf("--ctx %d exceeds model context %d", *contextSize, l.contextLength())
	}
	sampler, err := engine.NewSampler(engine.SamplerConfig{
		Temperature: float32(*temperature),
		TopK:        *topK,
		TopP:        float32(*topP),
		Seed:        *seed,
	}, l.vocabSize())
	if err != nil {
		return err
	}
	runner, err := buildEngine(l, *contextSize, sampler, *backend, *chunkSize, kvFormat)
	if err != nil {
		return err
	}
	defer runner.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	started := time.Now()
	var spec *engine.SpeculativeOptions
	if *speculative {
		spec = &engine.SpeculativeOptions{MaxDraft: *specDraft, MinMatch: *specMin}
	}
	result, err := runner.Generate(ctx, ids, engine.GenerateOptions{
		MaxTokens:   *maxTokens,
		Speculative: spec,
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
	if *speculative {
		fmt.Fprintf(os.Stderr, "speculative draft=%d accepted=%d\n",
			result.DraftTokens, result.AcceptedDraft)
	}
	return nil
}

func runBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	modelPath := fs.String("model", "", "GGUF model path")
	promptTokens := fs.Int("prompt-tokens", 128, "prompt length in tokens")
	decodeTokens := fs.Int("decode-tokens", 32, "generated tokens per repetition")
	contextSize := fs.Int("ctx", 0, "context capacity (default: prompt + decode + 64)")
	backend := fs.String("backend", "cpu", "execution backend: cpu or hip")
	chunkSize := fs.Int("chunk-size", 512, "HIP prompt chunk size")
	warmup := fs.Int("warmup", 1, "warmup repetitions")
	reps := fs.Int("reps", 3, "measured repetitions")
	hash := fs.Bool("hash", false, "print the model SHA-256")
	kvType := fs.String("kv-type", "f32", "KV cache format: f32, f16, or q8_0")
	speculative := fs.Bool("speculative", false, "enable greedy prompt-lookup speculative decoding")
	specDraft := fs.Int("spec-max-draft", 4, "maximum speculative draft tokens")
	specMin := fs.Int("spec-min-match", 2, "minimum prompt-lookup suffix length")
	npuOpt := fs.Bool("npu", false, "request NPU execution and fall back to the selected backend")
	if err := fs.Parse(args); err != nil {
		return err
	}
	requestNPU(*npuOpt)
	kvFormat, err := qwen35.ParseKVFormat(*kvType)
	if err != nil {
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
	if capacity > l.contextLength() {
		return fmt.Errorf("--ctx %d exceeds model context %d", capacity, l.contextLength())
	}
	sampler, err := engine.NewSampler(engine.SamplerConfig{}, l.vocabSize())
	if err != nil {
		return err
	}
	runner, err := buildEngine(l, capacity, sampler, *backend, *chunkSize, kvFormat)
	if err != nil {
		return err
	}
	defer runner.Close()
	var spec *engine.SpeculativeOptions
	if *speculative {
		spec = &engine.SpeculativeOptions{MaxDraft: *specDraft, MinMatch: *specMin}
	}
	run := func() (*engine.GenerateResult, error) {
		return runner.Generate(context.Background(), ids, engine.GenerateOptions{
			MaxTokens:   *decodeTokens,
			IgnoreEOS:   true,
			Speculative: spec,
		})
	}
	for range *warmup {
		if _, err := run(); err != nil {
			return err
		}
	}
	prefillSamples := make([]float64, 0, *reps)
	decodeSamples := make([]float64, 0, *reps)
	ttftSamples := make([]float64, 0, *reps)
	for i := 0; i < *reps; i++ {
		result, err := run()
		if err != nil {
			return err
		}
		prefillSamples = append(prefillSamples, result.PrefillTime.Seconds())
		decodeSamples = append(decodeSamples, result.DecodeTime.Seconds())
		ttftSamples = append(ttftSamples, result.TimeToFirst.Seconds())
	}
	fmt.Printf("model: %s\n", l.reader.Path())
	fmt.Printf("architecture: %s  quantization: %s  backend: %s  kv: %s\n",
		l.arch, l.quantizationLabel(), *backend, kvFormat)
	if *hash {
		sum, err := fileSHA256(l.reader.Path())
		if err != nil {
			return err
		}
		fmt.Printf("sha256: %s\n", sum)
	}
	fmt.Printf("prompt_tokens: %d  decode_tokens: %d  reps: %d  warmup: %d  context: %d\n",
		*promptTokens, *decodeTokens, *reps, *warmup, capacity)
	fmt.Printf("prefill: p50=%.3fs p95=%.3fs (%.2f tok/s p50)\n",
		percentile(prefillSamples, 0.50), percentile(prefillSamples, 0.95),
		float64(*promptTokens)/percentile(prefillSamples, 0.50))
	fmt.Printf("ttft: p50=%.3fs p95=%.3fs\n", percentile(ttftSamples, 0.50), percentile(ttftSamples, 0.95))
	fmt.Printf("decode: p50=%.3fs p95=%.3fs (%.2f tok/s p50)\n",
		percentile(decodeSamples, 0.50), percentile(decodeSamples, 0.95),
		float64(*decodeTokens-1)/percentile(decodeSamples, 0.50))
	if *speculative {
		result, err := run()
		if err != nil {
			return err
		}
		acceptance := 0.0
		if result.DraftTokens > 0 {
			acceptance = float64(result.AcceptedDraft) / float64(result.DraftTokens)
		}
		fmt.Printf("speculative: draft=%d accepted=%d acceptance=%.3f\n",
			result.DraftTokens, result.AcceptedDraft, acceptance)
	}
	return nil
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	index := int(math.Ceil(p*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func buildEngine(l *loadedModel, capacity int, sampler *engine.Sampler, backend string, chunkSize int, format qwen35.KVFormat) (*engine.Engine, error) {
	switch backend {
	case "cpu":
		return l.cpuEngine(capacity, sampler, format)
	case "hip":
		runtime, err := newHIPRuntime(l, capacity, format)
		if err != nil {
			return nil, err
		}
		if chunkSize <= 0 {
			_ = runtime.Close()
			return nil, fmt.Errorf("chunk size must be positive")
		}
		if configurer, ok := runtime.(interface{ SetPrefillChunk(int) error }); ok {
			if err := configurer.SetPrefillChunk(chunkSize); err != nil {
				_ = runtime.Close()
				return nil, err
			}
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
	case 7:
		return "Q8_0"
	case 10:
		return "Q2_K"
	case 11:
		return "Q3_K_S"
	case 12:
		return "Q3_K_M"
	case 13:
		return "Q3_K_L"
	case 14:
		return "Q4_K_S"
	case 15:
		return "Q4_K_M"
	case 16:
		return "Q5_K_S"
	case 17:
		return "Q5_K_M"
	case 18:
		return "Q6_K"
	default:
		return fmt.Sprintf("file_type=%d", raw)
	}
}

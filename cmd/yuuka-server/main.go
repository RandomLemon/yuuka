// Command yuuka-server exposes a loaded GGUF model through an
// OpenAI-compatible chat completions endpoint backed by the continuous
// scheduler, paged KV cache, prefix cache, and optional HIP Graph replay.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/yuuka-ai/yuuka/backend/npu"
	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/scheduler"
	"github.com/yuuka-ai/yuuka/serving"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

const maxRequestBody = 8 << 20

type backendConfig struct {
	ModelPath    string
	Backend      string
	Blocks       int
	BlockSize    int
	PrefillChunk int
	Graph        bool
	ModelKey     string
	KVFormat     qwen35.KVFormat
}

type server struct {
	manager     *serving.Manager
	tokenizer   *tokenizer.Tokenizer
	model       *qwen35.Model
	contextSize int
	modelName   string
	backend     string
	graph       bool
	started     time.Time
	requestSeq  atomic.Int64
}

func main() {
	modelPath := flag.String("model", "", "GGUF model path")
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	contextSize := flag.Int("ctx", 4096, "maximum prompt plus output tokens per request")
	modelName := flag.String("model-name", "yuuka", "model identifier returned by the API")
	backendName := flag.String("backend", "cpu", "execution backend: cpu or hip")
	blockSize := flag.Int("block-size", 16, "KV cache block size in tokens")
	kvBlocks := flag.Int("kv-blocks", 0, "total KV blocks (0 derives from ctx and concurrency)")
	concurrency := flag.Int("max-concurrency", 4, "maximum active requests")
	queue := flag.Int("max-queue", 64, "maximum pending requests")
	prefillChunk := flag.Int("prefill-chunk", 512, "prompt tokens processed per prefill opportunity")
	requestTimeout := flag.Duration("request-timeout", 5*time.Minute, "per-request deadline")
	queueTimeout := flag.Duration("queue-timeout", 30*time.Second, "maximum queue wait")
	prefixCache := flag.Bool("prefix-cache", true, "enable prompt prefix caching")
	prefixEntries := flag.Int("prefix-entries", 64, "maximum cached prefixes")
	prefixBlocks := flag.Int("prefix-blocks", 0, "maximum blocks held by the prefix cache (0 = pool)")
	graph := flag.Bool("graph", false, "enable HIP Graph decode replay with fallback")
	kvType := flag.String("kv-type", "f32", "KV cache format for qwen35: f32, f16, or q8_0")
	npuOpt := flag.Bool("npu", false, "diagnostic NPU probe; fall back to the GPU backend")
	modelKey := flag.String("model-key", "", "override the prefix-cache model identity")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}
	if *backendName != "cpu" && *backendName != "hip" {
		log.Fatalf("unknown backend %q (want cpu or hip)", *backendName)
	}
	if *npuOpt {
		caps := npu.Probe()
		device, reason := npu.Request(caps)
		log.Printf("npu: %s; execution device %s", reason, device)
	}
	if *contextSize <= 0 {
		log.Fatal("--ctx must be positive")
	}
	if *blockSize <= 0 {
		log.Fatal("--block-size must be positive")
	}
	if *concurrency <= 0 {
		log.Fatal("--max-concurrency must be positive")
	}
	if *queue < 0 {
		log.Fatal("--max-queue must not be negative")
	}
	kvFormat, err := qwen35.ParseKVFormat(*kvType)
	if err != nil {
		log.Fatal(err)
	}
	if *kvBlocks <= 0 {
		blocks := (*contextSize + *blockSize - 1) / *blockSize
		blocks++ // headroom for prefix copy-on-write and cache entries
		*kvBlocks = blocks * *concurrency
		if *kvBlocks < blocks {
			log.Fatal("--kv-blocks is too small for one context")
		}
	}
	key := *modelKey
	if key == "" {
		hashed, err := hashFile(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
		key = hashed
	}
	key += "|kv=" + kvFormat.String()

	reader, err := gguf.Open(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer reader.Close()
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		log.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		log.Fatal(err)
	}
	if *contextSize > model.Config.ContextLength {
		log.Fatalf("--ctx %d exceeds model context %d", *contextSize, model.Config.ContextLength)
	}
	backend, err := newBackend(reader, model, backendConfig{
		ModelPath:    *modelPath,
		Backend:      *backendName,
		Blocks:       *kvBlocks,
		BlockSize:    *blockSize,
		PrefillChunk: *prefillChunk,
		Graph:        *graph,
		ModelKey:     key,
		KVFormat:     kvFormat,
	})
	if err != nil {
		log.Fatal(err)
	}
	manager, err := serving.NewManager(backend, tok, serving.Options{
		ContextSize:          *contextSize,
		PrefixEnabled:        *prefixCache,
		PrefixEntries:        *prefixEntries,
		PrefixCapacityBlocks: *prefixBlocks,
		Scheduler: scheduler.Config{
			MaxActive:       *concurrency,
			MaxQueue:        *queue,
			MaxBatch:        *concurrency,
			PrefillChunk:    *prefillChunk,
			PrefillPerCycle: 1,
			DecodePerCycle:  *concurrency,
			RequestTimeout:  *requestTimeout,
			QueueTimeout:    *queueTimeout,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := manager.Start(context.Background()); err != nil {
		log.Fatal(err)
	}
	defer manager.Close()

	srv := &server{
		manager:     manager,
		tokenizer:   tok,
		model:       model,
		contextSize: *contextSize,
		modelName:   *modelName,
		backend:     *backendName,
		graph:       *graph,
		started:     time.Now(),
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", srv.handleHealth)
	mux.HandleFunc("/metrics", srv.handleMetrics)
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/v1/chat/completions", srv.handleChat)
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)

	httpServer := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("yuuka-server listening on %s model=%s backend=%s context=%d blocks=%d block_size=%d concurrency=%d prefix_cache=%v graph=%v",
		*addr, *modelPath, *backendName, *contextSize, *kvBlocks, *blockSize,
		*concurrency, *prefixCache, *graph)
	serveErr := make(chan error, 1)
	go func() { serveErr <- httpServer.ListenAndServe() }()
	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = httpServer.Shutdown(shutdownCtx)
		cancel()
		drainCtx, drainCancel := context.WithTimeout(context.Background(), 10*time.Second)
		_ = manager.Shutdown(drainCtx)
		drainCancel()
		_ = httpServer.Close()
		if err := <-serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("server stopped: %v", err)
		}
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required", "")
		return
	}
	metrics := s.manager.Metrics()
	limits := s.manager.Limits()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "ok",
		"model":        s.modelName,
		"architecture": s.model.Config.Architecture,
		"context":      s.contextSize,
		"backend":      s.backend,
		"graph":        s.graph,
		"prefix_cache": metrics.Prefix.MaxEntries > 0,
		"uptime_s":     time.Since(s.started).Seconds(),
		"queue_depth":  metrics.Scheduler.QueueDepth,
		"active":       metrics.Scheduler.Active,
		"kv_blocks":    metrics.KV.Total,
		"kv_used":      metrics.KV.Used,
		"limits":       limits,
	})
}

func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required", "")
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	metrics := s.manager.Metrics()
	limits := s.manager.Limits()
	sched := metrics.Scheduler
	kv := metrics.KV
	prefix := metrics.Prefix
	fmt.Fprintf(w, "# HELP yuuka_requests_total Accepted submissions.\n")
	fmt.Fprintf(w, "# TYPE yuuka_requests_total counter\nyuuka_requests_total %d\n", sched.Submissions)
	fmt.Fprintf(w, "# TYPE yuuka_requests_admitted_total counter\nyuuka_requests_admitted_total %d\n", sched.Admitted)
	fmt.Fprintf(w, "# TYPE yuuka_requests_completed_total counter\nyuuka_requests_completed_total %d\n", sched.Completed)
	fmt.Fprintf(w, "# TYPE yuuka_requests_failed_total counter\nyuuka_requests_failed_total %d\n", sched.Failures)
	fmt.Fprintf(w, "# TYPE yuuka_requests_cancelled_total counter\nyuuka_requests_cancelled_total %d\n", sched.Cancelled)
	fmt.Fprintf(w, "# TYPE yuuka_requests_timeout_total counter\nyuuka_requests_timeout_total %d\n", sched.Timeouts)
	fmt.Fprintf(w, "# TYPE yuuka_queue_full_total counter\nyuuka_queue_full_total %d\n", sched.QueueFull)
	fmt.Fprintf(w, "# TYPE yuuka_capacity_rejected_total counter\nyuuka_capacity_rejected_total %d\n", sched.CapacityRejected)
	fmt.Fprintf(w, "# TYPE yuuka_prompt_tokens_total counter\nyuuka_prompt_tokens_total %d\n", sched.PromptTokens)
	fmt.Fprintf(w, "# TYPE yuuka_tokens_generated_total counter\nyuuka_tokens_generated_total %d\n", sched.GeneratedTokens)
	fmt.Fprintf(w, "# TYPE yuuka_queue_wait_seconds_total counter\nyuuka_queue_wait_seconds_total %f\n", sched.QueueWaitTotal.Seconds())
	fmt.Fprintf(w, "# TYPE yuuka_prefill_seconds_total counter\nyuuka_prefill_seconds_total %f\n", sched.PrefillTotal.Seconds())
	fmt.Fprintf(w, "# TYPE yuuka_decode_seconds_total counter\nyuuka_decode_seconds_total %f\n", sched.DecodeTotal.Seconds())
	fmt.Fprintf(w, "# TYPE yuuka_sampling_seconds_total counter\nyuuka_sampling_seconds_total %f\n", sched.SamplingTotal.Seconds())
	fmt.Fprintf(w, "# TYPE yuuka_request_queue_depth gauge\nyuuka_request_queue_depth %d\n", sched.QueueDepth)
	fmt.Fprintf(w, "# TYPE yuuka_requests_active gauge\nyuuka_requests_active %d\n", sched.Active)
	fmt.Fprintf(w, "# TYPE yuuka_batch_size_last gauge\nyuuka_batch_size_last %d\n", sched.LastBatch)
	fmt.Fprintf(w, "# TYPE yuuka_batch_size_max gauge\nyuuka_batch_size_max %d\n", sched.MaxBatchObserved)
	fmt.Fprintf(w, "# TYPE yuuka_active_max gauge\nyuuka_active_max %d\n", sched.MaxActiveObserved)
	latency := func(name, help string, summary scheduler.LatencySummary) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n", name, help, name)
		for _, quantile := range []struct {
			label string
			value time.Duration
		}{
			{"0.5", summary.P50},
			{"0.95", summary.P95},
			{"0.99", summary.P99},
		} {
			fmt.Fprintf(w, "%s{quantile=%q} %f\n", name, quantile.label, quantile.value.Seconds())
		}
		fmt.Fprintf(w, "%s_max %f\n", name, summary.Max.Seconds())
	}
	latency("yuuka_ttft_seconds", "Time to first token.", sched.TTFT)
	latency("yuuka_tpot_seconds", "Time per output token.", sched.TPOT)
	fmt.Fprintf(w, "# TYPE yuuka_kv_blocks_total gauge\nyuuka_kv_blocks_total %d\n", kv.Total)
	fmt.Fprintf(w, "# TYPE yuuka_kv_blocks_used gauge\nyuuka_kv_blocks_used %d\n", kv.Used)
	fmt.Fprintf(w, "# TYPE yuuka_kv_blocks_free gauge\nyuuka_kv_blocks_free %d\n", kv.Free)
	fmt.Fprintf(w, "# TYPE yuuka_kv_blocks_shared gauge\nyuuka_kv_blocks_shared %d\n", kv.Shared)
	fmt.Fprintf(w, "# TYPE yuuka_kv_block_size gauge\nyuuka_kv_block_size %d\n", kv.BlockSize)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_entries gauge\nyuuka_prefix_entries %d\n", prefix.Entries)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_blocks gauge\nyuuka_prefix_blocks %d\n", prefix.Blocks)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_hits_total counter\nyuuka_prefix_hits_total %d\n", prefix.Hits)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_misses_total counter\nyuuka_prefix_misses_total %d\n", prefix.Misses)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_inserts_total counter\nyuuka_prefix_inserts_total %d\n", prefix.Inserts)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_evictions_total counter\nyuuka_prefix_evictions_total %d\n", prefix.Evictions)
	fmt.Fprintf(w, "# TYPE yuuka_model_loaded gauge\nyuuka_model_loaded 1\n")
	fmt.Fprintf(w, "# TYPE yuuka_context_size gauge\nyuuka_context_size %d\n", s.contextSize)
	fmt.Fprintf(w, "# TYPE yuuka_max_concurrency gauge\nyuuka_max_concurrency %d\n", limits.MaxActive)
	fmt.Fprintf(w, "# TYPE yuuka_max_queue gauge\nyuuka_max_queue %d\n", limits.MaxQueue)
	fmt.Fprintf(w, "# TYPE yuuka_max_batch gauge\nyuuka_max_batch %d\n", limits.MaxBatch)
	fmt.Fprintf(w, "# TYPE yuuka_prefill_chunk gauge\nyuuka_prefill_chunk %d\n", limits.PrefillChunk)
	fmt.Fprintf(w, "# TYPE yuuka_prefill_per_cycle gauge\nyuuka_prefill_per_cycle %d\n", limits.PrefillPerCycle)
	fmt.Fprintf(w, "# TYPE yuuka_decode_per_cycle gauge\nyuuka_decode_per_cycle %d\n", limits.DecodePerCycle)
	fmt.Fprintf(w, "# TYPE yuuka_request_timeout_seconds gauge\nyuuka_request_timeout_seconds %f\n", limits.RequestTimeoutSeconds)
	fmt.Fprintf(w, "# TYPE yuuka_queue_timeout_seconds gauge\nyuuka_queue_timeout_seconds %f\n", limits.QueueTimeoutSeconds)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_max_entries gauge\nyuuka_prefix_max_entries %d\n", limits.PrefixEntries)
	fmt.Fprintf(w, "# TYPE yuuka_prefix_max_blocks gauge\nyuuka_prefix_max_blocks %d\n", limits.PrefixBlocks)
	fmt.Fprintf(w, "# TYPE yuuka_uptime_seconds gauge\nyuuka_uptime_seconds %f\n", time.Since(s.started).Seconds())
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required", "")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []any{map[string]any{
			"id":       s.modelName,
			"object":   "model",
			"created":  s.started.Unix(),
			"owned_by": "yuuka",
			"meta": map[string]any{
				"architecture": s.model.Config.Architecture,
				"context":      s.contextSize,
				"vocab_size":   s.model.Config.VocabSize,
				"backend":      s.backend,
				"graph":        s.graph,
				"block_size":   s.manager.BlockSize(),
			},
		}},
	})
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type chatRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`
	Seed                *int64          `json:"seed"`
	Stream              bool            `json:"stream"`
	Stop                stopSequences   `json:"stop"`
	StopTokenIDs        []int32         `json:"stop_token_ids"`
	EnableThinking      *bool           `json:"enable_thinking"`
	ChatTemplateKwargs  map[string]any  `json:"chat_template_kwargs"`
	N                   *int            `json:"n"`
	PresencePenalty     *float64        `json:"presence_penalty"`
	FrequencyPenalty    *float64        `json:"frequency_penalty"`
	Logprobs            *bool           `json:"logprobs"`
	TopLogprobs         *int            `json:"top_logprobs"`
	ResponseFormat      json.RawMessage `json:"response_format"`
	Tools               json.RawMessage `json:"tools"`
	StreamOptions       json.RawMessage `json:"stream_options"`
	User                string          `json:"user"`
}

type stopSequences []string

func (s *stopSequences) UnmarshalJSON(data []byte) error {
	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*s = []string{single}
		return nil
	}
	var many []string
	if err := json.Unmarshal(data, &many); err != nil {
		return fmt.Errorf("stop must be a string or array of strings")
	}
	*s = many
	return nil
}

func (m chatMessage) text() (string, error) {
	var single string
	if err := json.Unmarshal(m.Content, &single); err == nil {
		return single, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err != nil {
		return "", fmt.Errorf("message content must be a string or text parts")
	}
	var out strings.Builder
	for _, part := range parts {
		switch part.Type {
		case "text":
			out.WriteString(part.Text)
		case "image_url", "input_image", "image":
			return "", fmt.Errorf("image content is not supported by this text-only build")
		default:
			return "", fmt.Errorf("unsupported content part %q", part.Type)
		}
	}
	return out.String(), nil
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required", "")
		return
	}
	id := fmt.Sprintf("chatcmpl-%d", s.requestSeq.Add(1))
	w.Header().Set("X-Request-ID", id)

	var req chatRequest
	decoder := json.NewDecoder(io.LimitReader(r.Body, maxRequestBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error(), id)
		return
	}
	if req.Model != "" && req.Model != s.modelName {
		writeError(w, http.StatusNotFound, "model_not_found", "requested model is not loaded", id)
		return
	}
	if err := validateOptionalParameters(&req); err != nil {
		writeError(w, http.StatusBadRequest, "unsupported_parameter", err.Error(), id)
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "messages must not be empty", id)
		return
	}
	messages := make([]tokenizer.Message, len(req.Messages))
	for i, message := range req.Messages {
		text, err := message.text()
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error(), id)
			return
		}
		messages[i] = tokenizer.Message{Role: message.Role, Content: text}
	}
	enableThinking := false
	if req.EnableThinking != nil {
		enableThinking = *req.EnableThinking
	} else if value, ok := req.ChatTemplateKwargs["enable_thinking"].(bool); ok {
		enableThinking = value
	}
	prompt, err := s.tokenizer.RenderChat(messages, tokenizer.ChatOptions{
		AddGenerationPrompt: true,
		EnableThinking:      enableThinking,
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error(), id)
		return
	}
	ids, err := s.tokenizer.Encode(prompt)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "tokenizer_error", err.Error(), id)
		return
	}
	if len(ids) >= s.contextSize {
		writeError(w, http.StatusBadRequest, "context_length_exceeded",
			fmt.Sprintf("prompt uses %d of %d context positions", len(ids), s.contextSize), id)
		return
	}
	maxTokens := 256
	if req.MaxCompletionTokens != nil {
		maxTokens = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	if maxTokens <= 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "max_tokens must be positive", id)
		return
	}
	if maxTokens > s.contextSize-len(ids) {
		maxTokens = s.contextSize - len(ids)
	}
	temperature := 0.0
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	topP := 1.0
	if req.TopP != nil {
		topP = *req.TopP
	}
	topK := 40
	if req.TopK != nil {
		topK = *req.TopK
	}
	seed := int64(0)
	if req.Seed != nil {
		seed = *req.Seed
	}
	sampler, err := engine.NewSampler(engine.SamplerConfig{
		Temperature: float32(temperature),
		TopK:        topK,
		TopP:        float32(topP),
		Seed:        seed,
	}, s.model.Config.VocabSize)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error(), id)
		return
	}

	stream := req.Stream
	var flusher http.Flusher
	if stream {
		flusher, _ = w.(http.Flusher)
	}
	var streamMu sync.Mutex
	streamStarted := false
	startStream := func() {
		if !stream {
			return
		}
		streamMu.Lock()
		defer streamMu.Unlock()
		if streamStarted {
			return
		}
		streamStarted = true
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		writeChunk(w, id, time.Now().Unix(), s.modelName, map[string]string{"role": "assistant"}, "")
		if flusher != nil {
			flusher.Flush()
		}
	}
	streamIsStarted := func() bool {
		streamMu.Lock()
		defer streamMu.Unlock()
		return streamStarted
	}
	sink := func(token int32, delta string) error {
		if !stream || delta == "" {
			return nil
		}
		startStream()
		writeChunk(w, id, time.Now().Unix(), s.modelName, map[string]string{"content": delta}, "")
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}

	result, genErr := s.manager.Generate(r.Context(), serving.Request{
		ID:          id,
		Prompt:      ids,
		MaxTokens:   maxTokens,
		Sampler:     sampler,
		StopTokens:  req.StopTokenIDs,
		StopStrings: req.Stop,
		Sink:        sink,
		Submitted:   startStream,
	})
	if genErr != nil {
		if stream && streamIsStarted() {
			code, message, status := classifyError(genErr)
			if r.Context().Err() == nil {
				writeSSEError(w, id, code, message, status)
				fmt.Fprint(w, "data: [DONE]\n\n")
				if flusher != nil {
					flusher.Flush()
				}
			}
			logRequest(id, 0, 0, genErr, result)
			return
		}
		writeAPIError(w, genErr, id)
		logRequest(id, 0, 0, genErr, result)
		return
	}
	finishReason := result.FinishReason
	if finishReason == "" {
		finishReason = "stop"
	}
	logRequest(id, result.PromptTokens, result.OutputTokens, nil, result)
	if stream {
		startStream()
		writeChunk(w, id, time.Now().Unix(), s.modelName, nil, finishReason)
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   s.modelName,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": result.Text},
			"finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens":     result.PromptTokens,
			"completion_tokens": result.OutputTokens,
			"total_tokens":      result.PromptTokens + result.OutputTokens,
		},
		"yuuka": map[string]any{
			"queue_ms":      result.QueueWait.Seconds() * 1000,
			"prefill_ms":    result.PrefillTime.Seconds() * 1000,
			"ttft_ms":       result.TimeToFirst.Seconds() * 1000,
			"decode_ms":     result.DecodeTime.Seconds() * 1000,
			"sampling_ms":   result.SamplingTime.Seconds() * 1000,
			"prefix_hit":    result.PrefixCacheHit,
			"prefix_tokens": result.PrefixTokens,
			"backend":       s.backend,
		},
	})
}

func logRequest(id string, promptTokens, outputTokens int, err error, result *serving.Result) {
	if err != nil {
		log.Printf("request=%s error=%v", id, err)
		return
	}
	finish := "stop"
	if result != nil {
		finish = result.FinishReason
	}
	log.Printf("request=%s finish=%s prompt_tokens=%d output_tokens=%d queue=%s prefill=%s decode=%s sampling=%s prefix_hit=%v",
		id, finish, promptTokens, outputTokens,
		result.QueueWait, result.PrefillTime, result.DecodeTime, result.SamplingTime,
		result.PrefixCacheHit)
}

func validateOptionalParameters(req *chatRequest) error {
	if req.N != nil && *req.N != 1 {
		return fmt.Errorf("n=%d is not supported; only n=1 is available", *req.N)
	}
	if req.PresencePenalty != nil && *req.PresencePenalty != 0 {
		return fmt.Errorf("presence_penalty is not supported")
	}
	if req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0 {
		return fmt.Errorf("frequency_penalty is not supported")
	}
	if req.Logprobs != nil && *req.Logprobs {
		return fmt.Errorf("logprobs is not supported")
	}
	if req.TopLogprobs != nil {
		return fmt.Errorf("top_logprobs is not supported")
	}
	if len(req.ResponseFormat) > 0 {
		return fmt.Errorf("response_format is not supported")
	}
	if len(req.Tools) > 0 {
		return fmt.Errorf("tools are not supported")
	}
	if len(req.StreamOptions) > 0 {
		return fmt.Errorf("stream_options is not supported")
	}
	for key := range req.ChatTemplateKwargs {
		if key != "enable_thinking" {
			return fmt.Errorf("chat_template_kwargs key %q is not supported", key)
		}
	}
	return nil
}

func classifyError(err error) (code, message string, status int) {
	switch {
	case errors.Is(err, scheduler.ErrQueueFull):
		return "server_overloaded", "the request queue is full", http.StatusTooManyRequests
	case errors.Is(err, scheduler.ErrCapacity), errors.Is(err, scheduler.ErrClosed):
		return "insufficient_capacity", err.Error(), http.StatusServiceUnavailable
	case errors.Is(err, serving.ErrContextLength):
		return "context_length_exceeded", err.Error(), http.StatusBadRequest
	case errors.Is(err, context.DeadlineExceeded):
		return "request_timeout", "the request exceeded its deadline", http.StatusGatewayTimeout
	case errors.Is(err, context.Canceled):
		return "client_closed", "the client cancelled the request", 499
	default:
		return "generation_error", err.Error(), http.StatusInternalServerError
	}
}

func writeAPIError(w http.ResponseWriter, err error, id string) {
	code, message, status := classifyError(err)
	if status == http.StatusTooManyRequests {
		w.Header().Set("Retry-After", "1")
	}
	writeError(w, status, code, message, id)
}

func writeChunk(w io.Writer, id string, created int64, model string, delta map[string]string, finish string) {
	if delta == nil {
		delta = map[string]string{}
	}
	choice := map[string]any{"index": 0, "delta": delta}
	if finish != "" {
		choice["finish_reason"] = finish
	} else {
		choice["finish_reason"] = nil
	}
	payload := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []any{choice},
	}
	data, _ := json.Marshal(payload)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

func writeSSEError(w io.Writer, id, code, message string, status int) {
	payload := map[string]any{"error": map[string]any{
		"message":    message,
		"type":       code,
		"code":       code,
		"request_id": id,
		"status":     status,
	}}
	data, _ := json.Marshal(payload)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

func writeError(w http.ResponseWriter, status int, code, message, id string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{
		"message":    message,
		"type":       code,
		"code":       code,
		"request_id": id,
		"retryable":  status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable,
	}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

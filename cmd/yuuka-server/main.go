// Command yuuka-server exposes a single loaded GGUF model through an
// OpenAI-compatible chat completions endpoint.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

type metrics struct {
	requests        atomic.Int64
	failures        atomic.Int64
	promptTokens    atomic.Int64
	generatedTokens atomic.Int64
	prefillNanos    atomic.Int64
	decodeNanos     atomic.Int64
}

type server struct {
	reader      *gguf.Reader
	tokenizer   *tokenizer.Tokenizer
	model       *qwen35.Model
	engine      *engine.Engine
	contextSize int
	modelName   string
	backend     string
	started     time.Time
	mu          sync.Mutex
	metrics     metrics
	requestSeq  atomic.Int64
}

func main() {
	modelPath := flag.String("model", "", "GGUF model path")
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	contextSize := flag.Int("ctx", 4096, "context capacity for the loaded model")
	modelName := flag.String("model-name", "yuuka", "model identifier returned by the API")
	backend := flag.String("backend", "cpu", "execution backend: cpu or hip")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}
	if *backend != "cpu" && *backend != "hip" {
		log.Fatalf("unknown backend %q (want cpu or hip)", *backend)
	}
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
	var runner *engine.Engine
	if *backend == "cpu" {
		runner, err = engine.New(model, tok, *contextSize, nil)
	} else {
		runtime, runtimeErr := buildRuntime(reader, model, *contextSize, *backend)
		if runtimeErr != nil {
			log.Fatal(runtimeErr)
		}
		runner, err = engine.NewWithRuntime(runtime, tok, nil)
	}
	if err != nil {
		log.Fatal(err)
	}
	defer runner.Close()
	srv := &server{
		reader:      reader,
		tokenizer:   tok,
		model:       model,
		engine:      runner,
		contextSize: *contextSize,
		modelName:   *modelName,
		backend:     *backend,
		started:     time.Now(),
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", srv.handleHealth)
	mux.HandleFunc("/metrics", srv.handleMetrics)
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/v1/chat/completions", srv.handleChat)
	httpServer := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
	}()
	log.Printf("yuuka-server listening on %s model=%s context=%d", *addr, *modelPath, *contextSize)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "ok",
		"model":        s.modelName,
		"architecture": s.model.Config.Architecture,
		"context":      s.contextSize,
		"backend":      s.backend,
		"uptime_s":     time.Since(s.started).Seconds(),
	})
}

func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# HELP yuuka_requests_total Chat completion requests.\n")
	fmt.Fprintf(w, "# TYPE yuuka_requests_total counter\nyuuka_requests_total %d\n", s.metrics.requests.Load())
	fmt.Fprintf(w, "# TYPE yuuka_requests_failed_total counter\nyuuka_requests_failed_total %d\n", s.metrics.failures.Load())
	fmt.Fprintf(w, "# TYPE yuuka_prompt_tokens_total counter\nyuuka_prompt_tokens_total %d\n", s.metrics.promptTokens.Load())
	fmt.Fprintf(w, "# TYPE yuuka_tokens_generated_total counter\nyuuka_tokens_generated_total %d\n", s.metrics.generatedTokens.Load())
	fmt.Fprintf(w, "# TYPE yuuka_prefill_seconds_total counter\nyuuka_prefill_seconds_total %f\n", float64(s.metrics.prefillNanos.Load())/1e9)
	fmt.Fprintf(w, "# TYPE yuuka_decode_seconds_total counter\nyuuka_decode_seconds_total %f\n", float64(s.metrics.decodeNanos.Load())/1e9)
	fmt.Fprintf(w, "# TYPE yuuka_model_loaded gauge\nyuuka_model_loaded 1\n")
	fmt.Fprintf(w, "# TYPE yuuka_context_size gauge\nyuuka_context_size %d\n", s.contextSize)
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
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
			},
		}},
	})
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type chatRequest struct {
	Model               string         `json:"model"`
	Messages            []chatMessage  `json:"messages"`
	MaxTokens           *int           `json:"max_tokens"`
	MaxCompletionTokens *int           `json:"max_completion_tokens"`
	Temperature         *float64       `json:"temperature"`
	TopP                *float64       `json:"top_p"`
	TopK                *int           `json:"top_k"`
	Seed                *int64         `json:"seed"`
	Stream              bool           `json:"stream"`
	Stop                stopSequences  `json:"stop"`
	EnableThinking      *bool          `json:"enable_thinking"`
	ChatTemplateKwargs  map[string]any `json:"chat_template_kwargs"`
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
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	s.metrics.requests.Add(1)
	var req chatRequest
	decoder := json.NewDecoder(io.LimitReader(r.Body, 8<<20))
	if err := decoder.Decode(&req); err != nil {
		s.metrics.failures.Add(1)
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if req.Model != "" && req.Model != s.modelName {
		s.metrics.failures.Add(1)
		writeError(w, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if len(req.Messages) == 0 {
		s.metrics.failures.Add(1)
		writeError(w, http.StatusBadRequest, "invalid_request", "messages must not be empty")
		return
	}
	messages := make([]tokenizer.Message, len(req.Messages))
	for i, message := range req.Messages {
		text, err := message.text()
		if err != nil {
			s.metrics.failures.Add(1)
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
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
		s.metrics.failures.Add(1)
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	ids, err := s.tokenizer.Encode(prompt)
	if err != nil {
		s.metrics.failures.Add(1)
		writeError(w, http.StatusInternalServerError, "tokenizer_error", err.Error())
		return
	}
	maxTokens := 256
	if req.MaxCompletionTokens != nil {
		maxTokens = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	if maxTokens <= 0 {
		s.metrics.failures.Add(1)
		writeError(w, http.StatusBadRequest, "invalid_request", "max_tokens must be positive")
		return
	}
	if len(ids) > s.contextSize {
		s.metrics.failures.Add(1)
		writeError(w, http.StatusBadRequest, "context_length_exceeded", "prompt exceeds the configured context")
		return
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
		s.metrics.failures.Add(1)
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// The current engine is single-request. Reject concurrent work rather than
	// interleaving state.
	if !s.mu.TryLock() {
		s.metrics.failures.Add(1)
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, "server_busy", "another request is in progress")
		return
	}
	defer s.mu.Unlock()
	s.engine.Sampler = sampler

	stop := newStopFilter(req.Stop)
	id := fmt.Sprintf("chatcmpl-%d", s.requestSeq.Add(1))
	created := time.Now().Unix()
	stream := req.Stream
	var flusher http.Flusher
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, _ = w.(http.Flusher)
		writeChunk(w, id, created, s.modelName, nil, "")
		if flusher != nil {
			flusher.Flush()
		}
	}

	var collected strings.Builder
	options := engine.GenerateOptions{
		MaxTokens: maxTokens,
		OnToken: func(_ int32, delta string) error {
			safe, stopped := stop.Feed(delta)
			if safe != "" {
				collected.WriteString(safe)
				if stream {
					writeChunk(w, id, created, s.modelName, map[string]string{"content": safe}, "")
					if flusher != nil {
						flusher.Flush()
					}
				}
			}
			if stopped {
				return engine.ErrStop
			}
			return nil
		},
	}
	result, err := s.engine.Generate(r.Context(), ids, options)
	if err != nil {
		s.metrics.failures.Add(1)
		if stream {
			writeSSEError(w, err)
			if flusher != nil {
				flusher.Flush()
			}
			return
		}
		writeError(w, http.StatusInternalServerError, "generation_error", err.Error())
		return
	}
	if !stop.Stopped() {
		tail := stop.Flush()
		if tail != "" {
			collected.WriteString(tail)
			if stream {
				writeChunk(w, id, created, s.modelName, map[string]string{"content": tail}, "")
				if flusher != nil {
					flusher.Flush()
				}
			}
		}
	}
	text := collected.String()
	finishReason := result.FinishReason
	s.metrics.promptTokens.Add(int64(result.PromptTokens))
	s.metrics.generatedTokens.Add(int64(result.OutputTokens))
	s.metrics.prefillNanos.Add(result.PrefillTime.Nanoseconds())
	s.metrics.decodeNanos.Add(result.DecodeTime.Nanoseconds())

	if stream {
		writeChunk(w, id, created, s.modelName, nil, finishReason)
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   s.modelName,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": text},
			"finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens":     result.PromptTokens,
			"completion_tokens": result.OutputTokens,
			"total_tokens":      result.PromptTokens + result.OutputTokens,
		},
		"yuuka": map[string]any{
			"prefill_ms": result.PrefillTime.Seconds() * 1000,
			"ttft_ms":    result.TimeToFirst.Seconds() * 1000,
			"decode_ms":  result.DecodeTime.Seconds() * 1000,
			"backend":    s.backend,
		},
	})
}

type stopFilter struct {
	stops   []string
	maxLen  int
	buf     strings.Builder
	stopped bool
}

func newStopFilter(stops []string) *stopFilter {
	filter := &stopFilter{stops: stops}
	for _, stop := range stops {
		if len(stop) > filter.maxLen {
			filter.maxLen = len(stop)
		}
	}
	return filter
}

func (f *stopFilter) Feed(delta string) (string, bool) {
	if f.stopped {
		return "", true
	}
	if len(f.stops) == 0 {
		return delta, false
	}
	f.buf.WriteString(delta)
	text := f.buf.String()
	for _, stop := range f.stops {
		if index := strings.Index(text, stop); index >= 0 {
			prefix := text[:index]
			f.buf.Reset()
			f.stopped = true
			return prefix, true
		}
	}
	keep := f.maxLen - 1
	if keep < 0 {
		keep = 0
	}
	if len(text) <= keep {
		f.buf.Reset()
		f.buf.WriteString(text)
		return "", false
	}
	out := text[:len(text)-keep]
	f.buf.Reset()
	f.buf.WriteString(text[len(text)-keep:])
	return out, false
}

func (f *stopFilter) Flush() string {
	if f.stopped {
		return ""
	}
	text := f.buf.String()
	f.buf.Reset()
	return text
}

func (f *stopFilter) Stopped() bool { return f.stopped }

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

func writeSSEError(w io.Writer, err error) {
	payload := map[string]any{"error": map[string]any{
		"message": err.Error(),
		"type":    "generation_error",
		"code":    "generation_error",
	}}
	data, _ := json.Marshal(payload)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{
		"message": message,
		"type":    code,
		"code":    code,
	}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

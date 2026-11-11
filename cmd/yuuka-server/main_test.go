package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/scheduler"
	"github.com/yuuka-ai/yuuka/serving"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestChatMessageTextParts(t *testing.T) {
	message := chatMessage{Content: json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`)}
	text, err := message.text()
	if err != nil {
		t.Fatal(err)
	}
	if text != "ab" {
		t.Fatalf("text %q", text)
	}
	bad := chatMessage{Content: json.RawMessage(`[{"type":"image_url"}]`)}
	if _, err := bad.text(); err == nil {
		t.Fatal("image content was accepted")
	}
}

func loadTiny(t *testing.T) (*gguf.Reader, *tokenizer.Tokenizer, *qwen35.Model) {
	t.Helper()
	// API contract tests use the deterministic tiny fixture so they stay fast
	// and independent of the optional real-model environment.
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	return reader, tok, model
}

func newTestServerWith(t *testing.T, backend serving.Backend, tok *tokenizer.Tokenizer, model *qwen35.Model) *server {
	t.Helper()
	manager, err := serving.NewManager(backend, tok, serving.Options{
		ContextSize: 128,
		Scheduler: scheduler.Config{
			MaxActive:       2,
			MaxQueue:        4,
			MaxBatch:        2,
			PrefillChunk:    16,
			PrefillPerCycle: 1,
			DecodePerCycle:  2,
			RequestTimeout:  time.Minute,
			QueueTimeout:    5 * time.Second,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = manager.Close() })
	return &server{
		manager:     manager,
		tokenizer:   tok,
		model:       model,
		contextSize: 128,
		modelName:   "yuuka",
		backend:     "cpu",
		started:     time.Now(),
	}
}

func newTestServer(t *testing.T) *server {
	t.Helper()
	_, tok, model := loadTiny(t)
	backend, err := serving.NewQwen35Backend(model, "tiny-server-test", 64, 8)
	if err != nil {
		t.Fatal(err)
	}
	return newTestServerWith(t, backend, tok, model)
}

func testMux(s *server) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/debug/pprof/", http.DefaultServeMux.ServeHTTP)
	return mux
}

func TestChatCompletions(t *testing.T) {
	s := newTestServer(t)
	httpServer := httptest.NewServer(testMux(s))
	defer httpServer.Close()

	body := `{"model":"yuuka","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":4,"temperature":0,"enable_thinking":false}`
	response, err := http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(response.Body)
		t.Fatalf("status %d: %s", response.StatusCode, data)
	}
	var payload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Yuuka struct {
			PrefixHit *bool `json:"prefix_hit"`
		} `json:"yuuka"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Choices) != 1 || payload.Choices[0].Message.Content == "" {
		t.Fatalf("unexpected payload: %+v", payload)
	}
	if payload.Choices[0].FinishReason != "length" {
		t.Fatalf("finish reason %q", payload.Choices[0].FinishReason)
	}
	if payload.Usage.PromptTokens == 0 || payload.Usage.CompletionTokens != 4 {
		t.Fatalf("usage %+v", payload.Usage)
	}
	if payload.Yuuka.PrefixHit == nil {
		t.Fatal("timing metadata is missing")
	}
}

func TestChatCompletionsStream(t *testing.T) {
	s := newTestServer(t)
	httpServer := httptest.NewServer(testMux(s))
	defer httpServer.Close()

	body := `{"model":"yuuka","messages":[{"role":"user","content":"1+1="}],"max_tokens":3,"temperature":0,"stream":true,"stop":["9"]}`
	response, err := http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status %d", response.StatusCode)
	}
	if response.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("content type %q", response.Header.Get("Content-Type"))
	}
	scanner := bufio.NewScanner(response.Body)
	var sawDone, sawFinish, sawRole bool
	for scanner.Scan() {
		line := scanner.Text()
		if line == "data: [DONE]" {
			sawDone = true
			break
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta        map[string]string `json:"delta"`
				FinishReason *string           `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatal(err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta["role"] == "assistant" {
				sawRole = true
			}
			if choice.FinishReason != nil {
				sawFinish = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if !sawRole || !sawFinish || !sawDone {
		t.Fatalf("stream role=%v finish=%v done=%v", sawRole, sawFinish, sawDone)
	}
}

func TestServerConcurrentRequests(t *testing.T) {
	s := newTestServer(t)
	httpServer := httptest.NewServer(testMux(s))
	defer httpServer.Close()

	var wg sync.WaitGroup
	var mu sync.Mutex
	var failures []string
	for _, text := range []string{"Hello world", "1+1=", "Once upon a time", "The capital of France is"} {
		wg.Add(1)
		go func(text string) {
			defer wg.Done()
			body := `{"messages":[{"role":"user","content":"` + text + `"}],"max_tokens":3,"temperature":0}`
			response, err := http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(body))
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				failures = append(failures, err.Error())
				return
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK {
				data, _ := io.ReadAll(response.Body)
				failures = append(failures, string(data))
				return
			}
			var payload struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
			}
			if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
				failures = append(failures, err.Error())
				return
			}
			if len(payload.Choices) != 1 || payload.Choices[0].Message.Content == "" {
				failures = append(failures, "empty concurrent completion")
			}
		}(text)
	}
	wg.Wait()
	for _, failure := range failures {
		t.Fatal(failure)
	}
	metrics := s.manager.Metrics()
	if metrics.Scheduler.Submissions < 4 || metrics.Scheduler.Completed < 4 {
		t.Fatalf("scheduler metrics: %+v", metrics.Scheduler)
	}
	if metrics.KV.Used != 0 {
		t.Fatalf("KV blocks leaked: %+v", metrics.KV)
	}
}

func TestServerRejectsUnsupportedAndBadRequests(t *testing.T) {
	s := newTestServer(t)
	httpServer := httptest.NewServer(testMux(s))
	defer httpServer.Close()

	cases := []struct {
		name   string
		body   string
		status int
		code   string
	}{
		{"empty messages", `{"messages":[]}`, http.StatusBadRequest, "invalid_request"},
		{"unknown field", `{"messages":[{"role":"user","content":"hi"}],"banana":1}`, http.StatusBadRequest, "invalid_request"},
		{"multiple choices", `{"messages":[{"role":"user","content":"hi"}],"n":2}`, http.StatusBadRequest, "unsupported_parameter"},
		{"tools", `{"messages":[{"role":"user","content":"hi"}],"tools":[{}]}`, http.StatusBadRequest, "unsupported_parameter"},
		{"bad model", `{"model":"other","messages":[{"role":"user","content":"hi"}]}`, http.StatusNotFound, "model_not_found"},
		{"bad tokens", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}`, http.StatusBadRequest, "invalid_request"},
	}
	for _, tc := range cases {
		response, err := http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(tc.body))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		var payload struct {
			Error struct {
				Code      string `json:"code"`
				RequestID string `json:"request_id"`
			} `json:"error"`
		}
		_ = json.NewDecoder(response.Body).Decode(&payload)
		response.Body.Close()
		if response.StatusCode != tc.status || payload.Error.Code != tc.code {
			t.Fatalf("%s: status %d code %q", tc.name, response.StatusCode, payload.Error.Code)
		}
		if payload.Error.RequestID == "" {
			t.Fatalf("%s: missing request id", tc.name)
		}
	}
}

func TestHealthMetricsAndPprof(t *testing.T) {
	s := newTestServer(t)
	httpServer := httptest.NewServer(testMux(s))
	defer httpServer.Close()

	health, err := http.Get(httpServer.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	var healthPayload map[string]any
	_ = json.NewDecoder(health.Body).Decode(&healthPayload)
	health.Body.Close()
	if health.StatusCode != http.StatusOK || healthPayload["status"] != "ok" {
		t.Fatalf("health %d %+v", health.StatusCode, healthPayload)
	}
	limits, ok := healthPayload["limits"].(map[string]any)
	if !ok {
		t.Fatalf("health limits missing: %+v", healthPayload)
	}
	if limits["max_active"] != float64(2) || limits["max_queue"] != float64(4) ||
		limits["prefill_chunk"] != float64(16) {
		t.Fatalf("health limits = %+v", limits)
	}

	metrics, err := http.Get(httpServer.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(metrics.Body)
	metrics.Body.Close()
	for _, name := range []string{
		"yuuka_requests_total",
		"yuuka_request_queue_depth",
		"yuuka_ttft_seconds{quantile=\"0.95\"}",
		"yuuka_kv_blocks_used",
		"yuuka_prefix_hits_total",
		"yuuka_sampling_seconds_total",
		"yuuka_max_concurrency",
		"yuuka_max_queue",
		"yuuka_prefix_max_entries",
	} {
		if !strings.Contains(string(data), name) {
			t.Fatalf("metrics missing %s:\n%s", name, data)
		}
	}

	profile, err := http.Get(httpServer.URL + "/debug/pprof/")
	if err != nil {
		t.Fatal(err)
	}
	profile.Body.Close()
	if profile.StatusCode != http.StatusOK {
		t.Fatalf("pprof status %d", profile.StatusCode)
	}
}

// slowBackend delays steps so an HTTP client can cancel mid-generation.
type slowBackend struct {
	serving.Backend
	delay time.Duration
}

func (b slowBackend) NewSession(table *memory.Table, capacity, position int, state []byte) (serving.Sequence, error) {
	seq, err := b.Backend.NewSession(table, capacity, position, state)
	if err != nil {
		return nil, err
	}
	return &slowSequence{Sequence: seq, delay: b.delay}, nil
}

type slowSequence struct {
	serving.Sequence
	delay time.Duration
}

func (s *slowSequence) Prefill(tokens []int32, final bool) ([]float32, error) {
	time.Sleep(s.delay)
	return s.Sequence.Prefill(tokens, final)
}

func (s *slowSequence) Decode(token int32) ([]float32, error) {
	time.Sleep(s.delay)
	return s.Sequence.Decode(token)
}

func TestStreamClientCancellationReleasesBlocks(t *testing.T) {
	_, tok, model := loadTiny(t)
	base, err := serving.NewQwen35Backend(model, "tiny-cancel-test", 64, 8)
	if err != nil {
		t.Fatal(err)
	}
	s := newTestServerWith(t, slowBackend{Backend: base, delay: 25 * time.Millisecond}, tok, model)
	httpServer := httptest.NewServer(testMux(s))
	defer httpServer.Close()

	ctx, cancel := context.WithCancel(context.Background())
	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":40,"temperature":0,"stream":true}`
	request, err := http.NewRequestWithContext(ctx, http.MethodPost,
		httpServer.URL+"/v1/chat/completions", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	reader := bufio.NewReader(response.Body)
	if _, err := reader.ReadString('\n'); err != nil {
		response.Body.Close()
		t.Fatalf("read first chunk: %v", err)
	}
	admitDeadline := time.Now().Add(2 * time.Second)
	for s.manager.Metrics().Scheduler.Admitted == 0 {
		if time.Now().After(admitDeadline) {
			response.Body.Close()
			t.Fatal("request was never admitted")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	response.Body.Close()

	deadline := time.Now().Add(3 * time.Second)
	for {
		metrics := s.manager.Metrics()
		if metrics.KV.Used == 0 && metrics.Scheduler.Active == 0 {
			if metrics.Scheduler.Cancelled == 0 {
				t.Fatalf("request was not recorded as cancelled: %+v", metrics.Scheduler)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("cancelled request leaked: %+v", metrics)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestClassifyErrorMapping(t *testing.T) {
	cases := []struct {
		err    error
		status int
		code   string
	}{
		{scheduler.ErrQueueFull, http.StatusTooManyRequests, "server_overloaded"},
		{scheduler.ErrCapacity, http.StatusServiceUnavailable, "insufficient_capacity"},
		{serving.ErrContextLength, http.StatusBadRequest, "context_length_exceeded"},
		{context.DeadlineExceeded, http.StatusGatewayTimeout, "request_timeout"},
		{context.Canceled, 499, "client_closed"},
		{io.ErrUnexpectedEOF, http.StatusInternalServerError, "generation_error"},
	}
	for _, tc := range cases {
		code, _, status := classifyError(tc.err)
		if status != tc.status || code != tc.code {
			t.Fatalf("%v -> %d %q, want %d %q", tc.err, status, code, tc.status, tc.code)
		}
	}
}

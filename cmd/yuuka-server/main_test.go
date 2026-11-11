package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func TestStopFilter(t *testing.T) {
	filter := newStopFilter([]string{"END"})
	first, stopped := filter.Feed("hello ")
	if stopped {
		t.Fatal("false stop")
	}
	out, stopped := filter.Feed("world END extra")
	if !stopped {
		t.Fatal("stop sequence was not detected")
	}
	if got := first + out; got != "hello world " {
		t.Fatalf("output %q, want %q", got, "hello world ")
	}
	if got := filter.Flush(); got != "" {
		t.Fatalf("flush after stop = %q", got)
	}
}

func TestStopFilterFlushWithoutStop(t *testing.T) {
	filter := newStopFilter([]string{"XYZ"})
	var got strings.Builder
	for _, part := range []string{"ab", "cd", "ef"} {
		out, _ := filter.Feed(part)
		got.WriteString(out)
	}
	got.WriteString(filter.Flush())
	if got.String() != "abcdef" {
		t.Fatalf("output %q", got.String())
	}
}

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

func newTestServer(t *testing.T) *server {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MODEL")
	contextSize := 64
	if path == "" {
		path = filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
		if err := fixture.WriteTinyQwen35(path); err != nil {
			t.Fatal(err)
		}
		contextSize = 128
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
	runner, err := engine.New(model, tok, contextSize, nil)
	if err != nil {
		t.Fatal(err)
	}
	return &server{
		reader:      reader,
		tokenizer:   tok,
		model:       model,
		engine:      runner,
		contextSize: contextSize,
		modelName:   "yuuka",
	}
}

func TestChatCompletions(t *testing.T) {
	s := newTestServer(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	httpServer := httptest.NewServer(mux)
	defer httpServer.Close()

	body := `{"model":"yuuka","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":4,"temperature":0,"enable_thinking":false}`
	response, err := http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status %d", response.StatusCode)
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
}

func TestChatCompletionsStream(t *testing.T) {
	s := newTestServer(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	httpServer := httptest.NewServer(mux)
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
	scanner := bufio.NewScanner(response.Body)
	var sawDone bool
	var sawFinish bool
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
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatal(err)
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				sawFinish = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if !sawFinish || !sawDone {
		t.Fatalf("stream missing finish=%v done=%v", sawFinish, sawDone)
	}
}

func TestServerRejectsBusyAndBadRequests(t *testing.T) {
	s := newTestServer(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	httpServer := httptest.NewServer(mux)
	defer httpServer.Close()

	bad := `{"messages":[]}`
	response, err := http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(bad))
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad request status %d", response.StatusCode)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	busy := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`
	response, err = http.Post(httpServer.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(busy))
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("busy status %d", response.StatusCode)
	}
}

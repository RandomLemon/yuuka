// Command serving_bench measures single-request latency and multi-request
// throughput for the continuous serving path.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yuuka-ai/yuuka/engine"
	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/scheduler"
	"github.com/yuuka-ai/yuuka/serving"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

type benchBackendConfig struct {
	Blocks       int
	BlockSize    int
	PrefillChunk int
	Graph        bool
	ModelKey     string
	Backend      string
}

type sample struct {
	latency  time.Duration
	queue    time.Duration
	ttft     time.Duration
	prefill  time.Duration
	decode   time.Duration
	sampling time.Duration
	tokens   int
}

func main() {
	modelPath := flag.String("model", "", "GGUF model path")
	backendName := flag.String("backend", "hip", "execution backend: cpu or hip")
	contextSize := flag.Int("ctx", 512, "context positions")
	blockSize := flag.Int("block-size", 16, "KV block size")
	kvBlocks := flag.Int("kv-blocks", 0, "KV blocks (0 derives from ctx and concurrency)")
	concurrency := flag.Int("concurrency", 1, "active requests")
	requests := flag.Int("requests", 1, "measured requests")
	promptTokens := flag.Int("prompt-tokens", 16, "prompt length")
	decodeTokens := flag.Int("decode-tokens", 16, "generated tokens")
	warmup := flag.Int("warmup", 1, "warmup requests")
	prefillChunk := flag.Int("prefill-chunk", 512, "prompt chunk size")
	prefixCache := flag.Bool("prefix-cache", false, "enable prefix cache")
	graph := flag.Bool("graph", false, "enable HIP Graph replay")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
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
	if *kvBlocks <= 0 {
		blocks := (*contextSize + *blockSize - 1) / *blockSize
		*kvBlocks = blocks * *concurrency
	}
	key, size, err := fileIdentity(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	backend, err := newBenchBackend(reader, model, benchBackendConfig{
		Blocks:       *kvBlocks,
		BlockSize:    *blockSize,
		PrefillChunk: *prefillChunk,
		Graph:        *graph,
		ModelKey:     key,
		Backend:      *backendName,
	})
	if err != nil {
		log.Fatal(err)
	}
	manager, err := serving.NewManager(backend, tok, serving.Options{
		ContextSize:   *contextSize,
		PrefixEnabled: *prefixCache,
		PrefixEntries: 32,
		Scheduler: scheduler.Config{
			MaxActive:       *concurrency,
			MaxQueue:        *concurrency * 2,
			MaxBatch:        *concurrency,
			PrefillChunk:    *prefillChunk,
			PrefillPerCycle: 1,
			DecodePerCycle:  *concurrency,
			RequestTimeout:  10 * time.Minute,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := manager.Start(context.Background()); err != nil {
		log.Fatal(err)
	}
	defer manager.Close()

	prompt := makePrompt(tok, *promptTokens)
	sampler, err := engine.NewSampler(engine.SamplerConfig{}, tok.VocabSize())
	if err != nil {
		log.Fatal(err)
	}
	run := func(count int) []sample {
		results := make([]sample, count)
		sem := make(chan struct{}, *concurrency)
		var wg sync.WaitGroup
		for i := 0; i < count; i++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				start := time.Now()
				result, err := manager.Generate(context.Background(), serving.Request{
					ID:        fmt.Sprintf("bench-%d", index),
					Prompt:    prompt,
					MaxTokens: *decodeTokens,
					Sampler:   sampler,
					IgnoreEOS: true,
				})
				if err != nil {
					log.Printf("request %d: %v", index, err)
					return
				}
				results[index] = sample{
					latency:  time.Since(start),
					queue:    result.QueueWait,
					ttft:     result.TimeToFirst,
					prefill:  result.PrefillTime,
					decode:   result.DecodeTime,
					sampling: result.SamplingTime,
					tokens:   result.OutputTokens,
				}
			}(i)
		}
		wg.Wait()
		return results
	}
	run(*warmup)
	start := time.Now()
	results := run(*requests)
	wall := time.Since(start)

	var totalTokens int
	valid := results[:0]
	for _, result := range results {
		if result.tokens > 0 {
			valid = append(valid, result)
			totalTokens += result.tokens
		}
	}
	if len(valid) == 0 {
		log.Fatal("no successful requests")
	}
	if *prefixCache {
		stats := manager.Metrics().Prefix
		fmt.Printf("prefix hits=%d misses=%d entries=%d\n", stats.Hits, stats.Misses, stats.Entries)
	}
	fmt.Printf("model=%s size=%d sha256=%s backend=%s context=%d block=%d blocks=%d concurrency=%d requests=%d prompt=%d decode=%d prefix_cache=%v graph=%v\n",
		*modelPath, size, key, *backendName, *contextSize, *blockSize, *kvBlocks,
		*concurrency, *requests, *promptTokens, *decodeTokens, *prefixCache, *graph)
	fmt.Printf("metadata commit=%s go=%s host_cpu=%q kernel=%s rocm_hip=%s power=%s sampling=greedy warmup=%d requests=%d concurrency=%d\n",
		buildRevision(), runtime.Version(), hostCPU(), hostKernel(), rocmVersion(), powerProfile(),
		*warmup, *requests, *concurrency)
	printSummary("latency", valid, func(s sample) time.Duration { return s.latency })
	printSummary("queue_wait", valid, func(s sample) time.Duration { return s.queue })
	printSummary("ttft", valid, func(s sample) time.Duration { return s.ttft })
	printSummary("prefill", valid, func(s sample) time.Duration { return s.prefill })
	printSummary("decode", valid, func(s sample) time.Duration { return s.decode })
	printSummary("sampling", valid, func(s sample) time.Duration { return s.sampling })
	fmt.Printf("throughput requests_per_s=%.3f output_tokens_per_s=%.3f total_tokens=%d wall=%.3fs\n",
		float64(len(valid))/wall.Seconds(),
		float64(totalTokens)/wall.Seconds(), totalTokens, wall.Seconds())
}

func printSummary(name string, values []sample, pick func(sample) time.Duration) {
	if len(values) == 0 {
		return
	}
	durations := make([]time.Duration, len(values))
	var sum time.Duration
	for i, value := range values {
		durations[i] = pick(value)
		sum += durations[i]
	}
	sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
	quantile := func(p float64) time.Duration {
		return durations[int(float64(len(durations)-1)*p)]
	}
	fmt.Printf("%s mean=%.6fs p50=%.6fs p95=%.6fs p99=%.6fs max=%.6fs\n",
		name, (sum / time.Duration(len(durations))).Seconds(),
		quantile(0.50).Seconds(), quantile(0.95).Seconds(),
		quantile(0.99).Seconds(), durations[len(durations)-1].Seconds())
}

func makePrompt(tok *tokenizer.Tokenizer, tokens int) []int32 {
	var ids []int32
	for len(ids) < tokens {
		encoded, err := tok.Encode("The quick brown fox jumps over the lazy dog. ")
		if err != nil || len(encoded) == 0 {
			log.Fatal("encode benchmark prompt")
		}
		ids = append(ids, encoded...)
	}
	return append([]int32(nil), ids[:tokens]...)
}

func fileIdentity(path string) (string, int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()
	hash := sha256.New()
	size, err := io.Copy(hash, file)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(hash.Sum(nil)), size, nil
}

// buildRevision reports the VCS revision embedded by the Go toolchain. A dirty
// working tree is marked so a result cannot be mistaken for a clean checkout.
func buildRevision() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	revision, modified := "", false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	if revision == "" {
		return "unknown"
	}
	if modified {
		return revision + "-dirty"
	}
	return revision
}

func hostKernel() string {
	data, err := os.ReadFile("/proc/sys/kernel/osrelease")
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(data))
}

func hostCPU() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "unknown"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "model name") {
			if _, value, ok := strings.Cut(line, ":"); ok {
				return strings.TrimSpace(value)
			}
		}
	}
	return "unknown"
}

func rocmVersion() string {
	candidates := []string{"/opt/rocm/bin/hipconfig"}
	if matches, _ := filepath.Glob("/opt/rocm/*/bin/hipconfig"); len(matches) > 0 {
		candidates = append(candidates, matches...)
	}
	candidates = append(candidates, "hipconfig")
	for _, candidate := range candidates {
		output, err := exec.Command(candidate, "--version").Output()
		if err != nil {
			continue
		}
		if version := strings.TrimSpace(string(output)); version != "" {
			return version
		}
	}
	matches, _ := filepath.Glob("/opt/rocm/*/.info/version")
	for _, path := range matches {
		if data, err := os.ReadFile(path); err == nil {
			if version := strings.TrimSpace(string(data)); version != "" {
				return version
			}
		}
	}
	return "unknown"
}

func powerProfile() string {
	matches, _ := filepath.Glob("/sys/class/drm/card*/device/power_dpm_force_performance_level")
	for _, path := range matches {
		if data, err := os.ReadFile(path); err == nil {
			if profile := strings.TrimSpace(string(data)); profile != "" {
				return profile
			}
		}
	}
	return "unknown"
}

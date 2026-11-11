package scheduler

import (
	"sort"
	"time"
)

// LatencySummary reports percentiles for a latency metric.
type LatencySummary struct {
	Count int           `json:"count"`
	Mean  time.Duration `json:"mean"`
	P50   time.Duration `json:"p50"`
	P95   time.Duration `json:"p95"`
	P99   time.Duration `json:"p99"`
	Max   time.Duration `json:"max"`
}

// Metrics is a point-in-time scheduler snapshot. QueueWait excludes time spent
// inside a task; Prefill and Decode totals come from task step reports, and
// Sampling is the subset of Decode spent selecting tokens.
type Metrics struct {
	Submissions      uint64 `json:"submissions"`
	Admitted         uint64 `json:"admitted"`
	Completed        uint64 `json:"completed"`
	Failures         uint64 `json:"failures"`
	Cancelled        uint64 `json:"cancelled"`
	Timeouts         uint64 `json:"timeouts"`
	QueueFull        uint64 `json:"queue_full"`
	CapacityRejected uint64 `json:"capacity_rejected"`

	Active            int `json:"active"`
	QueueDepth        int `json:"queue_depth"`
	BatchSize         int `json:"batch_size"`
	LastBatch         int `json:"last_batch"`
	MaxBatchObserved  int `json:"max_batch_observed"`
	MaxActiveObserved int `json:"max_active_observed"`

	PromptTokens    uint64        `json:"prompt_tokens"`
	GeneratedTokens uint64        `json:"generated_tokens"`
	QueueWaitTotal  time.Duration `json:"queue_wait_total"`
	PrefillTotal    time.Duration `json:"prefill_total"`
	DecodeTotal     time.Duration `json:"decode_total"`
	SamplingTotal   time.Duration `json:"sampling_total"`
	Steps           uint64        `json:"steps"`

	TTFT LatencySummary `json:"ttft"`
	TPOT LatencySummary `json:"tpot"`
}

type metricsState struct {
	submissions      uint64
	admitted         uint64
	completed        uint64
	failures         uint64
	cancelled        uint64
	timeouts         uint64
	queueFull        uint64
	capacityRejected uint64

	active            int
	queueDepth        int
	batchSize         int
	lastBatch         int
	maxBatchObserved  int
	maxActiveObserved int

	promptTokens    uint64
	generatedTokens uint64
	queueWaitTotal  time.Duration
	prefillTotal    time.Duration
	decodeTotal     time.Duration
	samplingTotal   time.Duration
	steps           uint64

	ttft latencyRecorder
	tpot latencyRecorder
}

const latencySamples = 4096

type latencyRecorder struct {
	samples [latencySamples]time.Duration
	next    int
	count   int
	total   int
}

func (r *latencyRecorder) add(value time.Duration) {
	if value <= 0 {
		return
	}
	r.samples[r.next] = value
	r.next = (r.next + 1) % latencySamples
	if r.count < latencySamples {
		r.count++
	}
	r.total++
}

func (r *latencyRecorder) summary() LatencySummary {
	if r.count == 0 {
		return LatencySummary{}
	}
	values := make([]time.Duration, r.count)
	copy(values, r.samples[:r.count])
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	var sum time.Duration
	for _, value := range values {
		sum += value
	}
	pick := func(percentile float64) time.Duration {
		index := int(float64(len(values)-1) * percentile)
		return values[index]
	}
	return LatencySummary{
		Count: r.count,
		Mean:  sum / time.Duration(len(values)),
		P50:   pick(0.50),
		P95:   pick(0.95),
		P99:   pick(0.99),
		Max:   values[len(values)-1],
	}
}

// Metrics returns a consistent snapshot of scheduler counters.
func (s *Scheduler) Metrics() Metrics {
	s.mu.Lock()
	defer s.mu.Unlock()
	return Metrics{
		Submissions:       s.metrics.submissions,
		Admitted:          s.metrics.admitted,
		Completed:         s.metrics.completed,
		Failures:          s.metrics.failures,
		Cancelled:         s.metrics.cancelled,
		Timeouts:          s.metrics.timeouts,
		QueueFull:         s.metrics.queueFull,
		CapacityRejected:  s.metrics.capacityRejected,
		Active:            len(s.active),
		QueueDepth:        len(s.queue),
		BatchSize:         s.metrics.batchSize,
		LastBatch:         s.metrics.lastBatch,
		MaxBatchObserved:  s.metrics.maxBatchObserved,
		MaxActiveObserved: s.metrics.maxActiveObserved,
		PromptTokens:      s.metrics.promptTokens,
		GeneratedTokens:   s.metrics.generatedTokens,
		QueueWaitTotal:    s.metrics.queueWaitTotal,
		PrefillTotal:      s.metrics.prefillTotal,
		DecodeTotal:       s.metrics.decodeTotal,
		SamplingTotal:     s.metrics.samplingTotal,
		Steps:             s.metrics.steps,
		TTFT:              s.metrics.ttft.summary(),
		TPOT:              s.metrics.tpot.summary(),
	}
}

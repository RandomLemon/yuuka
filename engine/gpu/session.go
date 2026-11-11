//go:build hip && linux

package gpu

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/memory"
)

// SessionOptions configures one request execution state.
type SessionOptions struct {
	// Capacity is the number of token positions reserved by the block table.
	// Zero uses every position mapped by the table.
	Capacity int
	// EnableGraph captures the decode plan after GraphMinSteps decode steps.
	EnableGraph   bool
	GraphMinSteps int
	// Trace allocates per-layer hidden-state buffers for ForwardTrace.
	Trace bool
}

// Session binds one block table and recurrent state to the shared model.
// Sessions must not execute concurrently; the model stream and activations are
// shared.
type Session struct {
	model    *Model
	table    *memory.Table
	capacity int
	position int
	tableBuf *hip.Buffer
	dyn      *hip.Buffer

	conv     []*hip.Buffer
	recState []*hip.Buffer

	plan      *hip.Plan
	tracePlan *hip.Plan
	traceBufs []*hip.Buffer

	prefillPlans map[int]*hip.Plan
	prefillGen   int64

	graph        *hip.Graph
	graphTried   bool
	graphEnabled bool
	graphMin     int
	decodeSteps  int

	logits []float32
	closed bool
}

// NewSession allocates per-request state and builds the decode plan.
func (m *Model) NewSession(table *memory.Table, position int, state []byte, opts SessionOptions) (*Session, error) {
	if m == nil || m.closed {
		return nil, fmt.Errorf("gpu: model is closed")
	}
	if table == nil {
		return nil, fmt.Errorf("gpu: block table is required")
	}
	capacity := opts.Capacity
	if capacity <= 0 {
		capacity = table.Bytes()
	}
	if capacity <= 0 || capacity > table.Bytes() {
		return nil, fmt.Errorf("gpu: capacity %d is not mapped by the block table", capacity)
	}
	if capacity > m.config.ContextLength {
		return nil, fmt.Errorf("gpu: capacity %d exceeds model context %d", capacity, m.config.ContextLength)
	}
	if position < 0 || position > capacity {
		return nil, fmt.Errorf("gpu: position %d is outside capacity %d", position, capacity)
	}
	nblocks := (capacity + m.blockSize - 1) / m.blockSize
	blocks := table.Blocks()
	if len(blocks) < nblocks {
		return nil, fmt.Errorf("gpu: block table maps %d blocks, need %d", len(blocks), nblocks)
	}
	s := &Session{
		model:        m,
		table:        table,
		capacity:     capacity,
		position:     position,
		conv:         make([]*hip.Buffer, m.config.BlockCount),
		recState:     make([]*hip.Buffer, m.config.BlockCount),
		prefillPlans: make(map[int]*hip.Plan),
		graphEnabled: opts.EnableGraph,
		graphMin:     opts.GraphMinSteps,
		logits:       make([]float32, m.config.VocabSize),
	}
	if s.graphMin <= 0 {
		s.graphMin = 1
	}
	cleanup := func() { _ = s.Close() }
	if err := s.allocState(); err != nil {
		cleanup()
		return nil, err
	}
	ids := make([]int32, nblocks)
	for i := range ids {
		ids[i] = int32(blocks[i])
	}
	buf, err := m.allocTransient(uint64(len(ids)) * 4)
	if err != nil {
		cleanup()
		return nil, err
	}
	s.tableBuf = buf
	if err := m.device.UploadInt32(buf, ids); err != nil {
		cleanup()
		return nil, err
	}
	dyn, err := m.allocTransient(8)
	if err != nil {
		cleanup()
		return nil, err
	}
	s.dyn = dyn
	if len(state) > 0 {
		if err := s.Restore(state); err != nil {
			cleanup()
			return nil, err
		}
	}
	plan, err := m.buildDecodePlan(s, false)
	if err != nil {
		cleanup()
		return nil, err
	}
	s.plan = plan
	if opts.Trace {
		s.traceBufs = make([]*hip.Buffer, m.config.BlockCount)
		for i := range s.traceBufs {
			buf, err := m.allocTransient(uint64(m.config.HiddenSize) * 4)
			if err != nil {
				cleanup()
				return nil, err
			}
			s.traceBufs[i] = buf
		}
		tracePlan, err := m.buildDecodePlan(s, true)
		if err != nil {
			cleanup()
			return nil, err
		}
		s.tracePlan = tracePlan
	}
	return s, nil
}

func (s *Session) allocState() error {
	cfg := s.model.config
	convDim := uint64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	convBytes := uint64(cfg.SSMConvKernel-1) * convDim * 4
	recBytes := uint64(cfg.SSMTimeStepRank*cfg.SSMStateSize*cfg.SSMStateSize) * 4
	for i := range cfg.BlockCount {
		if !cfg.IsRecurrent(i) {
			continue
		}
		conv, err := s.model.allocTransient(convBytes)
		if err != nil {
			return err
		}
		rec, err := s.model.allocTransient(recBytes)
		if err != nil {
			return err
		}
		if err := s.model.device.Memset(s.model.stream, conv, 0, convBytes); err != nil {
			return err
		}
		if err := s.model.device.Memset(s.model.stream, rec, 0, recBytes); err != nil {
			return err
		}
		s.conv[i] = conv
		s.recState[i] = rec
	}
	return s.model.stream.Synchronize()
}

// Capacity returns the reserved positions.
func (s *Session) Capacity() int { return s.capacity }

// Position returns the next absolute token position.
func (s *Session) Position() int { return s.position }

// GraphActive reports whether a captured graph is currently installed.
func (s *Session) GraphActive() bool { return s.graph != nil }

// Prefill processes prompt tokens. final selects whether the output logits are
// required and whether the plan tail runs.
func (s *Session) Prefill(tokens []int32, final bool) ([]float32, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("gpu: session is closed")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("gpu: empty token sequence")
	}
	if s.position+len(tokens) > s.capacity {
		return nil, fmt.Errorf("gpu: token sequence exceeds remaining context")
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= s.model.config.VocabSize {
			return nil, fmt.Errorf("gpu: token id %d is out of range", token)
		}
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
	s.syncPrefillPlans()
	for start := 0; start < len(tokens); {
		n := s.model.prefillRows
		if remaining := len(tokens) - start; remaining < n {
			n = remaining
		}
		plan, err := s.prefillPlan(n)
		if err != nil {
			return nil, fmt.Errorf("build prefill plan for %d rows: %w", n, err)
		}
		if err := s.model.device.UploadInt32(s.model.prefill.tokens, tokens[start:start+n]); err != nil {
			return nil, err
		}
		last := final && start+n == len(tokens)
		if err := plan.Execute(s.model.stream, 0, uint32(s.position), last, s.model.event); err != nil {
			return nil, err
		}
		if err := s.model.event.Wait(); err != nil {
			return nil, err
		}
		s.position += n
		start += n
	}
	if !final {
		return nil, nil
	}
	if err := s.model.device.DownloadFloats(s.logits, s.model.act.logits); err != nil {
		return nil, err
	}
	return s.logits, nil
}

func (s *Session) syncPrefillPlans() {
	if s.prefillGen == s.model.prefillGen {
		return
	}
	for _, plan := range s.prefillPlans {
		_ = plan.Close()
	}
	s.prefillPlans = make(map[int]*hip.Plan)
	s.prefillGen = s.model.prefillGen
}

func (s *Session) prefillPlan(rows int) (*hip.Plan, error) {
	if plan, ok := s.prefillPlans[rows]; ok {
		return plan, nil
	}
	plan, err := s.model.buildPrefillPlan(s, rows)
	if err != nil {
		return nil, err
	}
	s.prefillPlans[rows] = plan
	return plan, nil
}

// Decode feeds one token and returns next-position logits. Graph replay is
// attempted first and falls back to normal submission on any validation or
// launch failure.
func (s *Session) Decode(token int32) ([]float32, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("gpu: session is closed")
	}
	if token < 0 || int(token) >= s.model.config.VocabSize {
		return nil, fmt.Errorf("gpu: token id %d is out of range", token)
	}
	if s.position >= s.capacity {
		return nil, fmt.Errorf("gpu: context capacity %d exhausted", s.capacity)
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
	defer func() { s.decodeSteps++ }()

	if s.graph != nil {
		ok, err := s.replayGraph(token)
		if err != nil {
			return nil, err
		}
		if ok {
			return s.logits, nil
		}
	}
	if s.graph == nil && !s.graphTried && s.graphEnabled && s.decodeSteps+1 >= s.graphMin {
		s.graphTried = true
		graph, err := s.plan.CaptureGraph(s.model.stream, s.dyn, true)
		if err != nil {
			s.graphEnabled = false
		} else {
			s.graph = graph
			ok, replayErr := s.replayGraph(token)
			if replayErr != nil {
				return nil, replayErr
			}
			if ok {
				return s.logits, nil
			}
		}
	}
	if err := s.plan.Execute(s.model.stream, token, uint32(s.position), true, s.model.event); err != nil {
		return nil, err
	}
	if err := s.model.event.Wait(); err != nil {
		return nil, err
	}
	s.position++
	if err := s.model.device.DownloadFloats(s.logits, s.model.act.logits); err != nil {
		return nil, err
	}
	return s.logits, nil
}

// replayGraph validates the captured graph, uploads dynamic parameters, and
// launches it. Failure drops the graph and leaves normal execution possible.
func (s *Session) replayGraph(token int32) (bool, error) {
	if s.graph == nil || s.closed {
		return false, nil
	}
	if s.position >= s.capacity {
		s.dropGraph()
		return false, nil
	}
	needed := (s.position + 1 + s.model.blockSize - 1) / s.model.blockSize
	if needed > s.table.Len() {
		s.dropGraph()
		return false, nil
	}
	if err := s.model.device.SetDynamic(s.model.stream, s.dyn, token, uint32(s.position)); err != nil {
		s.dropGraph()
		return false, nil
	}
	if err := s.graph.Replay(s.model.stream, s.model.event); err != nil {
		s.dropGraph()
		return false, nil
	}
	if err := s.model.event.Wait(); err != nil {
		s.dropGraph()
		return false, err
	}
	if err := s.model.device.DownloadFloats(s.logits, s.model.act.logits); err != nil {
		return false, err
	}
	s.position++
	return true, nil
}

func (s *Session) dropGraph() {
	if s.graph != nil {
		_ = s.graph.Close()
		s.graph = nil
	}
	s.graphEnabled = false
	_ = s.model.stream.Synchronize()
}

// Snapshot serializes recurrent state. Full-attention KV is not copied.
func (s *Session) Snapshot() ([]byte, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("gpu: session is closed")
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
	total := 0
	for i := range s.conv {
		if s.conv[i] != nil {
			total += int(s.conv[i].Bytes())
		}
		if s.recState[i] != nil {
			total += int(s.recState[i].Bytes())
		}
	}
	data := make([]byte, total)
	offset := 0
	for i := range s.conv {
		for _, buf := range []*hip.Buffer{s.conv[i], s.recState[i]} {
			if buf == nil {
				continue
			}
			size := int(buf.Bytes())
			if err := s.model.device.Download(data[offset:offset+size], buf); err != nil {
				return nil, err
			}
			offset += size
		}
	}
	return data, nil
}

// Restore applies a Snapshot produced at the same geometry.
func (s *Session) Restore(data []byte) error {
	if s == nil || s.closed {
		return fmt.Errorf("gpu: session is closed")
	}
	expected := 0
	for i := range s.conv {
		if s.conv[i] != nil {
			expected += int(s.conv[i].Bytes())
		}
		if s.recState[i] != nil {
			expected += int(s.recState[i].Bytes())
		}
	}
	if len(data) != expected {
		return fmt.Errorf("gpu: state snapshot is %d bytes, want %d", len(data), expected)
	}
	offset := 0
	for i := range s.conv {
		for _, buf := range []*hip.Buffer{s.conv[i], s.recState[i]} {
			if buf == nil {
				continue
			}
			size := int(buf.Bytes())
			if err := s.model.device.Upload(buf, data[offset:offset+size]); err != nil {
				return err
			}
			offset += size
		}
	}
	return nil
}

// Reset releases the current blocks, allocates a fresh table, and clears
// recurrent state. It is used by the single-request runtime.
func (s *Session) Reset() error {
	if s == nil || s.closed {
		return fmt.Errorf("gpu: session is closed")
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
	convDim := uint64(2*s.model.config.SSMGroupCount*s.model.config.SSMStateSize + s.model.config.SSMInnerSize)
	convBytes := uint64(s.model.config.SSMConvKernel-1) * convDim * 4
	recBytes := uint64(s.model.config.SSMTimeStepRank*s.model.config.SSMStateSize*s.model.config.SSMStateSize) * 4
	for i := range s.conv {
		if s.conv[i] != nil {
			if err := s.model.device.Memset(s.model.stream, s.conv[i], 0, convBytes); err != nil {
				return err
			}
			if err := s.model.device.Memset(s.model.stream, s.recState[i], 0, recBytes); err != nil {
				return err
			}
		}
	}
	if err := s.model.stream.Synchronize(); err != nil {
		return err
	}
	s.position = 0
	s.decodeSteps = 0
	return nil
}

// ForwardTrace runs tokens through the trace plan and returns logits plus the
// hidden state after every layer for the last token.
func (s *Session) ForwardTrace(tokens []int32) ([]float32, [][]float32, error) {
	if s == nil || s.closed {
		return nil, nil, fmt.Errorf("gpu: session is closed")
	}
	if s.tracePlan == nil {
		return nil, nil, fmt.Errorf("gpu: session has no trace plan")
	}
	if len(tokens) == 0 {
		return nil, nil, fmt.Errorf("gpu: empty token sequence")
	}
	if s.position+len(tokens) > s.capacity {
		return nil, nil, fmt.Errorf("gpu: token sequence exceeds remaining context")
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
	for i, token := range tokens {
		if token < 0 || int(token) >= s.model.config.VocabSize {
			return nil, nil, fmt.Errorf("gpu: token id %d is out of range", token)
		}
		if err := s.tracePlan.Execute(s.model.stream, token, uint32(s.position), i == len(tokens)-1, s.model.event); err != nil {
			return nil, nil, err
		}
		if err := s.model.event.Wait(); err != nil {
			return nil, nil, err
		}
		s.position++
	}
	if err := s.model.device.DownloadFloats(s.logits, s.model.act.logits); err != nil {
		return nil, nil, err
	}
	traces := make([][]float32, len(s.traceBufs))
	for i, buf := range s.traceBufs {
		trace := make([]float32, s.model.config.HiddenSize)
		if err := s.model.device.DownloadFloats(trace, buf); err != nil {
			return nil, nil, err
		}
		traces[i] = trace
	}
	return s.logits, traces, nil
}

// ProfileStep executes one token with per-operation events on the normal plan.
func (s *Session) ProfileStep(token int32) ([]hip.OpTime, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("gpu: session is closed")
	}
	if token < 0 || int(token) >= s.model.config.VocabSize {
		return nil, fmt.Errorf("gpu: token id %d is out of range", token)
	}
	if s.position >= s.capacity {
		return nil, fmt.Errorf("gpu: context capacity %d exhausted", s.capacity)
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
	times, err := s.plan.ExecuteProfiled(s.model.stream, token, uint32(s.position), true, s.model.event)
	if err != nil {
		return nil, err
	}
	if err := s.model.event.Wait(); err != nil {
		return nil, err
	}
	s.position++
	return times, nil
}

// Close releases the session plans, state buffers, and block table references.
func (s *Session) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	if s.graph != nil {
		_ = s.graph.Close()
		s.graph = nil
	}
	if s.plan != nil {
		_ = s.plan.Close()
	}
	if s.tracePlan != nil {
		_ = s.tracePlan.Close()
	}
	for _, plan := range s.prefillPlans {
		_ = plan.Close()
	}
	s.prefillPlans = nil
	for i := len(s.traceBufs) - 1; i >= 0; i-- {
		if s.traceBufs[i] != nil {
			_ = s.traceBufs[i].Close()
		}
	}
	for i := range s.conv {
		if s.conv[i] != nil {
			_ = s.conv[i].Close()
		}
		if s.recState[i] != nil {
			_ = s.recState[i].Close()
		}
	}
	if s.tableBuf != nil {
		_ = s.tableBuf.Close()
	}
	if s.dyn != nil {
		_ = s.dyn.Close()
	}
	return nil
}

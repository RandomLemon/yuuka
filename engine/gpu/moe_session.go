//go:build hip && linux

package gpu

import (
	"fmt"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/memory"
)

// MoESession binds one block table to the shared MoE model.
type MoESession struct {
	model    *MoEModel
	table    *memory.Table
	capacity int
	position int
	tableBuf *hip.Buffer

	plan         *hip.Plan
	tracePlan    *hip.Plan
	traceBufs    []*hip.Buffer
	prefillPlans map[int]*hip.Plan

	logits []float32
	closed bool
}

// NewMoESession allocates per-request state and builds the decode plan.
func (m *MoEModel) NewMoESession(table *memory.Table, position int, capacity int) (*MoESession, error) {
	return m.newMoESession(table, position, capacity, false)
}

// NewMoESessionTrace is NewMoESession with per-layer hidden-state tracing.
func (m *MoEModel) NewMoESessionTrace(table *memory.Table, position int, capacity int) (*MoESession, error) {
	return m.newMoESession(table, position, capacity, true)
}

func (m *MoEModel) newMoESession(table *memory.Table, position int, capacity int, trace bool) (*MoESession, error) {
	if m == nil || m.closed {
		return nil, fmt.Errorf("gpu: MoE model is closed")
	}
	if table == nil {
		return nil, fmt.Errorf("gpu: block table is required")
	}
	if capacity <= 0 {
		capacity = table.Bytes()
	}
	if capacity <= 0 || capacity > table.Bytes() {
		return nil, fmt.Errorf("gpu: capacity %d is not mapped by the block table", capacity)
	}
	if capacity > m.config.Context {
		return nil, fmt.Errorf("gpu: capacity %d exceeds model context %d", capacity, m.config.Context)
	}
	if position < 0 || position > capacity {
		return nil, fmt.Errorf("gpu: position %d is outside capacity %d", position, capacity)
	}
	nblocks := (capacity + m.blockSize - 1) / m.blockSize
	blocks := table.Blocks()
	if len(blocks) < nblocks {
		return nil, fmt.Errorf("gpu: block table maps %d blocks, need %d", len(blocks), nblocks)
	}
	s := &MoESession{
		model:        m,
		table:        table,
		capacity:     capacity,
		position:     position,
		prefillPlans: make(map[int]*hip.Plan),
		logits:       make([]float32, m.config.VocabSize),
	}
	cleanup := func() { _ = s.Close() }
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
	plan, err := m.buildMoEDecodePlan(s, false)
	if err != nil {
		cleanup()
		return nil, err
	}
	s.plan = plan
	if trace {
		s.traceBufs = make([]*hip.Buffer, m.config.BlockCount)
		for i := range s.traceBufs {
			buf, err := m.allocTransient(uint64(m.config.HiddenSize) * 4)
			if err != nil {
				cleanup()
				return nil, err
			}
			s.traceBufs[i] = buf
		}
		tracePlan, err := m.buildMoEDecodePlan(s, true)
		if err != nil {
			cleanup()
			return nil, err
		}
		s.tracePlan = tracePlan
	}
	return s, nil
}

// Capacity returns the reserved positions.
func (s *MoESession) Capacity() int { return s.capacity }

// Position returns the next absolute token position.
func (s *MoESession) Position() int { return s.position }

// Prefill processes prompt tokens. final selects whether logits are required.
func (s *MoESession) Prefill(tokens []int32, final bool) ([]float32, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("gpu: MoE session is closed")
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
	chunk := s.model.act.maxRows
	if chunk <= 0 {
		return nil, fmt.Errorf("gpu: MoE prefill chunk is not configured")
	}
	for start := 0; start < len(tokens); {
		n := chunk
		if remaining := len(tokens) - start; remaining < n {
			n = remaining
		}
		plan, err := s.prefillPlan(n)
		if err != nil {
			return nil, fmt.Errorf("build MoE prefill plan for %d rows: %w", n, err)
		}
		if err := s.model.device.UploadInt32(s.model.act.tokens, tokens[start:start+n]); err != nil {
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

func (s *MoESession) prefillPlan(rows int) (*hip.Plan, error) {
	if plan, ok := s.prefillPlans[rows]; ok {
		return plan, nil
	}
	plan, err := s.model.buildMoEPrefillPlan(s, rows)
	if err != nil {
		return nil, err
	}
	s.prefillPlans[rows] = plan
	return plan, nil
}

// Decode feeds one token and returns next-position logits.
func (s *MoESession) Decode(token int32) ([]float32, error) {
	if s == nil || s.closed {
		return nil, fmt.Errorf("gpu: MoE session is closed")
	}
	if token < 0 || int(token) >= s.model.config.VocabSize {
		return nil, fmt.Errorf("gpu: token id %d is out of range", token)
	}
	if s.position >= s.capacity {
		return nil, fmt.Errorf("gpu: context capacity %d exhausted", s.capacity)
	}
	s.model.stepMu.Lock()
	defer s.model.stepMu.Unlock()
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

// ForwardTrace runs tokens through the trace plan and returns logits plus the
// hidden state after every layer for the last token.
func (s *MoESession) ForwardTrace(tokens []int32) ([]float32, [][]float32, error) {
	if s == nil || s.closed {
		return nil, nil, fmt.Errorf("gpu: MoE session is closed")
	}
	if s.tracePlan == nil {
		return nil, nil, fmt.Errorf("gpu: MoE session has no trace plan")
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

// Snapshot returns no state; Qwen3 MoE has no recurrent state outside KV.
func (s *MoESession) Snapshot() ([]byte, error) { return nil, nil }

// PositionSet restores the absolute position after a prefix-cache hit.
func (s *MoESession) SetPosition(pos int) error {
	if pos < 0 || pos > s.capacity {
		return fmt.Errorf("gpu: position %d exceeds capacity %d", pos, s.capacity)
	}
	s.position = pos
	return nil
}

// Close releases the plans and block table references.
func (s *MoESession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	if s.plan != nil {
		_ = s.plan.Close()
	}
	if s.tracePlan != nil {
		_ = s.tracePlan.Close()
	}
	for i := len(s.traceBufs) - 1; i >= 0; i-- {
		if s.traceBufs[i] != nil {
			_ = s.traceBufs[i].Close()
		}
	}
	for _, plan := range s.prefillPlans {
		_ = plan.Close()
	}
	s.prefillPlans = nil
	if s.tableBuf != nil {
		_ = s.tableBuf.Close()
	}
	return nil
}

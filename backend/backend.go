// Package backend defines the execution contracts shared by the CPU reference
// backend and the HIP backend. Higher layers depend on these interfaces and
// never on device runtime details.
package backend

import (
	"errors"
	"fmt"
)

// Capabilities describes what a backend can execute. The planner must consult
// this before selecting a kernel.
type Capabilities struct {
	Name             string
	Device           string
	SupportsF32      bool
	SupportsF16      bool
	SupportedQuant   []string
	SupportsGraph    bool
	MaxBatch         int
	TotalMemoryBytes uint64
}

// Buffer is a backend-owned allocation. The owner is responsible for calling
// Close exactly once unless ownership is explicitly transferred.
type Buffer interface {
	Bytes() uint64
	Close() error
}

// Event signals completion of submitted work. Wait may be called more than
// once; implementations must make it idempotent after completion.
type Event interface {
	Wait() error
}

// Executable is a compiled plan. Run submissions must not retain Go pointers
// after the call returns unless the backend copied them synchronously.
type Executable interface {
	Run(batch *Batch) (Event, error)
	Close() error
}

// Backend is the device execution seam.
type Backend interface {
	Name() string
	Capabilities() Capabilities
	Alloc(size uint64) (Buffer, error)
	Compile(plan *ExecutionPlan) (Executable, error)
	Synchronize() error
	Close() error
}

// OpCode identifies a planned operator.
type OpCode uint32

const (
	OpMatVec OpCode = iota + 1
	OpMatMul
	OpRMSNorm
	OpRoPE
	OpAttention
	OpSwiGLU
	OpGDN
	OpSample
)

// Op is one planned operation. Buffer indices refer to the plan's binding table.
type Op struct {
	Code   OpCode
	Inputs []int
	Out    int
	Attrs  map[string]any
}

// ExecutionPlan binds immutable weights and scratch regions for a model. The
// hot path submits a plan step; it does not re-plan every tensor operation.
type ExecutionPlan struct {
	ModelName string
	Pipeline  []Op
	Bindings  int
	Scratch   uint64
}

// Batch is the per-step execution input.
type Batch struct {
	Tokens     []int32
	Positions  []int32
	BatchSize  int
	SequenceID uint64
}

// ErrUnsupported marks a capability failure that planning must surface, not
// silently replace with a different numerical path.
var ErrUnsupported = errors.New("backend does not support the requested operation")

// UnsupportedError adds operation context to ErrUnsupported.
type UnsupportedError struct {
	Backend string
	Op      string
	Detail  string
}

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("%s: %s: %s", e.Backend, e.Op, e.Detail)
}

func (e *UnsupportedError) Unwrap() error { return ErrUnsupported }

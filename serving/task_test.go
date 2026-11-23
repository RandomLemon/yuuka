package serving

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
	"time"
)

type recordingSequence struct {
	events []string
	logits []float32
	state  []byte
}

func (s *recordingSequence) Prefill([]int32, bool) ([]float32, error) {
	s.events = append(s.events, "prefill")
	return s.logits, nil
}

func (s *recordingSequence) Decode(int32) ([]float32, error) {
	s.events = append(s.events, "decode")
	return s.logits, nil
}

func (s *recordingSequence) Snapshot() ([]byte, error) {
	s.events = append(s.events, "snapshot")
	return s.state, nil
}

func (s *recordingSequence) Close() error { return nil }

func newSnapshotTestTask(t *testing.T, capturePrefix bool) (*generationTask, *recordingSequence) {
	t.Helper()
	h := newTinyHarness(t)
	token := h.prompt(t, "a")[0]
	seq := &recordingSequence{logits: make([]float32, h.tok.VocabSize()), state: []byte{1, 2, 3}}
	seq.logits[token] = 1
	sink := func(_ int32, _ string) error {
		seq.events = append(seq.events, "emit")
		return nil
	}
	task := newGenerationTask("snapshot", []int32{token}, 2, seq, samplerFor(t, h.tok.VocabSize()),
		h.tok, nil, nil, true, capturePrefix, nil, nil, 0, sink, time.Now())
	return task, seq
}

func TestGenerationWithPrefixCacheDisabled(t *testing.T) {
	task, seq := newSnapshotTestTask(t, false)
	ctx := context.Background()
	if _, done, err := task.PrefillOnce(ctx, 1); err != nil || !done {
		t.Fatalf("PrefillOnce: done=%v, err=%v", done, err)
	}
	if _, done, err := task.DecodeOnce(ctx); err != nil || !done {
		t.Fatalf("DecodeOnce: done=%v, err=%v", done, err)
	}
	want := []string{"prefill", "emit", "decode", "emit"}
	if !reflect.DeepEqual(seq.events, want) {
		t.Fatalf("events = %v, want %v", seq.events, want)
	}
	if _, _, _, ready := task.PrefixReady(); ready || task.promptLogits != nil {
		t.Fatal("request without a cache captured prompt data")
	}
}

func TestGenerationCapturesPrefixAfterFirstEmission(t *testing.T) {
	task, seq := newSnapshotTestTask(t, true)
	var wantLogits bytes.Buffer
	if err := binary.Write(&wantLogits, binary.LittleEndian, seq.logits); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, done, err := task.PrefillOnce(ctx, 1); err != nil || !done {
		t.Fatalf("PrefillOnce: done=%v, err=%v", done, err)
	}
	seq.logits[task.prompt[0]] = 2
	if _, done, err := task.DecodeOnce(ctx); err != nil || !done {
		t.Fatalf("DecodeOnce: done=%v, err=%v", done, err)
	}
	want := []string{"prefill", "emit", "snapshot", "decode", "emit"}
	if !reflect.DeepEqual(seq.events, want) {
		t.Fatalf("events = %v, want %v", seq.events, want)
	}
	tokens, state, logits, ready := task.PrefixReady()
	if !ready || !reflect.DeepEqual(tokens, task.prompt) ||
		!bytes.Equal(state, seq.state) || !bytes.Equal(logits, wantLogits.Bytes()) {
		t.Fatal("cached prefix does not match the prompt boundary")
	}
}

func TestGenerationCapturesPrefixOnTerminalFirstStep(t *testing.T) {
	sinkErr := errors.New("disconnected sink")
	for _, tc := range []struct {
		name string
		want []string
		err  error
	}{
		{"length", []string{"prefill", "emit", "snapshot"}, nil},
		{"stop", []string{"prefill", "snapshot"}, nil},
		{"sink_error", []string{"prefill", "emit", "snapshot"}, sinkErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			task, seq := newSnapshotTestTask(t, true)
			switch tc.name {
			case "length":
				task.maxToken = 1
			case "stop":
				task.stops[task.prompt[0]] = struct{}{}
			case "sink_error":
				task.sink = func(_ int32, _ string) error {
					seq.events = append(seq.events, "emit")
					return sinkErr
				}
			}
			if _, done, err := task.PrefillOnce(context.Background(), 1); !done || !errors.Is(err, tc.err) {
				t.Fatalf("PrefillOnce: done=%v, err=%v, want %v", done, err, tc.err)
			}
			if !reflect.DeepEqual(seq.events, tc.want) {
				t.Fatalf("events = %v, want %v", seq.events, tc.want)
			}
			if _, _, _, ready := task.PrefixReady(); !ready {
				t.Fatal("terminal first step did not capture the prefix")
			}
		})
	}
}

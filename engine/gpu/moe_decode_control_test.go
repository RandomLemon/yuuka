//go:build hip && linux

package gpu

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
)

// moeDecodeFixtureSHA256 is the recorded digest of the text GGUF the decode
// plan targets. The control files only describe that model, so recording or
// replaying them against a different file must fail before any row is read.
const moeDecodeFixtureSHA256 = "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef"

// moeDecodeInputSHA256 is the digest of the 8320-token little-endian input the
// release driver writes from the fixed benchmark text. The control files carry
// the rows produced by that exact stream.
const moeDecodeInputSHA256 = "2367fa295341e579a00189d5a5f52df5b742c5e732f16c6b8f5b15f6709acf81"

// moeDecodeControlTolerance is the logit bound between the frozen baseline and
// the current build. Zero means every bit must match. A shipped change that
// deliberately reorders a floating-point reduction raises this to the
// documented 0.02 decode bound; arithmetic-preserving changes keep zero.
const moeDecodeControlTolerance = 0.0

// TestGPUMoEDecodeMatchesControl records or replays the frozen one-token decode
// rows at 512, 2048, and 8192 populated tokens. Record mode is enabled by
// YUUKA_TEST_MOE_DECODE_RECORD=1 with YUUKA_TEST_MOE_DECODE_CONTROL_DIR pointing
// at a fresh directory. Replay mode is selected by the control directory alone
// and compares every prefill and decode row for finite values, top-1 agreement,
// and the logit bound.
func TestGPUMoEDecodeMatchesControl(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	const length = 8320
	const steps = 128
	ids, err := tok.Encode(strings.Repeat(moePrefillBenchmarkText, length))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < length {
		t.Fatalf("benchmark prompt produced %d tokens, need %d", len(ids), length)
	}
	ids = ids[:length]
	if got := digestTokens(ids); got != moeDecodeInputSHA256 {
		t.Fatalf("benchmark input digest %s, want %s", got, moeDecodeInputSHA256)
	}

	dir := os.Getenv("YUUKA_TEST_MOE_DECODE_CONTROL_DIR")
	if os.Getenv("YUUKA_TEST_MOE_DECODE_RECORD") == "1" {
		if dir == "" {
			t.Fatal("YUUKA_TEST_MOE_DECODE_RECORD requires YUUKA_TEST_MOE_DECODE_CONTROL_DIR")
		}
		verifyDecodeFixture(t)
		prepareControlDir(t, dir, steps, model.Config.VocabSize)
		if err := os.WriteFile(filepath.Join(dir, "input.sha256"),
			[]byte(moeDecodeInputSHA256+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		for _, prompt := range []int{512, 2048, 8192} {
			t.Logf("recording P=%d", prompt)
			recordDecodeRows(t, reader, model, ids, dir, prompt, steps)
		}
		return
	}
	if dir == "" {
		// A Q8 selected-expert policy is not part of this build, so there is
		// no same-build route pair to compare. The frozen run is mandatory for
		// acceptance; this opt-in differential skips only without it.
		t.Skip("set YUUKA_TEST_MOE_DECODE_CONTROL_DIR or record with YUUKA_TEST_MOE_DECODE_RECORD=1")
	}
	verifyDecodeFixture(t)
	verifyControlInput(t, dir, ids)
	for _, prompt := range []int{512, 2048, 8192} {
		t.Run(fmt.Sprintf("P%d", prompt), func(t *testing.T) {
			verifyDecodeRows(t, reader, model, ids, dir, prompt, steps)
		})
	}
}

// digestTokens hashes the little-endian int32 token stream exactly as the
// release drivers write and read it.
func digestTokens(ids []int32) string {
	raw := make([]byte, 4*len(ids))
	for i, id := range ids {
		binary.LittleEndian.PutUint32(raw[4*i:], uint32(id))
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// verifyDecodeFixture checks the model file against the recorded digest before
// any control row is produced or consumed.
func verifyDecodeFixture(t *testing.T) {
	t.Helper()
	path := os.Getenv("YUUKA_TEST_MOE_MODEL")
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != moeDecodeFixtureSHA256 {
		t.Fatalf("model digest %s, want %s", got, moeDecodeFixtureSHA256)
	}
}

// prepareControlDir rejects a directory that already holds any control artifact
// so a recording can never overwrite a frozen baseline.
func prepareControlDir(t *testing.T, dir string, steps, vocab int) {
	t.Helper()
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("control directory: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("control path %q is not a directory", dir)
	}
	for _, name := range append([]string{"input.sha256"}, controlNames(steps)...) {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			t.Fatalf("control artifact %s already exists; refusing to overwrite", name)
		} else if !os.IsNotExist(err) {
			t.Fatal(err)
		}
	}
	if vocab <= 0 {
		t.Fatalf("invalid vocabulary size %d", vocab)
	}
}

func controlNames(steps int) []string {
	return []string{
		fmt.Sprintf("P512.f32"),
		fmt.Sprintf("P2048.f32"),
		fmt.Sprintf("P8192.f32"),
	}
}

// verifyControlInput rejects a control directory whose input digest differs
// from the generated stream.
func verifyControlInput(t *testing.T, dir string, ids []int32) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "input.sha256"))
	if err != nil {
		t.Fatalf("control input digest: %v", err)
	}
	got := strings.TrimSpace(string(raw))
	if got != digestTokens(ids) {
		t.Fatalf("control input digest %s, generated %s", got, digestTokens(ids))
	}
}

func recordControlPath(dir string, prompt int) string {
	return filepath.Join(dir, fmt.Sprintf("P%d.f32", prompt))
}

// recordDecodeRows writes the prefill row and 128 decode rows one row at a
// time. The runtime is closed before the next cell so only one model is
// resident.
func recordDecodeRows(t *testing.T, reader *gguf.Reader, model *qwen35.Model, ids []int32,
	dir string, prompt, steps int) {
	t.Helper()
	path := recordControlPath(dir, prompt)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	writer := bufio.NewWriterSize(file, 1<<20)
	rt, err := newRuntime(reader, model, prompt+steps,
		Options{KVType: tensor.F16, KVTypeSet: true})
	if err != nil {
		_ = file.Close()
		t.Fatalf("HIP runtime unavailable: %v", err)
	}
	if err := rt.SetPrefillChunk(512); err != nil {
		_ = rt.Close()
		_ = file.Close()
		t.Fatal(err)
	}
	s := rt.session
	writeRow := func(row []float32) {
		if len(row) != model.Config.VocabSize {
			t.Fatalf("row has %d logits, want %d", len(row), model.Config.VocabSize)
		}
		for _, value := range row {
			if err := binary.Write(writer, binary.LittleEndian, value); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	logits, err := s.Prefill(ids[:prompt], true)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Position(); got != prompt {
		t.Fatalf("position after prefill is %d, want %d", got, prompt)
	}
	writeRow(logits)
	for j := range steps {
		logits, err = s.Decode(ids[prompt+j])
		if err != nil {
			t.Fatalf("decode step %d: %v", j, err)
		}
		if got := s.Position(); got != prompt+j+1 {
			t.Fatalf("position after decode step %d is %d, want %d", j, got, prompt+j+1)
		}
		writeRow(logits)
	}
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := writer.Flush(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

// verifyDecodeRows compares the prefill row bitwise and every decode row
// against the frozen control, streaming one control row at a time.
func verifyDecodeRows(t *testing.T, reader *gguf.Reader, model *qwen35.Model, ids []int32,
	dir string, prompt, steps int) {
	t.Helper()
	vocab := model.Config.VocabSize
	path := recordControlPath(dir, prompt)
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("control file: %v", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	wantBytes := int64(steps+1) * int64(vocab) * 4
	if info.Size() != wantBytes {
		t.Fatalf("control file %s has %d bytes, want %d", path, info.Size(), wantBytes)
	}
	readerBuf := bufio.NewReaderSize(file, 1<<20)
	want := make([]float32, vocab)
	readRow := func() {
		for i := range want {
			if err := binary.Read(readerBuf, binary.LittleEndian, &want[i]); err != nil {
				t.Fatalf("control row: %v", err)
			}
		}
	}

	rt, err := newRuntime(reader, model, prompt+steps,
		Options{KVType: tensor.F16, KVTypeSet: true})
	if err != nil {
		t.Fatalf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(512); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	got, err := s.Prefill(ids[:prompt], true)
	if err != nil {
		t.Fatal(err)
	}
	if pos := s.Position(); pos != prompt {
		t.Fatalf("position after prefill is %d, want %d", pos, prompt)
	}
	readRow()
	if diff, equal := compareControlRow(got, want); !equal {
		t.Fatalf("prefill row is not bitwise equal; max diff %.6g", diff)
	}
	for j := range steps {
		got, err = s.Decode(ids[prompt+j])
		if err != nil {
			t.Fatalf("decode step %d: %v", j, err)
		}
		if pos := s.Position(); pos != prompt+j+1 {
			t.Fatalf("position after decode step %d is %d, want %d", j, pos, prompt+j+1)
		}
		readRow()
		for i, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("decode step %d logit %d is not finite", j, i)
			}
		}
		if top1(got) != top1(want) {
			t.Fatalf("decode step %d top-1 %d, control %d", j, top1(got), top1(want))
		}
		if diff, equal := compareControlRow(got, want); !equal &&
			diff > moeDecodeControlTolerance {
			t.Fatalf("decode step %d logit diff %.6g exceeds %.6g", j, diff,
				moeDecodeControlTolerance)
		}
	}
}

// compareControlRow returns the maximum absolute difference and whether the
// rows are bitwise equal. Signed zeros compare equal as values but not as bits;
// the bitwise result is what gates arithmetic-preserving changes.
func compareControlRow(got, want []float32) (float64, bool) {
	if len(got) != len(want) {
		return math.Inf(1), false
	}
	diff := 0.0
	equal := true
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			equal = false
		}
		d := math.Abs(float64(got[i] - want[i]))
		if d > diff || math.IsNaN(d) {
			diff = d
		}
	}
	return diff, equal
}

// TestGPUMoEGoldenDecodePrefixes generates greedy continuations after the fixed
// release prefixes. The expected tokens are recorded continuations, not forced
// inputs, so the test checks the model arithmetic through the shipping decode
// path at each length.
func TestGPUMoEGoldenDecodePrefixes(t *testing.T) {
	reader, tok, model := openMoEModel(t)
	const length = 8320
	ids, err := tok.Encode(strings.Repeat(moePrefillBenchmarkText, length))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < length {
		t.Fatalf("benchmark prompt produced %d tokens, need %d", len(ids), length)
	}
	ids = ids[:length]
	const continuation = 4
	want := map[int][]int32{
		512:  {13477, 37550, 33075, 888},
		2048: {5388, 13, 561, 3841},
		8192: {13477, 37550, 33075, 888},
	}
	rt, err := newRuntime(reader, model, 8192+continuation,
		Options{KVType: tensor.F16, KVTypeSet: true})
	if err != nil {
		t.Fatalf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	if err := rt.SetPrefillChunk(512); err != nil {
		t.Fatal(err)
	}
	s := rt.session
	for _, prompt := range []int{512, 2048, 8192} {
		if err := s.Reset(); err != nil {
			t.Fatal(err)
		}
		logits, err := s.Prefill(ids[:prompt], true)
		if err != nil {
			t.Fatalf("prefix %d: %v", prompt, err)
		}
		tokens := make([]int32, 0, continuation)
		for range continuation {
			tokens = append(tokens, int32(top1(logits)))
			if len(tokens) == continuation {
				break
			}
			logits, err = s.Decode(tokens[len(tokens)-1])
			if err != nil {
				t.Fatalf("prefix %d decode: %v", prompt, err)
			}
		}
		if !equalTokens(tokens, want[prompt]) {
			t.Fatalf("prefix %d greedy tokens %v, want %v", prompt, tokens, want[prompt])
		}
		t.Logf("prefix %d greedy tokens %v", prompt, tokens)
	}
}

//go:build hip && linux

package hip

import (
	"errors"
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

func TestAddGemvRejectsUndersizedBuffer(t *testing.T) {
	dev := openTestDevice(t)
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	w, err := dev.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	x, err := dev.Alloc(256 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	out, err := dev.Alloc(4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err := plan.AddGemv(w, x, out, 256, 4, tensor.Q4_K); err == nil {
		t.Fatal("undersized weight buffer was accepted")
	}
	if plan.Ops() != 0 {
		t.Fatalf("rejected operation was added: %d", plan.Ops())
	}
}

func TestAttentionHeadDimUnsupported(t *testing.T) {
	var plan Plan
	cases := []struct {
		name string
		err  error
	}{
		{
			name: "prefill",
			err:  plan.AddAttentionPrefillAt(nil, 0, nil, nil, nil, 0, 1, 1, 257, 257, 1, 1, 1),
		},
		{
			name: "paged decode",
			err:  plan.AddAttentionPagedAt(nil, nil, nil, nil, nil, nil, 1, 1, 257, 257, 1, 1, 1, 1),
		},
		{
			name: "paged prefill",
			err:  plan.AddAttentionPrefillPagedAt(nil, 0, nil, nil, nil, 0, nil, 1, 1, 257, 257, 1, 1, 1, 1),
		},
	}
	for _, tc := range cases {
		if !errors.Is(tc.err, ErrUnsupported) {
			t.Errorf("%s: error = %v, want ErrUnsupported", tc.name, tc.err)
		}
	}
}

func TestSelectPrefillPath(t *testing.T) {
	cases := []struct {
		name      string
		dtype     tensor.DType
		rows      int64
		nIn, nOut int64
		want      PrefillPath
	}{
		{"tiny batch", tensor.Q4_K, 1, 2560, 9216, PrefillBatchedGemv},
		{"q4 wide short", tensor.Q4_K, 4, 2560, 9216, PrefillWMMA},
		{"q4 tall short", tensor.Q4_K, 4, 9216, 2560, PrefillBatchedGemv},
		{"q4 tall batch8", tensor.Q4_K, 8, 9216, 2560, PrefillWMMA},
		{"q6 wide batch32", tensor.Q6_K, 32, 2560, 8192, PrefillWMMA},
		{"f16 small", tensor.F16, 2, 2560, 9216, PrefillBatchedGemv},
		{"f16 tall rocblas", tensor.F16, 4, 9216, 2560, PrefillBLAS},
		{"f16 wide batch4", tensor.F16, 4, 2560, 9216, PrefillBatchedGemv},
		{"f16 wide wmma", tensor.F16, 8, 2560, 9216, PrefillWMMA},
		{"f16 tall large wmma", tensor.F16, 128, 9216, 2560, PrefillWMMA},
		{"f16 wide large rocblas", tensor.F16, 128, 2560, 9216, PrefillBLAS},
		{"f32 wide", tensor.F32, 8, 2560, 9216, PrefillWMMA},
	}
	for _, c := range cases {
		if got := SelectPrefillPath(c.dtype, c.rows, c.nIn, c.nOut); got != c.want {
			t.Errorf("%s: got path %d, want %d", c.name, got, c.want)
		}
	}
}

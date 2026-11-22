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
	if plan.OpCount() != 0 {
		t.Fatalf("rejected operation was added: %d", plan.OpCount())
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
		{"q4k wide 128", tensor.Q4_K, 128, 2560, 9216, PrefillMMQ},
		{"q4k wide 256", tensor.Q4_K, 256, 2560, 9216, PrefillMMQ},
		{"q4k wide 8", tensor.Q4_K, 8, 2560, 9216, PrefillWMMA},
		{"q6k tall 256", tensor.Q6_K, 256, 9216, 2560, PrefillMMQ},
		{"q6k tall 512", tensor.Q6_K, 512, 9216, 2560, PrefillMMQ},
		{"q4k wide 64", tensor.Q4_K, 64, 2560, 9216, PrefillWMMA},
		{"q4k narrow out 256", tensor.Q4_K, 256, 2560, 32, PrefillWMMA},
		{"f16 small", tensor.F16, 2, 2560, 9216, PrefillBatchedGemv},
		{"f16 tall rocblas", tensor.F16, 4, 9216, 2560, PrefillBLAS},
		{"f16 wide batch4", tensor.F16, 4, 2560, 9216, PrefillBatchedGemv},
		{"f16 wide wmma", tensor.F16, 8, 2560, 9216, PrefillWMMA},
		{"f16 tall large wmma", tensor.F16, 128, 9216, 2560, PrefillWMMA},
		{"f16 wide large rocblas", tensor.F16, 128, 2560, 9216, PrefillBLAS},
		{"f32 wide", tensor.F32, 8, 2560, 9216, PrefillWMMA},
	}
	for _, c := range cases {
		got := SelectPrefillPath(c.dtype, c.rows, c.nIn, c.nOut)
		t.Logf("%s: dtype=%v rows=%d K=%d N=%d path=%s", c.name, c.dtype,
			c.rows, c.nIn, c.nOut, got)
		if got != c.want {
			t.Errorf("%s: got path %d, want %d", c.name, got, c.want)
		}
	}
}

func TestSelectMoEPrefillPath(t *testing.T) {
	cases := []struct {
		name      string
		dtype     tensor.DType
		rows      int64
		used      int64
		experts   int64
		nIn, nOut int64
		allowMMQ  bool
		want      MoEPrefillPath
		wantTile  int64
	}{
		// The opt-in is the only thing that enables the WMMA route, so a
		// K-quant MoE chunk keeps the FP32 grouped path by default.
		{"q4k chunk default", tensor.Q4_K, 512, 8, 256, 2048, 512, false,
			MoEPrefillF32Grouped, 0},
		{"q4k chunk opt-in", tensor.Q4_K, 512, 8, 256, 2048, 512, true,
			MoEPrefillMMQ, 16},
		{"q6k chunk 2048", tensor.Q6_K, 2048, 8, 256, 2048, 512, true,
			MoEPrefillMMQ, 64},
		{"q4k short chunk", tensor.Q4_K, 16, 8, 256, 2048, 512, true,
			MoEPrefillMMQ, 16},
		{"q4k below minimum rows", tensor.Q4_K, 8, 8, 256, 2048, 512, true,
			MoEPrefillF32Grouped, 0},
		{"q4k odd K width", tensor.Q4_K, 512, 8, 256, 2000, 512, true,
			MoEPrefillF32Grouped, 0},
		// The IQ codebooks take the WMMA route once the opt-in is on, at every
		// dtype and at the tile width the average bucket derives.
		{"iq2xxs chunk", tensor.IQ2XXS, 512, 8, 256, 2048, 512, true,
			MoEPrefillMMQ, 16},
		{"iq2s chunk 2048", tensor.IQ2S, 2048, 8, 256, 2048, 512, true,
			MoEPrefillMMQ, 64},
		{"iq3s odd width", tensor.IQ3S, 512, 8, 256, 2048, 500, true,
			MoEPrefillF32Grouped, 0},
		{"iq2s short chunk", tensor.IQ2S, 16, 8, 256, 2048, 512, true,
			MoEPrefillMMQ, 16},
		// Without the opt-in every dtype keeps its released route.
		{"iq2xxs chunk default", tensor.IQ2XXS, 512, 8, 256, 2048, 512, false,
			MoEPrefillQ8, 0},
		// The cropped one-row FFN of the last prefill chunk keeps the FP32
		// grouped route; the quantized tile would be almost entirely padding.
		{"iq2xxs cropped row", tensor.IQ2XXS, 1, 8, 256, 2048, 512, true,
			MoEPrefillF32Grouped, 0},
		{"q4k cropped row", tensor.Q4_K, 1, 8, 256, 2048, 512, true,
			MoEPrefillF32Grouped, 0},
		{"f16 experts", tensor.F16, 512, 8, 256, 2048, 512, true,
			MoEPrefillF32Grouped, 0},
	}
	for _, c := range cases {
		got, tile := SelectMoEPrefillPath(c.dtype, c.rows, c.used, c.experts, c.nIn,
			c.nOut, c.allowMMQ)
		t.Logf("%s: path=%s tile=%d", c.name, got, tile)
		if got != c.want {
			t.Errorf("%s: got path %s, want %s", c.name, got, c.want)
			continue
		}
		if c.want == MoEPrefillMMQ && tile != c.wantTile {
			t.Errorf("%s: got tile %d, want %d", c.name, tile, c.wantTile)
		}
	}
}

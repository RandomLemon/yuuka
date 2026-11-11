//go:build hip && linux

package hip

import (
	"testing"

	"github.com/yuuka-ai/yuuka/tensor"
)

func TestSelectGemvPath(t *testing.T) {
	cases := []struct {
		dtype tensor.DType
		batch int
		want  GemvPath
	}{
		{tensor.Q4_K, 1, GemvPerRow},
		{tensor.Q4_K, 2, GemvBatched},
		{tensor.Q6_K, 2, GemvBatched},
		{tensor.Q5_K, 8, GemvBatched},
		{tensor.Q8_0, 2, GemvPerRow},
		{tensor.Q8_0, 4, GemvBatched},
		{tensor.F16, 2, GemvPerRow},
		{tensor.F16, 4, GemvBatched},
		{tensor.Q4_K, 17, GemvPerRow},
		{tensor.Q4_K, 0, GemvPerRow},
	}
	for _, tc := range cases {
		if got := SelectGemvPath(tc.dtype, tc.batch); got != tc.want {
			t.Fatalf("SelectGemvPath(%s, %d) = %v, want %v", tc.dtype, tc.batch, got, tc.want)
		}
	}
}

func TestAddGemvAutoBatchOne(t *testing.T) {
	dev := openTestDevice(t)
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	w, err := dev.Alloc(4 * 144)
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
	if err := plan.AddGemvAuto(w, x, out, 256, 4, 1, tensor.Q4_K); err != nil {
		t.Fatal(err)
	}
	if plan.Ops() != 1 {
		t.Fatalf("batch one op count %d", plan.Ops())
	}
}

func TestAddGemvAutoDispatchAndFallback(t *testing.T) {
	dev := openTestDevice(t)
	w, err := dev.Alloc(4 * 144)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	x, err := dev.Alloc(17 * 256 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	out, err := dev.Alloc(17 * 4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	cases := []struct {
		name  string
		batch int64
		ops   int
	}{
		{"batched", 2, 1},
		{"above batch limit", 17, 17},
	}
	for _, tc := range cases {
		plan, err := dev.NewPlan()
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.AddGemvAuto(w, x, out, 256, 4, tc.batch, tensor.Q4_K); err != nil {
			plan.Close()
			t.Fatalf("%s: %v", tc.name, err)
		}
		if plan.Ops() != tc.ops {
			plan.Close()
			t.Fatalf("%s: op count %d, want %d", tc.name, plan.Ops(), tc.ops)
		}
		plan.Close()
	}
}

func TestAddGemvAutoRejectsInvalidInputs(t *testing.T) {
	dev := openTestDevice(t)
	plan, err := dev.NewPlan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	w, err := dev.Alloc(4 * 144)
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

	cases := []struct {
		name  string
		batch int64
		nIn   int64
		dtype tensor.DType
	}{
		{"zero batch", 0, 256, tensor.Q4_K},
		{"negative batch", -1, 256, tensor.Q4_K},
		{"misaligned width", 1, 100, tensor.Q4_K},
		{"unsupported dtype", 1, 256, tensor.Q4_0},
	}
	for _, tc := range cases {
		if err := plan.AddGemvAuto(w, x, out, tc.nIn, 4, tc.batch, tc.dtype); err == nil {
			t.Fatalf("%s was accepted", tc.name)
		}
	}
	if plan.Ops() != 0 {
		t.Fatalf("rejected operations were added: %d", plan.Ops())
	}
}

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

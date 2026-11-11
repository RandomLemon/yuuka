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
	w, err := dev.Alloc(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	x, err := dev.Alloc(256)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	out, err := dev.Alloc(256)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err := plan.AddGemvAuto(w, x, out, 64, 4, 1, tensor.Q4_K); err != nil {
		t.Fatal(err)
	}
	if plan.Ops() != 1 {
		t.Fatalf("batch one op count %d", plan.Ops())
	}
}

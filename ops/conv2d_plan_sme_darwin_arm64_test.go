//go:build darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: the selection test
// of Conv2DPlan's SME2 path (PLAN.md 2.2). It runs on darwin only, where
// tensor sets sme.GEMM; GitHub's macOS runners have no SME2 and skip it.

package ops

import (
	"fmt"
	"os"
	"syscall"
	"testing"

	"github.com/cwbudde/algo-tensor/internal/sme"
)

// TestConv2DPlanSMESelected checks that, whenever the CPU reports SME2
// (hw.optional.arm.FEAT_SME2, read here independently of tensor), sme.GEMM
// is set and, with the default thresholds, the plans of basic-pitch's
// conv2d_1, conv2d_2 and conv2d_4 pack their weights for it, while
// basic-pitch's other layers, go-whistle's stem and the shapes of
// BenchmarkConv2DRouting do not. Like tensor's TestGemmSMESelected it
// fails, rather than skips, when the CPU reports SME2 but the path is not
// selected (with CI set, noSMEEnv does not skip it), and a CPU without
// SME2 skips it.
func TestConv2DPlanSMESelected(t *testing.T) {
	ci := os.Getenv("CI")

	v, err := syscall.Sysctl("hw.optional.arm.FEAT_SME2")
	if err != nil || len(v) == 0 || v[0] != 1 {
		t.Skipf("CI=%q: the CPU does not report SME2 (sysctl %q, %v), so Conv2DPlan runs NEON", ci, v, err)
	}

	if os.Getenv(noSMEEnv) != "" && ci == "" {
		t.Skipf("%s is set", noSMEEnv)
	}

	if sme.GEMM == nil {
		t.Fatalf("CI=%q: the CPU reports SME2, but sme.GEMM is nil", ci)
	}

	takes := map[string]bool{"conv2d_1": true, "conv2d_2": true, "conv2d_4": true}

	type layer struct {
		name         string
		h, w         int
		kh, kw, c, o int
		p            Conv2DParams
		want         bool
	}

	layers := make([]layer, 0, len(basicPitchLayers)+len(benchWhistleStem)+3*6*5)

	for _, l := range basicPitchLayers {
		layers = append(layers, layer{"basic-pitch " + l.name, l.h, l.w, l.kh, l.kw, l.c, l.cout, l.params(), takes[l.name]})
	}

	for _, l := range benchWhistleStem {
		layers = append(layers, layer{"go-whistle stem " + l.name, l.h, l.w, l.kh, l.kw, l.c, l.cout, l.p, false})
	}

	// BenchmarkConv2DRouting's shapes.
	for _, hw := range []int{4, 8, 16} {
		for _, kc := range [][2]int{{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 9}, {3, 4}} {
			for _, o := range []int{1, 8, 32, 128, 255} {
				name := fmt.Sprintf("routing out=%dx%d/k=%d/o=%d", hw, hw, kc[0]*kc[0]*kc[1], o)
				layers = append(layers, layer{name, hw, hw, kc[0], kc[0], kc[1], o, samePadParams(hw, hw, kc[0], kc[0], 1, 1), false})
			}
		}
	}

	for _, l := range layers {
		kernel := mustTensorT(t, make([]float32, l.kh*l.kw*l.c*l.o), []int64{int64(l.kh), int64(l.kw), int64(l.c), int64(l.o)})

		plan, err := NewConv2DPlan(kernel, nil, l.h, l.w, l.p)
		if err != nil {
			t.Fatalf("%s: %v", l.name, err)
		}

		if got := plan.pb != nil; got != l.want {
			t.Errorf("CI=%q: %s (k=%d, outC=%d, %d×%d positions): packed for sme.GEMM %t, want %t",
				ci, l.name, plan.k, plan.outC, plan.outH, plan.outW, got, l.want)
		}
	}
}

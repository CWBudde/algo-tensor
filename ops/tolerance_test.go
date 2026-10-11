// From github.com/cwbudde/go-whistle internal/ops/tolerance_test.go @ 8926ee3;
// go-whistle's own, not in go-pocket-tts.
// algo-tensor change: pins all ten entries, the layer_norm, mlp and
// convtranspose1d ones included again.
// algo-tensor change: pins the conv2d entry too, all eleven.

package ops

import (
	"maps"
	"testing"
)

// TestKernelTolerancesPinned pins all eleven tolerances: the ten copied from
// go-pocket-tts internal/runtime/ops/tolerance.go @ 9faba38 and
// algo-tensor's conv2d. Any change must be a decision recorded with a note
// (AGENTS.md), not drift.
func TestKernelTolerancesPinned(t *testing.T) {
	want := map[string]Tolerance{
		"matmul":          {Abs: 1e-4, Rel: 1e-4},
		"linear":          {Abs: 1e-4, Rel: 1e-4},
		"softmax":         {Abs: 1e-4, Rel: 1e-4},
		"layer_norm":      {Abs: 1e-4, Rel: 1e-4},
		"causal_mask":     {Abs: 0, Rel: 0},
		"rope":            {Abs: 2e-4, Rel: 2e-4},
		"attention":       {Abs: 2e-4, Rel: 2e-4},
		"mlp":             {Abs: 2e-4, Rel: 2e-4},
		"conv1d":          {Abs: 2e-4, Rel: 2e-4},
		"conv2d":          {Abs: 2e-4, Rel: 2e-4},
		"convtranspose1d": {Abs: 2e-4, Rel: 2e-4},
	}

	if !maps.Equal(KernelTolerances, want) {
		t.Fatalf("KernelTolerances = %v, want %v", KernelTolerances, want)
	}
}

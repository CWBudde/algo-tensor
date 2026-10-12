//go:build !darwin || !arm64 || purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: sme.GEMM stays nil
// where gemmSME does not run (PLAN.md 2.2).

package tensor

import (
	"testing"

	"github.com/cwbudde/algo-tensor/internal/sme"
)

// TestGemmSMEPackedNotRegistered checks that tensor leaves sme.GEMM nil
// off darwin/arm64 and under purego, so that ops.Conv2DPlan stays on its
// NEON (or generic) GEMM there.
func TestGemmSMEPackedNotRegistered(t *testing.T) {
	t.Parallel()

	if sme.GEMM != nil {
		t.Fatal("sme.GEMM is set on a target without gemmSME")
	}
}

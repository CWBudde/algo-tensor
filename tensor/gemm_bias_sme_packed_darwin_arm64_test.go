//go:build darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: tests of gemmSME on
// operands packed by the caller (PLAN.md 2.2).

package tensor

import (
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/algo-tensor/internal/sme"
)

// smePackedRun runs sme.GEMM on o with b packed by sme.Pack and reports
// what it returned.
func smePackedRun(o *smeOperands) bool {
	pb := make([]float32, sme.PackedLen(o.n, o.k))
	sme.Pack(pb, o.b, o.n, o.k)

	return sme.GEMM(o.got, o.ldc, o.a, make([]float32, sme.PanelRows*sme.Steps(o.k)), pb, o.bias, o.m, o.n, o.k)
}

// TestGemmSMEPackedMatchesNEON runs sme.GEMM, set by this package, over
// the spike's sweep of TestGemmSMEMatchesNEON with b packed by sme.Pack
// and a as rows or packed as well, and compares every output, gap and
// guard element with gemmNEON4x4 bit for bit.
func TestGemmSMEPackedMatchesNEON(t *testing.T) {
	t.Parallel()
	skipWithoutSME(t)

	if sme.GEMM == nil {
		t.Fatal("gemmBiasUseSME is set, but sme.GEMM is nil")
	}

	rng := rand.New(rand.NewPCG(51, 52))
	shapes := 0

	for _, k := range []int{4, 5, 6, 7, 8, 9, 49, 200, 297, 672, 936} {
		for _, n := range []int{4, 8, 12, 16, 20, 32, 36, 64} {
			for m := 1; m <= 40; m++ {
				for _, ldc := range biasLdcs(n) {
					o := newSMEOperands(rng, m, n, k, ldc)
					o.neon()

					if !smePackedRun(o) {
						t.Fatalf("%s: sme.GEMM returned false on finite operands", o.name)
					}

					o.check(t, "sme.GEMM")

					shapes++
				}
			}
		}
	}

	t.Logf("%d shapes", shapes)
}

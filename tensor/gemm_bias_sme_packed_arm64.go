//go:build arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: gemmSME with the
// weights packed by the caller, for ops.Conv2DPlan (PLAN.md 2.2).

package tensor

import "github.com/cwbudde/algo-tensor/internal/sme"

// Sets sme.GEMM where MatMulTransBBias runs gemmSME, so that
// ops.Conv2DPlan reaches the kernel without an exported entry point.
var _ = registerSMEPacked()

func registerSMEPacked() bool {
	if gemmBiasUseSME {
		sme.GEMM = gemmBiasSMEPacked
	}

	return gemmBiasUseSME
}

// gemmBiasSMEPacked is sme.GEMM: gemmSME with b packed by sme.Pack into pb
// and a run panel by panel through po. It reports whether every output
// has gemmNEON4x4's bits; on false (a block given up after nine failed
// checks, or a NaN or infinite output) the caller computes them again,
// since it has no b in rows to do so here. The slices are bounded before
// their pointers reach the assembly.
func gemmBiasSMEPacked(c []float32, ldc int, a, po, pb, bias []float32, m, n, k int) bool {
	a = a[0 : m*k : len(a)]
	po = po[0 : sme.PanelRows*sme.Steps(k) : len(po)]
	pb = pb[0:sme.PackedLen(n, k):len(pb)]
	bias = bias[0:n:len(bias)]
	c = c[0 : (m-1)*ldc+n : len(c)]

	redos, done, nan := gemmSME(&c[0], ldc, &a[0], &pb[0], &bias[0], m, n, k, &po[0], &pb[0],
		0, 1, int(smeClobberK.Load()), int(smeClobberR.Load()))

	if redos > 0 {
		smeRedos.Add(int64(redos))
	}

	if done < m {
		smeFallbacks.Add(1)

		return false
	}

	if nan != 0 || (redos > 0 && !finiteOutputs(c, ldc, m, n)) {
		smeNaNs.Add(1)

		return false
	}

	return true
}

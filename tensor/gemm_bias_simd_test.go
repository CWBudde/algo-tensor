//go:build (amd64 || arm64) && !purego

// From github.com/cwbudde/algo-transcribe nn/gemm_simd_test.go @ 58c8332 (simdSweepK and
// checkKernelMatchesGo, which nn moved there from nn/gemm_arm64_test.go for its AVX2 tests).
// algo-tensor change: renamed to biasSIMDSweepK and checkBiasKernelMatchesGo, which takes
// the output row stride ldc; in a file of their own, built only with the kernels that use
// them, so that they are not unused under purego and js/wasm.

package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

// biasSIMDSweepK covers every k%4 and k%8 tail of the SIMD kernels (k >= 4)
// and every basic-pitch depth. The raw-kernel tests of each architecture
// use it.
var biasSIMDSweepK = []int{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 49, 200, 297, 672, 936}

// checkBiasKernelMatchesGo runs a raw SIMD kernel and the 4-wide Go tiles
// on the same data. They sum in different orders, so they agree to within
// twice the reference tolerance of checkBiasGemm, not bit for bit.
func checkBiasKernelMatchesGo(t *testing.T, rng *rand.Rand, name string, f biasGemmFunc, m, n, k, ldc int) {
	t.Helper()

	a := randGemmData(rng, m*k)
	b := randGemmData(rng, n*k)
	bias := randGemmData(rng, n)

	got := make([]float32, biasOutLen(m, n, ldc))
	want := make([]float32, len(got))

	f(got, ldc, a, b, bias, m, k)
	gemmBiasTiled(want, ldc, a, b, bias, m, k, 4)

	for i := range m {
		for j := range n {
			sumAbs := math.Abs(float64(bias[j]))
			for p := range k {
				sumAbs += math.Abs(float64(a[i*k+p]) * float64(b[j*k+p]))
			}

			g, w := float64(got[i*ldc+j]), float64(want[i*ldc+j])
			if d := math.Abs(g - w); d > 2*(1e-5*sumAbs+1e-6) {
				t.Fatalf("%s: SIMD c[%d,%d] = %v, Go tiles %v", name, i, j, g, w)
			}
		}
	}
}

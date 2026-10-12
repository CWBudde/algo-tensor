//go:build (!arm64 && !amd64) || purego

// From github.com/cwbudde/go-whistle internal/tensor/gemm_noasm.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/tensor/gemm_noarm64.go @ 9faba38.
// go-whistle change: renamed, || purego added to the build constraint.
// algo-tensor change: !amd64 added to the build constraint; amd64 has gemm_amd64.go (PLAN.md 1.4).

package tensor

func matMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	matMulTransBGeneric(c, ldc, a, b, m, n, k)
}

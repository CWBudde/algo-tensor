//go:build !arm64 || purego

// From github.com/cwbudde/go-whistle internal/tensor/gemm_noasm.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/tensor/gemm_noarm64.go @ 9faba38.
// go-whistle change: renamed, || purego added to the build constraint.

package tensor

func matMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	matMulTransBGeneric(c, ldc, a, b, m, n, k)
}

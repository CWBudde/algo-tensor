//go:build (!amd64 && !arm64) || purego

// From github.com/cwbudde/go-whistle internal/tensor/axpy_noasm.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/tensor/axpy_noamd64.go @ 9faba38.
// go-whistle change: renamed, || purego added to the build constraint.

package tensor

func axpyF32(dst []float32, alpha float32, src []float32) {
	axpyF32Generic(dst, alpha, src)
}

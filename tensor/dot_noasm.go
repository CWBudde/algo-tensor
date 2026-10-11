//go:build (!amd64 && !arm64) || purego

// From github.com/cwbudde/go-whistle internal/tensor/dot_noasm.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/tensor/dot_noasm.go @ 9faba38.
// go-whistle change: || purego added to the build constraint.

package tensor

func dotF32(a, b []float32) float32 {
	return dotF32Generic(a, b)
}

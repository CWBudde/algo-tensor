//go:build (!arm64 && !amd64) || purego

// From github.com/cwbudde/algo-transcribe nn/gemm_simd_stub.go @ 21ad164.
// algo-tensor change: renamed; haveSIMD and gemmSIMD renamed to gemmBiasHaveSIMD and
// gemmBiasSIMD, which takes the output row stride ldc; the build constraint of nn @ 58c8332,
// which adds the amd64 kernels.

package tensor

// gemmBiasHaveSIMD reports whether gemmBias has native SIMD kernels on
// this target.
const gemmBiasHaveSIMD = false

// gemmBiasSIMD has no kernels on this target and always leaves the work to
// the Go tiles.
func gemmBiasSIMD(_ []float32, _ int, _, _, _ []float32, _, _ int) bool { return false }

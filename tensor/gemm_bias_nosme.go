//go:build !arm64 || purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: no SME2 path off
// arm64 (PLAN.md 2.1).

package tensor

// matMulTransBBiasSME has no kernel on this target and always leaves the
// product to gemmBias.
func matMulTransBBiasSME(_ []float32, _ int, _, _, _ []float32, _, _, _ int) bool { return false }

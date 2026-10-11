//go:build !wasm

// From github.com/cwbudde/algo-transcribe nn/gemm_tile.go @ 21ad164.
// algo-tensor change: tileN renamed to gemmBiasTileN.

package tensor

// gemmBiasTileN is the output-channel width of the multi-channel
// MatMulTransBBias tile. Native targets have at least 16 (amd64) and
// usually 32 (arm64) floating-point registers and run the 4×4 tile
// fastest.
const gemmBiasTileN = 4

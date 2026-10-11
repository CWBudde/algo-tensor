//go:build wasm

// From github.com/cwbudde/algo-transcribe nn/gemm_tile_wasm.go @ 21ad164.
// algo-tensor change: tileN renamed to gemmBiasTileN.

package tensor

// gemmBiasTileN is the output-channel width of the multi-channel
// MatMulTransBBias tile. Go's wasm backend has 16 float32 registers, so the
// 4×4 tile spills and the 4×2 tile runs more than twice as fast.
const gemmBiasTileN = 2

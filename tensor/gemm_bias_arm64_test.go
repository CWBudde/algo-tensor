//go:build arm64 && !purego

// From github.com/cwbudde/algo-transcribe nn/gemm_arm64_test.go @ 21ad164.
// algo-tensor change: the kernels take the output row stride ldc, and both direct tests
// also run with ldc > n (gemmNEON8x1 with ldc > 1: its lane-by-lane store), gaps checked
// by checkBiasGemm; checkGemm renamed to checkBiasGemm; neonSweepK and checkNEONMatchesGo
// moved to gemm_bias_simd_test.go as biasSIMDSweepK and checkBiasKernelMatchesGo, as nn @
// 58c8332 moved them to gemm_simd_test.go.

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// TestGemmNEON4x4Direct calls the multi-channel assembly entry point
// directly, without gemmBiasSIMD, over every row remainder of its 4×4 and
// 1×4 tiles, with ldc = n and ldc > n, and compares it with the float64
// reference and the Go tiles.
func TestGemmNEON4x4Direct(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(7, 8))

	f := func(c []float32, ldc int, a, b, bias []float32, m, k int) {
		gemmNEON4x4(&c[0], &a[0], &b[0], &bias[0], m, len(bias), k, ldc)
	}

	for _, k := range biasSIMDSweepK {
		for m := 1; m <= 17; m++ {
			for _, n := range []int{4, 8, 12, 32} {
				for _, ldc := range []int{n, n + 1, n + 3} {
					name := fmt.Sprintf("k%d/m%d/n%d/ldc%d", k, m, n, ldc)
					checkBiasGemm(t, rng, name, f, m, n, k, ldc)
					checkBiasKernelMatchesGo(t, rng, name, f, m, n, k, ldc)
				}
			}
		}
	}
}

// TestGemmNEON8x1Direct calls the single-channel assembly entry point
// directly over every row remainder of its 8×1 tiles and 1×1 loop, with
// the contiguous store (ldc = 1) and the strided one (ldc > 1).
func TestGemmNEON8x1Direct(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(9, 10))

	f := func(c []float32, ldc int, a, b, bias []float32, m, k int) {
		gemmNEON8x1(&c[0], &a[0], &b[0], &bias[0], m, k, ldc)
	}

	for _, k := range biasSIMDSweepK {
		for m := 1; m <= 25; m++ {
			for _, ldc := range []int{1, 2, 3, 4, 7, 33} {
				name := fmt.Sprintf("k%d/m%d/ldc%d", k, m, ldc)
				checkBiasGemm(t, rng, name, f, m, 1, k, ldc)
				checkBiasKernelMatchesGo(t, rng, name, f, m, 1, k, ldc)
			}
		}
	}
}

// TestGemmNEONMatchesNN checks that a stride does not change the
// arithmetic: every output equals, bit for bit, the one the kernel
// computes for the same row with ldc = n.
func TestGemmNEONMatchesNN(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(11, 12))

	for _, k := range []int{4, 7, 13, 297} {
		for _, n := range []int{1, 4, 32} {
			m := 19
			a := randGemmData(rng, m*k)
			b := randGemmData(rng, n*k)
			bias := randGemmData(rng, n)

			want := make([]float32, m*n)
			if !gemmBiasSIMD(want, n, a, b, bias, m, k) {
				t.Fatalf("k%d/n%d: gemmBiasSIMD declined", k, n)
			}

			ldc := n + 5
			got := make([]float32, (m-1)*ldc+n)

			if !gemmBiasSIMD(got, ldc, a, b, bias, m, k) {
				t.Fatalf("k%d/n%d/ldc%d: gemmBiasSIMD declined", k, n, ldc)
			}

			for i := range m {
				for j := range n {
					if math.Float32bits(got[i*ldc+j]) != math.Float32bits(want[i*n+j]) {
						t.Fatalf("k%d/n%d: c[%d,%d] = %v with ldc %d, %v with ldc n", k, n, i, j, got[i*ldc+j], ldc, want[i*n+j])
					}
				}
			}
		}
	}
}

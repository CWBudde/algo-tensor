//go:build amd64 && !purego

// From github.com/cwbudde/algo-transcribe nn/gemm_amd64_test.go @ 58c8332.
// algo-tensor change: the kernels take the output row stride ldc, and the direct tests
// also run with ldc > n (gemmAVX2N1 with ldc > 1: its lane-by-lane store), gaps checked
// by checkBiasGemm; avx2SweepK, simdSweepK, haveSIMD, gemmSIMD, checkGemm,
// checkKernelMatchesGo, gemmTiled, gemmRandSlice and bpGemmShapes renamed to
// biasAVX2SweepK, biasSIMDSweepK, gemmBiasHaveSIMD, gemmBiasSIMD, checkBiasGemm,
// checkBiasKernelMatchesGo, gemmBiasTiled, randGemmData and biasBasicPitchShapes;
// TestGemmAVX2MaskedLanes also runs with ldc = n+3, whose gaps must keep gemmSentinel,
// and calls the kernels through gemmAVX2Direct; TestGemmAVX2Selected names the shape,
// not the layer, in its message; TestGemmAVX2MatchesNN, after TestGemmNEONMatchesNN,
// added.

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"golang.org/x/sys/cpu"
)

// biasAVX2SweepK adds the depths below 4, which gemmBiasSIMD declines but
// the kernels handle (the whole row is one masked step), to the common
// sweep.
var biasAVX2SweepK = append([]int{1, 2, 3}, biasSIMDSweepK...)

// skipWithoutAVX2 skips a test that calls the AVX2 kernels directly on a
// CPU that would fault on them.
func skipWithoutAVX2(t *testing.T) {
	t.Helper()

	if !gemmBiasHaveSIMD {
		t.Skip("the CPU lacks AVX2 or FMA")
	}
}

// gemmAVX2Direct calls the AVX2 kernel for n = len(bias) directly, without
// gemmBiasSIMD and its checks.
func gemmAVX2Direct(c []float32, ldc int, a, b, bias []float32, m, k int) {
	if len(bias) == 1 {
		gemmAVX2N1(&c[0], &a[0], &b[0], &bias[0], m, k, ldc)
	} else {
		gemmAVX2N4(&c[0], &a[0], &b[0], &bias[0], m, len(bias), k, ldc)
	}
}

// TestGemmAVX2N4Direct calls the multi-channel assembly entry point
// directly, without gemmBiasSIMD, over every row remainder of its 2×4 and
// 1×4 tiles, with ldc = n and ldc > n, and compares it with the float64
// reference and the Go tiles.
func TestGemmAVX2N4Direct(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(15, 16))

	f := func(c []float32, ldc int, a, b, bias []float32, m, k int) {
		gemmAVX2N4(&c[0], &a[0], &b[0], &bias[0], m, len(bias), k, ldc)
	}

	for _, k := range biasAVX2SweepK {
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

// TestGemmAVX2N1Direct calls the single-channel assembly entry point
// directly over every row remainder of its 8×1 tiles and 1×1 loop, with
// the contiguous store (ldc = 1) and the strided one (ldc > 1).
func TestGemmAVX2N1Direct(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(17, 18))

	f := func(c []float32, ldc int, a, b, bias []float32, m, k int) {
		gemmAVX2N1(&c[0], &a[0], &b[0], &bias[0], m, k, ldc)
	}

	for _, k := range biasAVX2SweepK {
		for m := 1; m <= 25; m++ {
			for _, ldc := range []int{1, 2, 3, 4, 7, 33} {
				name := fmt.Sprintf("k%d/m%d/ldc%d", k, m, ldc)
				checkBiasGemm(t, rng, name, f, m, 1, k, ldc)
				checkBiasKernelMatchesGo(t, rng, name, f, m, 1, k, ldc)
			}
		}
	}
}

// TestGemmAVX2MatchesNN checks that a stride does not change the
// arithmetic: every output equals, bit for bit, the one the kernel
// computes for the same row with ldc = n. m = 19 runs both the 8×1 tiles
// and the 1×1 loop, and both the 2×4 and the 1×4 tiles.
func TestGemmAVX2MatchesNN(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(21, 22))

	for _, k := range []int{1, 3, 4, 7, 13, 297} {
		for _, n := range []int{1, 4, 32} {
			m := 19
			a := randGemmData(rng, m*k)
			b := randGemmData(rng, n*k)
			bias := randGemmData(rng, n)

			want := make([]float32, m*n)
			gemmAVX2Direct(want, n, a, b, bias, m, k)

			for _, ldc := range []int{n + 1, n + 5} {
				got := make([]float32, biasOutLen(m, n, ldc))
				gemmAVX2Direct(got, ldc, a, b, bias, m, k)

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
}

// TestGemmAVX2MaskedLanes checks that the masked k%8 tail ignores the
// lanes past the end of a row. The tail of every row but the last overlaps
// the next row, and the tail of the last row the memory after the slice,
// which is NaN here; every output must still be finite and equal the Go
// tiles', and the gaps between the output rows (ldc > n) must keep
// gemmSentinel.
func TestGemmAVX2MaskedLanes(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(19, 20))
	nan := float32(math.NaN())

	for _, n := range []int{1, 4} {
		for _, k := range []int{1, 5, 7, 9, 13, 49} {
			for _, m := range []int{1, 2, 3, 8, 9} {
				for _, ldc := range []int{n, n + 3} {
					aBuf := make([]float32, m*k+8)
					bBuf := make([]float32, n*k+8)

					for i := range aBuf {
						aBuf[i] = nan
					}

					for i := range bBuf {
						bBuf[i] = nan
					}

					a := aBuf[:m*k]
					b := bBuf[:n*k]
					copy(a, randGemmData(rng, m*k))
					copy(b, randGemmData(rng, n*k))

					bias := randGemmData(rng, n)

					got := make([]float32, biasOutLen(m, n, ldc))
					want := make([]float32, len(got))

					for i := range got {
						got[i] = gemmSentinel
					}

					gemmAVX2Direct(got, ldc, a, b, bias, m, k)
					gemmBiasTiled(want, ldc, a, b, bias, m, k, 4)

					for i := range m {
						for j := range n {
							v, w := got[i*ldc+j], want[i*ldc+j]
							if math.IsNaN(float64(v)) || math.Abs(float64(v-w)) > 1e-4 {
								t.Fatalf("n%d/k%d/m%d/ldc%d: c[%d,%d] = %v, want %v", n, k, m, ldc, i, j, v, w)
							}
						}

						for g := i*ldc + n; g < min((i+1)*ldc, len(got)); g++ {
							if got[g] != gemmSentinel {
								t.Fatalf("n%d/k%d/m%d/ldc%d: gap c[%d] = %v was overwritten", n, k, m, ldc, g, got[g])
							}
						}
					}
				}
			}
		}
	}
}

// TestGemmAVX2Selected checks that gemmBias runs the AVX2 kernels for
// every basic-pitch layer shape whenever the CPU reports AVX2 and FMA. It
// fails, rather than skips, in that case, so the ubuntu CI job (GitHub
// Actions sets CI=true, and its x86 runners have both features) provably
// exercises the kernels instead of silently testing the Go tiles twice. A
// CPU without the features skips it locally but fails it under CI, where
// the amd64 job is meant to run the kernels.
func TestGemmAVX2Selected(t *testing.T) {
	t.Parallel()

	ci := os.Getenv("CI")

	if ci != "" && (!cpu.X86.HasAVX2 || !cpu.X86.HasFMA) {
		t.Fatalf("CI=%q: the runner lacks AVX2 (%v) or FMA (%v), so the AVX2 kernels cannot be tested",
			ci, cpu.X86.HasAVX2, cpu.X86.HasFMA)
	}

	if !cpu.X86.HasAVX2 || !cpu.X86.HasFMA {
		t.Skipf("CI=%q: the CPU lacks AVX2 (%v) or FMA (%v), so gemmBias runs the Go tiles",
			ci, cpu.X86.HasAVX2, cpu.X86.HasFMA)
	}

	if !gemmBiasHaveSIMD {
		t.Fatalf("CI=%q: the CPU reports AVX2 and FMA, but gemmBiasHaveSIMD is false", ci)
	}

	for _, s := range biasBasicPitchShapes() {
		m, n, k := s[0], s[1], s[2]
		c := make([]float32, m*n)

		if !gemmBiasSIMD(c, n, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, k) {
			t.Errorf("CI=%q: the AVX2 kernel was not selected for m=%d n=%d k=%d", ci, m, n, k)
		}
	}
}

// From github.com/cwbudde/go-whistle internal/tensor/asm_test.go @ 8926ee3;
// go-whistle's own, not in go-pocket-tts.
// algo-tensor change: TestGemmBiasMatchesGeneric for MatMulTransBBias's NEON kernels.
// algo-tensor change: TestGemmBiasMatchesGeneric also for its AVX2 kernels.
// algo-tensor change: TestLinearDot3MatchesDotF32 for Linear's AVX2 kernel.

package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The tests in this file compare each dispatching kernel (dotF32, axpyF32,
// matMulTransB, gemmBias, linearDot3) with its pure-Go fallback, as AGENTS.md requires
// for every assembly routine. With the purego build tag, or on an
// architecture without assembly, both sides are the fallback and the tests
// pass trivially; on arm64 (NEON) and amd64 with AVX2+FMA they exercise the
// assembly.

// unitRoundoff is the float32 unit roundoff 2^-24.
const unitRoundoff = 1.0 / (1 << 24)

// asmLengths covers both sides of every SIMD cutover (8, 48), every tail of
// the 4-, 8- and 16-wide loops, and a few long vectors.
func asmLengths() []int {
	ns := make([]int, 0, 136)
	for n := range 130 {
		ns = append(ns, n)
	}

	return append(ns, 255, 256, 257, 1023, 1024, 4099)
}

// TestDotF32MatchesGeneric: the SIMD dot product sums in a different order,
// so it may differ from the fallback by at most the two summation error
// bounds, 2·γ_n·Σ|a_i·b_i| with γ_n ≈ n·u. Offsets 1–3 start the operands
// off any vector alignment.
func TestDotF32MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))

	for _, n := range asmLengths() {
		for off := range 4 {
			a := randGemmData(rng, n+off)[off:]
			b := randGemmData(rng, n+off)[off:]

			var mag float64
			for i := range n {
				mag += math.Abs(float64(a[i]) * float64(b[i]))
			}

			got := float64(dotF32(a, b))
			want := float64(dotF32Generic(a, b))

			if bound := 2*float64(n+1)*unitRoundoff*mag + 1e-30; math.Abs(got-want) > bound {
				t.Fatalf("n=%d off=%d: dotF32 = %v, generic %v (|diff| %g > %g)", n, off, got, want, math.Abs(got-want), bound)
			}
		}
	}
}

// TestAxpyF32MatchesGeneric: each element is one multiply-add, so a fused
// (FMA) kernel and the fallback differ by at most one rounding of the
// product and one of the sum. The slack after n must stay untouched.
func TestAxpyF32MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))

	const slack = 3

	for _, n := range asmLengths() {
		for off := range 4 {
			src := randGemmData(rng, n+off)[off:]
			base := randGemmData(rng, n+off+slack)[off:]
			alpha := rng.Float32()*4 - 2

			got := append([]float32(nil), base...)
			want := append([]float32(nil), base...)

			if n > 0 {
				axpyF32(got[:n], alpha, src)
				axpyF32Generic(want[:n], alpha, src)
			}

			for i := range n {
				prod := math.Abs(float64(alpha) * float64(src[i]))
				bound := 2*unitRoundoff*(prod+math.Abs(float64(base[i]))+math.Abs(float64(want[i]))) + 1e-30

				if d := math.Abs(float64(got[i]) - float64(want[i])); d > bound {
					t.Fatalf("n=%d off=%d: dst[%d] = %v, generic %v (|diff| %g > %g)", n, off, i, got[i], want[i], d, bound)
				}
			}

			for i := n; i < len(got); i++ {
				if got[i] != base[i] {
					t.Fatalf("n=%d off=%d: slack dst[%d] = %v was overwritten (was %v)", n, off, i, got[i], base[i])
				}
			}
		}
	}
}

// TestMatMulTransBMatchesGeneric compares the blocked kernel with the
// per-output fallback element by element, under the dot-product bound, and
// checks that the gaps between output rows stay untouched.
func TestMatMulTransBMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))

	for _, k := range []int{1, 2, 3, 4, 5, 8, 12, 13, 48, 64, 67, 512, 1031} {
		for _, mn := range [][2]int{{1, 1}, {1, 4}, {1, 7}, {4, 4}, {5, 9}, {8, 12}, {13, 3}, {6, 70}} {
			m, n := mn[0], mn[1]
			a := randGemmData(rng, m*k)
			b := randGemmData(rng, n*k)
			ldc := n + 2

			got := make([]float32, m*ldc)
			want := make([]float32, m*ldc)

			for i := range got {
				got[i] = gemmSentinel
				want[i] = gemmSentinel
			}

			matMulTransB(got, ldc, a, b, m, n, k)
			matMulTransBGeneric(want, ldc, a, b, m, n, k)

			for i := range m {
				for j := range n {
					var mag float64
					for p := range k {
						mag += math.Abs(float64(a[i*k+p]) * float64(b[j*k+p]))
					}

					g, w := float64(got[i*ldc+j]), float64(want[i*ldc+j])
					if bound := 2*float64(k+1)*unitRoundoff*mag + 1e-30; math.Abs(g-w) > bound {
						t.Fatalf("m=%d n=%d k=%d: c[%d][%d] = %v, generic %v (|diff| %g > %g)", m, n, k, i, j, g, w, math.Abs(g-w), bound)
					}
				}

				for g := i*ldc + n; g < (i+1)*ldc; g++ {
					if got[g] != gemmSentinel {
						t.Fatalf("m=%d n=%d k=%d: gap c[%d] = %v was overwritten", m, n, k, g, got[g])
					}
				}
			}
		}
	}
}

// TestGemmBiasMatchesGeneric compares the dispatching gemmBias (the NEON
// or AVX2 kernels of gemmBiasSIMD where they apply) with the Go tiles
// element by element, under the dot-product bound with the bias as one
// more term, and checks that the gaps between output rows stay untouched.
func TestGemmBiasMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 18))

	for _, k := range []int{1, 2, 3, 4, 5, 8, 12, 13, 48, 64, 67, 512, 1031} {
		for _, mn := range [][2]int{{1, 1}, {1, 4}, {1, 7}, {4, 4}, {5, 9}, {8, 12}, {13, 3}, {6, 70}, {9, 1}, {17, 1}, {8, 32}, {3, 8}} {
			m, n := mn[0], mn[1]
			a := randGemmData(rng, m*k)
			b := randGemmData(rng, n*k)
			bias := randGemmData(rng, n)
			ldc := n + 2

			got := make([]float32, m*ldc)
			want := make([]float32, m*ldc)

			for i := range got {
				got[i] = gemmSentinel
				want[i] = gemmSentinel
			}

			gemmBias(got, ldc, a, b, bias, m, k)
			gemmBiasTiled(want, ldc, a, b, bias, m, k, gemmBiasTileN)

			for i := range m {
				for j := range n {
					mag := math.Abs(float64(bias[j]))
					for p := range k {
						mag += math.Abs(float64(a[i*k+p]) * float64(b[j*k+p]))
					}

					g, w := float64(got[i*ldc+j]), float64(want[i*ldc+j])
					if bound := 2*float64(k+1)*unitRoundoff*mag + 1e-30; math.Abs(g-w) > bound {
						t.Fatalf("m=%d n=%d k=%d: c[%d][%d] = %v, Go tiles %v (|diff| %g > %g)", m, n, k, i, j, g, w, math.Abs(g-w), bound)
					}
				}

				for g := i*ldc + n; g < (i+1)*ldc; g++ {
					if got[g] != gemmSentinel {
						t.Fatalf("m=%d n=%d k=%d: gap c[%d] = %v was overwritten", m, n, k, g, got[g])
					}
				}
			}
		}
	}
}

// TestLinearDot3MatchesDotF32: Linear's AVX2 kernel must give every output
// dotF32AVX2's bits, which is what dotF32 runs for 8 values or more with
// AVX2 and FMA, the only place Linear uses the kernel. Rows lie ldx and
// ldy apart and start off any vector alignment. Besides random rows, some
// hold infinities, subnormals, signed zeros and NaNs, and some hold only a
// NaN in x against a NaN in w at the same index, in the vector part or in
// the scalar tail, with a payload per row: which payload comes out shows
// which operand each FMA took as its second source. It skips where
// linearDot3 has no assembly.
func TestLinearDot3MatchesDotF32(t *testing.T) {
	if !linearDot3Asm {
		t.Skip("no Linear kernel in assembly on this configuration")
	}

	rng := rand.New(rand.NewPCG(19, 20))
	specials := []float32{
		math.Float32frombits(0x7fc00011), float32(math.Inf(1)), float32(math.Inf(-1)),
		math.Float32frombits(3), math.Float32frombits(0x807ffff0), float32(math.Copysign(0, -1)),
	}

	for _, n := range asmLengths() {
		if n < 8 {
			continue
		}

		for off := range 4 {
			for _, kind := range []string{"random", "specials", "NaN pair at n/2", "NaN pair at n-1"} {
				ldx, ldy := n+off, 5+off
				x := randGemmData(rng, 2*ldx+n+off)[off:]
				w := randGemmData(rng, n+off)[off:]

				switch kind {
				case "specials":
					for i, p := range []int{0, n / 3, n - 1, n / 2} {
						x[(i%3)*ldx+p] = specials[(i+off)%len(specials)]
						w[(p+7)%n] = specials[(i+off+2)%len(specials)]
					}
				case "NaN pair at n/2", "NaN pair at n-1":
					p := n / 2
					if kind == "NaN pair at n-1" {
						p = n - 1
					}

					for r := range 3 {
						x[r*ldx+p] = math.Float32frombits(0x7fc00020 + uint32(r))
					}

					w[p] = math.Float32frombits(0xffc00030)
				}

				y := make([]float32, 2*ldy+1)
				for i := range y {
					y[i] = gemmSentinel
				}

				linearDot3(y, ldy, x, ldx, w[:n])

				for i := range y {
					if i%ldy != 0 {
						if y[i] != gemmSentinel {
							t.Fatalf("n=%d off=%d: y[%d] = %v was overwritten", n, off, i, y[i])
						}

						continue
					}

					r := i / ldy
					want := dotF32(x[r*ldx:r*ldx+n], w[:n])

					if math.Float32bits(y[i]) != math.Float32bits(want) {
						t.Fatalf("n=%d off=%d %s: row %d = %v (%#08x), dotF32AVX2 %v (%#08x)",
							n, off, kind, r, y[i], math.Float32bits(y[i]), want, math.Float32bits(want))
					}
				}
			}
		}
	}
}

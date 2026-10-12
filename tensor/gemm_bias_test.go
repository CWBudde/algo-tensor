// From github.com/cwbudde/algo-transcribe nn/gemm_test.go and nn/gemm_simd_test.go @ 21ad164.
// algo-tensor change: both files in one; gemm, gemmTiled, gemmSIMD, haveSIMD, tileN and
// blockM renamed as in gemm_bias.go, and the helpers to biasSweepK, biasSweepN,
// biasGemmFunc, checkBiasGemm, biasSIMDCovers and biasBasicPitchShapes, with
// randGemmData and gemmSentinel from gemm_test.go for gemmRandSlice and 42; every shape
// also runs with an output row stride ldc = n+3 whose gaps must stay untouched; the
// basic-pitch shapes listed instead of taken from nn's layer table; the public
// MatMulTransBBias tests (nil bias, panics, empty products, allocations, worker counts)
// and BenchmarkMatMulTransBBias, after BenchmarkGemm, added.
// algo-tensor change: BenchmarkMatMulTransBBias also runs BenchmarkMatMulTransB's shapes
// with n a multiple of 4, and the allocation test skips the SME path's shapes under the
// race detector (PLAN.md 2.1).

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// biasSweepK and biasSweepN are the depths and channel counts of the GEMM
// sweeps: every k%4 and k%8 remainder, every basic-pitch depth (49, 200,
// 297, 672, 936), n = 1..9 for every channel remainder and n = 32 as in
// basic-pitch.
var (
	biasSweepK = []int{1, 2, 3, 5, 8, 49, 200, 297, 672, 936}
	biasSweepN = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 32}
)

// biasGemmFunc is the signature of gemmBias and its variants.
type biasGemmFunc func(c []float32, ldc int, a, b, bias []float32, m, k int)

// biasLdcs returns the output row strides every shape runs with: the
// contiguous n and a wider one with a gap after each row.
func biasLdcs(n int) []int { return []int{n, n + 3} }

// biasSIMDCovers reports whether gemmBiasSIMD is expected to take a shape.
func biasSIMDCovers(n, k int) bool {
	return gemmBiasHaveSIMD && k >= 4 && (n == 1 || n%4 == 0)
}

// biasOutLen is the length of an m×n output with row stride ldc.
func biasOutLen(m, n, ldc int) int {
	if m == 0 {
		return 0
	}

	return (m-1)*ldc + n
}

// biasBasicPitchShapes returns (m, n, k) of the gemmBias calls of the six
// basic-pitch convolutions: m = gemmBiasBlockM, n = cout, k = kh*kw*cin
// (conv2d_1, contours-reduced, conv2d_2, conv2d_3, conv2d_4, conv2d_5).
func biasBasicPitchShapes() [][3]int {
	return [][3]int{
		{gemmBiasBlockM, 8, 936},
		{gemmBiasBlockM, 1, 200},
		{gemmBiasBlockM, 32, 49},
		{gemmBiasBlockM, 1, 672},
		{gemmBiasBlockM, 32, 200},
		{gemmBiasBlockM, 1, 297},
	}
}

// TestGemmBiasTiled checks both Go tile widths on every architecture, and
// the SIMD kernels where the target has them, against a float64 reference,
// over sizes that hit every row and channel remainder of the 4×nr, 8×1 and
// SIMD tiles, with ldc = n and ldc > n. The SIMD variant runs only the
// shapes gemmBiasSIMD covers; the others fall back to the Go tiles
// (TestGemmBias checks that dispatch).
func TestGemmBiasTiled(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(3, 4))

	variants := []string{"nr4", "nr2"}
	if gemmBiasHaveSIMD {
		variants = append(variants, "simd")
	}

	for _, variant := range variants {
		var f biasGemmFunc

		switch variant {
		case "nr4", "nr2":
			nr := 4
			if variant == "nr2" {
				nr = 2
			}

			f = func(c []float32, ldc int, a, b, bias []float32, m, k int) {
				gemmBiasTiled(c, ldc, a, b, bias, m, k, nr)
			}
		case "simd":
			f = func(c []float32, ldc int, a, b, bias []float32, m, k int) {
				if !gemmBiasSIMD(c, ldc, a, b, bias, m, k) {
					t.Fatalf("gemmBiasSIMD declined m=%d n=%d k=%d", m, len(bias), k)
				}
			}
		}

		for _, k := range biasSweepK {
			for m := range 18 {
				for _, n := range biasSweepN {
					if variant == "simd" && !biasSIMDCovers(n, k) {
						continue
					}

					for _, ldc := range biasLdcs(n) {
						name := fmt.Sprintf("%s/k%d/m%d/n%d/ldc%d", variant, k, m, n, ldc)
						checkBiasGemm(t, rng, name, f, m, n, k, ldc)
					}
				}
			}
		}
	}
}

// checkBiasGemm runs f on random data and compares every output with a
// float64 reference. The gaps between the rows (ldc > n) and a guard
// element after the output must keep gemmSentinel.
func checkBiasGemm(t *testing.T, rng *rand.Rand, name string, f biasGemmFunc, m, n, k, ldc int) {
	t.Helper()

	a := randGemmData(rng, m*k)
	b := randGemmData(rng, n*k)
	bias := randGemmData(rng, n)

	size := biasOutLen(m, n, ldc)

	c := make([]float32, size+1) // guard element after the output
	for i := range c {
		c[i] = gemmSentinel
	}

	f(c[:size], ldc, a, b, bias, m, k)

	if c[size] != gemmSentinel {
		t.Fatalf("%s: wrote past the output", name)
	}

	checkBiasReference(t, name, c, ldc, a, b, bias, m, n, k)
}

// checkBiasReference compares the m×n output in c (row stride ldc) with a
// float64 reference, |got−want| ≤ 1e-5·Σ|terms| + 1e-6, and checks that the
// gaps between the rows still hold gemmSentinel.
func checkBiasReference(t *testing.T, name string, c []float32, ldc int, a, b, bias []float32, m, n, k int) {
	t.Helper()

	for i := range m {
		for j := range n {
			want, sumAbs := float64(bias[j]), math.Abs(float64(bias[j]))

			for p := range k {
				v := float64(a[i*k+p]) * float64(b[j*k+p])
				want += v
				sumAbs += math.Abs(v)
			}

			if got := float64(c[i*ldc+j]); math.Abs(got-want) > 1e-5*sumAbs+1e-6 {
				t.Fatalf("%s: c[%d,%d] = %v, want %v", name, i, j, got, want)
			}
		}

		if i == m-1 {
			break
		}

		for g := i*ldc + n; g < (i+1)*ldc; g++ {
			if c[g] != gemmSentinel {
				t.Fatalf("%s: gap c[%d] = %v was overwritten", name, g, c[g])
			}
		}
	}
}

func TestGemmBiasTileN(t *testing.T) {
	t.Parallel()

	if gemmBiasTileN != 4 && gemmBiasTileN != 2 {
		t.Fatalf("gemmBiasTileN = %d, want 4 or 2", gemmBiasTileN)
	}
}

// TestGemmBias checks the dispatching gemmBias, the SIMD kernels where they
// apply and the Go tiles elsewhere, against the float64 reference over the
// full sweep, and that gemmBiasSIMD takes exactly the shapes it should and
// never writes when it declines.
func TestGemmBias(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(5, 6))

	for _, k := range biasSweepK {
		for m := range 18 {
			for _, n := range biasSweepN {
				for _, ldc := range biasLdcs(n) {
					name := fmt.Sprintf("k%d/m%d/n%d/ldc%d", k, m, n, ldc)
					checkBiasGemm(t, rng, name, gemmBias, m, n, k, ldc)

					c := make([]float32, biasOutLen(m, n, ldc))
					for i := range c {
						c[i] = gemmSentinel
					}

					taken := gemmBiasSIMD(c, ldc, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, k)
					if want := biasSIMDCovers(n, k); taken != want {
						t.Fatalf("%s: gemmBiasSIMD = %v, want %v", name, taken, want)
					}

					if !taken {
						for i, v := range c {
							if v != gemmSentinel {
								t.Fatalf("%s: gemmBiasSIMD returned false but wrote c[%d]", name, i)
							}
						}
					}
				}
			}
		}
	}
}

// TestGemmBiasSIMDTails checks the SIMD kernels against the float64
// reference at the depths TestGemmBiasTiled's sweep lacks: together they
// give every k%8 remainder, so every k%4 tail mask and the single
// 4-element step of the NEON 8×1 kernel run, and every k%8 tail mask of
// the AVX2 kernels.
func TestGemmBiasSIMDTails(t *testing.T) {
	t.Parallel()

	if !gemmBiasHaveSIMD {
		t.Skip("no SIMD gemmBias on this target")
	}

	rng := rand.New(rand.NewPCG(7, 8))

	simd := func(c []float32, ldc int, a, b, bias []float32, m, k int) {
		if !gemmBiasSIMD(c, ldc, a, b, bias, m, k) {
			t.Fatalf("gemmBiasSIMD declined m=%d n=%d k=%d", m, len(bias), k)
		}
	}

	for _, k := range []int{4, 6, 7, 10, 11, 14, 15} {
		for m := range 18 {
			for _, n := range biasSweepN {
				if !biasSIMDCovers(n, k) {
					continue
				}

				for _, ldc := range biasLdcs(n) {
					checkBiasGemm(t, rng, fmt.Sprintf("k%d/m%d/n%d/ldc%d", k, m, n, ldc), simd, m, n, k, ldc)
				}
			}
		}
	}
}

// TestGemmBiasSIMDMatchesGo compares the SIMD kernels directly with the Go
// tiles of this target on random data.
func TestGemmBiasSIMDMatchesGo(t *testing.T) {
	t.Parallel()

	if !gemmBiasHaveSIMD {
		t.Skip("no SIMD gemmBias on this target")
	}

	rng := rand.New(rand.NewPCG(9, 10))

	shapes := append(biasBasicPitchShapes(), [][3]int{
		{17, 4, 4}, {5, 12, 7}, {9, 1, 13}, {3, 32, 297}, {16, 8, 936}, {1, 1, 5},
	}...)

	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		a := randGemmData(rng, m*k)
		b := randGemmData(rng, n*k)
		bias := randGemmData(rng, n)

		for _, ldc := range biasLdcs(n) {
			got := make([]float32, biasOutLen(m, n, ldc))
			want := make([]float32, len(got))

			if !gemmBiasSIMD(got, ldc, a, b, bias, m, k) {
				t.Fatalf("m%d/n%d/k%d/ldc%d: gemmBiasSIMD declined", m, n, k, ldc)
			}

			gemmBiasTiled(want, ldc, a, b, bias, m, k, gemmBiasTileN)

			for i := range m {
				for j := range n {
					sumAbs := math.Abs(float64(bias[j]))
					for p := range k {
						sumAbs += math.Abs(float64(a[i*k+p]) * float64(b[j*k+p]))
					}

					g, w := float64(got[i*ldc+j]), float64(want[i*ldc+j])
					if math.Abs(g-w) > 1e-5*sumAbs+1e-6 {
						t.Fatalf("m%d/n%d/k%d/ldc%d: c[%d,%d] = %v (SIMD), %v (Go)", m, n, k, ldc, i, j, g, w)
					}
				}
			}
		}
	}
}

// TestGemmBiasSIMDBasicPitchShapes checks that the SIMD path is taken for
// every basic-pitch layer on targets that have it, and never elsewhere.
func TestGemmBiasSIMDBasicPitchShapes(t *testing.T) {
	t.Parallel()

	for _, s := range biasBasicPitchShapes() {
		m, n, k := s[0], s[1], s[2]
		c := make([]float32, m*n)

		if got := gemmBiasSIMD(c, n, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, k); got != gemmBiasHaveSIMD {
			t.Errorf("m%d/n%d/k%d: gemmBiasSIMD = %v, want %v", m, n, k, got, gemmBiasHaveSIMD)
		}
	}
}

// TestGemmBiasDeterministic checks that two gemmBias calls on the same
// inputs give bit-identical results, for every basic-pitch shape and some
// remainders.
func TestGemmBiasDeterministic(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(11, 12))

	shapes := append(biasBasicPitchShapes(), [3]int{7, 4, 5}, [3]int{13, 1, 299}, [3]int{3, 12, 50})
	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		a := randGemmData(rng, m*k)
		b := randGemmData(rng, n*k)
		bias := randGemmData(rng, n)

		first := make([]float32, m*n)
		second := make([]float32, m*n)

		gemmBias(first, n, a, b, bias, m, k)
		gemmBias(second, n, a, b, bias, m, k)

		for i := range first {
			if math.Float32bits(first[i]) != math.Float32bits(second[i]) {
				t.Fatalf("m%d/n%d/k%d: c[%d] = %v, then %v", m, n, k, i, first[i], second[i])
			}
		}
	}
}

// TestGemmBiasTailInf checks that the masked k%4 tail of the SIMD kernels
// does not turn an infinity in the overlapping reload into NaN (0*Inf):
// the result must be +Inf as with the Go tiles. (That is the NEON tail; the
// masked k%8 step of the AVX2 kernels reloads nothing, and must give +Inf
// too.)
func TestGemmBiasTailInf(t *testing.T) {
	t.Parallel()

	for _, n := range []int{1, 4} {
		for _, k := range []int{5, 6, 7, 13} {
			for _, m := range []int{1, 4, 8, 9} {
				a := make([]float32, m*k)
				b := make([]float32, n*k)
				bias := make([]float32, n)

				for i := range a {
					a[i] = 1
				}

				for i := range b {
					b[i] = 1
				}

				// On NEON, element k-4 is summed in the last full step and
				// reloaded, masked, by the tail.
				for i := range m {
					a[i*k+k-4] = float32(math.Inf(1))
				}

				for _, f := range []biasGemmFunc{gemmBias, func(c []float32, ldc int, a, b, bias []float32, m, k int) {
					gemmBiasTiled(c, ldc, a, b, bias, m, k, gemmBiasTileN)
				}} {
					c := make([]float32, m*n)
					f(c, n, a, b, bias, m, k)

					for i, v := range c {
						if !math.IsInf(float64(v), 1) {
							t.Fatalf("n%d/k%d/m%d: c[%d] = %v, want +Inf", n, k, m, i, v)
						}
					}
				}
			}
		}
	}
}

// TestGemmBiasSIMDShortSlices checks that gemmBiasSIMD panics, rather than
// letting the assembly read or write past the end, when a slice is shorter
// than the shape needs, even if its capacity would suffice, and when ldc <
// n.
func TestGemmBiasSIMDShortSlices(t *testing.T) {
	t.Parallel()

	if !gemmBiasHaveSIMD {
		t.Skip("no SIMD gemmBias on this target")
	}

	const m, k = 8, 9

	type args struct {
		c, a, b []float32
		ldc, n  int
	}

	cases := map[string]func() args{
		"c": func() args {
			return args{make([]float32, m*4-1, m*4+16), make([]float32, m*k), make([]float32, 4*k), 4, 4}
		},
		"c ldc>n": func() args {
			return args{make([]float32, 7*7+3, 8*7), make([]float32, m*k), make([]float32, 4*k), 7, 4}
		},
		"c n=1 ldc>1": func() args {
			return args{make([]float32, 7*3, 8*3), make([]float32, m*k), make([]float32, k), 3, 1}
		},
		"a": func() args {
			return args{make([]float32, m*4), make([]float32, m*k-1, m*k+16), make([]float32, 4*k), 4, 4}
		},
		"b": func() args {
			return args{make([]float32, m*4), make([]float32, m*k), make([]float32, 4*k-1, 4*k+16), 4, 4}
		},
		"ldc<n": func() args {
			return args{make([]float32, m*4), make([]float32, m*k), make([]float32, 4*k), 3, 4}
		},
	}

	for name, mk := range cases {
		x := mk()

		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: short slice did not panic", name)
				}
			}()

			gemmBiasSIMD(x.c, x.ldc, x.a, x.b, make([]float32, x.n), m, k)
		}()
	}
}

// TestMatMulTransBBiasNilBias checks that a nil bias gives the same bits
// as an explicit zero bias, also beyond zeroBiasLen.
func TestMatMulTransBBiasNilBias(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))

	for _, s := range [][3]int{{1, 1, 5}, {9, 4, 13}, {17, 7, 3}, {8, 32, 49}, {3, zeroBiasLen + 1, 5}} {
		m, n, k := s[0], s[1], s[2]
		a := randGemmData(rng, m*k)
		b := randGemmData(rng, n*k)

		for _, ldc := range biasLdcs(n) {
			got := make([]float32, biasOutLen(m, n, ldc))
			want := make([]float32, len(got))

			MatMulTransBBias(got, ldc, a, b, nil, m, n, k)
			MatMulTransBBias(want, ldc, a, b, make([]float32, n), m, n, k)

			if !slices.EqualFunc(got, want, func(x, y float32) bool {
				return math.Float32bits(x) == math.Float32bits(y)
			}) {
				t.Fatalf("m%d/n%d/k%d/ldc%d: nil bias differs from a zero bias", m, n, k, ldc)
			}
		}
	}
}

// TestMatMulTransBBiasMatchesReference checks the public function, with
// and without bias, against the float64 reference, with gaps (ldc > n)
// that must stay untouched and rows beyond one block.
func TestMatMulTransBBiasMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))

	for _, k := range []int{1, 3, 4, 7, 49, 200} {
		for _, m := range []int{1, 7, 8, 9, 16, 23} {
			for _, n := range []int{1, 3, 4, 6, 8, 32} {
				for _, ldc := range biasLdcs(n) {
					for _, withBias := range []bool{true, false} {
						name := fmt.Sprintf("k%d/m%d/n%d/ldc%d/bias=%v", k, m, n, ldc, withBias)
						checkBiasGemm(t, rng, name, func(c []float32, ldc int, a, b, bias []float32, m, k int) {
							if !withBias {
								clear(bias)
								bias = nil
							}

							MatMulTransBBias(c, ldc, a, b, bias, m, n, k)
						}, m, n, k, ldc)
					}
				}
			}
		}
	}
}

// TestMatMulTransBBiasEmpty checks that m = 0 or n = 0 writes nothing and
// accepts empty operands, and that k = 0 gives the bias.
func TestMatMulTransBBiasEmpty(t *testing.T) {
	c := []float32{gemmSentinel, gemmSentinel}

	MatMulTransBBias(c, 1, nil, nil, nil, 0, 1, 4)
	MatMulTransBBias(c, 1, nil, nil, []float32{1}, 0, 1, 4)
	MatMulTransBBias(c, 0, nil, nil, nil, 1, 0, 4)
	MatMulTransBBias(nil, 0, nil, nil, nil, 3, 0, 4)

	if c[0] != gemmSentinel || c[1] != gemmSentinel {
		t.Fatalf("empty product wrote c = %v", c)
	}

	MatMulTransBBias(c, 1, nil, nil, []float32{2.5}, 2, 1, 0)

	if c[0] != 2.5 || c[1] != 2.5 {
		t.Fatalf("k=0: c = %v; want the bias 2.5", c)
	}

	MatMulTransBBias(c, 1, nil, nil, nil, 2, 1, 0)

	if c[0] != 0 || c[1] != 0 {
		t.Fatalf("k=0, nil bias: c = %v; want 0", c)
	}
}

func TestMatMulTransBBiasPanics(t *testing.T) {
	m, n, k := 4, 4, 8
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	c := make([]float32, m*n)
	bias := make([]float32, n)
	ldc := n + 3
	cw := make([]float32, (m-1)*ldc+n)

	cases := map[string]func(){
		"short a":          func() { MatMulTransBBias(c, n, a[:m*k-1], b, bias, m, n, k) },
		"short b":          func() { MatMulTransBBias(c, n, a, b[:n*k-1], bias, m, n, k) },
		"short c":          func() { MatMulTransBBias(c[:m*n-1], n, a, b, bias, m, n, k) },
		"short c, ldc > n": func() { MatMulTransBBias(cw[:len(cw)-1], ldc, a, b, bias, m, n, k) },
		"short bias":       func() { MatMulTransBBias(c, n, a, b, bias[:n-1], m, n, k) },
		"empty bias":       func() { MatMulTransBBias(c, n, a, b, []float32{}, m, n, k) },
		"ldc < n":          func() { MatMulTransBBias(c, n-1, a, b, bias, m, n, k) },
		"negative ldc":     func() { MatMulTransBBias(c, -1, a, b, bias, m, n, k) },
		"negative m":       func() { MatMulTransBBias(c, n, a, b, bias, -1, n, k) },
		"negative n":       func() { MatMulTransBBias(c, n, a, b, bias, m, -1, k) },
		"negative k":       func() { MatMulTransBBias(c, n, a, b, bias, m, n, -1) },
	}

	for name, call := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("no panic")
				}
			}()

			call()
		})
	}
}

// TestMatMulTransBBiasAllocs checks that MatMulTransBBias does not
// allocate on one worker, with and without bias, for small and split-sized
// products.
func TestMatMulTransBBiasAllocs(t *testing.T) {
	rng := rand.New(rand.NewPCG(25, 26))

	for _, s := range [][3]int{{8, 32, 49}, {9, 1, 297}, {37, 36, 1031}} {
		m, n, k := s[0], s[1], s[2]
		a := randGemmData(rng, m*k)
		b := randGemmData(rng, n*k)
		bias := randGemmData(rng, n)
		c := make([]float32, m*n)

		// The SME path's scratch comes from a sync.Pool, which drops items at
		// random under the race detector; TestMatMulTransBBiasSMEAllocs checks
		// it without.
		if raceEnabled && smeTakes(m, n, k) {
			continue
		}

		withWorkers(t, 1, func() {
			for _, bs := range [][]float32{bias, nil} {
				if allocs := testing.AllocsPerRun(10, func() {
					MatMulTransBBias(c, n, a, b, bs, m, n, k)
				}); allocs != 0 {
					t.Fatalf("m%d/n%d/k%d bias=%v: %v allocations per call", m, n, k, bs != nil, allocs)
				}
			}
		})
	}
}

// TestMatMulTransBBiasWorkers checks that the split product equals the
// one-worker product and the unsplit matMulTransBBias bit for bit, over
// the 4×4, 8×1 and 4-row/1×1 tail paths, leftover rows (m%8 ≠ 0), n a
// multiple of 4 and not, and an ldc wider than n.
func TestMatMulTransBBiasWorkers(t *testing.T) {
	rng := rand.New(rand.NewPCG(27, 28))

	shapes := [][3]int{
		{37, 36, 1031},
		{45, 13, 2000},
		{1100, 1, 1000},
		{75, 33, 500},
		{83, 64, 297},
	}

	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		if m*n*k < 2*gemmPieceFMAs || m <= gemmBiasBlockM {
			t.Fatalf("%v is below the split threshold", s)
		}

		// Parallel runs ceil(m/grain) units, so the split has at least two
		// pieces on two or more workers.
		if grain := gemmBiasGrain(n * k); m <= grain {
			t.Fatalf("%v: grain %d leaves a single piece", s, grain)
		}

		a, b, bias := randGemmData(rng, m*k), randGemmData(rng, n*k), randGemmData(rng, n)
		ldc := n + 3

		for _, bs := range [][]float32{bias, nil} {
			fill := func() []float32 {
				c := make([]float32, (m-1)*ldc+n+1)
				for i := range c {
					c[i] = gemmSentinel
				}

				return c
			}

			want := fill()
			matMulTransBBias(want, ldc, a, b, cmpBias(bs, n), m, n, k)

			if bs != nil {
				checkBiasReference(t, fmt.Sprintf("%v", s), want, ldc, a, b, bs, m, n, k)
			}

			for _, w := range []int{1, 2, 3, 4, 7, 16} {
				c := fill()

				withWorkers(t, w, func() { MatMulTransBBias(c, ldc, a, b, bs, m, n, k) })

				if !slices.EqualFunc(c, want, func(x, y float32) bool {
					return math.Float32bits(x) == math.Float32bits(y)
				}) {
					t.Fatalf("m %d n %d k %d bias=%v: %d workers differ from the unsplit product", m, n, k, bs != nil, w)
				}
			}
		}
	}
}

// cmpBias returns bias, or n zeros for nil, as MatMulTransBBias passes it
// on.
func cmpBias(bias []float32, n int) []float32 {
	if bias == nil {
		return make([]float32, n)
	}

	return bias
}

// BenchmarkMatMulTransBBias measures one MatMulTransBBias call for every
// basic-pitch layer shape (one Conv2D block, without the im2col around it),
// a product large enough to split over the workers, and the shapes of
// BenchmarkMatMulTransB with n a multiple of 4 (go-pocket-tts' and
// go-whistle's, which the SME2 path is for).
func BenchmarkMatMulTransBBias(b *testing.B) {
	rng := rand.New(rand.NewPCG(13, 14))

	type shape struct {
		name    string
		m, n, k int
	}

	shapes := make([]shape, 0, 18)
	for _, s := range append(biasBasicPitchShapes(), [3]int{256, 64, 576}) {
		shapes = append(shapes, shape{fmt.Sprintf("m%d_n%d_k%d", s[0], s[1], s[2]), s[0], s[1], s[2]})
	}

	shapes = append(shapes,
		shape{"conv_tile_64x56x576", 64, 56, 576},
		shape{"gemv_1x1024x1024", 1, 1024, 1024},
		shape{"linear_16x1024x1024", 16, 1024, 1024},
		shape{"pocket_flowlm_ffn_1x4096x1024", 1, 4096, 1024},
		shape{"pocket_flowlm_inproj_64x3072x1024", 64, 3072, 1024},
		shape{"pocket_flowlm_ffn_64x4096x1024", 64, 4096, 1024},
		shape{"pocket_mimi_conv_512x32x3584", 512, 32, 3584},
		shape{"pocket_mimi_inproj_1x1536x512", 1, 1536, 512},
		shape{"whistle_linear_375x512x512", 375, 512, 512},
		shape{"whistle_ffn_375x2048x512", 375, 2048, 512},
		shape{"whistle_values_64x64x375", 64, 64, 375},
	)

	for _, s := range shapes {
		m, n, k := s.m, s.n, s.k
		a := randGemmData(rng, m*k)
		w := randGemmData(rng, n*k)
		bias := randGemmData(rng, n)
		c := make([]float32, m*n)

		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				MatMulTransBBias(c, n, a, w, bias, m, n, k)
			}

			b.ReportMetric(float64(m*n*k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}

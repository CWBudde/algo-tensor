//go:build darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: tests of the SME2
// path of MatMulTransBBias (PLAN.md 2.1). They run on darwin only, where the
// path is selected; GitHub's macOS runners have no SME2 and skip them.

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"syscall"
	"testing"
)

// noSMEEnv names an environment variable that turns the SME path off in
// the test binary, for the NEON ("before") numbers of
// BenchmarkMatMulTransBBias (ops/PERFORMANCE.md).
const noSMEEnv = "ALGO_TENSOR_NO_SME"

//nolint:gochecknoinits // the switch must act before the first test or benchmark
func init() {
	if os.Getenv(noSMEEnv) != "" {
		gemmBiasUseSME = false
	}
}

// skipWithoutSME skips a test that runs gemmSME on a CPU without SME2,
// where it would fault.
func skipWithoutSME(t *testing.T) {
	t.Helper()

	if !gemmBiasUseSME {
		t.Skip("the CPU lacks SME2 (or " + noSMEEnv + " is set)")
	}
}

// setSMEThreshold sets the least rows, columns and multiply-adds of the
// products MatMulTransBBias runs on gemmSME until the test ends. Tests
// that call it must not run in parallel.
func setSMEThreshold(t *testing.T, rows, cols, fmas int64) {
	t.Helper()

	prevRows, prevCols, prevFMAs := smeMinRows.Load(), smeMinCols.Load(), smeMinFMAs.Load()

	t.Cleanup(func() {
		smeMinRows.Store(prevRows)
		smeMinCols.Store(prevCols)
		smeMinFMAs.Store(prevFMAs)
	})

	smeMinRows.Store(rows)
	smeMinCols.Store(cols)
	smeMinFMAs.Store(fmas)
}

// setSMEClobber sets the canary test hooks of gemmSME until the test ends.
// Tests that call it must not run in parallel.
func setSMEClobber(t *testing.T, k, r uint64) {
	t.Helper()

	t.Cleanup(func() {
		smeClobberK.Store(0)
		smeClobberR.Store(0)
	})

	smeClobberK.Store(k)
	smeClobberR.Store(r)
}

// smeOperands holds the operands of one product and the outputs of the
// kernel under test (got) and of gemmNEON4x4 (want), both pre-filled with
// gemmSentinel, with ldc-n gap elements after every row and one guard
// element after the last.
type smeOperands struct {
	m, n, k, ldc int
	a, b, bias   []float32
	got, want    []float32
	name         string
}

func newSMEOperands(rng *rand.Rand, m, n, k, ldc int) *smeOperands {
	o := &smeOperands{
		m: m, n: n, k: k, ldc: ldc,
		a:    randGemmData(rng, m*k),
		b:    randGemmData(rng, n*k),
		bias: randGemmData(rng, n),
		name: fmt.Sprintf("m%d/n%d/k%d/ldc%d", m, n, k, ldc),
	}
	o.want = sentinels(biasOutLen(m, n, ldc) + 1)
	o.got = sentinels(len(o.want))

	return o
}

// sentinels returns n copies of gemmSentinel.
func sentinels(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = gemmSentinel
	}

	return s
}

// neon fills want with gemmNEON4x4's product.
func (o *smeOperands) neon() {
	gemmNEON4x4(&o.want[0], &o.a[0], &o.b[0], &o.bias[0], o.m, o.n, o.k, o.ldc)
}

// check fails t unless got equals want bit for bit, gaps and guard
// included.
func (o *smeOperands) check(t *testing.T, what string) {
	t.Helper()

	for i := range o.want {
		if math.Float32bits(o.got[i]) != math.Float32bits(o.want[i]) {
			t.Fatalf("%s %s: c[%d] = %v (%#08x), gemmNEON4x4 gives %v (%#08x)", what, o.name, i,
				o.got[i], math.Float32bits(o.got[i]), o.want[i], math.Float32bits(o.want[i]))
		}
	}
}

// TestGemmSMEMatchesNEON runs gemmSME directly over the 1.1 sweep (the
// depths and channel counts of biasSweepK and biasSweepN that
// gemmNEON4x4 takes, m 1–17) and the spike's (every k%4 and k%8
// remainder, n with and without whole 16-column blocks, m 1–40, so both
// loop orders and every row and column remainder of the 16×16 blocks),
// with ldc = n and n+3, and compares every output, gap and guard element
// with gemmNEON4x4 bit for bit.
func TestGemmSMEMatchesNEON(t *testing.T) {
	t.Parallel()
	skipWithoutSME(t)

	rng := rand.New(rand.NewPCG(41, 42))

	type sweep struct {
		ks, ns []int
		maxM   int
	}

	var sweepK, sweepN []int

	for _, k := range biasSweepK {
		if k >= 4 {
			sweepK = append(sweepK, k)
		}
	}

	for _, n := range biasSweepN {
		if n%4 == 0 {
			sweepN = append(sweepN, n)
		}
	}

	sweeps := []sweep{
		{sweepK, sweepN, 17},
		{[]int{4, 5, 6, 7, 8, 9, 49, 200, 297, 672, 936}, []int{4, 8, 12, 16, 20, 32, 36, 64}, 40},
	}

	shapes := 0

	for _, s := range sweeps {
		for _, k := range s.ks {
			for _, n := range s.ns {
				for m := 1; m <= s.maxM; m++ {
					for _, ldc := range biasLdcs(n) {
						o := newSMEOperands(rng, m, n, k, ldc)
						o.neon()
						gemmBiasSMERun(o.got, ldc, o.a, o.b, o.bias, m, n, k)
						o.check(t, "gemmSME")

						shapes++
					}
				}
			}
		}
	}

	t.Logf("%d shapes", shapes)
}

// TestMatMulTransBBiasSMEWorkers runs MatMulTransBBias with the threshold
// lowered so that every shape reaches gemmSME, over both Parallel splits
// (by columns for n >= m, by rows otherwise) and for any worker count, and
// compares with gemmNEON4x4 bit for bit.
func TestMatMulTransBBiasSMEWorkers(t *testing.T) {
	skipWithoutSME(t)
	setSMEThreshold(t, 1, 4, 0)

	rng := rand.New(rand.NewPCG(43, 44))

	shapes := [][3]int{
		{1, 4, 4},
		{3, 20, 9},
		{40, 1028, 1031}, // split by columns, a 4-column last block
		{1100, 36, 1001}, // split by rows, a 12-row last block
		{37, 37 * 4, 297},
	}

	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]

		if !smeTakes(m, n, k) {
			t.Fatalf("%v: smeTakes declines it", s)
		}

		for _, ldc := range biasLdcs(n) {
			o := newSMEOperands(rng, m, n, k, ldc)
			o.neon()

			for _, w := range []int{1, 2, 3, 16} {
				copy(o.got, sentinels(len(o.got)))
				withWorkers(t, w, func() { MatMulTransBBias(o.got, ldc, o.a, o.b, o.bias, m, n, k) })
				o.check(t, fmt.Sprintf("MatMulTransBBias on %d workers", w))
			}
		}
	}
}

// smeSpecials are the values the special-value test mixes in: signed
// zeros and infinities, quiet and signalling NaNs with payloads of either
// sign, subnormals and the extremes of the normal range.
var smeSpecials = []float32{
	0, float32(math.Copysign(0, -1)),
	float32(math.Inf(1)), float32(math.Inf(-1)),
	math.Float32frombits(0x7fc00000), math.Float32frombits(0xffc12345),
	math.Float32frombits(0x7f800001), math.Float32frombits(0xff812345),
	math.Float32frombits(0x00000001), math.Float32frombits(0x807fffff),
	math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32 * 3,
}

// smeTinies are specials that cannot make a NaN: signed zeros, subnormals
// and the smallest normals, whose products underflow.
var smeTinies = []float32{
	0, float32(math.Copysign(0, -1)),
	math.Float32frombits(0x00000001), math.Float32frombits(0x807fffff),
	math.Float32frombits(0x00400000), math.Float32frombits(0x80800000),
	math.SmallestNonzeroFloat32 * 3, -1e-20, 1e-20,
}

// TestGemmSMESpecialValues compares gemmSME with gemmNEON4x4 bit for bit
// on special operands, over every k%4 tail:
//
//   - signed zeros, subnormals and underflowing products, and rows of -0
//     against a -0 bias, where the +0 that the masked tail lanes add
//     (fma(+0, +0, -0) = +0) decides the sign of the zero: no output is NaN
//     or infinite, so these are gemmSME's own bits;
//   - infinities of one sign against positive operands (infinite outputs,
//     no NaN) and any special value, NaNs with payloads included: a call
//     with a NaN or infinite output is computed again on NEON, since
//     gemmSME's NaNs are the default NaN, which drops the payload and sign
//     that gemmNEON4x4 keeps.
//
// It counts the calls computed again on NEON (smeNaNs) and so does not run
// in parallel with other tests.
func TestGemmSMESpecialValues(t *testing.T) {
	skipWithoutSME(t)

	rng := rand.New(rand.NewPCG(45, 46))

	sprinkle := func(s, from []float32, p float64) {
		for i := range s {
			if rng.Float64() < p {
				s[i] = from[rng.IntN(len(from))]
			}
		}
	}

	run := func(o *smeOperands, what string) {
		o.neon()
		gemmBiasSMERun(o.got, o.ldc, o.a, o.b, o.bias, o.m, o.n, o.k)
		o.check(t, what)
	}

	ks := []int{4, 5, 6, 7, 8, 13, 64, 99}
	ns := []int{4, 16, 20, 36}
	ms := []int{1, 5, 16, 19}
	ps := []float64{0.01, 0.1, 0.5, 1}

	nans := smeNaNs.Load()

	for _, k := range ks {
		for _, n := range ns {
			for _, m := range ms {
				for _, p := range ps {
					o := newSMEOperands(rng, m, n, k, n+3)
					sprinkle(o.a, smeTinies, p)
					sprinkle(o.b, smeTinies, p)
					sprinkle(o.bias, smeTinies, p)
					run(o, fmt.Sprintf("tiny p=%v", p))
				}
			}
		}
	}

	negZero := float32(math.Copysign(0, -1))

	for _, k := range []int{4, 5, 6, 7, 8} {
		o := newSMEOperands(rng, 3, 4, k, 4)
		for i := range o.a {
			o.a[i] = negZero
		}

		for i := range o.bias {
			o.bias[i] = negZero
		}

		run(o, "-0")
	}

	if d := smeNaNs.Load() - nans; d != 0 {
		t.Fatalf("%d calls with finite outputs went to NEON", d)
	}

	for _, k := range ks {
		for _, n := range ns {
			for _, m := range ms {
				for _, p := range ps {
					o := newSMEOperands(rng, m, n, k, n+3)
					for i := range o.b {
						o.b[i] = 0.5 + rng.Float32()/2
					}

					sprinkle(o.a, []float32{float32(math.Inf(1)), 0.75}, p)

					for i := range o.a {
						if i/k%2 == 1 { // one sign of infinity per row
							o.a[i] = -o.a[i]
						}
					}

					run(o, fmt.Sprintf("inf p=%v", p))

					o = newSMEOperands(rng, m, n, k, n+3)
					sprinkle(o.a, smeSpecials, p)
					sprinkle(o.b, smeSpecials, p)
					sprinkle(o.bias, smeSpecials, p)
					run(o, fmt.Sprintf("special p=%v", p))
				}
			}
		}
	}

	if smeNaNs.Load() == nans {
		t.Fatal("no call went to NEON for a NaN or infinite output")
	}
}

// smeClobberRun runs gemmSME on an m×n×297 product with the hooks set by
// the caller and checks every output against gemmNEON4x4; with nan, the
// first row of a holds a signalling NaN with a payload, so the first row
// of outputs is NaN. It returns the failed checks, the NEON fallbacks and
// the calls computed again for a NaN output that it caused.
func smeClobberRun(t *testing.T, m, n int, nan bool) (redos, fallbacks, nans int64) {
	t.Helper()

	rng := rand.New(rand.NewPCG(uint64(m), uint64(n)))
	o := newSMEOperands(rng, m, n, 297, n+3)

	if nan {
		o.a[5] = math.Float32frombits(0x7f812345)
	}

	o.neon()

	r0, f0, n0 := smeRedos.Load(), smeFallbacks.Load(), smeNaNs.Load()

	gemmBiasSMERun(o.got, o.ldc, o.a, o.b, o.bias, m, n, o.k)
	o.check(t, "clobbered")

	return smeRedos.Load() - r0, smeFallbacks.Load() - f0, smeNaNs.Load() - n0
}

// TestGemmSMERedo forces the signal guard through the test hooks, which
// zero a single lane (7) of the canary where a signal zeroes lanes 4–15,
// and the NaN collector z28:
// a check after the k loop or after the stores that finds the clobbered
// canary redoes the block (the reduction), up to 8 times; the 9th failure
// of one block finishes the product with gemmNEON4x4 from that block's
// rows (a outer) or columns (b outer) on. Every output must keep
// gemmNEON4x4's bits. The shapes are 40×36 (3×3 blocks, a outer, so the
// checks of the i-th row of blocks are numbered 3i..3i+2) and 20×64 (2×4
// blocks, b outer: the j-th column of blocks has checks 2j and 2j+1).
// With a NaN in the first block's outputs and the collector cleared at
// the last block, only the caller's own look at the outputs after a redo
// can find it. Signals can add redos, so the counts are lower bounds.
func TestGemmSMERedo(t *testing.T) {
	skipWithoutSME(t)

	cases := []struct {
		name     string
		k, r     uint64
		m, n     int
		redos    int64
		fallback bool
		nan      bool
	}{
		{name: "k loop, 3 redos", k: 3, m: 40, n: 36, redos: 3},
		{name: "k loop, 8 redos", k: 8, m: 40, n: 36, redos: 8},
		{name: "k loop, NEON after 9", k: 9, m: 40, n: 36, redos: 9, fallback: true},
		{name: "k loop, NEON from the 2nd row of blocks", k: 4<<32 | 9, m: 40, n: 36, redos: 9, fallback: true},
		{name: "k loop, NEON from the last block", k: 8<<32 | 9, m: 40, n: 36, redos: 9, fallback: true},
		{name: "k loop, NEON from the 2nd column of blocks", k: 2<<32 | 9, m: 20, n: 64, redos: 9, fallback: true},
		{name: "stores, 1 redo", r: 1, m: 40, n: 36, redos: 1},
		{name: "stores, 8 redos of the 5th block", r: 4<<32 | 8, m: 40, n: 36, redos: 8},
		{name: "stores, NEON after 9", r: 5<<32 | 9, m: 20, n: 64, redos: 9, fallback: true},
		{name: "both, NEON after 4 + 5", k: 4, r: 5, m: 40, n: 36, redos: 9, fallback: true},
		{name: "NaN, no redo", m: 40, n: 36, nan: true},
		{name: "NaN, collector cleared at the last block", k: 8<<32 | 1, m: 40, n: 36, redos: 1, nan: true},
		{name: "NaN, collector cleared at the last stores", r: 8<<32 | 1, m: 40, n: 36, redos: 1, nan: true},
		{name: "NaN, NEON after 9 in the 2nd row of blocks", k: 3<<32 | 9, m: 40, n: 36, redos: 9, fallback: true, nan: true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			setSMEClobber(t, c.k, c.r)

			redos, fallbacks, nans := smeClobberRun(t, c.m, c.n, c.nan)

			if redos < c.redos {
				t.Errorf("%d failed checks, want %d", redos, c.redos)
			}

			if want := map[bool]int64{false: 0, true: 1}[c.fallback]; fallbacks != want {
				t.Errorf("%d NEON fallbacks, want %d", fallbacks, want)
			}

			if want := map[bool]int64{false: 0, true: 1}[c.nan]; nans != want {
				t.Errorf("%d calls computed again for a NaN, want %d", nans, want)
			}
		})
	}
}

// TestGemmSMESelected checks that MatMulTransBBias takes the SME path
// whenever the CPU reports SME2 (hw.optional.arm.FEAT_SME2, read here
// independently of smeSupported), for go-pocket-tts' and go-whistle's
// shapes above the threshold, and NEON below it. Like
// TestGemmAVX2Selected it fails, rather than skips, when the CPU reports
// the feature but the path was not selected. Unlike it, a CPU without SME2
// skips it under CI too: GitHub's macOS runners (Apple M1) have none.
func TestGemmSMESelected(t *testing.T) {
	t.Parallel()

	ci := os.Getenv("CI")

	v, err := syscall.Sysctl("hw.optional.arm.FEAT_SME2")
	if err != nil || len(v) == 0 || v[0] != 1 {
		t.Skipf("CI=%q: the CPU does not report SME2 (sysctl %q, %v), so MatMulTransBBias runs NEON", ci, v, err)
	}

	if os.Getenv(noSMEEnv) != "" && ci == "" {
		t.Skipf("%s is set", noSMEEnv)
	}

	if !gemmBiasUseSME {
		t.Fatalf("CI=%q: the CPU reports SME2, but gemmBiasUseSME is false", ci)
	}

	for _, s := range [][3]int{{64, 3072, 1024}, {64, 4096, 1024}, {375, 2048, 512}, {375, 512, 512}, {512, 32, 3584}, {16, 1024, 1024}, {64, 56, 576}} {
		m, n, k := s[0], s[1], s[2]
		c := make([]float32, m*n)

		if !matMulTransBBiasSME(c, n, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, n, k) {
			t.Errorf("CI=%q: the SME path was not selected for m=%d n=%d k=%d", ci, m, n, k)
		}
	}

	for _, s := range [][3]int{{1, 4096, 1024}, {8, 32, 200}, {64, 8, 1024}, {64, 33, 1024}, {16, 16, 64}} {
		if m, n, k := s[0], s[1], s[2]; smeTakes(m, n, k) {
			t.Errorf("the SME path was selected for m=%d n=%d k=%d", m, n, k)
		}
	}
}

// TestMatMulTransBBiasSMEAllocs checks that the SME path does not
// allocate in steady state on one worker: its scratch comes from a pool.
// The race detector makes sync.Pool drop items at random, so it skips
// there.
func TestMatMulTransBBiasSMEAllocs(t *testing.T) {
	skipWithoutSME(t)

	if raceEnabled {
		t.Skip("sync.Pool drops items at random under the race detector")
	}

	rng := rand.New(rand.NewPCG(47, 48))

	for _, s := range [][3]int{{64, 64, 576}, {200, 36, 300}, {16, 1024, 1024}} {
		m, n, k := s[0], s[1], s[2]
		if !smeTakes(m, n, k) {
			t.Fatalf("%v: smeTakes declines it", s)
		}

		a, b, bias := randGemmData(rng, m*k), randGemmData(rng, n*k), randGemmData(rng, n)
		c := make([]float32, m*n)

		withWorkers(t, 1, func() {
			for _, bs := range [][]float32{bias, nil} {
				if allocs := testing.AllocsPerRun(20, func() {
					MatMulTransBBias(c, n, a, b, bs, m, n, k)
				}); allocs != 0 {
					t.Errorf("%v bias=%v: %v allocations per call", s, bs != nil, allocs)
				}
			}
		})
	}
}

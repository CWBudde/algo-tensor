//go:build darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: tests of gemmSME on
// operands packed by the caller (PLAN.md 2.2).

package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"syscall"
	"testing"

	"github.com/cwbudde/algo-tensor/internal/sme"
)

// skipWithoutSMEPacked skips a test of sme.GEMM where gemmSME does not
// run, and fails it if sme.GEMM is not set where it does.
func skipWithoutSMEPacked(t *testing.T) {
	t.Helper()
	skipWithoutSME(t)

	if sme.GEMM == nil {
		t.Fatal("gemmBiasUseSME is set, but sme.GEMM is nil")
	}
}

// smePackedRun runs sme.GEMM on o with b packed by sme.Pack and reports
// what it returned.
func smePackedRun(o *smeOperands) bool {
	pb := make([]float32, sme.PackedLen(o.n, o.k))
	sme.Pack(pb, o.b, o.n, o.k)

	return sme.GEMM(o.got, o.ldc, o.a, make([]float32, sme.PanelRows*sme.Steps(o.k)), pb, o.bias, o.m, o.n, o.k)
}

// TestGemmSMEPackedMatchesNEON runs sme.GEMM, set by this package, over
// the spike's sweep of TestGemmSMEMatchesNEON with b packed by sme.Pack
// and a as rows or packed as well, and compares every output, gap and
// guard element with gemmNEON4x4 bit for bit.
func TestGemmSMEPackedMatchesNEON(t *testing.T) {
	t.Parallel()
	skipWithoutSMEPacked(t)

	rng := rand.New(rand.NewPCG(51, 52))
	shapes := 0

	for _, k := range []int{4, 5, 6, 7, 8, 9, 49, 200, 297, 672, 936} {
		for _, n := range []int{4, 8, 12, 16, 20, 32, 36, 64} {
			for m := 1; m <= 40; m++ {
				for _, ldc := range biasLdcs(n) {
					o := newSMEOperands(rng, m, n, k, ldc)
					o.neon()

					if !smePackedRun(o) {
						t.Fatalf("%s: sme.GEMM returned false on finite operands", o.name)
					}

					o.check(t, "sme.GEMM")

					shapes++
				}
			}
		}
	}

	t.Logf("%d shapes", shapes)
}

// checkUpTo is check for the outputs of rows 0..rows-1 only, and for
// every gap and the guard: the outputs from row rows on are skipped. Where
// gemmNEON4x4's output is a NaN, got may be any NaN, since gemmSME's NaNs
// are the default NaN.
func (o *smeOperands) checkUpTo(t *testing.T, what string, rows int) {
	t.Helper()

	for i, w := range o.want {
		g := o.got[i]
		if i/o.ldc >= rows && i%o.ldc < o.n && i < len(o.want)-1 {
			continue
		}

		if math.Float32bits(g) == math.Float32bits(w) || (g != g && w != w) {
			continue
		}

		t.Fatalf("%s %s: c[%d] = %v (%#08x), gemmNEON4x4 gives %v (%#08x)", what, o.name, i,
			g, math.Float32bits(g), w, math.Float32bits(w))
	}
}

// TestGemmSMEPackedRedo forces the signal guard through the test hooks
// (see TestGemmSMERedo) on sme.GEMM, where b, the packed operand, is
// always the inner one and a runs panel by panel (a outer): in 40×36 (3×3
// blocks) the checks of the i-th row of blocks are numbered 3i..3i+2, in
// 20×64 (2×4 blocks) 4i..4i+3. Up to 8 failed checks of one block redo it,
// and sme.GEMM returns true with gemmNEON4x4's bits. The 9th failure gives
// the block up: sme.GEMM returns false, which leaves the outputs to the
// caller, with the rows of blocks before the given-up one finished. With a
// NaN (or infinity) in the first row of a, it returns false too, also
// when a cleared collector leaves only its own look at the outputs after
// a failed check to find it. Signals can add redos, so the counts of
// failed checks are lower bounds, and the finished rows are checked only
// when no signal added one.
func TestGemmSMEPackedRedo(t *testing.T) {
	skipWithoutSMEPacked(t)

	nan, inf := math.Float32frombits(0x7f812345), float32(math.Inf(1))

	cases := []struct {
		name    string
		k, r    uint64
		m, n    int
		redos   int64
		giveUp  int // first row of the given-up block, or -1
		special float32
	}{
		{name: "k loop, 3 redos", k: 3, m: 40, n: 36, redos: 3, giveUp: -1},
		{name: "k loop, 8 redos", k: 8, m: 40, n: 36, redos: 8, giveUp: -1},
		{name: "k loop, 8 redos of the 6th block", k: 5<<32 | 8, m: 20, n: 64, redos: 8, giveUp: -1},
		{name: "k loop, given up after 9", k: 9, m: 40, n: 36, redos: 9, giveUp: 0},
		{name: "k loop, given up in the 2nd row of blocks", k: 4<<32 | 9, m: 40, n: 36, redos: 9, giveUp: 16},
		{name: "k loop, given up at the last block", k: 8<<32 | 9, m: 40, n: 36, redos: 9, giveUp: 32},
		{name: "k loop, given up at the 4th block of the 1st row", k: 3<<32 | 9, m: 20, n: 64, redos: 9, giveUp: 0},
		{name: "k loop, given up in the 2nd row of blocks of 20×64", k: 5<<32 | 9, m: 20, n: 64, redos: 9, giveUp: 16},
		{name: "stores, 1 redo", r: 1, m: 40, n: 36, redos: 1, giveUp: -1},
		{name: "stores, 8 redos of the 5th block", r: 4<<32 | 8, m: 40, n: 36, redos: 8, giveUp: -1},
		{name: "stores, given up after 9", r: 5<<32 | 9, m: 20, n: 64, redos: 9, giveUp: 16},
		{name: "both, given up after 4 + 5", k: 4, r: 5, m: 40, n: 36, redos: 9, giveUp: 0},
		{name: "NaN, no redo", m: 40, n: 36, giveUp: -1, special: nan},
		{name: "NaN, collector cleared at the last block", k: 8<<32 | 1, m: 40, n: 36, redos: 1, giveUp: -1, special: nan},
		{name: "NaN, collector cleared at the last stores", r: 8<<32 | 1, m: 40, n: 36, redos: 1, giveUp: -1, special: nan},
		{name: "infinity, collector cleared at the last block", k: 8<<32 | 1, m: 40, n: 36, redos: 1, giveUp: -1, special: inf},
		{name: "infinity, collector cleared at the last stores", r: 7<<32 | 1, m: 20, n: 64, redos: 1, giveUp: -1, special: inf},
		{name: "NaN, given up in the 2nd row of blocks", k: 3<<32 | 9, m: 40, n: 36, redos: 9, giveUp: 16, special: nan},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			setSMEClobber(t, c.k, c.r)

			rng := rand.New(rand.NewPCG(uint64(c.m), uint64(c.n)))
			o := newSMEOperands(rng, c.m, c.n, 297, c.n+3)

			if c.special != 0 {
				o.a[5] = c.special
			}

			o.neon()

			r0, f0, n0 := smeRedos.Load(), smeFallbacks.Load(), smeNaNs.Load()
			ok := smePackedRun(o)
			redos, fallbacks, nans := smeRedos.Load()-r0, smeFallbacks.Load()-f0, smeNaNs.Load()-n0

			if redos < c.redos {
				t.Errorf("%d failed checks, want %d", redos, c.redos)
			}

			wantFallbacks, wantNaNs := int64(0), int64(0)

			switch {
			case c.giveUp >= 0:
				wantFallbacks = 1
			case c.special != 0:
				wantNaNs = 1
			}

			if fallbacks != wantFallbacks {
				t.Errorf("%d blocks given up, want %d", fallbacks, wantFallbacks)
			}

			if nans != wantNaNs {
				t.Errorf("%d calls reported for a NaN or infinite output, want %d", nans, wantNaNs)
			}

			if want := c.giveUp < 0 && c.special == 0; ok != want {
				t.Fatalf("sme.GEMM returned %v, want %v", ok, want)
			}

			switch {
			case c.giveUp < 0:
				o.checkUpTo(t, "sme.GEMM", c.m)
			case redos == c.redos:
				o.checkUpTo(t, fmt.Sprintf("sme.GEMM, rows before %d", c.giveUp), c.giveUp)
			}
		})
	}
}

// TestGemmSMEPackedSpecialValues runs sme.GEMM on the operands of
// TestGemmSMESpecialValues: with signed zeros, subnormals and underflowing
// products, and rows of -0 against a -0 bias, every output is finite,
// sme.GEMM returns true and the outputs have gemmNEON4x4's bits. With
// infinities and any special value it returns true exactly when every
// output of gemmNEON4x4 is finite, with its bits; when it returns false,
// it counts the call in smeNaNs, and its outputs are still gemmNEON4x4's
// except for the bits of a NaN. It counts smeNaNs and so does not run in
// parallel with other tests.
func TestGemmSMEPackedSpecialValues(t *testing.T) {
	skipWithoutSMEPacked(t)

	rng := rand.New(rand.NewPCG(53, 54))

	sprinkle := func(s, from []float32, p float64) {
		for i := range s {
			if rng.Float64() < p {
				s[i] = from[rng.IntN(len(from))]
			}
		}
	}

	ks := []int{4, 5, 6, 7, 8, 13, 64, 99}
	ns := []int{4, 16, 20, 36}
	ms := []int{1, 5, 16, 19}
	ps := []float64{0.01, 0.1, 0.5, 1}

	nans, fallbacks := smeNaNs.Load(), smeFallbacks.Load()
	falses := int64(0)

	// run runs o on sme.GEMM and checks what it returns and its outputs.
	run := func(o *smeOperands, what string) {
		o.neon()

		finite := finiteOutputs(o.want, o.ldc, o.m, o.n)
		if ok := smePackedRun(o); ok != finite {
			t.Fatalf("%s %s: sme.GEMM returned %v, but gemmNEON4x4's outputs finite = %v", what, o.name, ok, finite)
		}

		if !finite {
			falses++
		}

		o.checkUpTo(t, what, o.m)
	}

	for _, k := range ks {
		for _, n := range ns {
			for _, m := range ms {
				for _, p := range ps {
					o := newSMEOperands(rng, m, n, k, n+3)
					sprinkle(o.a, smeTinies, p)
					sprinkle(o.b, smeTinies, p)
					sprinkle(o.bias, smeTinies, p)
					run(o, fmt.Sprintf("tiny p=%v", p))
					o.check(t, "sme.GEMM") // bit for bit, no NaN
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
		o.check(t, "sme.GEMM -0")
	}

	if falses != 0 || smeNaNs.Load() != nans {
		t.Fatalf("%d calls with finite outputs returned false (smeNaNs grew by %d)", falses, smeNaNs.Load()-nans)
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

	if falses == 0 {
		t.Fatal("no call returned false for a NaN or infinite output")
	}

	if d := smeNaNs.Load() - nans; d != falses {
		t.Errorf("smeNaNs grew by %d, but %d calls returned false", d, falses)
	}

	if d := smeFallbacks.Load() - fallbacks; d != 0 {
		t.Errorf("%d blocks given up", d)
	}

	t.Logf("%d calls returned false", falses)
}

// TestGemmSMEPackedRegistered checks that tensor sets sme.GEMM to
// gemmBiasSMEPacked exactly where the CPU reports SME2
// (hw.optional.arm.FEAT_SME2, read here independently of smeSupported),
// so that ops.Conv2DPlan reaches the kernel wherever MatMulTransBBias
// does, and leaves it nil on a CPU without. ALGO_TENSOR_NO_SME turns
// gemmBiasUseSME off only after sme.GEMM was set (in this package's test
// init), so it does not change the outcome.
func TestGemmSMEPackedRegistered(t *testing.T) {
	t.Parallel()

	if gemmBiasUseSME && sme.GEMM == nil {
		t.Fatal("MatMulTransBBias runs gemmSME, but sme.GEMM is nil")
	}

	v, err := syscall.Sysctl("hw.optional.arm.FEAT_SME2")
	hw := err == nil && len(v) > 0 && v[0] == 1

	switch {
	case !hw && sme.GEMM != nil:
		t.Fatalf("the CPU does not report SME2 (sysctl %q, %v), but sme.GEMM is set", v, err)
	case hw && sme.GEMM == nil:
		t.Fatal("the CPU reports SME2, but sme.GEMM is nil")
	case hw && reflect.ValueOf(sme.GEMM).Pointer() != reflect.ValueOf(gemmBiasSMEPacked).Pointer():
		t.Fatal("sme.GEMM is not gemmBiasSMEPacked")
	}
}

// algo-tensor's own, not in go-whistle or go-pocket-tts: tests of Conv2DPlan
// on the SME2 kernel (PLAN.md 2.2).

package ops

import (
	"fmt"
	"math"
	"os"
	"sync/atomic"
	"testing"

	"github.com/cwbudde/algo-tensor/internal/sme"
	"github.com/cwbudde/algo-tensor/tensor"
)

// noSMEEnv names an environment variable that turns the SME path off in
// the test binary, as in tensor's tests, for the NEON ("before") numbers of
// the Conv2DPlan benchmarks (ops/PERFORMANCE.md).
const noSMEEnv = "ALGO_TENSOR_NO_SME"

//nolint:gochecknoinits // the switch must act before the first test or benchmark
func init() {
	if os.Getenv(noSMEEnv) != "" {
		sme.GEMM = nil
	}
}

// smeGEMMFunc is the type of sme.GEMM.
type smeGEMMFunc = func(c []float32, ldc int, a, po, pb, bias []float32, m, n, k int) bool

// Sentinels the outputs of the plan under test and of the NEON reference
// are filled with before a pass; they differ, so an output either plan
// leaves unwritten shows as a mismatch.
const (
	smeSentinel  = -12345.5
	neonSentinel = 54321.25
)

// skipWithoutSMEGEMM skips a test that needs sme.GEMM, which is nil on
// CPUs without SME2, under purego, on other platforms and with noSMEEnv
// set.
func skipWithoutSMEGEMM(t *testing.T) {
	t.Helper()

	if sme.GEMM == nil {
		t.Skip("sme.GEMM is nil: no SME2, purego, another platform, or " + noSMEEnv + " is set")
	}
}

// setConv2DPlanSMEThresholds sets the four thresholds of conv2d_plan_sme.go
// until the test ends. Tests that call it must not run in parallel.
func setConv2DPlanSMEThresholds(t *testing.T, minK, minWork, minPositions, minFMAs int) {
	t.Helper()

	prevK, prevWork, prevPositions, prevFMAs := conv2DPlanSMEMinK, conv2DPlanSMEMinWork, conv2DPlanSMEMinPositions, conv2DPlanSMEMinFMAs

	t.Cleanup(func() {
		conv2DPlanSMEMinK, conv2DPlanSMEMinWork, conv2DPlanSMEMinPositions, conv2DPlanSMEMinFMAs = prevK, prevWork, prevPositions, prevFMAs
	})

	conv2DPlanSMEMinK, conv2DPlanSMEMinWork, conv2DPlanSMEMinPositions, conv2DPlanSMEMinFMAs = minK, minWork, minPositions, minFMAs
}

// lowerConv2DPlanSMEThresholds lowers the thresholds until the test ends,
// so that every plan whose GEMM gemmNEON4x4 takes (k >= 4, outC a multiple
// of 4) packs its weights for sme.GEMM and every call of one row or more
// runs on it. Tests that call it must not run in parallel.
func lowerConv2DPlanSMEThresholds(t *testing.T) {
	t.Helper()
	setConv2DPlanSMEThresholds(t, 0, 0, 0, 1)
}

// setSMEGEMM replaces sme.GEMM by f until the test ends. Plans check
// sme.GEMM when they are built and call it when they run, so tests build
// their plans first. Tests that call it must not run in parallel.
func setSMEGEMM(t *testing.T, f smeGEMMFunc) {
	t.Helper()

	prev := sme.GEMM

	t.Cleanup(func() { sme.GEMM = prev })

	sme.GEMM = f
}

// smeGEMMCounts counts the calls of sme.GEMM and their results.
type smeGEMMCounts struct {
	calls, falses atomic.Int64
}

// countSMEGEMM wraps sme.GEMM until the test ends to count its calls and
// the calls that returned false.
func countSMEGEMM(t *testing.T) *smeGEMMCounts {
	t.Helper()

	counts := new(smeGEMMCounts)
	inner := sme.GEMM

	setSMEGEMM(t, func(c []float32, ldc int, a, po, pb, bias []float32, m, n, k int) bool {
		counts.calls.Add(1)

		ok := inner(c, ldc, a, po, pb, bias, m, n, k)
		if !ok {
			counts.falses.Add(1)
		}

		return ok
	})

	return counts
}

// sequentialConvWorkers sets the sequential default of SetConvWorkers
// until the test ends; the test may set other counts meanwhile.
func sequentialConvWorkers(t *testing.T) {
	t.Helper()

	prev := getConvWorkers()

	t.Cleanup(func() { SetConvWorkers(prev) })
	SetConvWorkers(0)
}

// newSMEPlanPair returns a plan of the kernel, bias and geometry that runs
// on sme.GEMM and a NEON reference plan of the same, with pb set to nil so
// that it runs MatMulTransBBias in blocks of conv2DPlanBlockM positions.
// It fails t unless the first one packed its weights.
func newSMEPlanPair(t *testing.T, kernel, bias *tensor.Tensor, inH, inW int, p Conv2DParams) (smePlan, neonPlan *Conv2DPlan) {
	t.Helper()

	smePlan, err := NewConv2DPlan(kernel, bias, inH, inW, p)
	if err != nil {
		t.Fatal(err)
	}

	if smePlan.pb == nil {
		t.Fatalf("plan k=%d outC=%d %d×%d positions did not pack its weights for sme.GEMM",
			smePlan.k, smePlan.outC, smePlan.outH, smePlan.outW)
	}

	neonPlan, err = NewConv2DPlan(kernel, bias, inH, inW, p)
	if err != nil {
		t.Fatal(err)
	}

	neonPlan.pb = nil

	return smePlan, neonPlan
}

// smePlanPair returns the case's plans (newSMEPlanPair) and input.
func (tc planCase) smePlanPair(t *testing.T) (smePlan, neonPlan *Conv2DPlan, x *tensor.Tensor) {
	t.Helper()

	kernel, bias, x := tc.fixture(t)
	smePlan, neonPlan = newSMEPlanPair(t, kernel, bias, tc.h, tc.w, tc.params())

	return smePlan, neonPlan, x
}

// setSMEChunk makes c run sme.GEMM on chunks of the given number of output
// positions. A smaller chunk than the plan chose fits in the scratch sized
// for its own, so it reports false, and changes nothing, for a larger one.
func setSMEChunk(c *Conv2DPlan, chunk int) bool {
	if chunk > c.smeChunk || chunk*c.k+sme.PanelRows*sme.Steps(c.k) > c.colLen {
		return false
	}

	c.smeChunk = chunk

	return true
}

// planForward runs c.Forward on x into a new output filled with sentinel
// and returns the output's values.
func planForward(t *testing.T, c *Conv2DPlan, x *tensor.Tensor, sentinel float32) []float32 {
	t.Helper()

	dst := planDst(t, c, sentinel)

	err := c.Forward(dst, x)
	if err != nil {
		t.Fatal(err)
	}

	return dst.RawData()
}

// firstBitDiff returns the first index at which got and want differ in
// their bits, or -1.
func firstBitDiff(got, want []float32) int {
	if len(got) != len(want) {
		return min(len(got), len(want))
	}

	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return i
		}
	}

	return -1
}

// assertSameBits fails t, naming the case, unless got equals want bit for
// bit.
func assertSameBits(t *testing.T, name string, got, want []float32) {
	t.Helper()

	i := firstBitDiff(got, want)
	if i < 0 {
		return
	}

	if i >= len(got) || i >= len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}

	t.Fatalf("%s: value %d = %v (%#08x), NEON gives %v (%#08x)",
		name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
}

// smeSweepCase is one convolution of the sweep of
// TestConv2DPlanSMEMatchesNEON.
type smeSweepCase struct {
	kh, kw, inC, outC int
	inH, inW          int
	p                 Conv2DParams
	bias              bool
	// multiChunk asks for a default chunk smaller than the output, with a
	// partial last chunk.
	multiChunk bool
}

func (sc smeSweepCase) name() string {
	return fmt.Sprintf("k=%dx%dx%d/outC=%d/in=%dx%d/s=%dx%d/pad=%d,%d,%d,%d/bias=%t",
		sc.kh, sc.kw, sc.inC, sc.outC, sc.inH, sc.inW, sc.p.StrideH, sc.p.StrideW,
		sc.p.PadTop, sc.p.PadBottom, sc.p.PadLeft, sc.p.PadRight, sc.bias)
}

// smeSweepCases returns the sweep of TestConv2DPlanSMEMatchesNEON: 1×1,
// 1×5, 2×3, 3×3 and 5×5 kernels on 1 to 9 input channels (k from 4 to
// 225, every k%4), 4 to 64 out channels, three strides and "same" or
// explicit asymmetric padding, on images whose output width is not a
// multiple of 16; and a few deep layers on images larger than one default
// chunk.
func smeSweepCases() []smeSweepCase {
	kernels := [][2]int{{1, 1}, {1, 5}, {2, 3}, {3, 3}, {5, 5}}
	outCs := []int{4, 8, 12, 16, 20, 36, 64}
	strides := [][2]int{{1, 1}, {2, 2}, {1, 3}}
	pads := [][4]int{{1, 0, 0, 2}, {0, 2, 3, 1}} // top, bottom, left, right
	inHs, inWs := []int{5, 7, 9, 6}, []int{13, 19, 23, 29, 37}

	var cases []smeSweepCase

	for _, kk := range kernels {
		for inC := 1; inC <= 9; inC++ {
			if kk[0]*kk[1]*inC < 4 {
				continue
			}

			for _, outC := range outCs {
				for _, s := range strides {
					for same := range 2 {
						i := len(cases)
						sc := smeSweepCase{
							kh: kk[0], kw: kk[1], inC: inC, outC: outC,
							inH: inHs[i%len(inHs)], inW: inWs[i%len(inWs)], bias: i%3 != 0,
						}
						sc.p = sc.params(s, same == 1, pads[i%len(pads)])

						// Chunks must span output rows.
						for sc.outW()%16 == 0 {
							sc.inW++
							sc.p = sc.params(s, same == 1, pads[i%len(pads)])
						}

						cases = append(cases, sc)
					}
				}
			}
		}
	}

	// k = 288, 300 and 35 on more positions than one default chunk, 3632,
	// 3488 and 29952.
	return append(cases,
		smeSweepCase{kh: 3, kw: 3, inC: 32, outC: 36, inH: 66, inW: 62, p: samePadParams(66, 62, 3, 3, 1, 1), bias: true, multiChunk: true},
		smeSweepCase{kh: 3, kw: 3, inC: 32, outC: 8, inH: 61, inW: 63, p: Conv2DParams{1, 1, 2, 0, 1, 3, 1}, multiChunk: true},
		smeSweepCase{kh: 5, kw: 5, inC: 12, outC: 64, inH: 60, inW: 61, p: samePadParams(60, 61, 5, 5, 1, 1), bias: true, multiChunk: true},
		smeSweepCase{kh: 1, kw: 5, inC: 7, outC: 4, inH: 200, inW: 163, p: samePadParams(200, 163, 1, 5, 1, 1), bias: true, multiChunk: true},
	)
}

// params returns "same" padding for strides s or the explicit padding pad.
func (sc smeSweepCase) params(s [2]int, same bool, pad [4]int) Conv2DParams {
	if same {
		return samePadParams(sc.inH, sc.inW, sc.kh, sc.kw, s[0], s[1])
	}

	return Conv2DParams{StrideH: s[0], StrideW: s[1], PadTop: pad[0], PadBottom: pad[1], PadLeft: pad[2], PadRight: pad[3], Groups: 1}
}

// outW returns the output width of the case's plan.
func (sc smeSweepCase) outW() int {
	return (sc.inW+sc.p.PadLeft+sc.p.PadRight-sc.kw)/sc.p.StrideW + 1
}

// TestConv2DPlanSMEMatchesNEON runs plans on sme.GEMM, with the thresholds
// lowered so that small shapes take it, over smeSweepCases with the plan's
// chunk and with chunks of 32 and 16 positions (many sme.GEMM calls per
// Forward, the last one partial), and compares every output with a NEON
// plan's bit for bit.
func TestConv2DPlanSMEMatchesNEON(t *testing.T) {
	skipWithoutSMEGEMM(t)
	lowerConv2DPlanSMEThresholds(t)
	sequentialConvWorkers(t)

	counts := countSMEGEMM(t)
	rng := planRNG(2200)

	for _, sc := range smeSweepCases() {
		name := sc.name()
		kernel := planRandTensor(t, rng, int64(sc.kh), int64(sc.kw), int64(sc.inC), int64(sc.outC))

		var bias *tensor.Tensor
		if sc.bias {
			bias = planRandTensor(t, rng, int64(sc.outC))
		}

		x := planRandTensor(t, rng, 1, int64(sc.inH), int64(sc.inW), int64(sc.inC))
		sp, np := newSMEPlanPair(t, kernel, bias, sc.inH, sc.inW, sc.p)

		if positions := sp.outH * sp.outW; sc.multiChunk && (sp.smeChunk >= positions || positions%sp.smeChunk == 0) {
			t.Fatalf("%s: chunk %d of %d positions, want several with a partial last one", name, sp.smeChunk, positions)
		}

		want := planForward(t, np, x, neonSentinel)

		for _, chunk := range []int{0, 32, 16} {
			if chunk > 0 && !setSMEChunk(sp, chunk) {
				continue
			}

			before := counts.calls.Load()
			got := planForward(t, sp, x, smeSentinel)

			if counts.calls.Load() == before {
				t.Fatalf("%s/chunk=%d: Forward made no sme.GEMM call", name, sp.smeChunk)
			}

			assertSameBits(t, fmt.Sprintf("%s/chunk=%d", name, sp.smeChunk), got, want)
		}
	}
}

// smeChunkLayer is a layer of the chunking, worker and fallback tests
// with the number of positions per sme.GEMM call: none of the output
// widths divides it, so ForwardRows' calls start their chunks elsewhere
// than Forward's, and chunks span output rows.
type smeChunkLayer struct {
	planCase

	chunk int
}

var smeChunkLayers = []smeChunkLayer{
	{planCase{name: "5x5/s1x3/outW=9", h: 13, w: 26, c: 3, kh: 5, kw: 5, cout: 12, sh: 1, sw: 3, seed: 2210}, 16},
	{planCase{name: "3x3/s2x2/outW=11", h: 12, w: 22, c: 8, kh: 3, kw: 3, cout: 16, sh: 2, sw: 2, seed: 2211}, 32},
	{planCase{name: "3x39/s1/outW=45", h: 16, w: 45, c: 2, kh: 3, kw: 39, cout: 8, sh: 1, sw: 1, seed: 2212}, 16},
	{planCase{name: "2x3/s1/outW=19", h: 14, w: 19, c: 5, kh: 2, kw: 3, cout: 36, sh: 1, sw: 1, noBias: true, seed: 2213}, 32},
}

// smePlans returns the layer's plans, the SME one set to the layer's
// chunk, the input and the NEON plan's output.
func (l smeChunkLayer) smePlans(t *testing.T) (smePlan *Conv2DPlan, x *tensor.Tensor, full []float32) {
	t.Helper()

	sp, np, x := l.smePlanPair(t)
	if !setSMEChunk(sp, l.chunk) {
		t.Fatalf("chunk %d does not fit the plan's %d", l.chunk, sp.smeChunk)
	}

	return sp, x, planForward(t, np, x, neonSentinel)
}

// smeChunkings returns chunkings of h rows as step sizes: one step, single
// rows, steps of 2 and 3, and random steps of 0 to 5 rows.
func smeChunkings(h int, seed uint64) map[string][]int {
	chunkings := map[string][]int{"all": {h}}

	for _, size := range []int{1, 2, 3} {
		steps := make([]int, (h+size-1)/size)
		for i := range steps {
			steps[i] = size
		}

		chunkings[fmt.Sprintf("by-%d", size)] = steps
	}

	rng := planRNG(seed)

	for i := range 4 {
		var steps []int

		for sum := 0; sum < h; {
			s := rng.IntN(6)
			steps = append(steps, s)
			sum += s
		}

		chunkings[fmt.Sprintf("random-%d", i)] = steps
	}

	return chunkings
}

// forwardSteps runs ForwardRows over all rows of c in the given steps into
// a new output filled with sentinel and returns its values.
func forwardSteps(t *testing.T, c *Conv2DPlan, x *tensor.Tensor, steps []int, sentinel float32) []float32 {
	t.Helper()

	h, _, _ := c.OutShape()
	dst := planDst(t, c, sentinel)
	r0 := 0

	for _, n := range steps {
		r1 := min(r0+n, h)

		err := c.ForwardRows(dst, x, r0, r1)
		if err != nil {
			t.Fatal(err)
		}

		r0 = r1
	}

	if r0 != h {
		t.Fatalf("chunking covers %d of %d rows", r0, h)
	}

	return dst.RawData()
}

// TestConv2DPlanSMEForwardRows checks that ForwardRows on sme.GEMM, in
// chunkings of single rows, steps of 2 and 3 and random steps (whose
// positions start inside Forward's chunks), on one to four workers, gives
// the NEON plan's bits and writes only the rows asked for.
func TestConv2DPlanSMEForwardRows(t *testing.T) {
	skipWithoutSMEGEMM(t)
	lowerConv2DPlanSMEThresholds(t)
	sequentialConvWorkers(t)

	for _, l := range smeChunkLayers {
		t.Run(l.name, func(t *testing.T) {
			sp, x, full := l.smePlans(t)
			h, _, _ := sp.OutShape()

			for workers := 1; workers <= 4; workers++ {
				SetConvWorkers(workers)

				assertSameBits(t, fmt.Sprintf("workers=%d/Forward", workers), planForward(t, sp, x, smeSentinel), full)

				for name, steps := range smeChunkings(h, l.seed+uint64(workers)) {
					got := forwardSteps(t, sp, x, steps, smeSentinel)
					assertSameBits(t, fmt.Sprintf("workers=%d/%s", workers, name), got, full)
				}

				for _, rr := range [][2]int{{0, 0}, {0, 1}, {1, 2}, {h - 1, h}, {1, h - 1}, {2, 5}, {3, 3}, {h, h}} {
					t.Run(fmt.Sprintf("workers=%d/only[%d,%d)", workers, rr[0], rr[1]), func(t *testing.T) {
						dst := planDst(t, sp, smeSentinel)

						err := sp.ForwardRows(dst, x, rr[0], rr[1])
						if err != nil {
							t.Fatal(err)
						}

						assertRows(t, sp, dst.RawData(), full, rr[0], rr[1], smeSentinel)
					})
				}
			}
		})
	}
}

// TestConv2DPlanSMEMixedPaths sets the per-call threshold between one
// row's and four rows' multiply-adds, so that calls of ForwardRows (or a
// worker's share) of up to three rows run on NEON and longer ones on
// sme.GEMM within the same plan, and checks that the bits match the NEON
// plan's either way.
func TestConv2DPlanSMEMixedPaths(t *testing.T) {
	skipWithoutSMEGEMM(t)
	lowerConv2DPlanSMEThresholds(t)
	sequentialConvWorkers(t)

	for _, l := range smeChunkLayers {
		t.Run(l.name, func(t *testing.T) {
			sp, x, full := l.smePlans(t)
			h, _, _ := sp.OutShape()

			// Four rows hold enough work, three do not.
			setConv2DPlanSMEThresholds(t, 0, 0, 0, 3*sp.MACsPerRow()+1)

			counts := countSMEGEMM(t)

			for workers := 1; workers <= 4; workers++ {
				SetConvWorkers(workers)

				for name, steps := range smeChunkings(h, l.seed+10+uint64(workers)) {
					before := counts.calls.Load()
					got := forwardSteps(t, sp, x, steps, smeSentinel)
					assertSameBits(t, fmt.Sprintf("workers=%d/%s", workers, name), got, full)

					if calls := counts.calls.Load() - before; (name == "by-1" || name == "by-2" || name == "by-3") && calls != 0 {
						t.Errorf("workers=%d/%s: %d sme.GEMM calls for steps of at most three rows, want 0", workers, name, calls)
					}
				}

				assertSameBits(t, fmt.Sprintf("workers=%d/Forward", workers), planForward(t, sp, x, smeSentinel), full)
			}

			if counts.calls.Load() == 0 {
				t.Error("no call ran on sme.GEMM")
			}
		})
	}
}

// TestConv2DPlanSMEFallback replaces sme.GEMM by kernels that report a
// chunk as not bit-exact, after scribbling over its outputs or without
// computing anything, on every call or every other one, and checks that
// the NEON recomputation gives the NEON plan's bits.
func TestConv2DPlanSMEFallback(t *testing.T) {
	skipWithoutSMEGEMM(t)
	lowerConv2DPlanSMEThresholds(t)
	sequentialConvWorkers(t)

	inner := sme.GEMM
	garbage := math.Float32frombits(0x7fc0dead)

	scribble := func(c []float32, ldc, m, n int) {
		for i := range m {
			row := c[i*ldc : i*ldc+n]
			for j := range row {
				row[j] = garbage
			}
		}
	}

	var alternate atomic.Int64

	kernels := []struct {
		name string
		f    smeGEMMFunc
	}{
		{"scribble", func(c []float32, ldc int, a, po, pb, bias []float32, m, n, k int) bool {
			inner(c, ldc, a, po, pb, bias, m, n, k)
			scribble(c, ldc, m, n)

			return false
		}},
		{"no-op", func([]float32, int, []float32, []float32, []float32, []float32, int, int, int) bool {
			return false
		}},
		{"every-other", func(c []float32, ldc int, a, po, pb, bias []float32, m, n, k int) bool {
			ok := inner(c, ldc, a, po, pb, bias, m, n, k)
			if alternate.Add(1)%2 == 0 {
				scribble(c, ldc, m, n)

				return false
			}

			return ok
		}},
	}

	for _, l := range smeChunkLayers {
		t.Run(l.name, func(t *testing.T) {
			sp, x, full := l.smePlans(t)
			h, _, _ := sp.OutShape()

			for _, kern := range kernels {
				t.Run(kern.name, func(t *testing.T) {
					setSMEGEMM(t, kern.f)

					counts := countSMEGEMM(t)

					for workers := 1; workers <= 3; workers++ {
						SetConvWorkers(workers)

						assertSameBits(t, fmt.Sprintf("workers=%d/Forward", workers), planForward(t, sp, x, smeSentinel), full)

						dst := planDst(t, sp, smeSentinel)

						err := sp.ForwardRows(dst, x, 1, h-1)
						if err != nil {
							t.Fatal(err)
						}

						assertRows(t, sp, dst.RawData(), full, 1, h-1, smeSentinel)
					}

					if counts.falses.Load() == 0 {
						t.Fatal("sme.GEMM never returned false")
					}
				})
			}
		})
	}
}

// TestConv2DPlanSMENonFinite checks the outputs sme.GEMM reports as not
// bit-exact: an infinite weight, which times the zero padding gives NaN
// and times the image infinities (as in TestConv2DPlanPaddingTimesWeight),
// in every chunk, and a NaN input, which reaches a few chunks only. The
// plan must give the NEON plan's bits, NaN payloads included.
func TestConv2DPlanSMENonFinite(t *testing.T) {
	skipWithoutSMEGEMM(t)
	lowerConv2DPlanSMEThresholds(t)
	sequentialConvWorkers(t)

	t.Run("inf-weight", func(t *testing.T) {
		rng := planRNG(41)
		x := planRandTensor(t, rng, 1, 8, 8, 4)
		kernel := planRandTensor(t, rng, 3, 3, 4, 8)
		kernel.RawData()[0] = float32(math.Inf(1)) // tap (0, 0), input channel 0, out channel 0
		p := Conv2DParams{StrideH: 1, StrideW: 1, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1}

		sp, np := newSMEPlanPair(t, kernel, planRandTensor(t, rng, 8), 8, 8, p)
		want := planForward(t, np, x, neonSentinel)

		if v := want[0]; !math.IsNaN(float64(v)) {
			t.Fatalf("NEON output (0, 0, 0) = %v, want NaN", v)
		}

		if v := want[(1*8+1)*8]; !math.IsInf(float64(v), 0) {
			t.Fatalf("NEON output (1, 1, 0) = %v, want infinite", v)
		}

		counts := countSMEGEMM(t)

		for _, chunk := range []int{0, 16} {
			if chunk > 0 && !setSMEChunk(sp, chunk) {
				t.Fatalf("chunk %d does not fit the plan's %d", chunk, sp.smeChunk)
			}

			assertSameBits(t, fmt.Sprintf("chunk=%d", sp.smeChunk), planForward(t, sp, x, smeSentinel), want)
		}

		if counts.falses.Load() == 0 {
			t.Fatal("sme.GEMM never returned false")
		}
	})

	t.Run("nan-input", func(t *testing.T) {
		tc := planCase{h: 12, w: 20, c: 4, kh: 3, kw: 3, cout: 16, sh: 1, sw: 1, seed: 2220}
		sp, np, x := tc.smePlanPair(t)
		x.RawData()[(5*20+7)*4+2] = float32(math.NaN())

		want := planForward(t, np, x, neonSentinel)

		if !setSMEChunk(sp, 16) {
			t.Fatalf("chunk 16 does not fit the plan's %d", sp.smeChunk)
		}

		counts := countSMEGEMM(t)
		assertSameBits(t, "chunk=16", planForward(t, sp, x, smeSentinel), want)

		if calls, falses := counts.calls.Load(), counts.falses.Load(); falses == 0 || falses == calls {
			t.Fatalf("%d of %d sme.GEMM calls returned false, want some but not all", falses, calls)
		}
	})
}

// TestConv2DPlanSMEPerCall checks the per-call rule with the default
// thresholds on basic-pitch's conv2d_2 over 12 frames (k = 49, 32 out
// channels, 12 × 88 positions, which the plan takes): a call of fewer rows
// than hold conv2DPlanSMEMinFMAs multiply-adds makes no sme.GEMM call, one
// of that many rows makes at least one, and both give the NEON plan's
// bits.
func TestConv2DPlanSMEPerCall(t *testing.T) {
	skipWithoutSMEGEMM(t)
	sequentialConvWorkers(t)

	tc := planCase{h: 12, w: 264, c: 1, kh: 7, kw: 7, cout: 32, sh: 1, sw: 3, seed: 2230}
	sp, np, x := tc.smePlanPair(t)
	full := planForward(t, np, x, neonSentinel)

	// 2^19 multiply-adds take 4 rows of 88 · 49 · 32 = 137984.
	need := (conv2DPlanSMEMinFMAs + sp.MACsPerRow() - 1) / sp.MACsPerRow()
	if need != 4 {
		t.Fatalf("%d rows hold conv2DPlanSMEMinFMAs = %d multiply-adds, want 4", need, conv2DPlanSMEMinFMAs)
	}

	counts := countSMEGEMM(t)

	for _, rr := range [][2]int{{0, 1}, {5, 6}, {0, 3}, {9, 12}, {0, 4}, {3, 10}, {0, 12}} {
		before := counts.calls.Load()
		dst := planDst(t, sp, smeSentinel)

		err := sp.ForwardRows(dst, x, rr[0], rr[1])
		if err != nil {
			t.Fatal(err)
		}

		assertRows(t, sp, dst.RawData(), full, rr[0], rr[1], smeSentinel)

		calls := counts.calls.Load() - before
		if rows := rr[1] - rr[0]; rows < need && calls != 0 {
			t.Errorf("rows [%d, %d): %d sme.GEMM calls, want 0 below %d rows", rr[0], rr[1], calls, need)
		} else if rows >= need && calls == 0 {
			t.Errorf("rows [%d, %d): no sme.GEMM call, want at least one from %d rows on", rr[0], rr[1], need)
		}
	}
}

// TestConv2DPlanSMEZeroAllocs checks that Forward and ForwardRows on
// sme.GEMM do not allocate on one worker: the plan owns the packed
// weights and the scratch.
func TestConv2DPlanSMEZeroAllocs(t *testing.T) {
	skipWithoutSMEGEMM(t)
	lowerConv2DPlanSMEThresholds(t)
	sequentialConvWorkers(t)

	// conv2d_4 geometry over 8 frames.
	tc := planCase{h: 8, w: 264, c: 8, kh: 5, kw: 5, cout: 32, sh: 1, sw: 3, seed: 2240}
	sp, _, x := tc.smePlanPair(t)
	dst := planDst(t, sp, 0)
	counts := countSMEGEMM(t)

	for _, chunk := range []int{0, 16} {
		if chunk > 0 && !setSMEChunk(sp, chunk) {
			t.Fatalf("chunk %d does not fit the plan's %d", chunk, sp.smeChunk)
		}

		var err error

		if allocs := testing.AllocsPerRun(5, func() { err = sp.ForwardRows(dst, x, 2, 5) }); allocs != 0 {
			t.Errorf("chunk=%d: ForwardRows: %v allocations per run, want 0", sp.smeChunk, allocs)
		}

		if err != nil {
			t.Fatal(err)
		}

		if allocs := testing.AllocsPerRun(3, func() { err = sp.Forward(dst, x) }); allocs != 0 {
			t.Errorf("chunk=%d: Forward: %v allocations per run, want 0", sp.smeChunk, allocs)
		}

		if err != nil {
			t.Fatal(err)
		}
	}

	if counts.calls.Load() == 0 {
		t.Fatal("no call ran on sme.GEMM")
	}
}

// TestConv2DPlanSMETakes pins the plans conv2DPlanSMETakes selects with the
// default thresholds. It runs on every platform.
func TestConv2DPlanSMETakes(t *testing.T) {
	if conv2DPlanSMEMinK != 32 || conv2DPlanSMEMinWork != 1536 || conv2DPlanSMEMinPositions != 1024 || conv2DPlanSMEMinFMAs != 1<<19 {
		t.Fatalf("thresholds k %d, work %d, positions %d, FMAs %d, want 32, 1536, 1024, 2^19",
			conv2DPlanSMEMinK, conv2DPlanSMEMinWork, conv2DPlanSMEMinPositions, conv2DPlanSMEMinFMAs)
	}

	cases := []struct {
		name               string
		k, outC, positions int
		want               bool
	}{
		{"basic-pitch conv2d_1", 936, 8, 172 * 264, true},
		{"basic-pitch conv2d_2", 49, 32, 172 * 88, true},
		{"basic-pitch conv2d_4", 200, 32, 172 * 88, true},
		{"basic-pitch contours-reduced", 200, 1, 172 * 264, false},
		{"basic-pitch conv2d_3", 672, 1, 172 * 88, false},
		{"basic-pitch conv2d_5", 297, 1, 172 * 88, false},
		{"go-whistle stem", 9, 128, 1500 * 40, false},
		{"go-whistle stem chunk", 9, 128, 35 * 40, false},
		{"outC 33", 200, 33, 172 * 88, false},
		{"outC 6", 936, 6, 172 * 264, false},
		{"outC 30", 512, 30, 4096, false},
		{"k=288 outC=8: half-empty tiles", 288, 8, 172 * 264, false},
		{"k=384 outC=8: work 1536", 384, 8, 4096, true},
		{"k=380 outC=8: work 1520", 380, 8, 4096, false},
		{"k=32 outC=48: work 1536", 32, 48, 4096, true},
		{"k=31", 31, 128, 4096, false},
		{"k=3", 3, 128, 4096, false},
		{"k=36 outC=128 at 1024 positions", 36, 128, 1024, true},
		{"k=36 outC=128 at 1023 positions", 36, 128, 1023, false},
		{"k=36 outC=128 at 256 positions", 36, 128, 256, false},
	}

	for _, tc := range cases {
		if got := conv2DPlanSMETakes(tc.k, tc.outC, tc.positions); got != tc.want {
			t.Errorf("%s: conv2DPlanSMETakes(%d, %d, %d) = %t, want %t", tc.name, tc.k, tc.outC, tc.positions, got, tc.want)
		}
	}
}

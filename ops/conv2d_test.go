// From github.com/cwbudde/go-whistle internal/ops/conv2d_test.go @ 8926ee3;
// go-whistle's own, not in go-pocket-tts.
// algo-tensor change: the tensor import is github.com/cwbudde/algo-tensor/tensor.
// algo-tensor change: Conv2D and its plan path against conv2DDirect, the per-pixel path, on the
// golden configurations and the basic-pitch shapes; the routing rule; batches, a nil bias, conv
// workers and zero sizes on the plan path; the plan's NaN where an infinite weight meets
// padding (PLAN 1.3).

package ops

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// conv2DRef is a float64 transcription of lax.conv_general_dilated in
// NHWC/HWIO with explicit padding and feature groups. It also returns
// Σ|x·k| per output for the rounding bound.
func conv2DRef(x []float32, shape [4]int, k []float32, kh, kw, o int, b []float32, p Conv2DParams,
) (out, absSum []float64, oh, ow int) {
	n, h, w, c := shape[0], shape[1], shape[2], shape[3]
	oh = (h+p.PadTop+p.PadBottom-kh)/p.StrideH + 1
	ow = (w+p.PadLeft+p.PadRight-kw)/p.StrideW + 1
	inPer, outPer := c/p.Groups, o/p.Groups
	out, absSum = make([]float64, n*oh*ow*o), make([]float64, n*oh*ow*o)

	for idx := range out {
		oc := idx % o
		z := idx / o % ow
		y := idx / o / ow % oh
		b0 := idx / o / ow / oh

		var s, a float64
		if b != nil {
			s, a = float64(b[oc]), math.Abs(float64(b[oc]))
		}

		for i := range kh {
			for j := range kw {
				iy, iz := y*p.StrideH+i-p.PadTop, z*p.StrideW+j-p.PadLeft
				if iy < 0 || iy >= h || iz < 0 || iz >= w {
					continue
				}

				for ci := range inPer {
					xv := float64(x[((b0*h+iy)*w+iz)*c+oc/outPer*inPer+ci])
					kv := float64(k[((i*kw+j)*inPer+ci)*o+oc])
					s += xv * kv
					a += math.Abs(xv * kv)
				}
			}
		}

		out[idx], absSum[idx] = s, a
	}

	return out, absSum, oh, ow
}

func TestConv2DMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 14))

	cases := []struct {
		name   string
		shape  [4]int // n, h, w, c
		o      int
		kh, kw int
		p      Conv2DParams
		bias   bool
	}{
		{"stem conv stride 2 same", [4]int{1, 12, 10, 1}, 8, 3, 3, Conv2DParams{2, 2, 0, 1, 0, 1, 1}, false},
		{"depthwise stride 2 pad 1", [4]int{1, 9, 8, 8}, 8, 3, 3, Conv2DParams{2, 2, 1, 1, 1, 1, 8}, false},
		{"depthwise multiplier 2", [4]int{2, 7, 5, 3}, 6, 3, 3, Conv2DParams{1, 1, 1, 1, 1, 1, 3}, true},
		{"groups 2", [4]int{1, 6, 6, 4}, 6, 2, 3, Conv2DParams{1, 2, 0, 1, 1, 0, 2}, true},
		{"valid stride 1", [4]int{2, 5, 7, 3}, 4, 3, 2, Conv2DParams{1, 1, 0, 0, 0, 0, 1}, true},
		{"asymmetric stride", [4]int{1, 11, 4, 2}, 5, 3, 3, Conv2DParams{3, 1, 2, 0, 1, 1, 1}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, h, w, c := tc.shape[0], tc.shape[1], tc.shape[2], tc.shape[3]
			x := randSlice(rng, n*h*w*c)
			k := randSlice(rng, tc.kh*tc.kw*c/tc.p.Groups*tc.o)

			var (
				b  []float32
				bt *tensor.Tensor
			)

			if tc.bias {
				b = randSlice(rng, tc.o)
				bt = mustTensorT(t, b, []int64{int64(tc.o)})
			}

			want, absSum, oh, ow := conv2DRef(x, tc.shape, k, tc.kh, tc.kw, tc.o, b, tc.p)

			xt := mustTensorT(t, x, []int64{int64(n), int64(h), int64(w), int64(c)})
			kt := mustTensorT(t, k, []int64{int64(tc.kh), int64(tc.kw), int64(c / tc.p.Groups), int64(tc.o)})

			got, err := Conv2D(xt, kt, bt, tc.p)
			if err != nil {
				t.Fatal(err)
			}

			if s := got.Shape(); s[0] != int64(n) || s[1] != int64(oh) || s[2] != int64(ow) || s[3] != int64(tc.o) {
				t.Fatalf("shape %v, want [%d %d %d %d]", s, n, oh, ow, tc.o)
			}

			// Float32 accumulation of m terms: |Δ| ≤ γ(m)·Σ|x·k| with γ(m) = m·u/(1 − m·u).
			terms := float64(tc.kh*tc.kw*c/tc.p.Groups + 2)
			gamma := terms * 0x1p-24 / (1 - terms*0x1p-24)

			for i, v := range got.RawData() {
				if d := math.Abs(float64(v) - want[i]); d > gamma*absSum[i] {
					t.Fatalf("out[%d] = %v, reference %v (|Δ| %g > %g)", i, v, want[i], d, gamma*absSum[i])
				}
			}
		})
	}
}

func TestConv2DHandComputed(t *testing.T) {
	// 4×4 ramp, 3×3 box kernel, stride 2, SAME (0 before, 1 after): output
	// (i, j) sums rows 2i…2i+2 and columns 2j…2j+2 inside the image.
	x := make([]float32, 16)
	for i := range x {
		x[i] = float32(i)
	}

	k := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}

	got, err := Conv2D(mustTensorT(t, x, []int64{1, 4, 4, 1}), mustTensorT(t, k, []int64{3, 3, 1, 1}), nil,
		Conv2DParams{StrideH: 2, StrideW: 2, PadBottom: 1, PadRight: 1, Groups: 1})
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{
		0 + 1 + 2 + 4 + 5 + 6 + 8 + 9 + 10, 2 + 3 + 6 + 7 + 10 + 11,
		8 + 9 + 10 + 12 + 13 + 14, 10 + 11 + 14 + 15,
	}
	if !equalApprox(got.RawData(), want, 0) {
		t.Fatalf("got %v, want %v", got.RawData(), want)
	}
}

func TestSamePadding(t *testing.T) {
	cases := []struct{ n, k, s, lo, hi, out int }{
		{80, 3, 2, 0, 1, 40},
		{75, 3, 2, 1, 1, 38},
		{1, 3, 2, 1, 1, 1},
		{2, 3, 2, 0, 1, 1},
		{10, 3, 1, 1, 1, 10},
		{9, 9, 1, 4, 4, 9},
		{7, 1, 2, 0, 0, 4},
	}

	for _, c := range cases {
		lo, hi := SamePadding(c.n, c.k, c.s)
		if lo != c.lo || hi != c.hi {
			t.Errorf("SamePadding(%d, %d, %d) = %d, %d, want %d, %d", c.n, c.k, c.s, lo, hi, c.lo, c.hi)
		}

		if out := (c.n+lo+hi-c.k)/c.s + 1; out != c.out {
			t.Errorf("n %d k %d s %d: %d outputs, want ceil = %d", c.n, c.k, c.s, out, c.out)
		}
	}
}

func TestConv2DErrors(t *testing.T) {
	x := mustTensorT(t, seqDataT(2*3*4), []int64{1, 2, 3, 4})
	k := mustTensorT(t, seqDataT(3*3*4*2), []int64{3, 3, 4, 2})
	ok := Conv2DParams{StrideH: 1, StrideW: 1, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1}

	_, err := Conv2D(nil, k, nil, ok)
	assertErrContains(t, err, "non-nil")

	bad := ok
	bad.StrideH = 0
	_, err = Conv2D(x, k, nil, bad)
	assertErrContains(t, err, "> 0")

	bad = ok
	bad.PadLeft = -1
	_, err = Conv2D(x, k, nil, bad)
	assertErrContains(t, err, ">= 0")

	_, err = Conv2D(mustTensorT(t, seqDataT(12), []int64{3, 4}), k, nil, ok)
	assertErrContains(t, err, "rank 4")

	bad = ok
	bad.Groups = 2
	_, err = Conv2D(x, k, nil, bad)
	assertErrContains(t, err, "groups")

	_, err = Conv2D(x, k, mustTensorT(t, seqDataT(3), []int64{3}), ok)
	assertErrContains(t, err, "bias")

	_, err = Conv2D(x, k, nil, Conv2DParams{StrideH: 1, StrideW: 1, Groups: 1})
	assertErrContains(t, err, "smaller than kernel")
}

// conv2DDiff returns the largest |got − want|, the largest
// |got − want| / |want| (over want ≠ 0) and the want where that is.
func conv2DDiff(got, want []float32) (maxAbs, maxRel float64, at float32) {
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		maxAbs = max(maxAbs, d)

		if rel := d / math.Abs(float64(want[i])); want[i] != 0 && rel > maxRel {
			maxRel, at = rel, want[i]
		}
	}

	return maxAbs, maxRel, at
}

// conv2DPaths runs Conv2D, conv2DDirect and the conv2DPlanned path on the
// same arguments and checks that Conv2D gave the planned bits where
// planGains holds for groups = 1 and the direct bits elsewhere.
func conv2DPaths(t *testing.T, x, kernel, bias *tensor.Tensor, p Conv2DParams) (got, direct, planned []float32) {
	t.Helper()

	run := func(path conv2DPath) *tensor.Tensor {
		out, err := conv2D(x, kernel, bias, p, path)
		if err != nil {
			t.Fatal(err)
		}

		return out
	}

	out, dir, pl := run(conv2DRouted), run(conv2DPixels), run(conv2DPlanned)
	if !slices.Equal(out.Shape(), dir.Shape()) || !slices.Equal(pl.Shape(), dir.Shape()) {
		t.Fatalf("shapes %v (Conv2D), %v (direct), %v (planned)", out.Shape(), dir.Shape(), pl.Shape())
	}

	g, err := prepareConv2D(x, kernel, bias, p)
	if err != nil {
		t.Fatal(err)
	}

	if p.Groups == 1 && g.planGains() {
		assertBitIdentical(t, out.RawData(), pl.RawData())
	} else {
		assertBitIdentical(t, out.RawData(), dir.RawData())
	}

	return out.RawData(), dir.RawData(), pl.RawData()
}

// TestConv2DMatchesDirect runs the three convolutions of the Conv2D golden
// case (golden_cases_whistle_test.go). Groups = 1 runs through Conv2DPlan
// (except on amd64) and stays within KernelTolerance("conv2d") of
// conv2DDirect, the per-pixel path it ran before v0.3.0; groups 2 and 4
// still are that path, bit for bit. It logs the largest differences of the
// plan's groups = 1 output.
func TestConv2DMatchesDirect(t *testing.T) {
	r := &goldenRand{21}
	x := r.tensor(t, 2, 9, 8, 4)
	top, bottom := SamePadding(9, 3, 2)
	left, right := SamePadding(8, 3, 2)

	cases := []struct {
		name         string
		kernel, bias *tensor.Tensor
		p            Conv2DParams
	}{
		{"groups 1", r.tensor(t, 3, 3, 4, 6), r.tensor(t, 6), Conv2DParams{1, 1, 1, 1, 1, 1, 1}},
		{"groups 2", r.tensor(t, 3, 3, 2, 6), nil, Conv2DParams{1, 2, 0, 1, 2, 0, 2}},
		{"depthwise multiplier 2", r.tensor(t, 3, 3, 1, 8), r.tensor(t, 8), Conv2DParams{2, 2, top, bottom, left, right, 4}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, direct, planned := conv2DPaths(t, x, tc.kernel, tc.bias, tc.p)
			if tc.p.Groups > 1 {
				assertBitIdentical(t, got, direct)

				return
			}

			assertConv2DTolerance(t, got, direct)
			assertConv2DTolerance(t, planned, direct)

			maxAbs, maxRel, at := conv2DDiff(planned, direct)
			t.Logf("%s: Conv2D runs the plan: %v; plan against direct on %d values: max |Δ| %.3g, max |Δ|/|direct| %.3g (at direct = %.3g)",
				goldenConfig(), slices.Equal(got, planned), len(got), maxAbs, maxRel, at)
		})
	}
}

// TestConv2DBasicPitchMatchesDirect runs Conv2D, conv2DDirect and the plan
// path on the six basic-pitch shapes (conv2d_plan_test.go), on the first
// 12 rows of the input under reduced work.
func TestConv2DBasicPitchMatchesDirect(t *testing.T) {
	for i, tc := range basicPitchLayers {
		tc.seed = uint64(600 + i)
		if planReducedWork() {
			tc.h = 12
		}

		t.Run(tc.name, func(t *testing.T) {
			kernel, bias, x := tc.fixture(t)
			got, direct, _ := conv2DPaths(t, x, kernel, bias, tc.params())
			assertConv2DTolerance(t, got, direct)
		})
	}
}

// TestConv2DPlanGains pins Conv2D's routing (planGains): only groups = 1
// geometries with a GEMM depth of at least 4, at least 64 output positions
// over the batch and at most 16 out channels per unit of depth run through
// a Conv2DPlan, and none on amd64.
func TestConv2DPlanGains(t *testing.T) {
	cases := []struct {
		name              string
		batch, h, w, c    int
		kh, kw, o, sh, sw int
		want              bool
	}{
		{"golden groups 1", 2, 9, 8, 4, 3, 3, 6, 1, 1, true},
		{"go-whistle stem chunk", 1, 71, 80, 1, 3, 3, 128, 2, 2, true},
		{"basic-pitch conv2d_1", 1, 172, 264, 8, 3, 39, 8, 1, 1, true},
		{"basic-pitch conv2d_3", 1, 172, 88, 32, 7, 3, 1, 1, 1, true},
		{"depth 3", 1, 16, 16, 3, 1, 1, 8, 1, 1, false},
		{"depth 4", 1, 16, 16, 4, 1, 1, 8, 1, 1, true},
		{"63 positions", 1, 9, 7, 4, 1, 1, 8, 1, 1, false},
		{"64 positions over a batch of 4", 4, 4, 4, 4, 1, 1, 8, 1, 1, true},
		{"16·k out channels", 1, 8, 8, 9, 1, 1, 144, 1, 1, true},
		{"16·k + 1 out channels", 1, 8, 8, 9, 1, 1, 145, 1, 1, false},
		{"no out channels", 1, 8, 8, 9, 1, 1, 0, 1, 1, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := samePadParams(tc.h, tc.w, tc.kh, tc.kw, tc.sh, tc.sw)
			x := mustTensorT(t, make([]float32, tc.batch*tc.h*tc.w*tc.c), []int64{int64(tc.batch), int64(tc.h), int64(tc.w), int64(tc.c)})
			k := mustTensorT(t, make([]float32, tc.kh*tc.kw*tc.c*tc.o), []int64{int64(tc.kh), int64(tc.kw), int64(tc.c), int64(tc.o)})

			g, err := prepareConv2D(x, k, nil, p)
			if err != nil {
				t.Fatal(err)
			}

			want := tc.want && runtime.GOARCH != "amd64"
			if got := g.planGains(); got != want {
				t.Errorf("planGains() = %v on %s, want %v", got, runtime.GOARCH, want)
			}
		})
	}
}

// TestConv2DBatchAndNilBias: on the plan path every image of a batch gives
// the bits of the plan on that image alone, a nil bias the bits of a zero
// bias, and conv workers the bits of the sequential path; Conv2D gives
// them too (except on amd64, where it runs the per-pixel path).
func TestConv2DBatchAndNilBias(t *testing.T) {
	defer SetConvWorkers(getConvWorkers())

	SetConvWorkers(0)

	rng := planRNG(31)
	p := samePadParams(11, 13, 3, 5, 2, 1)
	x := planRandTensor(t, rng, 3, 11, 13, 4)
	kernel := planRandTensor(t, rng, 3, 5, 4, 9)

	zero, err := tensor.Zeros([]int64{9})
	if err != nil {
		t.Fatal(err)
	}

	planned := func(x, bias *tensor.Tensor) []float32 {
		out, err := conv2D(x, kernel, bias, p, conv2DPlanned)
		if err != nil {
			t.Fatal(err)
		}

		return out.RawData()
	}

	got, direct, batch := conv2DPaths(t, x, kernel, nil, p)
	assertConv2DTolerance(t, got, direct)
	assertConv2DTolerance(t, batch, direct)
	assertBitIdentical(t, planned(x, zero), batch)

	img := len(batch) / 3

	for n := range 3 {
		assertBitIdentical(t, planned(planImage(t, x, n), nil), batch[n*img:(n+1)*img])
	}

	SetConvWorkers(3)
	assertBitIdentical(t, planned(x, nil), batch)
}

// TestConv2DZeroSizes: shapes with a zero size, which Conv2DPlan does not
// take, run on the per-pixel path also where the plan path is asked for:
// no input channels give the bias, an empty batch an empty output.
func TestConv2DZeroSizes(t *testing.T) {
	p := Conv2DParams{StrideH: 1, StrideW: 1, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1}
	bias := mustTensorT(t, []float32{1, -2, 3}, []int64{3})

	for _, path := range []conv2DPath{conv2DRouted, conv2DPlanned} {
		got, err := conv2D(mustTensorT(t, nil, []int64{1, 2, 2, 0}), mustTensorT(t, nil, []int64{3, 3, 0, 3}), bias, p, path)
		if err != nil {
			t.Fatal(err)
		}

		want := []float32{1, -2, 3, 1, -2, 3, 1, -2, 3, 1, -2, 3}
		if !slices.Equal(got.Shape(), []int64{1, 2, 2, 3}) || !slices.Equal(got.RawData(), want) {
			t.Fatalf("no input channels, path %d: %v %v, want [1 2 2 3] %v", path, got.Shape(), got.RawData(), want)
		}

		got, err = conv2D(mustTensorT(t, nil, []int64{0, 2, 2, 1}), mustTensorT(t, seqDataT(27), []int64{3, 3, 1, 3}), bias, p, path)
		if err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(got.Shape(), []int64{0, 2, 2, 3}) || got.ElemCount() != 0 {
			t.Fatalf("empty batch, path %d: %v, want [0 2 2 3] and no values", path, got.Shape())
		}
	}
}

// TestConv2DPlanPaddingTimesWeight pins a difference of the plan path: it
// multiplies the zero padding by the weights, so an infinite weight that
// only meets padding at an output gives NaN there (0·Inf), where the
// per-pixel path skips the tap and stays finite.
func TestConv2DPlanPaddingTimesWeight(t *testing.T) {
	rng := planRNG(41)
	x := planRandTensor(t, rng, 1, 8, 8, 4)
	kernel := planRandTensor(t, rng, 3, 3, 4, 8)
	kernel.RawData()[0] = float32(math.Inf(1)) // tap (0, 0), input channel 0, out channel 0
	p := Conv2DParams{StrideH: 1, StrideW: 1, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1}

	direct, err := conv2DDirect(x, kernel, nil, p)
	if err != nil {
		t.Fatal(err)
	}

	planned, err := conv2D(x, kernel, nil, p, conv2DPlanned)
	if err != nil {
		t.Fatal(err)
	}

	// Output (0, 0) reads tap (0, 0) from the top-left padding; output
	// (1, 1) reads it from the image.
	if d, q := direct.RawData()[0], planned.RawData()[0]; math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) || !math.IsNaN(float64(q)) {
		t.Errorf("output (0, 0): direct %v, want finite; plan %v, want NaN", d, q)
	}

	inner := (1*8 + 1) * 8
	if d, q := direct.RawData()[inner], planned.RawData()[inner]; !math.IsInf(float64(d), 0) || !math.IsInf(float64(q), 0) {
		t.Errorf("output (1, 1): direct %v, plan %v, want both infinite", d, q)
	}
}

func randSlice(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}

	return out
}

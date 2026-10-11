// From github.com/cwbudde/algo-transcribe nn/conv_test.go @ 21ad164.
// algo-tensor change: the plan is checked against ops.Conv2D within KernelTolerance("conv2d")
// instead of nn's float64 reference, on tensor.Tensor images with Conv2DParams padding; nn's
// reference cases with SamePadding, the groups = 1 cases of TestConv2DMatchesReference and the
// golden configuration added; the basic-pitch layers under reduced work compare row windows.
// algo-tensor change: chunkings of 2, 3 and 7 rows and random partitions; the SetConvWorkers
// split; errors for permuted and batch-2 shapes, groups and padding; the tests renamed
// Conv2DPlan…; nn's samePadding, golden and Padding tests left out (ops has its own).

package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// planReducedWork reports whether expensive tests should check a subset of
// the work: under -short and under js/wasm, where Conv2D over a full
// basic-pitch layer is slow.
func planReducedWork() bool {
	return testing.Short() || runtime.GOARCH == "wasm"
}

// planRNG returns a deterministic generator for a test case.
func planRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
}

// planRandTensor returns a tensor of values uniformly distributed in [-1, 1).
func planRandTensor(t testing.TB, rng *rand.Rand, shape ...int64) *tensor.Tensor {
	t.Helper()

	n := int64(1)
	for _, d := range shape {
		n *= d
	}

	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}

	x, err := tensor.New(v, shape)
	if err != nil {
		t.Fatal(err)
	}

	return x
}

// planDst returns a [1, h, w, c] output tensor of the plan filled with v.
func planDst(t testing.TB, c *Conv2DPlan, v float32) *tensor.Tensor {
	t.Helper()

	h, w, ch := c.OutShape()

	dst, err := tensor.Full([]int64{1, int64(h), int64(w), int64(ch)}, v)
	if err != nil {
		t.Fatal(err)
	}

	return dst
}

// planImage returns image n of the NHWC tensor x as a [1, H, W, C] tensor.
func planImage(t testing.TB, x *tensor.Tensor, n int) *tensor.Tensor {
	t.Helper()

	h, w, c := x.Dim(1), x.Dim(2), x.Dim(3)
	img := int(h * w * c)

	out, err := tensor.New(x.RawData()[n*img:(n+1)*img], []int64{1, h, w, c})
	if err != nil {
		t.Fatal(err)
	}

	return out
}

// assertConv2DTolerance checks |got − want| ≤ tol.Abs + tol.Rel·|want| for
// KernelTolerance("conv2d").
func assertConv2DTolerance(t *testing.T, got, want []float32) {
	t.Helper()

	tol, err := KernelTolerance("conv2d")
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(want) {
		t.Fatalf("%d values, want %d", len(got), len(want))
	}

	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if d := math.Abs(g - w); !(d <= tol.Abs+tol.Rel*math.Abs(w)) {
			t.Fatalf("value %d = %v, Conv2D %v (|Δ| %g > %g)", i, g, w, d, tol.Abs+tol.Rel*math.Abs(w))
		}
	}
}

func assertBitIdentical(t *testing.T, got, want []float32) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}

	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d = %v (%#08x), want %v (%#08x)",
				i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// samePadParams returns the Conv2DParams of TensorFlow "same" padding for
// an inH × inW input, a kh × kw kernel and strides (sh, sw), groups 1.
func samePadParams(inH, inW, kh, kw, sh, sw int) Conv2DParams {
	top, bottom := SamePadding(inH, kh, sh)
	left, right := SamePadding(inW, kw, sw)

	return Conv2DParams{StrideH: sh, StrideW: sw, PadTop: top, PadBottom: bottom, PadLeft: left, PadRight: right, Groups: 1}
}

// checkPlanAgainstConv2D runs the plan on every image of x and compares
// with Conv2D within KernelTolerance("conv2d").
func checkPlanAgainstConv2D(t *testing.T, x, kernel, bias *tensor.Tensor, p Conv2DParams) {
	t.Helper()

	want, err := Conv2D(x, kernel, bias, p)
	if err != nil {
		t.Fatal(err)
	}

	plan, err := NewConv2DPlan(kernel, bias, int(x.Dim(1)), int(x.Dim(2)), p)
	if err != nil {
		t.Fatal(err)
	}

	if h, w, c := plan.OutShape(); int64(h) != want.Dim(1) || int64(w) != want.Dim(2) || int64(c) != want.Dim(3) {
		t.Fatalf("OutShape() = (%d, %d, %d), Conv2D gives %v", h, w, c, want.Shape())
	}

	got := make([]float32, 0, want.ElemCount())

	for n := range int(x.Dim(0)) {
		dst := planDst(t, plan, float32(math.NaN()))

		err := plan.Forward(dst, planImage(t, x, n))
		if err != nil {
			t.Fatal(err)
		}

		got = append(got, dst.RawData()...)
	}

	assertConv2DTolerance(t, got, want.RawData())
}

// TestConv2DPlanGoldenConfig runs the groups = 1 configuration of the
// Conv2D golden case (golden_cases_whistle_test.go) on each image.
func TestConv2DPlanGoldenConfig(t *testing.T) {
	r := &goldenRand{21}
	x := r.tensor(t, 2, 9, 8, 4)
	checkPlanAgainstConv2D(t, x, r.tensor(t, 3, 3, 4, 6), r.tensor(t, 6), Conv2DParams{1, 1, 1, 1, 1, 1, 1})
}

// TestConv2DPlanMatchesConv2D covers the groups = 1 cases of
// TestConv2DMatchesReference and explicit padding beyond the kernel.
func TestConv2DPlanMatchesConv2D(t *testing.T) {
	rng := planRNG(14)

	cases := []struct {
		name   string
		shape  [4]int64 // n, h, w, c
		o      int64
		kh, kw int64
		p      Conv2DParams
		bias   bool
	}{
		{"stem conv stride 2 same", [4]int64{1, 12, 10, 1}, 8, 3, 3, Conv2DParams{2, 2, 0, 1, 0, 1, 1}, false},
		{"valid stride 1", [4]int64{2, 5, 7, 3}, 4, 3, 2, Conv2DParams{1, 1, 0, 0, 0, 0, 1}, true},
		{"asymmetric stride", [4]int64{1, 11, 4, 2}, 5, 3, 3, Conv2DParams{3, 1, 2, 0, 1, 1, 1}, false},
		{"padding beyond kernel", [4]int64{1, 3, 4, 2}, 3, 3, 3, Conv2DParams{1, 2, 4, 4, 5, 5, 1}, true},
		{"padding one side only", [4]int64{1, 6, 9, 3}, 9, 2, 4, Conv2DParams{2, 1, 3, 0, 0, 6, 1}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := planRandTensor(t, rng, tc.shape[:]...)
			k := planRandTensor(t, rng, tc.kh, tc.kw, tc.shape[3], tc.o)

			var b *tensor.Tensor
			if tc.bias {
				b = planRandTensor(t, rng, tc.o)
			}

			checkPlanAgainstConv2D(t, x, k, b, tc.p)
		})
	}
}

// planCase is one "same" convolution of a single image.
type planCase struct {
	name          string
	h, w, c       int
	kh, kw, cout  int
	sh, sw        int
	noBias        bool
	seed          uint64
	wantH, wantW  int
	wantMACsPerRw int
}

func (tc planCase) params() Conv2DParams {
	return samePadParams(tc.h, tc.w, tc.kh, tc.kw, tc.sh, tc.sw)
}

// fixture returns the case's random kernel, bias (nil for noBias) and input.
func (tc planCase) fixture(t testing.TB) (kernel, bias, x *tensor.Tensor) {
	t.Helper()

	rng := planRNG(tc.seed)
	kernel = planRandTensor(t, rng, int64(tc.kh), int64(tc.kw), int64(tc.c), int64(tc.cout))

	if !tc.noBias {
		bias = planRandTensor(t, rng, int64(tc.cout))
	}

	x = planRandTensor(t, rng, 1, int64(tc.h), int64(tc.w), int64(tc.c))

	return kernel, bias, x
}

// plan returns the case's plan and input.
func (tc planCase) plan(t testing.TB) (*Conv2DPlan, *tensor.Tensor) {
	t.Helper()

	kernel, bias, x := tc.fixture(t)

	plan, err := NewConv2DPlan(kernel, bias, tc.h, tc.w, tc.params())
	if err != nil {
		t.Fatal(err)
	}

	return plan, x
}

// TestConv2DPlanSameReference runs nn's TestConv2DMatchesReference cases,
// "same" padding by SamePadding, against Conv2D.
func TestConv2DPlanSameReference(t *testing.T) {
	cases := []planCase{
		{name: "1x1/s1", h: 5, w: 7, c: 3, kh: 1, kw: 1, cout: 4, sh: 1, sw: 1},
		{name: "1x1/s1x3", h: 4, w: 10, c: 8, kh: 1, kw: 1, cout: 5, sh: 1, sw: 3},
		{name: "1x1/s3x3/stride>kernel", h: 7, w: 8, c: 3, kh: 1, kw: 1, cout: 9, sh: 3, sw: 3},
		{name: "1x1/1x1x1", h: 1, w: 1, c: 1, kh: 1, kw: 1, cout: 1, sh: 1, sw: 1},
		{name: "1x3/s1", h: 3, w: 9, c: 3, kh: 1, kw: 3, cout: 2, sh: 1, sw: 1},
		{name: "1x3/s1x2/nilbias", h: 2, w: 9, c: 1, kh: 1, kw: 3, cout: 33, sh: 1, sw: 2, noBias: true},
		{name: "3x1/s1", h: 6, w: 5, c: 1, kh: 3, kw: 1, cout: 3, sh: 1, sw: 1},
		{name: "3x1/s2x1/nilbias", h: 7, w: 5, c: 3, kh: 3, kw: 1, cout: 8, sh: 2, sw: 1, noBias: true},
		{name: "2x2/s1/asym", h: 5, w: 6, c: 3, kh: 2, kw: 2, cout: 7, sh: 1, sw: 1},
		{name: "2x2/s2x2/asym", h: 7, w: 9, c: 3, kh: 2, kw: 2, cout: 8, sh: 2, sw: 2},
		{name: "3x3/s1", h: 6, w: 11, c: 8, kh: 3, kw: 3, cout: 9, sh: 1, sw: 1},
		{name: "3x3/s2x2", h: 9, w: 10, c: 3, kh: 3, kw: 3, cout: 33, sh: 2, sw: 2},
		{name: "3x3/s1x3/w-not-divisible", h: 4, w: 17, c: 8, kh: 3, kw: 3, cout: 32, sh: 1, sw: 3},
		{name: "4x4/s1/asym", h: 6, w: 7, c: 1, kh: 4, kw: 4, cout: 3, sh: 1, sw: 1},
		{name: "4x4/s1x3/asym", h: 6, w: 13, c: 3, kh: 4, kw: 4, cout: 5, sh: 1, sw: 3},
		{name: "4x4/s3x3", h: 10, w: 11, c: 1, kh: 4, kw: 4, cout: 4, sh: 3, sw: 3},
		{name: "5x5/s1/H=1", h: 1, w: 7, c: 3, kh: 5, kw: 5, cout: 2, sh: 1, sw: 1},
		{name: "5x5/s1/W=2", h: 4, w: 2, c: 3, kh: 5, kw: 5, cout: 3, sh: 1, sw: 1},
		{name: "5x5/s1x3/w-not-divisible", h: 7, w: 23, c: 8, kh: 5, kw: 5, cout: 32, sh: 1, sw: 3},
		{name: "5x5/s2x1", h: 9, w: 6, c: 3, kh: 5, kw: 5, cout: 1, sh: 2, sw: 1},
		{name: "5x5/s2x2/h-not-divisible", h: 11, w: 9, c: 1, kh: 5, kw: 5, cout: 9, sh: 2, sw: 2},
		{name: "7x3/s1", h: 10, w: 9, c: 8, kh: 7, kw: 3, cout: 1, sh: 1, sw: 1},
		{name: "7x3/s3x3/kernel>H", h: 5, w: 4, c: 3, kh: 7, kw: 3, cout: 7, sh: 3, sw: 3},
		{name: "7x7/s1x3", h: 9, w: 26, c: 1, kh: 7, kw: 7, cout: 32, sh: 1, sw: 3},
		{name: "3x39/s1", h: 4, w: 45, c: 2, kh: 3, kw: 39, cout: 8, sh: 1, sw: 1},
		{name: "3x39/s1/kernel>W", h: 3, w: 20, c: 1, kh: 3, kw: 39, cout: 3, sh: 1, sw: 1},
		{name: "3x39/s1x3", h: 3, w: 50, c: 3, kh: 3, kw: 39, cout: 5, sh: 1, sw: 3},
		{name: "9x9/kernel>input", h: 1, w: 1, c: 3, kh: 9, kw: 9, cout: 4, sh: 1, sw: 1},
		{name: "6x6/s2x3/kernel>input", h: 2, w: 3, c: 8, kh: 6, kw: 6, cout: 33, sh: 2, sw: 3},
	}

	// nn's TestConv2DBlockRemainders: every output width 1..17 (every
	// remainder of the 8-position blocks) against many channel counts.
	couts := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 33}
	cins := []int{1, 3, 8}

	for w := 1; w <= 17; w++ {
		for j, cout := range couts {
			cin := cins[(w+j)%len(cins)]
			sw := 1 + (w+j)%3
			cases = append(cases, planCase{
				name: fmt.Sprintf("outW=%d/cout=%d/cin=%d/sw=%d", w, cout, cin, sw),
				h:    3, w: w * sw, c: cin, kh: 3, kw: 3, cout: cout, sh: 1, sw: sw,
			})
		}
	}

	for i, tc := range cases {
		tc.seed = uint64(i)
		t.Run(tc.name, func(t *testing.T) {
			kernel, bias, x := tc.fixture(t)
			checkPlanAgainstConv2D(t, x, kernel, bias, tc.params())
		})
	}
}

// basicPitchLayers are the six convolutions of basic-pitch for one
// 172-frame window, all "same" (algo-transcribe docs/port-spec.md §4).
var basicPitchLayers = []planCase{
	{name: "conv2d_1", h: 172, w: 264, c: 8, kh: 3, kw: 39, cout: 8, sh: 1, sw: 1, wantH: 172, wantW: 264, wantMACsPerRw: 264 * 936 * 8},
	{name: "contours-reduced", h: 172, w: 264, c: 8, kh: 5, kw: 5, cout: 1, sh: 1, sw: 1, wantH: 172, wantW: 264, wantMACsPerRw: 264 * 200 * 1},
	{name: "conv2d_2", h: 172, w: 264, c: 1, kh: 7, kw: 7, cout: 32, sh: 1, sw: 3, wantH: 172, wantW: 88, wantMACsPerRw: 88 * 49 * 32},
	{name: "conv2d_3", h: 172, w: 88, c: 32, kh: 7, kw: 3, cout: 1, sh: 1, sw: 1, wantH: 172, wantW: 88, wantMACsPerRw: 88 * 672 * 1},
	{name: "conv2d_4", h: 172, w: 264, c: 8, kh: 5, kw: 5, cout: 32, sh: 1, sw: 3, wantH: 172, wantW: 88, wantMACsPerRw: 88 * 200 * 32},
	{name: "conv2d_5", h: 172, w: 88, c: 33, kh: 3, kw: 3, cout: 1, sh: 1, sw: 1, wantH: 172, wantW: 88, wantMACsPerRw: 88 * 297 * 1},
}

// conv2DRows is Conv2D's output rows [o0, o1) of the single image x: Conv2D
// on the input rows they read, padded where the whole image is and
// nowhere else, so every output reads the same taps as in Conv2D on x.
func conv2DRows(t *testing.T, x, kernel, bias *tensor.Tensor, p Conv2DParams, o0, o1 int) []float32 {
	t.Helper()

	inH, inW, inC, kh := int(x.Dim(1)), int(x.Dim(2)), int(x.Dim(3)), int(kernel.Dim(0))
	lo := max(0, o0*p.StrideH-p.PadTop)
	hi := min(inH, (o1-1)*p.StrideH-p.PadTop+kh)

	q := p
	q.PadTop = p.PadTop + lo - o0*p.StrideH
	q.PadBottom = max(0, (o1-o0-1)*p.StrideH+kh-(hi-lo)-q.PadTop)

	crop, err := tensor.New(x.RawData()[lo*inW*inC:hi*inW*inC], []int64{1, int64(hi - lo), int64(inW), int64(inC)})
	if err != nil {
		t.Fatal(err)
	}

	out, err := Conv2D(crop, kernel, bias, q)
	if err != nil {
		t.Fatal(err)
	}

	if out.Dim(1) != int64(o1-o0) {
		t.Fatalf("cropped Conv2D has %d rows, want %d", out.Dim(1), o1-o0)
	}

	return out.RawData()
}

// TestConv2DPlanBasicPitchLayers checks the six basic-pitch shapes against
// Conv2D: all rows, or under reduced work rows at both edges (which see the
// top and bottom padding) and in the middle.
func TestConv2DPlanBasicPitchLayers(t *testing.T) {
	for i, tc := range basicPitchLayers {
		tc.seed = uint64(500 + i)
		t.Run(tc.name, func(t *testing.T) {
			kernel, bias, x := tc.fixture(t)

			plan, err := NewConv2DPlan(kernel, bias, tc.h, tc.w, tc.params())
			if err != nil {
				t.Fatal(err)
			}

			h, w, c := plan.OutShape()
			if h != tc.wantH || w != tc.wantW || c != tc.cout {
				t.Fatalf("OutShape() = (%d, %d, %d), want (%d, %d, %d)", h, w, c, tc.wantH, tc.wantW, tc.cout)
			}

			if got := plan.MACsPerRow(); got != tc.wantMACsPerRw {
				t.Errorf("MACsPerRow() = %d, want %d", got, tc.wantMACsPerRw)
			}

			windows := [][2]int{{0, h}}
			if planReducedWork() {
				windows = [][2]int{{0, 4}, {85, 87}, {h - 4, h}}
			}

			dst := planDst(t, plan, float32(math.NaN()))
			rowLen := w * c

			for _, win := range windows {
				err := plan.ForwardRows(dst, x, win[0], win[1])
				if err != nil {
					t.Fatal(err)
				}

				want := conv2DRows(t, x, kernel, bias, tc.params(), win[0], win[1])
				assertConv2DTolerance(t, dst.RawData()[win[0]*rowLen:win[1]*rowLen], want)
			}
		})
	}
}

// chunkLayers are the layers of the chunking and conv-worker tests.
var chunkLayers = []planCase{
	{name: "5x5/s1x3", h: 13, w: 23, c: 3, kh: 5, kw: 5, cout: 9, sh: 1, sw: 3, seed: 700},
	{name: "3x3/s2x2", h: 12, w: 10, c: 8, kh: 3, kw: 3, cout: 7, sh: 2, sw: 2, seed: 701},
	{name: "2x2/s1", h: 9, w: 7, c: 1, kh: 2, kw: 2, cout: 33, sh: 1, sw: 1, seed: 702},
	{name: "3x39/s1/W=45", h: 16, w: 45, c: 2, kh: 3, kw: 39, cout: 8, sh: 1, sw: 1, seed: 703},
}

// assertRows checks that rows [r0, r1) of dst equal full bit for bit and
// that every other value is still sentinel.
func assertRows(t *testing.T, plan *Conv2DPlan, dst, full []float32, r0, r1 int, sentinel float32) {
	t.Helper()

	h, w, c := plan.OutShape()
	n := w * c

	for y := range h {
		got := dst[y*n : (y+1)*n]
		if y >= r0 && y < r1 {
			assertBitIdentical(t, got, full[y*n:(y+1)*n])

			continue
		}

		for x, v := range got {
			if math.Float32bits(v) != math.Float32bits(sentinel) {
				t.Fatalf("row %d outside [%d, %d) written at %d: %v", y, r0, r1, x, v)
			}
		}
	}
}

func TestConv2DPlanForwardRowsChunking(t *testing.T) {
	for _, l := range chunkLayers {
		t.Run(l.name, func(t *testing.T) {
			plan, x := l.plan(t)
			h, _, _ := plan.OutShape()

			full := planDst(t, plan, 0)

			err := plan.Forward(full, x)
			if err != nil {
				t.Fatal(err)
			}

			// Each chunking lists step sizes; zero steps are empty ranges.
			chunkings := map[string][]int{
				"all":         {h},
				"empty-first": {0, 0, h - 1, 1, 0},
				"halves":      {h / 2, h - h/2},
				"uneven":      {2, 0, 3, 1, h},
			}

			for _, size := range []int{1, 2, 3, 7} {
				steps := make([]int, (h+size-1)/size)
				for i := range steps {
					steps[i] = size
				}

				chunkings[fmt.Sprintf("by-%d", size)] = steps
			}

			rng := planRNG(l.seed)

			for i := range 3 {
				var steps []int

				for sum := 0; sum < h; {
					s := rng.IntN(5)
					steps = append(steps, s)
					sum += s
				}

				chunkings[fmt.Sprintf("random-%d", i)] = steps
			}

			for name, steps := range chunkings {
				t.Run(name, func(t *testing.T) {
					dst := planDst(t, plan, float32(math.NaN()))

					r0 := 0
					for _, n := range steps {
						r1 := min(r0+n, h)

						err := plan.ForwardRows(dst, x, r0, r1)
						if err != nil {
							t.Fatal(err)
						}

						r0 = r1
					}

					if r0 != h {
						t.Fatalf("chunking covers %d of %d rows", r0, h)
					}

					assertBitIdentical(t, dst.RawData(), full.RawData())
				})
			}

			// A partial range writes exactly rows [r0, r1).
			for _, rr := range [][2]int{{0, 0}, {h, h}, {2, 2}, {1, 4}, {0, 1}, {h - 1, h}} {
				t.Run(fmt.Sprintf("only[%d,%d)", rr[0], rr[1]), func(t *testing.T) {
					const sentinel = 12345

					dst := planDst(t, plan, sentinel)

					err := plan.ForwardRows(dst, x, rr[0], rr[1])
					if err != nil {
						t.Fatal(err)
					}

					assertRows(t, plan, dst.RawData(), full.RawData(), rr[0], rr[1], sentinel)
				})
			}
		})
	}
}

// TestConv2DPlanConvWorkers checks that splitting the rows over
// SetConvWorkers goroutines gives the sequential bits and writes only the
// requested rows, for worker counts that rise (growing the scratch) and
// exceed the row count.
func TestConv2DPlanConvWorkers(t *testing.T) {
	defer SetConvWorkers(getConvWorkers())

	for _, l := range chunkLayers {
		t.Run(l.name, func(t *testing.T) {
			plan, x := l.plan(t)
			h, _, _ := plan.OutShape()

			SetConvWorkers(0)

			full := planDst(t, plan, 0)

			err := plan.Forward(full, x)
			if err != nil {
				t.Fatal(err)
			}

			for _, workers := range []int{2, 3, 5, 64} {
				SetConvWorkers(workers)

				for _, rr := range [][2]int{{0, h}, {1, h - 1}, {2, 5}, {3, 4}, {h - 2, h}, {1, 1}} {
					t.Run(fmt.Sprintf("workers=%d/rows[%d,%d)", workers, rr[0], rr[1]), func(t *testing.T) {
						const sentinel = -777

						dst := planDst(t, plan, sentinel)

						err := plan.ForwardRows(dst, x, rr[0], rr[1])
						if err != nil {
							t.Fatal(err)
						}

						assertRows(t, plan, dst.RawData(), full.RawData(), rr[0], rr[1], sentinel)
					})
				}
			}
		})
	}
}

func TestConv2DPlanZeroAllocs(t *testing.T) {
	defer SetConvWorkers(getConvWorkers())

	SetConvWorkers(0)

	// conv2d_4 geometry with fewer time rows to keep the runtime small.
	tc := planCase{h: 8, w: 264, c: 8, kh: 5, kw: 5, cout: 32, sh: 1, sw: 3, seed: 900}
	plan, x := tc.plan(t)
	dst := planDst(t, plan, 0)

	var err error

	allocs := testing.AllocsPerRun(5, func() {
		err = plan.ForwardRows(dst, x, 2, 5)
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("ForwardRows: %v allocations per run, want 0", allocs)
	}

	allocs = testing.AllocsPerRun(3, func() {
		err = plan.Forward(dst, x)
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("Forward: %v allocations per run, want 0", allocs)
	}
}

func TestNewConv2DPlanErrors(t *testing.T) {
	k := mustTensorT(t, seqDataT(3*3*4*2), []int64{3, 3, 4, 2})
	ok := Conv2DParams{StrideH: 1, StrideW: 1, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1}

	with := func(f func(p *Conv2DParams)) Conv2DParams {
		p := ok
		f(&p)

		return p
	}

	cases := []struct {
		name     string
		kernel   *tensor.Tensor
		bias     *tensor.Tensor
		inH, inW int
		p        Conv2DParams
		want     string
	}{
		{"nil kernel", nil, nil, 5, 6, ok, "nil kernel"},
		{"kernel rank 3", mustTensorT(t, seqDataT(3*3*4), []int64{3, 3, 4}), nil, 5, 6, ok, "kernel shape"},
		{"kernel rank 5", mustTensorT(t, seqDataT(3*3*4*2), []int64{1, 3, 3, 4, 2}), nil, 5, 6, ok, "kernel shape"},
		{"kernel zero out channels", mustTensorT(t, nil, []int64{3, 3, 4, 0}), nil, 5, 6, ok, "kernel shape"},
		{"kernel zero height", mustTensorT(t, nil, []int64{0, 3, 4, 2}), nil, 5, 6, ok, "kernel shape"},
		{"bias rank 2", k, mustTensorT(t, seqDataT(2), []int64{1, 2}), 5, 6, ok, "bias shape"},
		{"bias short", k, mustTensorT(t, seqDataT(1), []int64{1}), 5, 6, ok, "bias shape"},
		{"bias long", k, mustTensorT(t, seqDataT(3), []int64{3}), 5, 6, ok, "bias shape"},
		{"groups 0", k, nil, 5, 6, with(func(p *Conv2DParams) { p.Groups = 0 }), "groups must be 1"},
		{"groups 2", k, nil, 5, 6, with(func(p *Conv2DParams) { p.Groups = 2 }), "groups must be 1"},
		{"stride h 0", k, nil, 5, 6, with(func(p *Conv2DParams) { p.StrideH = 0 }), "strides"},
		{"stride w negative", k, nil, 5, 6, with(func(p *Conv2DParams) { p.StrideW = -1 }), "strides"},
		{"pad top negative", k, nil, 5, 6, with(func(p *Conv2DParams) { p.PadTop = -1 }), "padding"},
		{"pad right negative", k, nil, 5, 6, with(func(p *Conv2DParams) { p.PadRight = -2 }), "padding"},
		{"pad too large", k, nil, 5, 6, with(func(p *Conv2DParams) { p.PadBottom = math.MaxInt }), "too large"},
		{"inH 0", k, nil, 0, 6, ok, "input size"},
		{"inW negative", k, nil, 5, -1, ok, "input size"},
		{"inH too large", k, nil, math.MaxInt, 6, ok, "too large"},
		{"padded smaller than kernel", k, nil, 2, 2, Conv2DParams{StrideH: 1, StrideW: 1, Groups: 1}, "smaller than kernel"},
		{"padded width smaller than kernel", k, nil, 5, 1, with(func(p *Conv2DParams) { p.PadLeft, p.PadRight = 0, 1 }), "smaller than kernel"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := NewConv2DPlan(tc.kernel, tc.bias, tc.inH, tc.inW, tc.p)
			assertErrContains(t, err, tc.want)
			assertErrContains(t, err, "ops: conv2d plan: ")

			if plan != nil {
				t.Errorf("NewConv2DPlan returned a plan with error %v", err)
			}
		})
	}

	// The valid counterpart of the cases above succeeds.
	_, err := NewConv2DPlan(k, mustTensorT(t, seqDataT(2), []int64{2}), 5, 6, ok)
	if err != nil {
		t.Fatalf("valid plan: %v", err)
	}
}

func TestConv2DPlanForwardRowsErrors(t *testing.T) {
	// 6×4 input, 3×3 kernel, stride (2, 1), pad 1: output 3×4×5.
	tc := planCase{h: 6, w: 4, c: 3, kh: 3, kw: 3, cout: 5, sh: 2, sw: 1, seed: 950}
	plan, x := tc.plan(t)
	dst := planDst(t, plan, 0)

	tensorOf := func(shape ...int64) *tensor.Tensor {
		return planRandTensor(t, planRNG(1), shape...)
	}

	cases := []struct {
		name   string
		dst, x *tensor.Tensor
		r0, r1 int
		want   string
	}{
		{"nil x", dst, nil, 0, 1, "nil x or dst"},
		{"nil dst", nil, x, 0, 1, "nil x or dst"},
		{"x rank 3", dst, tensorOf(6, 4, 3), 0, 1, "x shape"},
		{"x rank 5", dst, tensorOf(1, 1, 6, 4, 3), 0, 1, "x shape"},
		{"x channels", dst, tensorOf(1, 6, 4, 2), 0, 1, "x shape"},
		{"x transposed", dst, tensorOf(1, 4, 6, 3), 0, 1, "x shape"},
		{"x batch 2 half height", dst, tensorOf(2, 3, 4, 3), 0, 1, "x shape"},
		{"x channels and width swapped", dst, tensorOf(1, 6, 3, 4), 0, 1, "x shape"},
		{"dst rank 3", tensorOf(3, 4, 5), x, 0, 1, "dst shape"},
		{"dst channels", tensorOf(1, 3, 4, 6), x, 0, 1, "dst shape"},
		{"dst transposed", tensorOf(1, 4, 3, 5), x, 0, 1, "dst shape"},
		{"dst batch 3 one row", tensorOf(3, 1, 4, 5), x, 0, 1, "dst shape"},
		{"dst unstrided", tensorOf(1, 6, 4, 5), x, 0, 1, "dst shape"},
		{"r0 negative", dst, x, -1, 1, "rows"},
		{"r1 beyond", dst, x, 0, 4, "rows"},
		{"r0 > r1", dst, x, 2, 1, "rows"},
		{"r0 beyond", dst, x, 4, 4, "rows"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Errors must be reported before anything is written.
			var before []float32
			if tc.dst != nil {
				before = append(before, tc.dst.RawData()...)
			}

			assertErrContains(t, plan.ForwardRows(tc.dst, tc.x, tc.r0, tc.r1), tc.want)

			if tc.dst != nil {
				assertBitIdentical(t, tc.dst.RawData(), before)
			}
		})
	}

	assertErrContains(t, plan.Forward(nil, x), "nil x or dst")
	assertErrContains(t, plan.Forward(dst, tensorOf(1, 4, 6, 3)), "x shape")
}

func TestConv2DPlanOutShapeAndMACs(t *testing.T) {
	// conv2d_1 of basic-pitch: [1, 172, 264, 8] input, 3×39 kernel, 8 filters
	// (nn's ExampleConv2D_MACsPerRow).
	k := mustTensorT(t, make([]float32, 3*39*8*8), []int64{3, 39, 8, 8})

	plan, err := NewConv2DPlan(k, nil, 172, 264, samePadParams(172, 264, 3, 39, 1, 1))
	if err != nil {
		t.Fatal(err)
	}

	h, w, c := plan.OutShape()
	if h != 172 || w != 264 || c != 8 {
		t.Errorf("OutShape() = (%d, %d, %d), want (172, 264, 8)", h, w, c)
	}

	if got, total := plan.MACsPerRow(), plan.MACsPerRow()*h; got != 1976832 || total != 340015104 {
		t.Errorf("MACsPerRow() = %d (total %d), want 1976832 (340015104)", got, total)
	}

	// Explicit padding and strides: (9 + 2 + 0 − 3)/2 + 1 = 5 rows and
	// (8 + 0 + 4 − 2)/3 + 1 = 4 columns.
	k = mustTensorT(t, make([]float32, 3*2*5*6), []int64{3, 2, 5, 6})

	plan, err = NewConv2DPlan(k, nil, 9, 8, Conv2DParams{StrideH: 2, StrideW: 3, PadTop: 2, PadRight: 4, Groups: 1})
	if err != nil {
		t.Fatal(err)
	}

	if h, w, c := plan.OutShape(); h != 5 || w != 4 || c != 6 {
		t.Errorf("OutShape() = (%d, %d, %d), want (5, 4, 6)", h, w, c)
	}

	if got := plan.MACsPerRow(); got != 4*3*2*5*6 {
		t.Errorf("MACsPerRow() = %d, want %d", got, 4*3*2*5*6)
	}
}

// TestConv2DPlanCopiesParameters checks that NewConv2DPlan copies the kernel
// and the bias: changing the caller's tensors afterwards must not change
// the output.
func TestConv2DPlanCopiesParameters(t *testing.T) {
	tc := planCase{h: 5, w: 9, c: 3, kh: 3, kw: 3, cout: 6, sh: 1, sw: 2, seed: 42}
	kernel, bias, x := tc.fixture(t)

	plan, err := NewConv2DPlan(kernel, bias, tc.h, tc.w, tc.params())
	if err != nil {
		t.Fatal(err)
	}

	before := planDst(t, plan, 0)

	err = plan.Forward(before, x)
	if err != nil {
		t.Fatal(err)
	}

	for i := range kernel.RawData() {
		kernel.RawData()[i] = 100
	}

	for i := range bias.RawData() {
		bias.RawData()[i] = -100
	}

	after := planDst(t, plan, 0)

	err = plan.Forward(after, x)
	if err != nil {
		t.Fatal(err)
	}

	assertBitIdentical(t, after.RawData(), before.RawData())
}

// TestConv2DPlanDoesNotModifyInput checks that x is read only.
func TestConv2DPlanDoesNotModifyInput(t *testing.T) {
	tc := planCase{h: 7, w: 11, c: 3, kh: 5, kw: 5, cout: 8, sh: 1, sw: 3, seed: 77}
	plan, x := tc.plan(t)
	orig := append([]float32(nil), x.RawData()...)

	err := plan.Forward(planDst(t, plan, 0), x)
	if err != nil {
		t.Fatal(err)
	}

	assertBitIdentical(t, x.RawData(), orig)
}

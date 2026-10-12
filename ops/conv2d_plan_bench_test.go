// From github.com/cwbudde/algo-transcribe nn/conv_bench_test.go @ 21ad164.
// algo-tensor change: BenchmarkConv2DWindow is BenchmarkConv2DPlanBasicPitch, next to
// BenchmarkConv2DBasicPitch for ops.Conv2D on the same layers, both on one worker; the layers
// are basicPitchLayers of conv2d_plan_test.go; BenchmarkConv2DPlanWhistleStem and
// BenchmarkConv2DWhistleStem for go-whistle's stem conv added; no BenchmarkConv2DRow.
// algo-tensor change: go-whistle's stem chunk; BenchmarkConv2DRouting for Conv2D's routing (PLAN 1.3).

package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// benchConvLayer is one benchmarked convolution of a single image.
type benchConvLayer struct {
	name         string
	h, w, c      int
	kh, kw, cout int
	p            Conv2DParams
	bias         bool
}

// benchBasicPitchLayers are basic-pitch's six layers, "same" padding,
// with bias.
func benchBasicPitchLayers() []benchConvLayer {
	layers := make([]benchConvLayer, 0, len(basicPitchLayers))
	for _, l := range basicPitchLayers {
		layers = append(layers, benchConvLayer{
			name: l.name, h: l.h, w: l.w, c: l.c, kh: l.kh, kw: l.kw, cout: l.cout, p: l.params(), bias: true,
		})
	}

	return layers
}

// benchWhistleStem is the first conv of go-whistle's encoder stem
// (github.com/cwbudde/go-whistle internal/encoder/stem.go @ 1c826b2,
// stem.conv for stage 0 under the Best hypothesis: StrideAll,
// PadSymmetric) on a 30 s clip: 3000 frames × 80 mels (frontend.Mels),
// conv 3×3, 1 → 128 channels (stem/w), stride 2, padding 1 on every side,
// no bias; out 1500 × 40 × 128. go-whistle runs it in chunks (stem.chunk
// @ 98603c6, stemChunk = 8 output frames): "chunk" is one chunk away from
// the clip's edges, 71 input rows with no padding above or below, out
// 35 × 40 × 128.
var benchWhistleStem = []benchConvLayer{{
	name: "conv", h: 3000, w: 80, c: 1, kh: 3, kw: 3, cout: 128,
	p: Conv2DParams{StrideH: 2, StrideW: 2, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1},
}, {
	name: "chunk", h: 71, w: 80, c: 1, kh: 3, kw: 3, cout: 128,
	p: Conv2DParams{StrideH: 2, StrideW: 2, PadLeft: 1, PadRight: 1, Groups: 1},
}}

// benchConvSetup returns the layer's random kernel, bias (nil without) and
// input and sets one worker for tensor kernels and Conv workers, restored
// when the benchmark ends.
func benchConvSetup(b *testing.B, l benchConvLayer) (kernel, bias, x *tensor.Tensor) {
	b.Helper()

	prevTensor, prevConv := tensor.Workers(), getConvWorkers()

	b.Cleanup(func() {
		tensor.SetWorkers(prevTensor)
		SetConvWorkers(prevConv)
	})
	tensor.SetWorkers(1)
	SetConvWorkers(0)

	rng := rand.New(rand.NewPCG(1, 2))
	kernel = planRandTensor(b, rng, int64(l.kh), int64(l.kw), int64(l.c), int64(l.cout))

	if l.bias {
		bias = planRandTensor(b, rng, int64(l.cout))
	}

	x = planRandTensor(b, rng, 1, int64(l.h), int64(l.w), int64(l.c))

	return kernel, bias, x
}

// BenchmarkConv2DPlanBasicPitch runs Conv2DPlan.Forward over each layer's
// whole image on one worker and reports multiply-accumulates per second.
func BenchmarkConv2DPlanBasicPitch(b *testing.B) { benchConv2DPlan(b, benchBasicPitchLayers()) }

// BenchmarkConv2DBasicPitch runs Conv2D on the same layers, one worker.
func BenchmarkConv2DBasicPitch(b *testing.B) { benchConv2D(b, benchBasicPitchLayers()) }

// BenchmarkConv2DPlanWhistleStem runs Conv2DPlan.Forward on go-whistle's
// stem conv, one worker.
func BenchmarkConv2DPlanWhistleStem(b *testing.B) { benchConv2DPlan(b, benchWhistleStem) }

// BenchmarkConv2DWhistleStem runs Conv2D on go-whistle's stem conv, one
// worker.
func BenchmarkConv2DWhistleStem(b *testing.B) { benchConv2D(b, benchWhistleStem) }

func benchConv2DPlan(b *testing.B, layers []benchConvLayer) {
	b.Helper()

	for _, l := range layers {
		b.Run(l.name, func(b *testing.B) {
			kernel, bias, x := benchConvSetup(b, l)

			plan, err := NewConv2DPlan(kernel, bias, l.h, l.w, l.p)
			if err != nil {
				b.Fatal(err)
			}

			dst := planDst(b, plan, 0)
			h, _, _ := plan.OutShape()
			macs := float64(plan.MACsPerRow() * h)

			b.ReportAllocs()

			for b.Loop() {
				err := plan.Forward(dst, x)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(macs*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}

func benchConv2D(b *testing.B, layers []benchConvLayer) {
	b.Helper()

	for _, l := range layers {
		b.Run(l.name, func(b *testing.B) {
			kernel, bias, x := benchConvSetup(b, l)

			plan, err := NewConv2DPlan(kernel, bias, l.h, l.w, l.p) // for the MAC count only
			if err != nil {
				b.Fatal(err)
			}

			h, _, _ := plan.OutShape()
			macs := float64(plan.MACsPerRow() * h)

			b.ReportAllocs()

			for b.Loop() {
				_, err := Conv2D(x, kernel, bias, l.p)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(macs*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}

// BenchmarkConv2DRouting runs groups = 1 shapes around Conv2D's routing
// thresholds (planGains) through a Conv2DPlan ("plan", conv2D's
// conv2DPlanned path, packing included) and the per-pixel path
// ("direct", conv2DDirect), one worker: images of 4×4, 8×8 and 16×16
// outputs, GEMM depths k of 1, 2, 3, 4 and 9 (1×1 kernels on k input
// channels) and 36 (3×3 on 4, "same"), and 1 to 255 out channels, with
// bias. Run it with -benchtime=0.2s; ops/PERFORMANCE.md has the result.
func BenchmarkConv2DRouting(b *testing.B) {
	type shape struct{ hw, kk, c, o int }

	sizes, kernels, outs := []int{4, 8, 16}, [][2]int{{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 9}, {3, 4}}, []int{1, 8, 32, 128, 255}
	shapes := make([]shape, 0, len(sizes)*len(kernels)*len(outs))

	for _, hw := range sizes {
		for _, kc := range kernels {
			for _, o := range outs {
				shapes = append(shapes, shape{hw, kc[0], kc[1], o})
			}
		}
	}

	for _, s := range shapes {
		l := benchConvLayer{h: s.hw, w: s.hw, c: s.c, kh: s.kk, kw: s.kk, cout: s.o, p: samePadParams(s.hw, s.hw, s.kk, s.kk, 1, 1), bias: true}
		name := fmt.Sprintf("out=%dx%d/k=%d/o=%d", s.hw, s.hw, s.kk*s.kk*s.c, s.o)

		for _, path := range []struct {
			name string
			path conv2DPath
		}{{"plan", conv2DPlanned}, {"direct", conv2DPixels}} {
			b.Run(name+"/"+path.name, func(b *testing.B) {
				kernel, bias, x := benchConvSetup(b, l)

				b.ReportAllocs()

				for b.Loop() {
					_, err := conv2D(x, kernel, bias, l.p, path.path)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

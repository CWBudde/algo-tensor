// From github.com/cwbudde/go-whistle internal/ops/ops_bench_test.go @ 8926ee3,
// originally github.com/cwbudde/go-pocket-tts internal/runtime/ops/ops_bench_test.go @ 9faba38.
// go-whistle change: benchmarks use Whistle shapes instead of FlowLM/Mimi ones; the helpers are unchanged.
// algo-tensor change: the FlowLM/Mimi benchmarks restored from go-pocket-tts @ 9faba38 next to them.

package ops

import (
	"fmt"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// The benchmarks use Whistle's shapes (PLAN.md reference card): width 512,
// 8 query heads of qk 48 and v 64, 375 encoder frames for a 30 s clip, and
// the depthwise kernel-9 conv of the encoder's conv module.

func BenchmarkLinearWhistleQProj(b *testing.B) {
	x := mustTensor(b, seqData(375*512), []int64{1, 375, 512})
	w := mustTensor(b, seqData(384*512), []int64{384, 512})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.Linear(x, w, nil)
		if err != nil {
			b.Fatalf("linear: %v", err)
		}
	}
}

func BenchmarkMatMulWhistleScores(b *testing.B) {
	q := mustTensor(b, seqData(8*375*48), []int64{1, 8, 375, 48})
	kT := mustTensor(b, seqData(8*48*375), []int64{1, 8, 48, 375})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.MatMul(q, kT)
		if err != nil {
			b.Fatalf("matmul: %v", err)
		}
	}
}

func BenchmarkAttentionWhistleEncoder(b *testing.B) {
	q := mustTensor(b, seqData(8*375*48), []int64{1, 8, 375, 48})
	k := mustTensor(b, seqData(8*375*48), []int64{1, 8, 375, 48})
	v := mustTensor(b, seqData(8*375*64), []int64{1, 8, 375, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, false, 0)
		if err != nil {
			b.Fatalf("attention: %v", err)
		}
	}
}

func BenchmarkAttentionWhistleDecodeStep(b *testing.B) {
	q := mustTensor(b, seqData(8*48), []int64{1, 8, 1, 48})
	k := mustTensor(b, seqData(8*64*48), []int64{1, 8, 64, 48})
	v := mustTensor(b, seqData(8*64*64), []int64{1, 8, 64, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, true, 63)
		if err != nil {
			b.Fatalf("attention: %v", err)
		}
	}
}

// BenchmarkConv1DWhistleDepthwise runs the grouped (scalar) path; a
// depthwise kernel would replace it.
func BenchmarkConv1DWhistleDepthwise(b *testing.B) {
	input := mustTensor(b, seqData(512*375), []int64{1, 512, 375})
	kernel := mustTensor(b, seqData(512*9), []int64{512, 1, 9})
	bias := mustTensor(b, seqData(512), []int64{512})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Conv1D(input, kernel, bias, 1, 4, 1, 512)
		if err != nil {
			b.Fatalf("conv1d: %v", err)
		}
	}
}

func BenchmarkConv1DWhistlePointwiseParallel(b *testing.B) {
	for _, workers := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			SetConvWorkers(workers)
			defer SetConvWorkers(1)

			input := mustTensor(b, seqData(512*375), []int64{1, 512, 375})
			kernel := mustTensor(b, seqData(1024*512), []int64{1024, 512, 1})
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				_, err := Conv1D(input, kernel, nil, 1, 0, 1, 1)
				if err != nil {
					b.Fatalf("conv1d: %v", err)
				}
			}
		})
	}
}

// The go-pocket-tts benchmarks, at its FlowLM and Mimi shapes.

func BenchmarkMatMulFlowLM(b *testing.B) {
	a := mustTensor(b, seqData(1*16*64), []int64{1, 16, 64})
	w := mustTensor(b, seqData(1*64*64), []int64{1, 64, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.MatMul(a, w)
		if err != nil {
			b.Fatalf("matmul: %v", err)
		}
	}
}

func BenchmarkLayerNormFlowLM(b *testing.B) {
	x := mustTensor(b, seqData(1*64*1024), []int64{1, 64, 1024})
	w := mustTensor(b, seqData(1024), []int64{1024})
	bias := mustTensor(b, seqData(1024), []int64{1024})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.LayerNorm(x, w, bias, 1e-5)
		if err != nil {
			b.Fatalf("layernorm: %v", err)
		}
	}
}

func BenchmarkAttentionFlowLM(b *testing.B) {
	q := mustTensor(b, seqData(1*8*32*64), []int64{1, 8, 32, 64})
	k := mustTensor(b, seqData(1*8*32*64), []int64{1, 8, 32, 64})
	v := mustTensor(b, seqData(1*8*32*64), []int64{1, 8, 32, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, true, 0)
		if err != nil {
			b.Fatalf("attention: %v", err)
		}
	}
}

func BenchmarkConv1DMimi(b *testing.B) {
	input := mustTensor(b, seqData(1*256*128), []int64{1, 256, 128})
	kernel := mustTensor(b, seqData(512*256*3), []int64{512, 256, 3})
	bias := mustTensor(b, seqData(512), []int64{512})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Conv1D(input, kernel, bias, 1, 1, 1, 1)
		if err != nil {
			b.Fatalf("conv1d: %v", err)
		}
	}
}

func BenchmarkConv1DMimiParallel(b *testing.B) {
	for _, workers := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			SetConvWorkers(workers)
			defer SetConvWorkers(1)

			input := mustTensor(b, seqData(1*256*128), []int64{1, 256, 128})
			kernel := mustTensor(b, seqData(512*256*3), []int64{512, 256, 3})
			bias := mustTensor(b, seqData(512), []int64{512})
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				_, err := Conv1D(input, kernel, bias, 1, 1, 1, 1)
				if err != nil {
					b.Fatalf("conv1d: %v", err)
				}
			}
		})
	}
}

func BenchmarkFrameDecodeThroughput(b *testing.B) {
	latent := mustTensor(b, seqData(1*24*32), []int64{1, 24, 32})
	projW := mustTensor(b, seqData(512*32), []int64{512, 32})
	projB := mustTensor(b, seqData(512), []int64{512})
	mlpW1 := mustTensor(b, seqData(256*512), []int64{256, 512})
	mlpW2 := mustTensor(b, seqData(512*256), []int64{512, 256})
	deconvK := mustTensor(b, seqData(512*1*3), []int64{512, 1, 3})
	deconvB := mustTensor(b, seqData(1), []int64{1})

	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		h, err := tensor.Linear(latent, projW, projB)
		if err != nil {
			b.Fatalf("proj linear: %v", err)
		}

		h, err = MLP(h, mlpW1, nil, mlpW2, nil)
		if err != nil {
			b.Fatalf("mlp: %v", err)
		}
		// [1, T, C] -> [1, C, T] for conv-transpose.
		h, err = h.Transpose(1, 2)
		if err != nil {
			b.Fatalf("transpose: %v", err)
		}

		_, err = ConvTranspose1D(h, deconvK, deconvB, 2, 1, 0, 1, 1)
		if err != nil {
			b.Fatalf("convtranspose1d: %v", err)
		}
	}
}

func seqData(n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32((i%17)-8) / 17
	}

	return out
}

func mustTensor(tb testing.TB, data []float32, shape []int64) *tensor.Tensor {
	tb.Helper()

	t, err := tensor.New(data, shape)
	if err != nil {
		tb.Fatalf("new tensor: %v", err)
	}

	return t
}

package ops

import (
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// TestGoldenHashes: every operator gives the same bits as in the project it
// was taken from, sequentially and with 4 tensor and 3 conv workers. The
// hashes in golden_hashes_test.go were made as tensor/golden_test.go
// describes, in go-whistle internal/ops @ 8926ee3 and go-pocket-tts
// internal/runtime/ops @ 9faba38. go-pocket-tts' ConvTranspose1D gives other
// bits with conv workers on arm64, there as here: its channel split does not
// keep MatMulTransB's 4-row blocks. goldenTranscribe holds algo-tensor's own
// hashes for Conv2DPlan, made as tensor/golden_test.go describes for its
// goldenTranscribe; Conv2DPlan gave the same bits as algo-transcribe nn's
// Conv2D @ 21ad164 on the same inputs (checked once with a throwaway module
// on darwin/arm64 and linux/amd64, with and without purego, and on js/wasm).
// Its amd64/avx2 hash, added with nn's AVX2 kernels @ 58c8332 (PLAN 1.2), gave
// that commit's bits on every configuration but the one whose out channels
// (7) the AVX2 kernels decline: there the 4×4 Go tile keeps nn @ 21ad164's
// bits, where nn @ 58c8332 runs the 4×2 tile on amd64.
// Conv2D's hashes on arm64 and wasm are algo-tensor's since v0.3.0, where its
// groups = 1 convolution runs through Conv2DPlan (PLAN 1.3).
func TestGoldenHashes(t *testing.T) {
	cfg := goldenConfig()

	prevWorkers, prevConv := tensor.Workers(), getConvWorkers()

	t.Cleanup(func() {
		tensor.SetWorkers(prevWorkers)
		SetConvWorkers(prevConv)
	})

	sources := []struct {
		name  string
		table map[string]map[string][2]string
		cases []goldenCase
	}{
		{"go-whistle @ 8926ee3 (Conv2D on arm64 and wasm: algo-tensor v0.3.0)", goldenWhistle, append(commonGoldenCases(), whistleGoldenCases()...)},
		{"go-pocket-tts @ 9faba38", goldenPocket, append(commonGoldenCases(), pocketGoldenCases()...)},
		{"algo-tensor, algo-transcribe's kernels @ 21ad164", goldenTranscribe, transcribeGoldenCases()},
	}

	checked := 0

	for _, src := range sources {
		want, ok := src.table[cfg]
		if !ok {
			t.Logf("%s: no hashes for %s", src.name, cfg)
			continue
		}

		for _, c := range src.cases {
			for i, w := range [][2]int{{1, 0}, {4, 3}} {
				tensor.SetWorkers(w[0])
				SetConvWorkers(w[1])

				if got := goldenHash(c.run(t)); got != want[c.name][i] {
					t.Errorf("%s at %d/%d workers on %s: hash %s, %s gave %q", c.name, w[0], w[1], cfg, got, src.name, want[c.name][i])
				}

				checked++
			}
		}
	}

	if checked == 0 {
		t.Skipf("no golden hashes for %s", cfg)
	}
}

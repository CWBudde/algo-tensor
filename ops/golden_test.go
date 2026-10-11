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
// keep MatMulTransB's 4-row blocks.
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
		{"go-whistle @ 8926ee3", goldenWhistle, append(commonGoldenCases(), whistleGoldenCases()...)},
		{"go-pocket-tts @ 9faba38", goldenPocket, append(commonGoldenCases(), pocketGoldenCases()...)},
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

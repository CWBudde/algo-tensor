package tensor

import "testing"

// TestGoldenHashes: every kernel gives the same bits as in the project it
// was taken from, at 1 and at 4 workers. golden_hashes_test.go holds the
// hashes. They were made by copying golden_cases_test.go, the
// golden_cases_<origin>_test.go of the source and golden_config_test.go
// (with golden_purego_test.go and golden_asm_test.go) into go-whistle
// internal/tensor @ 8926ee3 and go-pocket-tts internal/runtime/tensor @
// 9faba38, and printing goldenHash(c.run(t)) of every case at SetWorkers(1)
// and SetWorkers(4): natively on darwin/arm64, with and without purego, in
// Docker on linux/amd64 (AVX2+FMA), also with and without purego, and on
// js/wasm under Node. The kernels both projects have gave the same hashes in
// both, on every configuration both cover. goldenTranscribe holds
// algo-tensor's own hashes for the kernels taken from algo-transcribe nn @
// 21ad164 (golden_cases_transcribe_test.go), printed the same way here on
// every configuration but amd64/avx2: there MatMulTransBBias runs the Go
// tiles, with the amd64/go bits, until PLAN.md 1.2 gives it AVX2 kernels.
func TestGoldenHashes(t *testing.T) {
	cfg := goldenConfig()

	prev := Workers()

	t.Cleanup(func() { SetWorkers(prev) })

	sources := []struct {
		name  string
		table map[string]map[string][2]string
		cases []goldenCase
	}{
		{"go-whistle @ 8926ee3", goldenWhistle, append(commonGoldenCases(), whistleGoldenCases()...)},
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
			for i, workers := range []int{1, 4} {
				SetWorkers(workers)

				if got := goldenHash(c.run(t)); got != want[c.name][i] {
					t.Errorf("%s at %d workers on %s: hash %s, %s gave %q", c.name, workers, cfg, got, src.name, want[c.name][i])
				}

				checked++
			}
		}
	}

	if checked == 0 {
		t.Skipf("no golden hashes for %s", cfg)
	}
}

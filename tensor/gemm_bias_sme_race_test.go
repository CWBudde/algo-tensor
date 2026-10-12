//go:build race

// algo-tensor's own, not in go-whistle or go-pocket-tts: whether the race
// detector is on (PLAN.md 2.1).

package tensor

// raceEnabled reports whether the tests run under the race detector, which
// makes sync.Pool drop items at random, so the SME path's pooled
// scratch allocates.
const raceEnabled = true

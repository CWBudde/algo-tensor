//go:build !darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: no SME2 path outside
// darwin (PLAN.md 2.1).

package tensor

// smeSupported reports false: gemmSME's signal guard is built for darwin,
// where a signal clobbers the Z registers of a thread in streaming mode,
// and has only been tested there. Other arm64 systems run NEON.
func smeSupported() bool { return false }

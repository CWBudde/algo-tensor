//go:build darwin && arm64 && !purego

// algo-tensor's own, not in go-whistle or go-pocket-tts: SME2 detection on
// darwin (PLAN.md 2.1).

package tensor

import "syscall"

// smeSupported reports whether the CPU has SME2, from the sysctl
// hw.optional.arm.FEAT_SME2 (1 on Apple M4 and later, 0 or missing before).
// golang.org/x/sys/cpu does not report SME2. The sysctl is a 32-bit integer,
// which syscall.Sysctl returns as its bytes, little-endian, the last zero
// byte dropped.
func smeSupported() bool {
	v, err := syscall.Sysctl("hw.optional.arm.FEAT_SME2")

	return err == nil && len(v) > 0 && v[0] == 1
}

package ops

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// goldenConfig names the kernel set the golden hashes depend on:
//
//   - arm64/neon and amd64/avx2: the assembly kernels;
//   - <arch>/go: the pure-Go fallbacks, under the purego tag, on amd64
//     without AVX2+FMA and on every other architecture. They differ between
//     architectures because the compiler fuses x*y+z into one FMA on arm64
//     (and ppc64, s390x, riscv64) but not on amd64 or wasm.
func goldenConfig() string {
	switch {
	case goldenPurego:
	case runtime.GOARCH == "arm64":
		return "arm64/neon"
	case runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 && cpu.X86.HasFMA:
		return "amd64/avx2"
	}

	return runtime.GOARCH + "/go"
}

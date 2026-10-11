// Package tensor is a dense, row-major float32 tensor with the kernels the
// model blocks build on: MatMulTransB (GEMM), DotProduct and Axpy, plus
// Linear, MatMul, Softmax, LayerNorm, broadcasting and shape operations.
//
// The kernels have NEON assembly on arm64 and AVX2+FMA assembly on amd64
// (chosen at run time from golang.org/x/sys/cpu). Every assembly routine has
// a pure-Go fallback that other architectures use and that the purego build
// tag forces everywhere; asm_test.go compares each kernel with its fallback.
//
// Workers, SetWorkers and Parallel are one bounded worker pool for the whole
// process. The default is GOMAXPROCS at start-up; SetWorkers(1) makes every
// kernel single-threaded. MatMulTransB and Parallel give the same bits for
// any worker count (docs/FINDINGS.md F-038).
//
// From github.com/cwbudde/go-whistle internal/tensor @ 8926ee3, which copied
// github.com/cwbudde/go-pocket-tts internal/runtime/tensor @ 9faba38 without
// LayerNorm (docs/FINDINGS.md F-008); LayerNorm is back from go-pocket-tts.
// Each file names its source.
//
// The F-numbers, docs/FINDINGS.md and PLAN.md in this module's comments refer
// to go-whistle's docs/FINDINGS.md and PLAN.md.
package tensor

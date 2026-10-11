# PLAN

Open work for `algo-tensor`. Each checkbox is one pull request, written to be picked up by a headless session
(ccbatch): the task text is the whole first paragraph of the item, so it stays self-contained. When an item is done,
tick it and append a one-line note with what was covered, the resulting numbers and any deliberate gap. `PLAN.md`
references inside copied code comments still mean go-whistle's plan (see AGENTS.md).

## Dependencies and sequencing

```
1.1                       (now)
1.2                       (after 1.1 merges and algo-transcribe PR #9, its PLAN 4.2, merges)
→ release v0.2.0 (additive: no golden hash changes); algo-transcribe then drops its own kernels (its PLAN 5.1)
1.3                       (after v0.2.0; a numerics change, released as v0.3.0)
```

## Phase 1: Fast small-channel convolution

`ops.Conv2D` loops over output pixels and runs one `tensor.Axpy` over the output channels per input tap, so layers with
few output channels get no vector speed-up and every call allocates. On the six convolutions of Spotify's basic-pitch
CNN (1, 8 or 32 output channels), measured on 2026-10-11 on an Apple M5 Pro with one worker, it takes 311 ms per
172-row window against 14.5 ms (0 allocs) for algo-transcribe's `nn` package, which uses im2col plus a small-m GEMM
with the bias fused and the weights packed once, Go tiles that also run fast under js/wasm, and NEON (and soon AVX2)
assembly. Phase 1 moves that code here so that algo-transcribe, go-whistle and go-pocket-tts share one set of kernels.

- [ ] 1.1 `conv2d-plan` — Port algo-transcribe's GEMM and convolution (github.com/cwbudde/algo-transcribe, package
      `nn`, main @ 21ad164: `gemm.go`, `gemm_tile.go`, `gemm_tile_wasm.go`, `gemm_arm64.go`, `gemm_arm64.s`,
      `gemm_simd_stub.go`, `neon_fp.h`, `conv.go` and their tests and benchmarks) as two additive APIs, with
      provenance headers (`From github.com/cwbudde/algo-transcribe nn/<file> @ 21ad164` plus `algo-tensor change:`
      lines). (1) `tensor.MatMulTransBBias(c []float32, ldc int, a, b, bias []float32, m, n, k int)`:
      `c[i*ldc+j] = bias[j] + dot(a[i*k:(i+1)*k], b[j*k:(j+1)*k])` for i < m, j < n, with a nil bias meaning zero,
      the same argument checks and panics as `MatMulTransB`, and `nn`'s kernels: the Go tiles (4×4, 4×2 under js/wasm
      where Go has 16 float32 registers, 8×1 for n = 1, the `dot` tail) and the NEON kernels `gemmNEON4x4` and
      `gemmNEON8x1` with their shape guards (k ≥ 4, n = 1 or a multiple of 4; other shapes take the Go tiles), rows
      processed in blocks of 8. Generalise `nn`'s contiguous output (ldc = n) to `ldc`. Products of at least
      `2·gemmPieceFMAs` may split over `Parallel` in whole 8-row blocks, which keeps every output bit-identical for any
      worker count. `MatMulTransB` itself does not change. (2) `ops.Conv2DPlan`, built by
      `ops.NewConv2DPlan(kernel, bias *tensor.Tensor, inH, inW int, p Conv2DParams) (*Conv2DPlan, error)` from an HWIO
      kernel (bias may be nil), groups = 1 only (an error otherwise), with explicit padding as in `Conv2DParams` (so
      `SamePadding` gives TensorFlow "same"): it packs the transposed weights once and owns the im2col scratch, and
      has `OutShape() (h, w, c int)`, `MACsPerRow() int`, `Forward(dst, x *tensor.Tensor) error` and
      `ForwardRows(dst, x *tensor.Tensor, r0, r1 int) error` for one NHWC image (`[1, inH, inW, inC]`) into a
      caller-allocated `[1, outH, outW, outC]` dst, with 0 allocations and both shapes checked exactly; since
      `Shape()` returns a copy, add (3) a nonallocating `(*tensor.Tensor).Dim(i int) int64` accessor (panics on an
      out-of-range i, like a slice index) and validate through `Rank` and `Dim`; a plan is not safe for concurrent use
      (document it), and it may split rows over the `SetConvWorkers` goroutines (default sequential) because rows are
      independent. Tests: `nn`'s float64-reference sweep for the GEMM (k ∈ {1, 2, 3, 5, 8, 49, 200, 297, 672, 936}, m
      0–17, n 1–9 and 32, a guard element after the output, ldc > n) on every Go tile and the NEON path; an
      asm-versus-Go comparison in `tensor/asm_test.go`; the worker-count bit-identity test; `Conv2DPlan` against
      `ops.Conv2D` within `KernelTolerance("conv2d")` on the groups = 1 golden configurations and on the six
      basic-pitch shapes (input 172×264×8 with kernel 3×39→8 and 5×5→1, 172×264×1 with 7×7→32 at stride (1,3),
      172×88×32 with 7×3→1, 172×264×8 with 5×5→32 at stride (1,3), 172×88×33 with 3×3→1; all "same"); `ForwardRows`
      in any chunking bit-identical to `Forward`; 0 allocs for `Forward` and `ForwardRows`; errors for bad shapes
      (including ones with the right rank and element count) and groups ≠ 1; `Dim` against `Shape`, with 0 allocs.
      Add golden cases and hashes (asm and purego) for `MatMulTransBBias` and `Conv2DPlan`, and check once, with a
      throwaway module outside the repository that is not committed, that `Conv2DPlan` gives the same bits as `nn`'s
      `Conv2D` at 21ad164 on the same inputs; state the result in the PR. The existing golden hashes must not change.
      Benchmarks: the six basic-pitch layers for `Conv2DPlan` and `ops.Conv2D` (one worker) and one go-whistle encoder
      `Conv2D` shape, recorded with date, CPU and Go version in `ops/PERFORMANCE.md`; also run them under
      `just test-wasm`-style js/wasm and record the result. Update README (kernel table and the `ops` list) and add an
      `Example`.
- [ ] 1.2 `matmul-bias-avx2` — Port the amd64 AVX2 + FMA GEMM kernels of algo-transcribe's PLAN 4.2 (package `nn`,
      `gemm_amd64.go` and `gemm_amd64.s`, at the main commit that merged its PR #9) into `MatMulTransBBias`, with
      provenance headers: the run-time `golang.org/x/sys/cpu` check (`HasAVX2 && HasFMA`, read once), no VEX/SSE
      mixing and `VZEROUPPER` before every return, and the Go tiles as the fallback (tags
      `(!arm64 && !amd64) || purego`). Tests: the 1.1 sweep and asm-versus-Go comparison on amd64; golden hashes for
      the amd64 assembly (they must equal `nn`'s bits at that commit; check once with a throwaway module as in 1.1);
      a test that fails, not skips, when the `CI` environment variable is set and the CPU reports AVX2 and FMA but
      the AVX2 kernel was not selected; `just test-amd64` passes. Benchmarks on linux/amd64 next to 1.1's in
      `ops/PERFORMANCE.md` (name the CPU; Rosetta under Docker is not a real x86 measurement, say so if that is all
      there is). Update the README kernel table. Existing golden hashes must not change.
- [ ] 1.3 `route-existing-ops` — A numerics change, after v0.2.0: route `ops.Conv2D` with groups = 1 through
      `Conv2DPlan`'s im2col and `MatMulTransBBias` path (still allocating its output, as today), and give
      `tensor.MatMulTransB` on amd64 a register-blocked AVX2 + FMA kernel in place of one `DotProduct` per output
      (reusing 1.2's kernels where the layout allows). Do each only where `ops_bench_test.go` shows a gain on
      go-whistle's and go-pocket-tts' shapes, and report the numbers before and after. Update `golden_hashes_test.go`
      in the same commit and describe in the commit message and PR which kernels change, on which configurations, and
      by how much against the previous output (max abs/relative difference on the golden cases, within the pinned
      `KernelTolerances`); release as v0.3.0. Update MIGRATION.md's behaviour column.

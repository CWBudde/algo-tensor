# PLAN

Open work for `algo-tensor`. Each checkbox is one pull request, written to be picked up by a headless session
(ccbatch): the task text is the whole first paragraph of the item, so it stays self-contained. When an item is done,
tick it and append a one-line note with what was covered, the resulting numbers and any deliberate gap. `PLAN.md`
references inside copied code comments still mean go-whistle's plan (see AGENTS.md).

## Dependencies and sequencing

```
1.1                       (done)
→ release v0.2.0 (tagged after 1.1, at b33d487; additive: no golden hash changes); algo-transcribe then drops its own
  kernels (its PLAN 5.1)
1.3                       (done, after v0.2.0; a numerics change)
→ release v0.3.0
→ release v0.3.1 (tagged at f4a4dbc: #10, a bit-exact `Linear` over cache-sized weight panels with an AVX2 kernel; v0.3.0
  plus #10 only, without 1.2)
1.2                       (done, after algo-transcribe PR #9, its PLAN 4.2, merged at 58c8332; a numerics change on
                          AVX2 machines)
1.4                       (after 1.2; a numerics change)
→ release v0.4.0 (both amd64 numerics changes in one minor release)
```

## Phase 1: Fast small-channel convolution

`ops.Conv2D` loops over output pixels and runs one `tensor.Axpy` over the output channels per input tap, so layers with
few output channels get no vector speed-up and every call allocates. On the six convolutions of Spotify's basic-pitch
CNN (1, 8 or 32 output channels), measured on 2026-10-11 on an Apple M5 Pro with one worker, it takes 311 ms per
172-row window against 14.5 ms (0 allocs) for algo-transcribe's `nn` package, which uses im2col plus a small-m GEMM
with the bias fused and the weights packed once, Go tiles that also run fast under js/wasm, and NEON (and soon AVX2)
assembly. Phase 1 moves that code here so that algo-transcribe, go-whistle and go-pocket-tts share one set of kernels.

- [x] 1.1 `conv2d-plan` — Port algo-transcribe's GEMM and convolution (github.com/cwbudde/algo-transcribe, package
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
      Done: `tensor.MatMulTransBBias` (nn's Go tiles and NEON kernels with an `ldc`; `gemmNEON8x1` stores lane by lane
      for ldc > 1), `(*Tensor).Dim`, `ops.Conv2DPlan` with nn's 8-position im2col blocks, the sweep, asm-versus-Go,
      worker and conv-worker bit-identity, 0-alloc and error tests, three `Example`s. `Conv2DPlan` gave nn's `Conv2D`
      bits @ 21ad164 on 14 shapes (the six basic-pitch layers and eight Go-tile shapes) on arm64 NEON and purego,
      linux/amd64 AVX2 and purego, and js/wasm; for ldc = n `MatMulTransBBias` gave nn's `gemm` bits on 3,900 shapes.
      Golden hashes for arm64/neon, arm64/go, amd64/go and wasm/go, none for amd64/avx2 (1.2 adds them); no existing
      hash changed. One worker on an Apple M5 Pro: the basic-pitch window takes 13.9 ms against 328.9 ms for `Conv2D`
      (nn: 13.5 ms; the plan is 2–7 % slower per layer, not profiled), 94 ms against 748 ms under js/wasm; go-whistle's
      stem conv 3.6 ms against 9.6 ms. The plan text named `KernelTolerance("conv2d")`, which did not exist: added as
      `{2e-4, 2e-4}` (conv1d's), pinned, no other value changed.
- [x] 1.2 `matmul-bias-avx2` — Port the amd64 AVX2 + FMA GEMM kernels of algo-transcribe's PLAN 4.2 (package `nn`,
      `gemm_amd64.go` and `gemm_amd64.s`, at the main commit that merged its PR #9) into `MatMulTransBBias`, with
      provenance headers: the run-time `golang.org/x/sys/cpu` check (`HasAVX2 && HasFMA`, read once), no VEX/SSE
      mixing and `VZEROUPPER` before every return, and the Go tiles as the fallback (tags
      `(!arm64 && !amd64) || purego`). Tests: the 1.1 sweep and asm-versus-Go comparison on amd64; golden hashes for
      the amd64 assembly (they must equal `nn`'s bits at that commit; check once with a throwaway module as in 1.1);
      a test that fails, not skips, when the `CI` environment variable is set and the CPU reports AVX2 and FMA but
      the AVX2 kernel was not selected; `just test-amd64` passes. Benchmarks on linux/amd64 next to 1.1's in
      `ops/PERFORMANCE.md` (name the CPU; Rosetta under Docker is not a real x86 measurement, say so if that is all
      there is). Update the README kernel table. Existing golden hashes must not change. The AVX2 kernel still moves
      `MatMulTransBBias`'s bits on AVX2 machines (today the Go tiles' amd64/go bits), so it is a numerics change and
      is released with 1.4 as v0.4.0.
      Done: nn's `gemmAVX2N4` (2×4, 1×4) and `gemmAVX2N1` (8×1, 1×1) @ 58c8332 with an `ldc` (N4 keeps its channel
      offset in elements and bounds it by n in memory, to free R11 for ldc·4; N1 stores lane by lane for ldc > 1),
      chosen once by `HasAVX2 && HasFMA` for the NEON shapes (k ≥ 4, n = 1 or a multiple of 4); the direct-kernel,
      stride, NaN-past-the-end and `TestGemmAVX2Selected` tests (it also fails under CI on a runner without the
      features, as nn's does; checked with `GODEBUG=cpu.avx2=off`); `just test-amd64` passes. Golden hashes for
      amd64/avx2 (`MatMulTransBBias`, `Conv2DPlan`); no existing hash changed. Throwaway check: on all 4,704 AVX2 shapes
      of a 9,996-shape sweep (ldc = n and n+3, 1 and 4 workers) and the six basic-pitch layers the bits are nn @
      58c8332's. Found: in the same PR nn switched amd64's Go tile to 4×2 (1.6–1.8× on its Xeon without AVX2); not
      ported, because it moves the amd64/go hashes, so on 1,099 declined shapes (n ∈ {2, 3, 6, 7}, m mod 8 ≥ 4, at most 8
      outputs, |Δ| ≤ 3.2 × 10⁻⁵) the bits are nn @ 21ad164's, not 58c8332's: golden `MatMulTransBBias/13x6x7` and the
      7-channel configuration of `Conv2DPlan`. Against the Go tiles that amd64 with AVX2 ran before (v0.2.0), the golden
      cases move by at most 1.9 × 10⁻⁵ (`MatMulTransBBias`) and 2.5 × 10⁻⁵ (`Conv2DPlan`), so it is released with
      1.4 as v0.4.0. Only under Docker's x86 emulation (Rosetta), not real x86: the basic-pitch window
      35.4 ms with AVX2 against 201.3 ms with the Go tiles (nn: 35.2 and 98.9 ms with its 4×2 tile), the GEMMs 5.2–8.3×;
      the routing sweep now favours the plan on 21 of the 24 routed shapes. Gaps: the 4×2 amd64 tile (a numerics change
      of its own), and routing `Conv2D` on amd64. On a real Xeon Gold 5520+ (ops/PERFORMANCE.md) all tests pass with
      `CI=true` and the same amd64/avx2 hashes; the plan runs the window in 24.1 ms with AVX2 (2.4 % behind nn,
      interleaved) and the routing sweep favours it on 22 of the 24 routed shapes (the two losers: 255 out channels).
- [x] 1.3 `route-existing-ops` — A numerics change, after v0.2.0: route `ops.Conv2D` with groups = 1 through
      `Conv2DPlan`'s im2col and `MatMulTransBBias` path (still allocating its output, as today), and give
      `tensor.MatMulTransB` on amd64 a register-blocked AVX2 + FMA kernel in place of one `DotProduct` per output
      (reusing 1.2's kernels where the layout allows). Do each only where `ops_bench_test.go` shows a gain on
      go-whistle's and go-pocket-tts' shapes, and report the numbers before and after. Update `golden_hashes_test.go`
      in the same commit and describe in the commit message and PR which kernels change, on which configurations, and
      by how much against the previous output (max abs/relative difference on the golden cases, within the pinned
      `KernelTolerances`); release as v0.3.0. Update MIGRATION.md's behaviour column.
      Done, the `Conv2D` half only: the `MatMulTransB` AVX2 half moved to 1.4, after 1.2, by the user's decision on
      2026-10-12. `Conv2D` with groups = 1 builds a `Conv2DPlan` per call and runs each batch image through its raw-slice
      runner (`ForwardRows` = check + `run`, bits and 0 allocs kept); groups > 1 and the old path (`conv2DDirect`, the
      tests' reference) unchanged. It routes only where `BenchmarkConv2DRouting` (90 shapes) gained on every shape:
      k = kh·kw·inC ≥ 4, batch·outH·outW ≥ 64, outC ≤ 16·k, and not on amd64, where `MatMulTransBBias` has only Go tiles
      and the plan lost on most routed shapes under Docker's x86 emulation (go-whistle's stem chunk 2.0× slower with
      AVX2; no real x86 at hand). One worker on an Apple M5 Pro: the basic-pitch window 291.5 → 14.0 ms, go-whistle's
      stem 9.51 → 3.93 ms, its 8-frame chunk 235 → 116 µs; js/wasm 741 → 91 ms and 78.6 → 26.2 ms; 5 more allocations
      per call. Only the `Conv2D` golden hashes changed, on arm64/neon and arm64/go (to `ca4adf10951d3251`) and wasm/go
      (`d6b681f444405320`); amd64/avx2 and amd64/go keep go-whistle's bits. Golden case, groups = 1 part: max |Δ|
      9.5 × 10⁻⁷ and max |Δ|/|old| 1.0 × 10⁻³ (at an output of 7 × 10⁻⁵) on arm64, 1.9 × 10⁻⁶ and 8.5 × 10⁻⁴ on
      js/wasm, within the unchanged `{2e-4, 2e-4}`. Found: the plan multiplies the zero padding by the weights, so an
      infinite weight that only meets padding gives NaN where the per-pixel path stays finite (pinned in a test). Gap:
      `Conv2D` on amd64 stays on the per-pixel path until 1.2's kernels can be measured on real x86; routing it there
      is a numerics change of its own.
- [ ] 1.4 `matmul-transb-avx2` — A numerics change, after 1.2: give `tensor.MatMulTransB` on amd64 a register-blocked
      AVX2 + FMA kernel in place of one `DotProduct` per output (reusing 1.2's kernels where the layout allows), only
      where `ops_bench_test.go` shows a gain on go-whistle's and go-pocket-tts' shapes, and report the numbers before
      and after (name the CPU; Rosetta under Docker is not a real x86 measurement, say so if that is all there is).
      `Linear`, `MatMul`, `Conv1D` and `ConvTranspose1D` run through it, so the hashes of what builds on it can move too.
      Update `golden_hashes_test.go` in the same commit and describe in the commit message and PR which kernels change,
      on which configurations, and by how much against the previous output (max abs/relative difference on the golden
      cases, within the pinned `KernelTolerances`); release as v0.4.0, together with 1.2. Update MIGRATION.md's behaviour
      column.

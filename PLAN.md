# PLAN

Open work for `algo-tensor`. Each checkbox is one pull request, written to be picked up by a headless session
(ccbatch): the task text is the whole first paragraph of the item, so it stays self-contained. When an item is done,
tick it and append a one-line note with what was covered, the resulting numbers and any deliberate gap; a declined
item is marked `[-]`, keeps its text and says why. `PLAN.md` references inside copied code comments still mean
go-whistle's plan (see AGENTS.md).

## Dependencies and sequencing

```
1.1                       (done)
→ release v0.2.0 (tagged after 1.1, at b33d487; additive: no golden hash changes); algo-transcribe can then drop its
  own kernels
1.3                       (done, after v0.2.0; a numerics change)
→ release v0.3.0
→ release v0.3.1 (tagged at f4a4dbc: #10, a bit-exact `Linear` over cache-sized weight panels with an AVX2 kernel; v0.3.0
  plus #10 only, without 1.2)
1.2                       (done, after algo-transcribe PR #9, its PLAN 4.2, merged at 58c8332; a numerics change on
                          AVX2 machines)
1.4                       (done, after 1.2; bit-exact, by the user's decision on 2026-10-12)
→ release v0.4.0 (tagged at e4f3f89: 1.2's amd64 numerics change, which reaches none of the consumers' calls, plus 1.4)
2.1                       (done; bit-exact)
2.2                       (done; bit-exact)
→ release v0.5.0 (tagged at b9afdfb, the merge of #15: 2.1 and 2.2 move no bits, a minor version for the speed-up)
2.3                       (declined by the user's decision on 2026-10-13: about 10–15 % over 2.1's kernel on large
                          products, not worth arm64/sme golden hashes that CI cannot check)
3.1, 3.2, 3.3, 3.4        (independent; additive: a new golden set `goldenSeparate`, no existing hash changes)
3.5                       (a decision item, after 3.1–3.3; adds code only if its measurement says so)
→ release v0.6.0 (minor, additive); consumed by algo-separate (github.com/cwbudde/algo-separate, its PLAN 2.2) and
  algo-audio-editor Phase 36 (stem separation)
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
- [x] 1.4 `matmul-transb-avx2` — Bit-exact, after 1.2: give `tensor.MatMulTransB` on amd64 an AVX2 + FMA kernel in
      place of one `DotProduct` per output that keeps every output's bits: #10's `linearDot3AVX2`, which replays the
      AVX2 `DotProduct` for three rows against one, fits A · Bᵀ with the rows of A for `Linear`'s tokens. Use it only
      where `BenchmarkMatMulTransB` shows a gain on go-whistle's and go-pocket-tts' shapes, and report the numbers before
      and after (name the CPU; Rosetta under Docker is not a real x86 measurement, say so if that is all there is).
      `ops.Conv1D` and `ops.ConvTranspose1D` run through it (`Linear` and `MatMul` do not). No golden hash may change;
      a test compares every output with `DotProduct` bit for bit, on special values and for any worker count. Release as
      v0.4.0, together with 1.2. Update MIGRATION.md's behaviour column.
      Done (2026-10-12): `gemm_amd64.go` runs `Linear`'s loop for A · Bᵀ (256 KiB panels of B rows, three rows of A
      per row of B in `linearDot3AVX2`, leftover rows through `DotProduct`) from three rows of A and k ≥ 8 with AVX2 +
      FMA; `TestMatMulTransBBitIdentical` (k 0–1031, m 0–17, n 1–600, ldc > n, special values, misaligned rows, 1 to
      GOMAXPROCS workers) and `TestMatMulTransBWorkersBits` compare with `DotProduct` bit for bit, and fail with the
      leftover rows in `dotF32Generic`; no golden hash changed. On a Xeon Gold 5520+ (one pinned worker, load 320–1200,
      ops/PERFORMANCE.md): go-pocket-tts' FlowLM over a 64-token prompt 6.7× (FFN) and 3.0× (in_proj), go-whistle's
      FFN 2.9×, linear 1.7×, attention scores 1.8× and values 1.6×; GEMVs (m < 3, go-pocket-tts' per-frame FlowLM)
      and the Mimi conv tile unchanged. Not done: the 1-row-of-A × 3-rows-of-B kernel, because the GEMVs are bound by
      memory (30 GB/s for 4 MiB of B); and default worker counts, not measurable on the loaded VM. Gap: the Mimi conv
      tile, where A (7 MiB of kernel rows) is much larger than B, reads A once per panel of B. The plan text named a
      numerics change and 1.2's kernels; the user chose the bit-exact kernel on 2026-10-12.

## Phase 2: SME2 on Apple M4 and later

Apple's M4 and M5 have SME2 (`hw.optional.arm.FEAT_SME2`, 512-bit streaming vectors, so four 16×16 fp32 ZA tiles fed
by `FMOPA` outer products). A throwaway spike on 2026-10-12 (Apple M5 Pro, Go 1.27.1, one worker) found:

- **Bit-exact is possible.** Replaying `gemmNEON4x4`'s arithmetic, one ZA tile per k%4 lane, gave `MatMulTransBBias`'s
  arm64 bits on 7,040 shapes and on special values (Inf, NaN, −0, subnormals). The layout: A in 16-row and B in
  16-column panels, k-major, kp = 4⌈k/4⌉ steps; step p < 4⌊k/4⌋ is column p, and the tail group replays
  `gemmNEON4x4`'s reload of columns k−4..k−1 with the already-summed lanes zeroed in both operands (so a lane adds
  fma(+0, +0, acc), as NEON does); tile t accumulates the steps p ≡ t (mod 4) from +0, and each output is
  ((t0 + t1) + (t2 + t3)) + bias, the `FADDP` order of `REDUCE4`.
- **It is fast on large products only.** Against `MatMulTransBBias` on NEON: go-pocket-tts' FlowLM in_proj
  (64×3072×1024) 3.61 → 0.52 ms, its FFN (64×4096×1024) 4.67 → 0.96 ms, `linear_16x1024x1024` 278 → 37 µs,
  go-whistle's FFN (375×2048×512) 6.99 → 1.23 ms; the 22–34 µs shapes (conv tile, attention scores and values) tie or
  lose, and the basic-pitch conv GEMMs lost 2.5–7× because the spike packed A and reduced ZA in Go (packing alone
  took 70 % of the time on 15136×32×200, the kernel 8 %). Split by rows, the SME units scale: 616 GFLOPS with one
  worker, 1,132 with 2, 1,622 with 4, 2,319 with 8, 1,891 with 15.
- **Darwin loses Z registers on signals.** When a signal handler returns while the thread is in streaming mode, the
  kernel zeroes lanes 4–15 of every Z register; ZA, the P registers and SVCR survive (reproduced in C, so not Go's
  doing). Go's preemption (SIGURG) and profiler (SIGPROF) signals hit SME kernels constantly: unguarded, the bit sweep
  failed on every run and passed with `GODEBUG=asyncpreemptoff=1`. Guard: load a canary into z31 at the start of each
  output block, check it once after the block's k loop, and redo the block from a zeroed ZA if it was hit; under a
  CPU profile, a SIGUSR1 storm, GC and 2× oversubscribed goroutines, 602,242 GEMMs gave 0 wrong results with 42,018
  redos. One check per k step instead cost 6–12× throughput (the result travels from the SME unit to the core).
- **Toolchain.** Go's assembler has no SME mnemonics: emit `WORD`s, each commented with its mnemonic as in
  `neon_fp.h`, taken from `xcrun clang -c -mcpu=apple-m4 -march=armv9.2-a+sme2` and `xcrun objdump -d` (e.g.
  `smstart` d503477f, `smstop` d503467f, `zero {za}` c00800ff, `fmopa za0.s, p0/m, p0/m, z0.s, z4.s` 80840000). A C
  test program must be built with `-mcpu=apple-m4` alone: `-march=armv9.2-a` lets clang emit non-streaming SVE,
  which Apple's CPUs lack (SIGILL). GitHub's macOS runners have no SME, so the SME tests run on a developer machine
  only.

- [x] 2.1 `sme-gemm-bias` — Bit-exact: give `tensor.MatMulTransBBias` on arm64 an SME2 path for the shapes
      `gemmNEON4x4` takes (k ≥ 4, n a multiple of 4) that gives every output `gemmNEON4x4`'s bits, so no golden hash
      changes, in new files (`arm64 && !purego`, a header saying they are algo-tensor's own as in `gemm_amd64.go`).
      Read Phase 2's notes above first: the panel layout with its tail group, the four tiles per 16×16 output block,
      the reduction order, the signal guard and the `WORD` encodings come from the spike. Select it once at start-up
      where the CPU reports SME2 (on darwin `syscall.Sysctl("hw.optional.arm.FEAT_SME2")`; use `golang.org/x/sys/cpu`
      instead if it reports SME2; NEON everywhere else), and only for products where `BenchmarkMatMulTransBBias`
      (add go-whistle's and go-pocket-tts' `BenchmarkMatMulTransB` shapes with n a multiple of 4) shows a gain: a
      size threshold, with the smaller products on NEON. Do the whole call in one `SMSTART`/`SMSTOP` per worker: pack
      A (in assembly, or in Go only if it does not dominate, see the notes) and B into scratch (`MatMulTransBBias` has
      no plan, so B is packed per call; 0 allocations in steady state, e.g. a `sync.Pool`), and reduce the four tiles,
      add the bias and store into c with `ldc` in assembly, predicated for m%16 and n%16, instead of dumping ZA to
      memory. Guard every block: the canary in z31 is checked once after the block's k loop by a compare that redoes
      the block when any canary lane is zero (the spike's check only caught all lanes zero); after 8 redos of one
      block finish the call with NEON, which gives the same bits. The `Parallel` split may stay as it is, since every
      output's bits are independent of it; measure the default worker count against the spike's scaling. Tests: the
      1.1 sweep and the spike's (k ∈ {4, 5, 6, 7, 8, 9, 49, 200, 297, 672, 936}, n ∈ {4, 8, 12, 16, 20, 32, 36, 64},
      m 1–40, ldc = n and n+3, a guard element after the output) comparing the SME path with `gemmNEON4x4` bit for
      bit, with the threshold lowered by a test hook so small shapes reach SME; special values; a stress test that
      runs SME GEMMs on 2×GOMAXPROCS goroutines for a few seconds under a CPU profile, a SIGUSR1 storm and
      `runtime.GC` and checks every bit; the redo and the NEON fallback, forced by a test hook that clobbers the
      canary; a test that fails, not skips, under `CI` when the CPU reports SME2 but the SME path was not selected,
      as `TestGemmAVX2Selected`. All tests pass with and without `-tags purego`, under `-race`, `just test-amd64` and
      the js/wasm build; `TestGoldenHashes` passes unchanged. Record the benchmarks before and after in
      `ops/PERFORMANCE.md` (date, CPU, Go version, one worker and the default), and update the README kernel table.
      Done (2026-10-12): `gemm_bias_sme_arm64.s` packs both operands itself, through tile za0, the operand with fewer
      rows whole and the other panel by panel, so each is read once, and runs Phase 2's layout, four tiles per 16×16
      block and `REDUCE4`'s order with predicated `ldc` stores, in one `SMSTART`/`SMSTOP` per worker; selected once
      by the sysctl (x/sys/cpu v0.40.0 has no SME2) for k ≥ 4, n a multiple of 4, m ≥ 16, n ≥ 16 and m·n·k ≥ 2¹⁵,
      0 allocations on one worker (`sync.Pool`). The canary is set once and checked after every k loop and after every
      block's stores (any lane zero; a failure after the stores redoes only the reduction from the intact ZA); the 9th
      failure of one block finishes on NEON from that block's rows (columns). Tests: the 1.1 and spike sweeps (7,754
      shapes) and both `Parallel` splits on 1–16 workers bit for bit against `gemmNEON4x4`, special values, the hooks
      (redo, fallback, a lost NaN), a 3 s stress test (8.4 million GEMMs, 2,000–3,500 redos, no wrong bit),
      `TestGemmSMESelected`; no golden hash changed (`MatMulTransBBias/200x36x300` runs on SME). One worker on an Apple
      M5 Pro: FlowLM in_proj 3.42 → 0.45 ms, its FFN 4.65 → 0.64 ms, go-whistle's FFN 6.91 → 0.91 ms and linear
      1.68 → 0.23 ms, Mimi's conv tile 0.95 → 0.14 ms, the conv tile and attention values (25–35 µs) 4.7× and 4.4×; on the default 15 workers
      1.8–3.3×. All workers together reach about 2.1 TFLOPS (spike: 1.9–2.3), within 15 % of the best worker count,
      so the split stays (one piece per worker, at least 2²⁴ multiply-adds). Found: `FMOPA` gives the default NaN (ZA
      instructions behave as if FPCR.DN were set), so NaN outputs lost the payload and sign NEON keeps, which the
      spike's check missed; the kernel collects 0·output in z28 and the call is computed again on NEON for a NaN or
      infinite output (4–7 %; a check per row or block cost 30–40 %). 8-row products gain alone, but on SME they made
      `Conv2DPlan`'s conv2d_4 slower (2.05 → 2.5 ms), so the threshold starts at 16 rows and the plan is unchanged
      (2.2). Gaps: darwin only (the guard is untested elsewhere); measured on an M5 Pro under a load of 3.5–4, not on
      an M4.
- [x] 2.2 `sme-conv2d-plan` — Bit-exact, after 2.1: run `ops.Conv2DPlan`'s GEMM on 2.1's SME kernel where 2.1
      selects SME and the plan's shape is one `gemmNEON4x4` takes (k ≥ 4, outC a multiple of 4; outC = 1 stays on
      `gemmNEON8x1`), so no golden hash changes. Today the plan calls `tensor.MatMulTransBBias` on 8-position im2col
      blocks, far below 2.1's threshold. Instead: pack the weights once into SME panels in `NewConv2DPlan`, have
      im2col write A directly in the 16-row k-major panel layout (with the tail group), and run whole panels through
      the kernel, which needs an unexported entry point in `tensor` for prepacked operands, or an exported one only
      if `ops` cannot reach it otherwise (no breaking change). Keep `Forward` and `ForwardRows` at 0 allocations and
      bit-identical for any chunking and worker count. Use it only where `BenchmarkConv2DPlanBasicPitch`,
      `BenchmarkConv2DPlanWhistleStem` and `BenchmarkConv2DRouting` show a gain (the spike: the kernel took 8 % of a
      basic-pitch GEMM, packing the rest; the outC = 8 layer fills half a tile, so it may stay on NEON) and report
      the numbers before and after in `ops/PERFORMANCE.md`. `ops.Conv2D` with groups = 1 runs through the plan, so
      its numbers move too: report them, and rerun `BenchmarkConv2DRouting` to see whether the routing rule should
      change (a routing change is a numerics change of its own; propose it, do not make it).
      Done (2026-10-12): `NewConv2DPlan` packs the weights once (`internal/sme.Pack`) and a call runs the im2col rows of up
      to 2²⁰ values of output positions, across output rows, through 2.1's kernel with the inner operand prepacked
      (`gemmSME`'s new `innerPacked`), which `ops` reaches through `sme.GEMM` in the new package `internal/sme` (set by
      `tensor` at start-up, nil elsewhere; nothing exported). Selected for k ≥ 32, k·outC·outC/(16⌈outC/16⌉) ≥ 1536 and
      outH·outW ≥ 1024, per call from 2¹⁹ multiply-adds; a chunk with a NaN or infinite output or a given-up block is
      computed again on NEON. Tests: 1,768 plan shapes, any chunking, 1–4 workers, mixed NEON/SME calls, fake failing
      kernels and NaN payloads bit for bit against the NEON plan, 0 allocations (also under -race), `sme.Pack` against
      the Phase 2 layout, `sme.GEMM`'s redo and NaN paths, `TestConv2DPlanSMESelected`; no golden hash changed. One
      worker on an Apple M5 Pro: conv2d_1 7.67 → 5.48 ms, conv2d_2 0.90 → 0.64 ms, conv2d_4 2.04 → 0.85 ms, the
      basic-pitch window 13.8 → 10.2 ms (`Conv2D` 14.0 → 10.6 ms, one more allocation and 3–4 MB more scratch per call
      on those layers); the stem (k = 9), the outC = 1 layers and all 90 routing shapes stay on NEON, so the routing
      sweep is unchanged and the proposal is no routing change. Found: (1) im2col writing the panel layout directly, as
      planned, was 1.5–3.8× slower in Go (conv2d_1 13.8 or 25.6 against 6.8 ms) than contiguous rows plus the kernel's
      own transpose through `za0`, so only the weights are prepacked; (2) the SME unit is slow to start after Go code ran
      (a 6.6 µs call takes 10 µs after 1 µs of Go, 16.8 µs after 66 µs), so chunks must be large (conv2d_1 7.2 ms in
      chunks of 256 positions, 5.4 ms in 1024) and small calls stay on NEON; (3) the reduction of the four tiles, one
      row at a time, dominated small k: the kernel now reduces four rows at once (same bits), which also made
      `MatMulTransBBias`' SME shapes 1.03–1.26× faster. The outC = 8 layer (conv2d_1) gains despite filling half a
      tile. Gaps: the stem stays on NEON (4.7 against 3.6 ms on SME; the check after each block's stores costs about a
      quarter on such blocks, but checking once per panel would change 2.1's guard); up to 4 MiB of scratch per plan and
      worker; measured on an M5 Pro under a load of 2.4–4.4, not on an M4.
- [-] 2.3 `sme-fast-gemm` — Declined (2026-10-13), see the note below. A numerics change, only after the user
  decides on it: an SME kernel with one FMA chain per output, 32×32 output blocks over the four tiles (two A and two
  B vectors per k step, four `FMOPA`s), for `MatMulTransBBias` on SME2 machines. In the spike it ran FlowLM's FFN in
  0.55 ms against 0.96 ms for the bit-exact kernel and in_proj in 0.40 against 0.52 ms; it ties on the smaller
  shapes. It moves the arm64 bits on SME machines only, so the golden hashes would depend on the CPU within
  arm64/neon: add an arm64/sme configuration to `golden_hashes_test.go`, say in the commit and PR which kernels
  change and by how much (within the pinned `KernelTolerances`), and release it as a new minor version.
  Declined (2026-10-13), by the user's decision: the shipped bit-exact kernel runs FlowLM's FFN in 0.64 ms and
  in_proj in 0.45 ms (v0.5.0, after 2.2's four-row reduction), against the spike's 0.55 and 0.40 ms for this
  kernel, so it would gain about 10–15 % on the large products and nothing on the small ones, in exchange for
  arm64/sme golden hashes that depend on the CPU and that CI cannot check (GitHub's macOS runners have no SME).
  Phase 2 ends with 2.2; reopen this if SME machines become a target where that gain matters.

## Phase 3: Layers for HTDemucs

[algo-separate](https://github.com/cwbudde/algo-separate) ports Meta's Demucs v4 `htdemucs` (Hybrid Transformer Demucs,
demucs 4.0.1, MIT, 41,984,456 parameters stored as F16) to pure Go for algo-audio-editor's stem separation (its Phase
36). Most of the network runs on kernels this module already has: `ops.Conv1D` with dilation and groups (the DConv
branches, kernel 3, dilation 1 and 2), `ops.ConvTranspose1D` (the time decoder), `ops.Conv2DPlan` (the decoder's 3×3
rewrite convolutions), `tensor.Linear` (the transformer's projections and feed-forward, 512 → 2048 → 512),
`tensor.LayerNorm`, `tensor.Softmax` and `ops.Attention`. Missing are GroupNorm, the exact erf GELU, GLU, a tested
non-causal cross-attention at the model's sequence lengths, and a public safetensors reader for the weights (today
private to go-pocket-tts). Per 7.8 s segment the model costs roughly 150 GMAC by our estimate (algo-separate measures
it in its PLAN 1.3), about 30 of them in attention over 2,688 frequency tokens and 1,344 time tokens, so the attention
item benchmarks those shapes. Every item adds its cases to a new golden set: `separateGoldenCases()` in new
`golden_cases_separate_test.go` files (in `tensor` and/or `ops`, whichever package holds the operator) with a
`goldenSeparate` table in `golden_hashes_test.go`, wired into `TestGoldenHashes` next to `goldenTranscribe` as
`{"algo-tensor, algo-separate's layers", goldenSeparate, separateGoldenCases()}`, with hashes for every configuration
the existing sets cover (arm64/neon, amd64/avx2, purego, js/wasm). Existing hashes must not change. New files are
algo-tensor's own (a header saying so, as `golden_cases_transcribe_test.go`), except 3.4's, which carry go-pocket-tts
provenance.

- [ ] 3.1 `groupnorm` — Add `GroupNorm(x, weight, bias *Tensor, numGroups int, eps float32) (*Tensor, error)` to
      `tensor` (next to `LayerNorm` in `nn_ops.go`) with `torch.nn.GroupNorm` semantics: `x` is channel-first
      [B, C, ...] (rank ≥ 2, any number of trailing spatial dims including none), C must be divisible by `numGroups`,
      mean and biased variance are taken per batch item and group over (C/numGroups) × all spatial positions, the
      result is `(x − mean) / sqrt(var + eps)`, then the per-channel affine `weight[c]·y + bias[c]` when weight and
      bias are non-nil (both or neither; shape [C]). HTDemucs needs `numGroups = 1` on [N, C, T] (the DConv branches'
      `nn.GroupNorm(1, C)`, C from 6 to 768, with time folded from frequency into N) and on [B, C, T] after a
      transpose for the transformer's `MyGroupNorm(1, 512)`, which normalises over all tokens and channels together
      (2,688 or 1,344 tokens × 512); test `numGroups` > 1 too (e.g. 4, as torch's default `norm_groups`). Reduce in
      float64 per group (one pass for the sum and sum of squares is not acceptable for large means; use two passes
      or Welford) and state the choice in the doc comment; the output is float32. Errors for nil input, rank < 2,
      C % numGroups ≠ 0, eps ≤ 0 and wrong affine shapes, wrapped like the other `tensor` errors. Split the work over
      batch × group with `Parallel` only if a benchmark shows a gain, and keep the bits independent of the worker
      count (`TestParallelPieces` style). Tests: table-driven against a float64 reference (`Abs`/`Rel` ≤ 1e-5 on
      inputs with |x| ≤ 10 and a mean offset of 1e3, plus a constant group whose variance is 0), the HTDemucs shapes
      above, a case equal to `LayerNorm` (C = 1 with spatial = last dim after a reshape) to cross-check; add a
      `"group_norm"` entry to `ops.KernelTolerances` (pinned, `TestKernelTolerancesPinned`) and golden cases at small
      and HTDemucs shapes. Benchmarks: [2688, 512] after the transpose, [8·336, 384, 336]-style DConv shapes
      ([2688, 48, 336] is too big for a benchmark; use [336, 48, 2048] and [336, 768, 8]), native and
      `GOOS=js GOARCH=wasm`; record them in `ops/PERFORMANCE.md`. Update README's operator table and MIGRATION.md.
- [ ] 3.2 `gelu-glu` — Add exact activations to `tensor`: `GELU(x *Tensor) (*Tensor, error)` and an in-place
      `GELUInPlace([]float32)` computing `0.5·x·(1 + erf(x/√2))` (torch's default `approximate='none'`; go-pocket-tts'
      private `geluTanh` is the tanh approximation and stays where it is), and `GLU(x *Tensor, dim int)
    (*Tensor, error)`: split `dim` (size must be even) into halves a and b and return `a · sigmoid(b)`, as
      `torch.nn.functional.glu`; HTDemucs uses `dim = 1` on channel-first [N, 2C, T] and [B, 2C, F, T]. Evaluate erf
      in float64 (`math.Erf`) per element unless a float32 polynomial reaches ≤ 1 ulp-level agreement with it over
      the whole float32 range and is measurably faster natively and under wasm; either way, document the choice and
      the measured worst error against a float64 reference over a dense sweep of [−10, 10] plus ±Inf, NaN, ±0,
      subnormals and |x| > 10. `sigmoid(b)` as `1 / (1 + exp(−b))` in float64 or a stable float32 form, again with the
      sweep. Both must be deterministic per element (no dependence on vector width or workers) so purego, NEON, AVX2
      and wasm give the same bits; if that rules out assembly, say so. Errors for nil input, an odd split dim and an
      out-of-range dim (negative dims count from the end, as in `Softmax`). Tests: the sweeps, GLU on dims 0, 1 and
      −1 against a reference, aliasing safety of the in-place variant; `"gelu"` and `"glu"` entries in
      `ops.KernelTolerances`; golden cases. Benchmarks on 336·2048·48 and 2688·2048 elements, native and wasm, in
      `ops/PERFORMANCE.md`. Update README and MIGRATION.md.
- [ ] 3.3 `cross-attention` — Verify and test `ops.Attention` without a causal mask for tq ≠ tk at HTDemucs'
      shapes: the cross-transformer has 5 layers per branch, alternating self-attention (layers 0, 2, 4) and
      cross-attention (layers 1, 3), dim 512, 8 heads of 64, where the frequency branch's 2,688 tokens (8 bins ×
      336 frames) attend to the time branch's 1,344 tokens and the time branch attends back, batch 1. Add golden
      cases for `Attention(q, k, v, false, 0)` on [1, 8, 2688, 64] × [1, 8, 1344, 64] and the reverse (inputs from
      `goldenRand`, hashed) plus a small tq ≠ tk shape, and a test against a float64 reference at a reduced shape
      ([1, 2, 37, 16] × [1, 2, 23, 16]) within the pinned `"attention"` tolerance. Then decide whether a helper for
      torch's `nn.MultiheadAttention` belongs here: HTDemucs stores `in_proj_weight` [3C, C], `in_proj_bias` [3C],
      `out_proj.weight` [C, C] and `out_proj.bias` [C] (batch_first, no mask, scale 1/√64); the forward is Q = x_q·Wqᵀ,
      K = x_kv·Wkᵀ, V = x_kv·Wvᵀ (rows 0–C, C–2C, 2C–3C of the packed weight), heads split from the channel dim,
      attention per head, heads merged, out_proj. If composing `tensor.Linear`, reshapes and `ops.Attention` costs
      more than 5 % over a fused version at these shapes, or needs copies the caller cannot avoid, add
      `ops.MultiheadAttention(q, kv, inProjW, inProjB, outProjW, outProjB *tensor.Tensor, heads int)` with a test
      against the composition and golden cases; otherwise document the composition in an `Example` and add no API.
      Benchmarks (`ops/PERFORMANCE.md`): self-attention at 2,688 and 1,344 tokens and both cross directions, with
      and without the projections, native (one worker and the default) and under js/wasm in Node, reported as ms and
      GMAC/s; the score matrix for 2,688 × 2,688 × 8 heads is 231 MB in float32, so report peak memory too and, if the
      current implementation materialises it whole, propose (do not make) a blocked variant with its expected saving.
- [ ] 3.4 `safetensors` — Extract go-pocket-tts' safetensors reader into a public package
      `github.com/cwbudde/algo-tensor/safetensors` (from github.com/cwbudde/go-pocket-tts `internal/safetensors` at
      origin/main @ 4b72d3a: `store.go`, `writer.go` and the generic part of `reader.go`, with their tests), keeping
      the generic API: `Store`, `StoreOptions` (with `KeyMapper` and `RemapMode`), `OpenStore`,
      `OpenStoreFromBytes`, `Store.Names`, `Has`, `Tensor`, `TensorWithShape`, `ReadAll`, `Close`, the `Tensor`
      type, `EncodeTensors` and `WriteFile`, dtypes F32, F16, BF16 and I64 decoded to float32 (I64 as today). Leave
      the voice- and Mimi-specific parts (`VoiceFileKind`, `VoiceModelState`, `LoadFirstTensor*`,
      `LoadVoiceEmbedding*`, `InspectVoiceFile*`, `LoadVoiceModelState*`, `ValidateModelKeys`,
      `HasMimiEncoderWeights`, `EncodeVoiceModelState`, `WriteVoiceModelState` and their private helpers) in
      go-pocket-tts. Keep `safetensors.Tensor` (name, shape, data)
      as it is, and add a conversion to `*tensor.Tensor` only if it needs no copy (`tensor.FromOwned`). F16 and BF16 to float32 must be exact, including subnormals, ±Inf, NaN payload
      class and −0 (HTDemucs' 533 weight tensors are stored as F16): test every one of the 65,536 F16 and BF16 bit
      patterns against a reference decoder. Bound header size and tensor byte ranges as the source does (and fuzz
      the header parser: `FuzzOpenStoreFromBytes`), never trust offsets, and reject overlapping or out-of-range
      entries. Provenance headers as AGENTS.md requires (`From github.com/cwbudde/go-pocket-tts internal/safetensors/…
    @ 4b72d3a`, then `algo-tensor change:` lines). Tests: the copied tests, the dtype sweep, round trip through the
      writer, the fuzz target, and `GOOS=js GOARCH=wasm` (the editor's separation worker loads an 84 MB F16 file from
      a byte slice, so `OpenStoreFromBytes` must not copy the data more than once; measure peak memory for an 84 MB
      file in the PR). README, MIGRATION.md (a new section: go-pocket-tts may switch to this package in its own PR,
      which is not this repository's work).
- [ ] 3.5 `conv-freq-fold` — Decision item, after 3.1–3.3. HTDemucs' frequency branch uses `nn.Conv2d` and
      `nn.ConvTranspose2d` with kernel [8, 1], stride [4, 1] and padding [2, 0] on [B, C, F, T] (T = 336, F = 2048 →
      512 → 128 → 32 → 8), which equal `ops.Conv1D` / `ops.ConvTranspose1D` along F with T folded into the batch,
      after a layout change to [B·T, C, F] and back; the DConv branches fold the other way ([B·F, C, T]). Measure the
      folded Conv1D path against `Conv2DPlan` with a [8, 1] kernel at the encoder and decoder shapes (native, one
      worker and the default, and wasm), including the transposes. Then decide, and record the decision and numbers
      here: either nothing is added (algo-separate does the folding itself; document the equivalence in an `Example`),
      or a layout helper (e.g. `tensor.Permute` for rank-4 channel-first tensors, if `Transpose` is not enough) or a
      dedicated `[K, 1]` path lands here with golden cases. Prefer the smallest public API that avoids a copy per
      layer.

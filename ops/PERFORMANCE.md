# Runtime Kernel Performance Notes

<!-- From github.com/cwbudde/go-pocket-tts internal/runtime/ops/PERFORMANCE.md @ 9faba38; go-whistle dropped it.
algo-tensor change: the package path, go-whistle's benchmarks next to go-pocket-tts', and the state of the acceleration
points; the Conv2DPlan benchmarks; Conv2D through Conv2DPlan (v0.3.0) and its routing sweep; Linear's weight panels and
amd64 kernel. -->

This package intentionally started with deterministic, single-threaded CPU kernels as the correctness baseline for
safetensors-native inference.

## Current benchmark entry points

Run:

```bash
go test -bench . -run ^$ ./ops
```

Benchmarks include, at go-pocket-tts' FlowLM and Mimi shapes:

- `BenchmarkMatMulFlowLM`
- `BenchmarkLayerNormFlowLM`
- `BenchmarkAttentionFlowLM`
- `BenchmarkConv1DMimi`, `BenchmarkConv1DMimiParallel`
- `BenchmarkFrameDecodeThroughput`

and at go-whistle's shapes (width 512, 8 query heads of qk 48 and v 64, 375 encoder frames for a 30 s clip):

- `BenchmarkLinearWhistleQProj`
- `BenchmarkMatMulWhistleScores`
- `BenchmarkAttentionWhistleEncoder`, `BenchmarkAttentionWhistleDecodeStep`
- `BenchmarkConv1DWhistleDepthwise`, `BenchmarkConv1DWhistlePointwiseParallel`

`go test -bench . -run ^$ ./tensor` runs the kernel benchmarks (dot, axpy, GEMM, softmax, layer norm).

## Conv2DPlan against Conv2D

`BenchmarkConv2DPlanBasicPitch` and `BenchmarkConv2DBasicPitch` run `Conv2DPlan.Forward` and `Conv2D` on the six
convolutions of Spotify's basic-pitch CNN for one 172-row window ("same" padding, with bias);
`BenchmarkConv2DPlanWhistleStem` and `BenchmarkConv2DWhistleStem` run go-whistle's first encoder stem conv on a 30 s clip
(3000 × 80 × 1, 3 × 3 → 128, stride 2, padding 1, no bias). All on one worker (`tensor.SetWorkers(1)`,
`SetConvWorkers(0)`):

```bash
go test -run '^$' -bench 'Conv2DPlanBasicPitch|Conv2DBasicPitch|WhistleStem' -benchmem -count=3 ./ops
```

Measured on 2026-10-11 on an Apple M5 Pro (darwin/arm64, NEON), Go 1.27.1, median of three runs; js/wasm is the same
test binary under Node 24.12.0 (`go_js_wasm_exec`, as `just test-wasm` runs it, `-test.benchtime=3x`, two runs) on the
same machine, with the Go tiles (4 × 2) and one thread:

| layer                                    | `Conv2DPlan` arm64 | `Conv2D` arm64 | `Conv2DPlan` wasm | `Conv2D` wasm |
| ---------------------------------------- | -----------------: | -------------: | ----------------: | ------------: |
| conv2d_1 (172×264×8, 3×39 → 8)           |            7.73 ms |       226.7 ms |           59.5 ms |      506.2 ms |
| contours-reduced (172×264×8, 5×5 → 1)    |            1.55 ms |        30.6 ms |            4.4 ms |       40.3 ms |
| conv2d_2 (172×264×1, 7×7 → 32, (1, 3))   |            0.89 ms |         5.8 ms |            6.1 ms |       29.2 ms |
| conv2d_3 (172×88×32, 7×3 → 1)            |            1.14 ms |        30.8 ms |            3.6 ms |       41.1 ms |
| conv2d_4 (172×264×8, 5×5 → 32, (1, 3))   |            2.05 ms |        21.3 ms |           19.0 ms |      112.6 ms |
| conv2d_5 (172×88×33, 3×3 → 1)            |            0.52 ms |        13.7 ms |            1.6 ms |       18.2 ms |
| **basic-pitch window**                   |        **13.9 ms** |   **328.9 ms** |       **94.3 ms** |  **747.5 ms** |
| go-whistle stem (3000×80×1, 3×3 → 128/2) |            3.61 ms |         9.6 ms |           25.6 ms |       79.5 ms |

`Conv2DPlan` allocates nothing per call; `Conv2D` allocates its output (7 allocations, 30.7 MB for the stem). On the
basic-pitch window it is 24× faster natively and 8× under js/wasm. The plan's bits equal algo-transcribe nn's `Conv2D` @
21ad164, whose `BenchmarkConv2DWindow` gave 13.5 ms for the window in the same session (7.55, 1.45, 0.87, 1.08, 2.02 and
0.50 ms): the plan is 2–7 % slower per layer, most on the single-channel layers; not profiled, the likely cause is the
checks and the block loop of the exported `tensor.MatMulTransBBias` around nn's kernels. A first, noisier run of the
same command measured conv2d_1 between 8.5 and 13.9 ms; the table is from a second run on an otherwise idle machine.

## Conv2D through Conv2DPlan (v0.3.0)

From v0.3.0 on, `Conv2D` runs a groups = 1 convolution through a `Conv2DPlan` built for the call (packed weights, a bias
copy and the im2col scratch, every image of the batch through it), where `BenchmarkConv2DRouting` showed a gain; the
rest, groups > 1 included, stays on the per-pixel path (`conv2DDirect` in the tests). go-whistle's stem runs in chunks
of 8 output frames (`stem.chunk` @ 98603c6), so `BenchmarkConv2DWhistleStem/chunk` adds one chunk away from the clip's
edges (71 × 80 × 1, 3 × 3 → 128, stride 2, padding left and right only).

```bash
go test -run '^$' -bench 'Conv2DBasicPitch|Conv2DWhistleStem|Conv2DPlanWhistleStem|Conv2DPlanBasicPitch' -benchmem -count=5 ./ops
```

Measured on 2026-10-12 on an Apple M5 Pro (darwin/arm64, NEON), Go 1.27.1, one worker, median of five runs; before is
v0.2.0 (b33d487) with the chunk benchmark added. js/wasm is the same test binary under Node 24.12.0
(`-test.benchtime=3x`, two runs), Go tiles (4 × 2):

| `Conv2D`                                 | arm64 before | arm64 after | wasm before | wasm after |
| ---------------------------------------- | -----------: | ----------: | ----------: | ---------: |
| conv2d_1 (172×264×8, 3×39 → 8)           |     192.0 ms |     7.74 ms |    499.4 ms |    58.3 ms |
| contours-reduced (172×264×8, 5×5 → 1)    |      29.2 ms |     1.56 ms |     39.7 ms |     4.3 ms |
| conv2d_2 (172×264×1, 7×7 → 32, (1, 3))   |      5.78 ms |     0.94 ms |     29.7 ms |     6.1 ms |
| conv2d_3 (172×88×32, 7×3 → 1)            |      30.2 ms |     1.12 ms |     40.7 ms |     3.4 ms |
| conv2d_4 (172×264×8, 5×5 → 32, (1, 3))   |      20.9 ms |     2.09 ms |    113.8 ms |    17.4 ms |
| conv2d_5 (172×88×33, 3×3 → 1)            |      13.5 ms |     0.53 ms |     18.0 ms |     1.6 ms |
| **basic-pitch window**                   | **291.5 ms** | **14.0 ms** |  **741 ms** |  **91 ms** |
| go-whistle stem (3000×80×1, 3×3 → 128/2) |      9.51 ms |     3.93 ms |     78.6 ms |    26.2 ms |
| go-whistle stem chunk (71×80×1)          |       235 µs |      116 µs |     1.74 ms |    0.59 ms |

`Conv2DPlan.Forward` itself did not change (7.57, 1.52, 0.89, 1.12, 2.03 and 0.52 ms; stem 3.59 ms, chunk 84 µs):
`Conv2D` adds the output allocation and zeroing, which is most of the stem's remaining gap (30.7 MB), and the plan's
set-up, 5 more allocations per call (12 for the basic-pitch layers, 11 for the stem). The window was 328.9 ms for
`Conv2D` in the 2026-10-11 table above; the before column is from this session. On amd64 `Conv2D` does not route (see
below), so its speed and bits there are v0.2.0's.

### Routing

`BenchmarkConv2DRouting` runs 90 groups = 1 shapes through the plan ("plan", packing included) and the per-pixel path
("direct"), one worker: 4 × 4, 8 × 8 and 16 × 16 outputs; GEMM depths k = kh·kw·inC of 1, 2, 3, 4 and 9 (1 × 1 kernels)
and 36 (3 × 3 on 4 channels); 1, 8, 32, 128 and 255 out channels; with bias.

```bash
go test -run '^$' -bench Conv2DRouting -benchtime=0.2s -count=3 ./ops
```

`Conv2D` routes a geometry when k ≥ 4, batch·outH·outW ≥ 64 and outC ≤ 16·k (`planGains` in `conv2d.go`), and not on
amd64. Time of the plan path over the direct path on 2026-10-12, Go 1.27.1, the M5 Pro as above (three runs; js/wasm
and amd64 two):

| configuration                 | 24 routed shapes | 66 other shapes | other shapes faster on the plan |
| ----------------------------- | ---------------: | --------------: | ------------------------------: |
| arm64/neon                    |        0.13–0.92 |       0.35–2.04 |                              30 |
| arm64/go (`-tags purego`)     |        0.26–0.83 |       0.37–1.62 |                              39 |
| wasm/go (Node 24.12.0)        |        0.23–0.71 |       0.36–1.54 |                              43 |
| amd64/avx2 (Docker, emulated) |        0.20–3.14 |       0.41–4.02 |                              16 |
| amd64/go (Docker, emulated)   |        0.40–1.77 |       0.75–2.45 |                               5 |

The thresholds follow the shapes that lost on arm64 or js/wasm, in this sweep and in a first, wider one (also 1 × 1 and
2 × 2 outputs, and 65 to 511 out channels on 1 × 1 kernels). On 16 output positions the plan lost up to 1.5×, on 1 or 4
several times over: packing the weights for the call does not pay off. With k < 4 and many out channels it lost up to
2.0×: the per-pixel NEON `Axpy` over the out channels against the Go GEMM tiles (the NEON GEMM needs k ≥ 4). With
outC > 16·k and out channels that are not 1 or a multiple of 4, where the NEON GEMM falls back to the Go tiles, k = 4
lost up to 39 % and k = 9 up to 11 %. The rules are conservative: they leave the gains of the other shapes in the last
column on the table.

The amd64 rows ran in `debian:bookworm-slim` under Docker Desktop's x86 emulation (Rosetta, "VirtualApple" CPU with
AVX2 and FMA) on the same Mac: not a real x86 measurement, and the only one available. There the plan lost on most
routed shapes, go-whistle's stem chunk included (2.0× with AVX2, 1.6× with purego in the first sweep): `MatMulTransBBias`
has only Go tiles on amd64 (PLAN 1.2 adds AVX2 + FMA), against the AVX2 `Axpy`. So `Conv2D` does not route on amd64
until 1.2's kernels are measured on real hardware.

## Linear: weight panels and an AVX2 kernel, same bits

`Linear` used to compute one dot product per output, token by token, so it streamed the whole weight matrix once per
token: for go-laya's [512, 1024] · [3072, 1024]ᵀ that is 6 GiB through a 12 MiB L3 per call, and a profile put 87% of
go-laya's native CPU time in it (go-laya Backlog B.20). It now runs every token over a 256 KiB panel of weight rows
before it moves on to the next panel, on every configuration; on amd64 with AVX2 + FMA, `linearDot3AVX2` runs three
tokens over one weight row at once (twelve ymm accumulators, one load of each weight chunk for three FMAs). Every output
is still the dot product `DotProduct` gives, plus the bias in float32, bit for bit: the kernel replays the AVX2
`DotProduct` per output (accumulators, fold, tails, reduction and FMA operand order), the leftover tokens and every
other configuration call `DotProduct` itself, and the split over workers (now `Parallel`, over outputs or tokens) puts
each output in one call. No golden hash changed. `TestLinearBitIdentical`, `TestLinearWorkers` and
`TestLinearDot3MatchesDotF32` compare against `DotProduct` with `math.Float32bits`.

`BenchmarkLinearLaya` runs go-laya's shapes, x [512, in] · weight [out, in]ᵀ without bias (ModernBERT's and mmBERT's;
1152 → 768 is mmBERT-base's MLP `Wo`), and one batch-1 product, on one worker and on the default count:

```bash
go test -run '^$' -bench BenchmarkLinearLaya -count=1 ./tensor   # five times, alternating before and after
```

Measured on 2026-10-12 on an Intel Core i7-1255U (linux/amd64, AVX2 + FMA; 2 P-cores with two threads each and 8
E-cores, GOMAXPROCS 12), Go 1.26.8. Before is `origin/main` (91fe50a) with the benchmark copied in; one test binary each,
five rounds alternating before and after, `benchstat` medians in ms with the range of the five runs. The machine was
shared: the 1-minute load average was 5.6–7.8 during the runs, and the scheduler moves a single worker between P- and
E-cores (pinned, the kernel took 45 ms on a P-core and 89 ms on an E-core for 1024 → 3072), so the ranges are wide.

| in → out (512 tokens) | 1 worker before  | 1 worker after   | speed-up | default before   | default after    | speed-up |
| --------------------- | ---------------- | ---------------- | -------: | ---------------- | ---------------- | -------: |
| 1024 → 3072           | 250.8 (185–341)  | 44.2 (42.2–105)  |     5.7× | 48.7 (38.6–61.1) | 13.1 (12.6–19.1) |     3.7× |
| 1024 → 5248           | 522.2 (476–533)  | 77.0 (69.8–78.8) |     6.8× | 89.7 (78.6–105)  | 25.6 (21.4–27.5) |     3.5× |
| 2624 → 1024           | 168.2 (130–179)  | 38.6 (34.9–42.1) |     4.4× | 35.7 (28.9–51.2) | 19.8 (16.8–23.4) |     1.8× |
| 1024 → 1024           | 37.4 (36.2–39.4) | 15.0 (13.6–29.9) |     2.5× | 11.5 (10.1–12.8) | 5.5 (4.9–7.0)    |     2.1× |
| 768 → 2304            | 71.3 (60.4–90.9) | 24.7 (23.7–24.8) |     2.9× | 21.3 (17.0–26.1) | 9.0 (8.4–9.8)    |     2.4× |
| 1152 → 768            | 29.4 (28.5–32.1) | 12.4 (11.4–15.0) |     2.4× | 10.3 (8.9–11.3)  | 4.8 (4.3–5.7)    |     2.2× |
| 768 → 768             | 19.6 (19.0–20.6) | 8.3 (7.4–14.3)   |     2.4× | 6.5 (6.1–7.0)    | 3.2 (2.9–4.9)    |     2.0× |
| 1024 → 3072, 1 token  | 0.48 (0.37–0.49) | 0.46 (0.36–1.02) |        – | 0.23 (0.17–0.24) | 0.17 (0.15–0.23) |        – |

A single token reads every weight once either way and gains nothing (benchstat: no significant change); the 512-token
products gain 2.4–6.8× on one worker, most where the weights are largest. On the default count they gain less
(1.8–3.7×), presumably because the cores share L3 and memory bandwidth; 2624 → 1024, the least, re-reads its 5.4 MiB of x
once per 24-row panel. A split over tokens instead of outputs measured the same within noise.

Two shapes were measured for the kernel, both bit-exact: three tokens against one weight row (the one kept) and one
token against three weight rows. On a pinned P-core (E-core) the latter took 60 (104) ms for 1024 → 3072 against 45 (89)
ms, and lost on 2624 → 1024, 1152 → 768 and 768 → 768 as well, except 2624 → 1024 on the E-core, a tie. The panel budget
(one run per budget on four shapes, pinned as above) was flat from 128 to 512 KiB on both kinds of core and lost 8–38 % at
32 KiB and up to 9 % at 4 MiB (capped at 512 rows); 256 KiB stays. More than three tokens per weight row do not fit in
16 ymm registers: replaying `DotProduct` takes four accumulators per output.

In go-laya (`BenchmarkForwardDefaultWorkers`, its Task 8.9 cell: one question, 512 tokens, 4 options, default workers,
p50 of three passes), this branch against v0.2.0 through a `go.work`, interleaved, two sweeps, 1-minute load average
7.3–11.6: `laya` 9583 → 5431 and 9831 → 5553 ms (1.8×), `laya-multilingual` 3693 → 2685 and 4490 → 2677 ms (1.4–1.7×),
`laya-typed-decisions` 10250 → 5563 and 10248 → 5620 ms (1.8×). Its pinned forward digests did not change.

## Optional acceleration points (correctness contract unchanged)

1. `MatMul` and `Linear`: SIMD/assembly kernels for inner-product loops. Done: `tensor.MatMulTransB` (NEON 4×4 blocks
   on arm64), `tensor.MatMulTransBBias` (NEON 4×4 and 8×1 tiles on arm64, Go tiles elsewhere), `DotProduct` and `Axpy`
   (NEON, AVX2+FMA), `Linear` (weight panels everywhere, a three-token AVX2 kernel on amd64, same bits).
2. `Softmax` + masking: fuse max/exp/sum/normalize pass and mask application.
3. `LayerNorm`: vectorized mean/variance reduction and affine fuse.
4. `Attention`: fuse `QK^T` scale + mask + softmax where memory reuse is possible.
5. `Conv1D`/`ConvTranspose1D`: im2col + GEMM or direct SIMD kernels for common kernel sizes. Done for groups=1: both
   run tiled through `tensor.MatMulTransB`. `Conv2DPlan` does the same for 2-D convolutions with groups=1, through
   `tensor.MatMulTransBBias`, and `Conv2D` runs its groups = 1 convolutions through it where that gains (not on amd64);
   the rest still runs one `Axpy` per input tap.
6. Parallelization: shard batch/head/channel loops with worker pools while keeping deterministic reduction order where
   parity requires it. `tensor.Parallel` is the shared, bounded pool; work split on it in multiples of the kernel's
   block gives the same bits for any worker count.

Any acceleration must preserve the tolerance targets in `tolerance.go` and, unless it comes with a numerics note, the
golden hashes in `golden_hashes_test.go`.

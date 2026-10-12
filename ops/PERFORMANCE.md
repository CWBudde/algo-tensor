# Runtime Kernel Performance Notes

<!-- From github.com/cwbudde/go-pocket-tts internal/runtime/ops/PERFORMANCE.md @ 9faba38; go-whistle dropped it.
algo-tensor change: the package path, go-whistle's benchmarks next to go-pocket-tts', and the state of the acceleration
points; the Conv2DPlan benchmarks; Conv2D through Conv2DPlan (v0.3.0) and its routing sweep; the amd64 AVX2 benchmarks
of MatMulTransBBias and Conv2DPlan. -->

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

### linux/amd64: AVX2 + FMA (PLAN 1.2)

`MatMulTransBBias` runs algo-transcribe nn's AVX2 + FMA kernels @ 58c8332 on amd64 CPUs with AVX2 and FMA (2 × 4 and
8 × 1 tiles for k ≥ 4 and n = 1 or a multiple of 4), the Go tiles elsewhere. The same benchmarks, with
`BenchmarkMatMulTransBBias` (one `MatMulTransBBias` call of each layer's GEMM, m = 8) and, for comparison, nn's
`BenchmarkConv2DWindow` and `BenchmarkGemm` @ 58c8332, as linux/amd64 test binaries with and without `-tags purego`,
alternating, three rounds, median:

```bash
GOOS=linux GOARCH=amd64 go test -c -o ops.test ./ops   # and -tags purego, and ./tensor
docker run --rm --platform linux/amd64 -v "$PWD:/t:ro" debian:bookworm-slim \
    /t/ops.test -test.run '^$' -test.bench 'Conv2DPlanBasicPitch|Conv2DBasicPitch|WhistleStem' -test.benchmem
```

**Not a real x86 measurement**: measured on 2026-10-12 in `debian:bookworm-slim` under Docker Desktop's x86 emulation
(Rosetta, "VirtualApple @ 2.50GHz" with AVX2 and FMA, 15 CPUs) on the Apple M5 Pro, Go 1.27.1, the only x86 at hand. It
shows the relative order of the paths, not x86 speed; nn measured its kernels on an Intel Xeon Gold 5520+ (algo-transcribe
`docs/benchmarks/2026-10-11-simd-gemm.md`: the basic-pitch window 19.0 ms with AVX2 against 115 ms with its 4 × 2 Go
tiles).

| layer                                    | `Conv2DPlan` AVX2 |     nn AVX2 | `Conv2DPlan` purego | nn purego (4 × 2) | `Conv2D` AVX2 | `Conv2D` purego |
| ---------------------------------------- | ----------------: | ----------: | ------------------: | ----------------: | ------------: | --------------: |
| conv2d_1 (172×264×8, 3×39 → 8)           |          22.03 ms |    21.90 ms |            137.9 ms |           63.7 ms |      358.4 ms |        187.2 ms |
| contours-reduced (172×264×8, 5×5 → 1)    |           2.21 ms |     2.15 ms |             5.35 ms |           5.16 ms |       43.8 ms |         26.6 ms |
| conv2d_2 (172×264×1, 7×7 → 32, (1, 3))   |           2.51 ms |     2.53 ms |            10.88 ms |           5.12 ms |       11.2 ms |          9.1 ms |
| conv2d_3 (172×88×32, 7×3 → 1)            |           1.67 ms |     1.63 ms |             5.19 ms |           5.03 ms |       45.3 ms |         29.8 ms |
| conv2d_4 (172×264×8, 5×5 → 32, (1, 3))   |           6.11 ms |     6.17 ms |             39.7 ms |           17.6 ms |       42.3 ms |         31.9 ms |
| conv2d_5 (172×88×33, 3×3 → 1)            |           0.86 ms |     0.83 ms |             2.28 ms |           2.26 ms |       20.1 ms |         13.1 ms |
| **basic-pitch window**                   |       **35.4 ms** | **35.2 ms** |        **201.3 ms** |       **98.9 ms** |  **521.2 ms** |    **297.7 ms** |
| go-whistle stem (3000×80×1, 3×3 → 128/2) |          14.54 ms |             |            35.48 ms |                   |      17.76 ms |        23.43 ms |
| go-whistle stem chunk (71×80×1)          |            336 µs |             |              822 µs |                   |        427 µs |          566 µs |

| `MatMulTransBBias` (m = 8)    |    AVX2 | Go tiles (purego) | AVX2 vs Go | nn `gemm` AVX2 | nn `gemm` purego (4 × 2) |
| ----------------------------- | ------: | ----------------: | ---------: | -------------: | -----------------------: |
| conv2d_1 (n 8, k 936)         | 3303 ns |          23408 ns |      7.1 × |        3155 ns |                 10295 ns |
| contours-reduced (n 1, k 200) |   90 ns |            613 ns |      6.8 × |          79 ns |                   587 ns |
| conv2d_2 (n 32, k 49)         | 1019 ns |           5336 ns |      5.2 × |        1001 ns |                  2358 ns |
| conv2d_3 (n 1, k 672)         |  246 ns |           2029 ns |      8.3 × |         239 ns |                  2004 ns |
| conv2d_4 (n 32, k 200)        | 2857 ns |          20765 ns |      7.3 × |        2866 ns |                  9071 ns |
| conv2d_5 (n 1, k 297)         |  132 ns |            919 ns |      6.9 × |         122 ns |                   869 ns |

- With AVX2 the plan runs the window 5.7 × faster than with its Go tiles and 15 × faster than `Conv2D` (which on amd64
  stays on the per-pixel path), within 1 % of nn @ 58c8332, whose bits it gives on these layers. Every layer takes the
  AVX2 kernels; none allocates.
- The multi-channel Go tiles on amd64 are the 4 × 4 tile, which spills on amd64's 16 vector registers: under purego
  conv2d_1, conv2d_2 and conv2d_4 take 2.1–2.3 × as long as nn's 4 × 2 tile, which nn @ 58c8332 adopted for amd64 (1.6–1.8
  × on its Xeon). algo-tensor keeps the 4 × 4 tile so that the amd64/go hashes do not change; the single-channel layers
  run the 8 × 1 tile on both sides and match. It matters on amd64 without AVX2 or FMA, under purego, and for shapes the
  AVX2 kernels decline.
- `MatMulTransBBias` takes up to 14 % longer than nn's bare `gemm` (most on the short single-channel GEMMs; conv2d_4
  the same): its argument checks and 8-row block loop around the same kernels.
- On the stem (k = 9, 128 out channels) the plan is now faster than `Conv2D` (14.5 against 17.8 ms; the chunk 336 against
  427 µs); with the Go tiles it was not (see Routing below). Under this emulation `Conv2D` runs faster with purego than
  with its AVX2 `Axpy`, one more sign that it does not predict x86.

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
had only Go tiles on amd64 then, against the AVX2 `Axpy`. So `Conv2D` does not route on amd64 until PLAN 1.2's kernels
are measured on real hardware.

With those kernels (PLAN 1.2, 2026-10-12, the same emulation, two runs) the plan path takes 0.15–3.12 of the direct
path's time on the 24 routed shapes and 0.33–4.09 on the 66 others, 21 of them faster. Of the routed shapes it now loses
on three: 255 out channels at k = 36 (2.8 and 3.1 ×), which the AVX2 kernels decline (not a multiple of 4) and the 4 × 4
Go tiles run, and 8 × 8 outputs at k = 9 and 128 out channels (1.06 ×). Routing on amd64 stays off: it needs a real x86
measurement and is a numerics change of its own (`Conv2D`'s amd64 bits would move).

## Optional acceleration points (correctness contract unchanged)

1. `MatMul` and `Linear`: SIMD/assembly kernels for inner-product loops. Done: `tensor.MatMulTransB` (NEON 4×4 blocks
   on arm64), `tensor.MatMulTransBBias` (NEON 4×4 and 8×1 tiles on arm64, AVX2+FMA 2×4 and 8×1 tiles on amd64, Go tiles
   elsewhere), `DotProduct` and `Axpy` (NEON, AVX2+FMA).
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

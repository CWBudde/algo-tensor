# Runtime Kernel Performance Notes

<!-- From github.com/cwbudde/go-pocket-tts internal/runtime/ops/PERFORMANCE.md @ 9faba38; go-whistle dropped it.
algo-tensor change: the package path, go-whistle's benchmarks next to go-pocket-tts', and the state of the acceleration
points; the Conv2DPlan benchmarks; Conv2D through Conv2DPlan (v0.3.0) and its routing sweep; the amd64 AVX2 benchmarks
of MatMulTransBBias and Conv2DPlan; Linear's weight panels and amd64 kernel. -->

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
# algo-tensor: ops and tensor, each with AVX2 and with the Go tiles (purego)
GOOS=linux GOARCH=amd64 go test -c -o ops.test ./ops
GOOS=linux GOARCH=amd64 go test -c -tags purego -o ops-go.test ./ops
GOOS=linux GOARCH=amd64 go test -c -o tensor.test ./tensor
GOOS=linux GOARCH=amd64 go test -c -tags purego -o tensor-go.test ./tensor
# nn @ 58c8332, in an algo-transcribe checkout; copy the binaries next to the ones above
GOOS=linux GOARCH=amd64 go test -c -o nn.test ./nn
GOOS=linux GOARCH=amd64 go test -c -tags purego -o nn-go.test ./nn

run() { docker run --rm --platform linux/amd64 -v "$PWD:/t:ro" debian:bookworm-slim "$@"; }
for v in "" -go; do  # alternate the variants, three rounds
    run /t/ops$v.test -test.run '^$' -test.bench 'Conv2DPlanBasicPitch|Conv2DBasicPitch|WhistleStem' -test.benchmem
    run /t/tensor$v.test -test.run '^$' -test.bench 'MatMulTransBBias' -test.benchmem
    run /t/nn$v.test -test.run '^$' -test.bench 'Conv2DWindow|Gemm' -test.benchmem
done
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

#### On a real x86: Intel Xeon Gold 5520+

The same test binaries (this PR's head, 87d9a2e), built with Go 1.25.0 on the Xeon that nn measured on, a shared VM with 80
vCPUs, pinned with `taskset -c 77`, minimum of five runs, 2026-10-12. The VM ran a load average of 130 to 230 during
the runs, so absolute times are only good to about ±40 % (nn's window took 16.6 ms there the day before and 23.2 ms
now); the comparisons within one column and the interleaved nn comparison below are the reliable part. All tensor
and ops tests pass there with `CI=true`, with and without purego and under `-race`, with no AVX2 test skipped; the
amd64/avx2 golden hashes are the ones Rosetta gave.

| layer                  | `Conv2DPlan` AVX2 | `Conv2DPlan` purego | `Conv2D` AVX2 | `Conv2D` purego |
| ---------------------- | ----------------: | ------------------: | ------------: | --------------: |
| conv2d_1               |          12.35 ms |            163.1 ms |      273.6 ms |        292.3 ms |
| contours-reduced       |           2.74 ms |             6.10 ms |       51.8 ms |         32.9 ms |
| conv2d_2               |           2.05 ms |             13.5 ms |        9.0 ms |         18.0 ms |
| conv2d_3               |           2.01 ms |             5.39 ms |       52.2 ms |         32.7 ms |
| conv2d_4               |           3.94 ms |             51.1 ms |       28.9 ms |         47.5 ms |
| conv2d_5               |           1.01 ms |             2.88 ms |       24.7 ms |         13.6 ms |
| **basic-pitch window** |       **24.1 ms** |        **242.0 ms** |  **440.2 ms** |    **437.0 ms** |
| go-whistle stem        |           9.13 ms |             54.8 ms |       16.0 ms |         39.9 ms |
| go-whistle stem chunk  |            187 µs |             1039 µs |        301 µs |          920 µs |

- Interleaved with nn @ 58c8332's `BenchmarkConv2DWindow` (AVX2, five alternating rounds on the same CPU), the plan
  takes 23.8 ms for the six layers against nn's 23.2 ms, 2.4 % longer, as on arm64.
- `MatMulTransBBias` with AVX2 runs 11–21 × as fast as the Go tiles (m = 8: 1560 against 30689 ns for n 8, k 936;
  1310 against 27000 ns for n 32, k 200; 63 against 724 ns for n 1, k 200; m 256, n 64, k 576: 0.25 against 4.9 ms).
- With AVX2 the plan beats the per-pixel `Conv2D` on every layer here, the stem by 1.75 × and its chunk by 1.6 ×. With
  the 4 × 4 Go tiles it loses on the stem (1.4 ×) and conv2d_4, as under emulation.

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

On the real Xeon Gold 5520+ (setup as in the amd64 section above, three runs, minimum per path) the sweep agrees with
the emulation. With AVX2 the plan takes 0.15–4.43 of the direct path's time on the 24 routed shapes and 0.34–4.71 on
the 66 others, 15 of them faster; of the routed shapes it loses only on the two with 255 out channels at k = 36 (4.4
and 4.1 ×), which the AVX2 kernels decline. With purego's 4 × 4 Go tiles it loses on 15 of the 24 routed shapes (up to
1.65 ×). Routing on amd64 with AVX2 and out channels of 1 or a multiple of 4 would therefore gain; it stays a numerics
change for a later item.

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

## MatMulTransB on amd64: Linear's kernel, same bits (PLAN 1.4)

On amd64, `tensor.MatMulTransB` was one `DotProduct` per output. It now runs `Linear`'s loop (`gemm_amd64.go`): the
rows of A take the place of the tokens and the rows of B of the weight rows, over 256 KiB panels of B, with three rows
of A against each row of B in `linearDot3AVX2` where `DotProduct` would run its AVX2 loop (k ≥ 8, AVX2 + FMA). So every
output keeps the bits of one `DotProduct`, on every configuration and for any worker count; no golden hash changed.
Fewer than three rows of A (a GEMV), k < 8, purego and machines without AVX2 + FMA keep one `DotProduct` per output.
`TestMatMulTransBBitIdentical` and `TestMatMulTransBWorkersBits` compare against `DotProduct` with `math.Float32bits`
(special values, misaligned rows, ldc > n, 1 to GOMAXPROCS workers).

`BenchmarkMatMulTransB` runs go-pocket-tts' shapes (FlowLM at d_model 1024 per frame and over a 64-token prompt, a
Mimi conv as `ops.Conv1D` runs it, a Mimi `in_proj` per frame) and go-whistle's (needle's projections over a 30 s clip,
one attention score and value tile of 64 queries); `m × n × k` is A [m, k] · B [n, k]ᵀ:

```bash
GOOS=linux GOARCH=amd64 go test -c -o after.test ./tensor   # and before.test from origin/main with the same benchmark
taskset -c 77 ./after.test -test.run '^$' -test.bench 'BenchmarkMatMulTransB/.*/kernel' -test.count=1
```

Measured on 2026-10-12 on the Intel Xeon Gold 5520+ of PLAN 1.2 (a shared VM with 80 vCPUs), one worker pinned with
`taskset -c 77`, binaries built with Go 1.27.1. Before is `origin/main` (f0f8835) with the benchmark copied in; five
rounds alternating before and after, `benchstat` medians in ms with the range of the five runs. The VM ran a 1-minute
load average of 320 to 1200 during the runs, so the ranges are wide; the speed-ups below are the ones benchstat calls
significant (p = 0.008).

| m × n × k                               | before              | after               | speed-up |
| --------------------------------------- | ------------------- | ------------------- | -------: |
| pocket FlowLM FFN, 64 × 4096 × 1024     | 109.1 (85.1–185)    | 16.3 (12.4–18.2)    |     6.7× |
| pocket FlowLM in_proj, 64 × 3072 × 1024 | 33.9 (32.0–74.0)    | 11.4 (10.1–15.3)    |     3.0× |
| whistle FFN, 375 × 2048 × 512           | 52.2 (52.0–58.2)    | 18.1 (14.5–21.3)    |     2.9× |
| 16 × 1024 × 1024                        | 2.27 (2.09–2.31)    | 0.880 (0.714–0.896) |     2.6× |
| conv tile, 64 × 56 × 576                | 0.148 (0.129–0.170) | 0.078 (0.071–0.085) |     1.9× |
| whistle scores, 64 × 375 × 48           | 0.238 (0.166–0.279) | 0.135 (0.106–0.148) |     1.8× |
| whistle linear, 375 × 512 × 512         | 8.07 (6.72–8.96)    | 4.67 (3.50–5.73)    |     1.7× |
| whistle values, 64 × 64 × 375           | 0.167 (0.153–0.178) | 0.104 (0.085–0.115) |     1.6× |
| pocket Mimi conv, 512 × 32 × 3584       | 5.02 (4.08–5.27)    | 4.24 (3.73–5.39)    |        – |
| pocket FlowLM FFN, 1 × 4096 × 1024      | 1.34 (0.92–2.33)    | 2.22 (1.08–2.99)    |        – |
| pocket Mimi in_proj, 1 × 1536 × 512     | 0.106 (0.095–0.106) | 0.103 (0.098–0.109) |        – |
| 1 × 1024 × 1024                         | 0.134 (0.131–0.152) | 0.144 (0.142–0.152) |        – |

The GEMVs run the same code before and after; ten more alternating rounds of the two small ones gave medians of 0.138
against 0.137 ms (1 × 1024 × 1024) and 0.095 against 0.096 ms (1 × 1536 × 512). A kernel that runs one row of A against
three rows of B (bit-exact as well, measured for `Linear` in #10) was not written: 1 × 1024 × 1024 streams its 4 MiB of
B at about 30 GB/s, about six times the time its 2²⁰ FMAs need at two 8-wide FMAs per cycle, so it is bound by memory,
not by the arithmetic a kernel could save. go-pocket-tts' per-frame FlowLM products are such GEMVs, so on amd64 only
its prompt, its voice conditioning and Mimi gain. The Mimi conv tile (A is the 7 MiB of kernel rows, B the 458 KiB of
patches) gained nothing significant: with panels of 18 rows of B, A is read twice. Default worker counts were not
measured: at that load a multi-threaded time on the shared VM says little.

## Optional acceleration points (correctness contract unchanged)

1. `MatMul` and `Linear`: SIMD/assembly kernels for inner-product loops. Done: `tensor.MatMulTransB` (NEON 4×4 blocks
   on arm64), `tensor.MatMulTransBBias` (NEON 4×4 and 8×1 tiles on arm64, AVX2+FMA 2×4 and 8×1 tiles on amd64, Go tiles
   elsewhere), `DotProduct` and `Axpy` (NEON, AVX2+FMA), `Linear` (weight panels everywhere, a three-token AVX2 kernel
   on amd64, same bits).
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

# Runtime Kernel Performance Notes

<!-- From github.com/cwbudde/go-pocket-tts internal/runtime/ops/PERFORMANCE.md @ 9faba38; go-whistle dropped it.
algo-tensor change: the package path, go-whistle's benchmarks next to go-pocket-tts', and the state of the acceleration
points; the Conv2DPlan benchmarks. -->

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

## Optional acceleration points (correctness contract unchanged)

1. `MatMul` and `Linear`: SIMD/assembly kernels for inner-product loops. Done: `tensor.MatMulTransB` (NEON 4×4 blocks
   on arm64), `tensor.MatMulTransBBias` (NEON 4×4 and 8×1 tiles on arm64, Go tiles elsewhere), `DotProduct` and `Axpy`
   (NEON, AVX2+FMA).
2. `Softmax` + masking: fuse max/exp/sum/normalize pass and mask application.
3. `LayerNorm`: vectorized mean/variance reduction and affine fuse.
4. `Attention`: fuse `QK^T` scale + mask + softmax where memory reuse is possible.
5. `Conv1D`/`ConvTranspose1D`: im2col + GEMM or direct SIMD kernels for common kernel sizes. Done for groups=1: both
   run tiled through `tensor.MatMulTransB`. `Conv2DPlan` does the same for 2-D convolutions with groups=1, through
   `tensor.MatMulTransBBias`; `Conv2D` itself still runs one `Axpy` per input tap.
6. Parallelization: shard batch/head/channel loops with worker pools while keeping deterministic reduction order where
   parity requires it. `tensor.Parallel` is the shared, bounded pool; work split on it in multiples of the kernel's
   block gives the same bits for any worker count.

Any acceleration must preserve the tolerance targets in `tolerance.go` and, unless it comes with a numerics note, the
golden hashes in `golden_hashes_test.go`.

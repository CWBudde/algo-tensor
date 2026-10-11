# algo-tensor

Pure-Go tensor kernels and neural-network ops, shared by [go-whistle](https://github.com/CWBudde/go-whistle) (speech to
text) and [go-pocket-tts](https://github.com/CWBudde/go-pocket-tts) (text to speech). No cgo; the hot kernels have NEON
and AVX2 assembly with pure-Go fallbacks.

```go
import (
	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
)
```

## What is in it

**`tensor`**: a dense, row-major float32 `Tensor` (`New`, `Zeros`, `Full`, `Reshape`, `Transpose`, `Narrow`, `Gather`,
`Concat`, broadcasting `BroadcastAdd`/`BroadcastMul`), the layers `Linear`, `MatMul`, `Softmax` and `LayerNorm`, and
the slice kernels (AVX2 + FMA is checked at run time; shorter slices and tails run the Go loops):

| kernel                                 | arm64                        | amd64                       | elsewhere, or `-tags purego` |
| -------------------------------------- | ---------------------------- | --------------------------- | ---------------------------- |
| `MatMulTransB` (C = A · Bᵀ, row-major) | NEON, register-blocked 4 × 4 | one `DotProduct` per output | Go                           |
| `DotProduct`                           | NEON from 48 values          | AVX2 + FMA from 8 values    | Go, four accumulators        |
| `Axpy`                                 | NEON from 8 values           | AVX2 + FMA from 8 values    | Go                           |

**`ops`**: scaled dot-product attention (`Attention` with a causal mask and offset, `CausalMask`, and
`AttentionWithPositions` with absolute positions, padding keys and a context window), interleaved `RoPE` from cos/sin
tables, `Conv1D`/`Conv1DLeftPad` (im2col through `MatMulTransB` for groups = 1, a direct loop for grouped and depthwise
convolutions), `ConvTranspose1D` with right trim and pre-packed kernels, `Conv2D` in NHWC/HWIO with groups and
`SamePadding`, `MLP`, and the per-kernel parity tolerances `KernelTolerance`.

## Consumers

The code was extracted from both projects, which switch to the module by changing import paths only;
[MIGRATION.md](MIGRATION.md) lists every identifier each project uses and the commands.

- **go-whistle** used `internal/tensor` and `internal/ops` (main @ 8926ee3). It added the `purego` tag, the worker pool
  and `Conv2D`, and dropped what it did not need.
- **go-pocket-tts** used `internal/runtime/tensor` and `internal/runtime/ops` (@ 9faba38), the original code, with
  `LayerNorm`, `ConvTranspose1D`, `MLP` and the position attention that go-whistle dropped.

Every file names its source in its header. A golden test per package (`golden_test.go`) checks that each kernel gives
the same bits as in the project it was taken from. The F-numbers in comments refer to go-whistle's `docs/FINDINGS.md`.

## Worker pool

`tensor.SetWorkers(n)`, `tensor.Workers()` and `tensor.Parallel(n, grain, fn)` are one bounded pool for the whole
process. The default is GOMAXPROCS at start-up; `SetWorkers(1)` makes every kernel single-threaded. A helper goroutine
starts only while fewer than `Workers() − 1` run, counted over all callers, so nested calls (a GEMM inside a parallel
loop) and concurrent ones (several decoder beams) never add more than `Workers() − 1` goroutines together.

`Parallel` cuts [0, n) into pieces that start at multiples of `grain`; the pieces depend on n, grain and `Workers()`
only. `MatMulTransB` splits products of at least 2²⁰ multiply-adds over it in whole 4-row blocks, so its output is the
same bit for bit for any worker count, and so are `Linear` and `MatMul`. `ops.SetConvWorkers(n)` separately sets the
goroutines of the convolutions (default sequential); `ConvTranspose1D` on arm64 gives other bits with conv workers than
without (as in go-pocket-tts).

## Build tags

- `purego` forces the pure-Go fallbacks on every architecture, for comparison and debugging. The tests pass with and
  without it; `tensor/asm_test.go` compares every assembly kernel with its fallback.
- Without assembly (`js/wasm`, `386`, `riscv64`, …) the fallbacks are used automatically. `GOOS=js GOARCH=wasm go
build ./...` is part of CI.
- amd64 picks AVX2 + FMA at run time from `golang.org/x/sys/cpu`, the only dependency.

## Checks

```bash
just test          # go test -v ./...
just test-race     # go test -race ./...
just test-purego   # go test -tags purego ./...
just build-wasm    # GOOS=js GOARCH=wasm go build ./...
just test-wasm     # the tests as js/wasm under Node
just test-amd64    # the linux/amd64 tests (AVX2 and purego) in Docker, for arm64 hosts
just lint          # golangci-lint (.golangci.yml)
just fmt           # treefmt (gofumpt, gci, prettier, shfmt)
just ci            # check-formatted + test + lint + check-tidy
just bench         # all benchmarks
```

CI runs `just ci`, `just test-purego` and `just build-wasm` on Linux (amd64).

## License

MIT, see [LICENSE](LICENSE).

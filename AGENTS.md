# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

Pure-Go tensor kernels (`tensor`) and neural-network operators (`ops`), extracted from
[go-whistle](https://github.com/CWBudde/go-whistle) and [go-pocket-tts](https://github.com/CWBudde/go-pocket-tts), which
both depend on them. See `README.md` for the contents and `MIGRATION.md` for what each consumer uses.

**Module:** `github.com/cwbudde/algo-tensor`
**Go version:** 1.25+

## Ground Rules

- **Pure Go.** No cgo. The only dependency is `golang.org/x/sys/cpu`; add none without a reason in the PR.
- **Assembly only with a fallback and an equivalence test.** Every Plan 9 assembly routine has a pure-Go fallback
  (`*_noasm.go`, built with `(!amd64 && !arm64) || purego`, or a run-time CPU check) and a test that compares both
  (`tensor/asm_test.go`). The tests must pass with and without `-tags purego`, under `-race`, and on linux/amd64
  (`just test-amd64` from an arm64 host); `GOOS=js GOARCH=wasm go build ./...` must succeed.
- **Numerics changes need a note.** Both consumers check their outputs against fixtures, so a change that moves any bit
  of a result is a numerics change: it fails `TestGoldenHashes` (`golden_test.go` in both packages). Update the hashes
  in `golden_hashes_test.go` in the same commit, and say in the commit message and the PR which kernels change, on
  which configurations, and why. Release it as a new minor version. Refactors and speed-ups must keep the hashes; a
  parallel split must keep the bits for any worker count (`TestMatMulTransBWorkers`, `TestParallelPieces`).
- **Tolerances are pinned.** `ops.KernelTolerances` changes only with a note, like a numerics change
  (`TestKernelTolerancesPinned`).
- **Provenance.** Every file says where it came from in its header (`From github.com/cwbudde/go-whistle … @ 8926ee3,
originally github.com/cwbudde/go-pocket-tts … @ 9faba38`) followed by `go-whistle change:` and `algo-tensor change:`
  lines. Add an `algo-tensor change:` line when you change a copied file. F-numbers (F-008, F-038, …), `docs/FINDINGS.md`
  and `PLAN.md` in comments refer to go-whistle's; this repository's own [PLAN.md](PLAN.md) lists its open work.
- **No breaking API changes without a major version.** Both consumers switch by import path only; renaming or changing
  an exported identifier breaks them.

## Web demo

The browser demo (<https://cwbudde.github.io/algo-tensor/>, deployed by `.github/workflows/pages.yml`) has four demos,
keyed `conv2d`, `attention`, `bench` and `signals`. Each demo lives in its own files:

| file                                      | what                                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| `cmd/demo/<key>_js.go`                    | js/wasm glue: an `init` that calls `register("<key>", …)`, argument parsing   |
| `internal/demo/<key>.go`, `<key>_test.go` | the computation, host-tested and linted; identifiers start with the demo name |
| `web/src/wasm/<key>.ts`                   | TypeScript types and typed wrappers; adds the demo to `AlgoTensorDemos`       |
| `web/src/demos/<Name>.tsx`                | the tab's React component (`Conv2D`, `Attention`, `Benchmark`, `Signals`)     |
| `web/src/wasm/bench.worker.ts`            | the bench demo's module worker                                                |

- A demo PR touches only its own files. The shared files (`cmd/demo/main_js.go`, `bridge_js.go`, `doc.go`,
  `main_other.go`; `internal/demo/convert.go`, `image.go`; `web/src/wasm/loader.ts`, `web/src/App.tsx`, the
  `web/src/components/ui/` components, the configs) change only in a PR of their own.
- `web/package.json` and `web/package-lock.json` change only with a reason in the PR; the scaffold installed every
  dependency the demos need (shadcn/ui components, recharts).
- Go functions return `{error: "…"}` instead of throwing; the wrappers pass every result through `unwrap` from
  `loader.ts`, which throws it. `cmd/demo/bridge_js.go` has the argument and typed-array helpers.
- The `_js.go` files build for `GOOS=js` only: `just lint` skips them, `just lint-wasm` lints them. `web/go.mod` keeps
  `web/` (and the Go files in `node_modules`) out of the root module's `./...`.
- New files under `cmd/demo`, `internal/demo` and `web/` are algo-tensor's own and have no provenance header.

## Commands

```bash
just test              # go test -v ./...
just test-race         # go test -race ./...
just test-purego       # go test -tags purego ./...
just build-wasm        # GOOS=js GOARCH=wasm go build ./...
just test-amd64        # linux/amd64 tests (AVX2, purego) in Docker
just lint              # golangci-lint (config in .golangci.yml)
just fmt               # treefmt (gofumpt, gci, prettier, shfmt)
just ci                # check-formatted + test + lint + check-tidy
just dev-demo          # web demo: build demo.wasm, start Vite (http://localhost:5173/algo-tensor/)
just build-demo        # web demo: production build into web/dist
just lint-wasm         # golangci-lint on cmd/demo for js/wasm
just lint-web          # ESLint + tsc -b on web/

go test -v -run TestName ./tensor/...
```

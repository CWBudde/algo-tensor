set shell := ["bash", "-uc"]

# Default recipe - show available commands
default:
    @just --list

# Format all code using treefmt
fmt:
    treefmt --allow-missing-formatter

# Check if code is formatted correctly
check-formatted:
    treefmt --allow-missing-formatter --fail-on-change

# Run linters
lint:
    golangci-lint run --timeout=5m ./...

# Run linters with auto-fix
lint-fix:
    golangci-lint run --fix --timeout=5m ./...

# Ensure go.mod is tidy
check-tidy:
    go mod tidy
    git diff --exit-code go.mod go.sum

# Run all tests
test:
    go test -v ./...

# Run tests with race detector
test-race:
    go test -race ./...

# Run all tests on the pure-Go fallbacks (no assembly)
test-purego:
    go test -tags purego ./...

# Build for js/wasm (pure-Go fallbacks)
build-wasm:
    GOOS=js GOARCH=wasm go build ./...

# Run the tests as js/wasm under Node; the environment is cleared because wasm_exec limits its size
test-wasm:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$(mktemp -d)
    trap 'rm -rf "$dir"' EXIT
    exec_wasm="$(go env GOROOT)/lib/wasm/go_js_wasm_exec"
    for pkg in tensor ops; do
        GOOS=js GOARCH=wasm go test -c -o "$dir/$pkg.test" "./$pkg"
        (cd "$pkg" && env -i PATH="$PATH" HOME="$HOME" "$exec_wasm" "$dir/$pkg.test" -test.count=1)
    done

# Build the browser demo: cmd/demo as js/wasm into web/public, then the Vite app into web/dist
build-demo: _demo-wasm
    npm --prefix web ci
    npm --prefix web run build

# Serve the browser demo with Vite's dev server (http://localhost:5173/algo-tensor/)
dev-demo: _demo-wasm
    npm --prefix web run dev

# Lint the js/wasm-only demo glue in cmd/demo
lint-wasm:
    GOOS=js GOARCH=wasm golangci-lint run --timeout=5m ./cmd/demo/...

# Lint and type-check the demo front end (needs web/node_modules, e.g. from just build-demo)
lint-web:
    npm --prefix web run lint
    npm --prefix web run typecheck

# Build cmd/demo as web/public/demo.wasm and copy the Go toolchain's wasm_exec.js next to it
_demo-wasm:
    GOOS=js GOARCH=wasm go build -trimpath -ldflags='-s -w' -o web/public/demo.wasm ./cmd/demo
    cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" web/public/

# Run the linux/amd64 tests (AVX2 assembly, then purego) in Docker, for arm64 hosts without Rosetta
test-amd64:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$(mktemp -d)
    trap 'rm -rf "$dir"' EXIT
    for pkg in tensor ops; do
        GOOS=linux GOARCH=amd64 go test -c -o "$dir/$pkg.test" "./$pkg"
        GOOS=linux GOARCH=amd64 go test -c -tags purego -o "$dir/$pkg-purego.test" "./$pkg"
    done
    docker run --rm --platform linux/amd64 -v "$dir:/t:ro" debian:bookworm-slim sh -c '
        grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo && echo "cpu: avx2 fma" || echo "cpu: no avx2+fma"
        for t in /t/tensor.test /t/ops.test /t/tensor-purego.test /t/ops-purego.test; do
            echo "== ${t#/t/}"; "$t" -test.count=1 || exit 1
        done'

# Run the benchmarks
bench *args:
    go test -run '^$' -bench . -benchmem {{args}} ./...

# Run tests with coverage report
test-coverage:
    go test -v -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Run all checks (formatting, linting, tests, tidiness)
ci: check-formatted test lint check-tidy

# Clean build artifacts
clean:
    rm -rf bin coverage.out coverage.html web/dist web/public/demo.wasm web/public/wasm_exec.js

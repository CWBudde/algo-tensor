//go:build !js

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "cmd/demo runs in the browser only; build it with: just build-demo")
	os.Exit(1)
}

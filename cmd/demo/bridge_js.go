package main

import (
	"errors"
	"fmt"
	"syscall/js"

	"github.com/cwbudde/algo-tensor/internal/demo"
)

// Errors reach JavaScript as a returned object {error: "<message>"}, never as
// a thrown exception (syscall/js cannot throw): every function either returns
// its result or that object. The typed wrappers in web/src/wasm/<key>.ts pass
// each result through unwrap (web/src/wasm/loader.ts), which throws an Error
// with the message. A panic inside a function is recovered and returned the
// same way, so a bad call does not end the Go program.

// export wraps fn as a JavaScript function; name ("<key>.<name>") prefixes
// its error messages.
func export(name string, fn Func) js.Func {
	return js.FuncOf(func(_ js.Value, args []js.Value) (result any) {
		defer func() {
			if r := recover(); r != nil {
				result = errorValue(fmt.Errorf("%s: panic: %v", name, r))
			}
		}()

		v, err := fn(args)
		if err != nil {
			return errorValue(fmt.Errorf("%s: %w", name, err))
		}

		// Convert here, inside the recover: js.ValueOf panics on types it
		// does not know, such as []float32.
		return js.ValueOf(v)
	})
}

// errorValue is the {error: "<message>"} object a failing function returns.
func errorValue(err error) js.Value {
	return js.ValueOf(map[string]any{"error": err.Error()})
}

var errMissing = errors.New("missing")

// arg returns argument i, named name in error messages.
func arg(args []js.Value, i int, name string) (js.Value, error) {
	if i >= len(args) || args[i].IsUndefined() {
		return js.Undefined(), fmt.Errorf("argument %d (%s): %w", i, name, errMissing)
	}

	return args[i], nil
}

// argFloat returns argument i as a number.
func argFloat(args []js.Value, i int, name string) (float64, error) {
	v, err := arg(args, i, name)
	if err != nil {
		return 0, err
	}

	if v.Type() != js.TypeNumber {
		return 0, fmt.Errorf("argument %d (%s): want a number, got %s", i, name, v.Type())
	}

	return v.Float(), nil
}

// argInt returns argument i as an int; it must be a whole number.
func argInt(args []js.Value, i int, name string) (int, error) {
	f, err := argFloat(args, i, name)
	if err != nil {
		return 0, err
	}

	n, ok := demo.ExactInt(f)
	if !ok {
		return 0, fmt.Errorf("argument %d (%s): want an integer, got %v", i, name, f)
	}

	return n, nil
}

// argBool returns argument i as a bool.
func argBool(args []js.Value, i int, name string) (bool, error) {
	v, err := arg(args, i, name)
	if err != nil {
		return false, err
	}

	if v.Type() != js.TypeBoolean {
		return false, fmt.Errorf("argument %d (%s): want a boolean, got %s", i, name, v.Type())
	}

	return v.Bool(), nil
}

// argString returns argument i as a string.
func argString(args []js.Value, i int, name string) (string, error) {
	v, err := arg(args, i, name)
	if err != nil {
		return "", err
	}

	if v.Type() != js.TypeString {
		return "", fmt.Errorf("argument %d (%s): want a string, got %s", i, name, v.Type())
	}

	return v.String(), nil
}

// argBytes copies argument i, a Uint8Array or Uint8ClampedArray (such as
// ImageData.data), to Go.
func argBytes(args []js.Value, i int, name string) ([]byte, error) {
	v, err := arg(args, i, name)
	if err != nil {
		return nil, err
	}

	if !v.InstanceOf(js.Global().Get("Uint8Array")) && !v.InstanceOf(js.Global().Get("Uint8ClampedArray")) {
		return nil, fmt.Errorf("argument %d (%s): want a Uint8Array or Uint8ClampedArray", i, name)
	}

	b := make([]byte, v.Get("length").Int())
	js.CopyBytesToGo(b, v)

	return b, nil
}

// argFloat32s copies argument i, a Float32Array, to Go through a byte view
// of its buffer.
func argFloat32s(args []js.Value, i int, name string) ([]float32, error) {
	v, err := arg(args, i, name)
	if err != nil {
		return nil, err
	}

	if !v.InstanceOf(js.Global().Get("Float32Array")) {
		return nil, fmt.Errorf("argument %d (%s): want a Float32Array", i, name)
	}

	view := js.Global().Get("Uint8Array").New(v.Get("buffer"), v.Get("byteOffset"), v.Get("byteLength"))
	b := make([]byte, view.Get("length").Int())
	js.CopyBytesToGo(b, view)

	return demo.Float32sFromBytes(b)
}

// bytesToJS copies b to a new Uint8ClampedArray, ready for new ImageData.
func bytesToJS(b []byte) js.Value {
	out := js.Global().Get("Uint8ClampedArray").New(len(b))
	js.CopyBytesToJS(out, b)

	return out
}

// float32sToJS copies values to a new Float32Array.
func float32sToJS(values []float32) js.Value {
	b := demo.BytesFromFloat32s(values)
	view := js.Global().Get("Uint8Array").New(len(b))
	js.CopyBytesToJS(view, b)

	return js.Global().Get("Float32Array").New(view.Get("buffer"))
}

// stringsToJS converts s to a JavaScript array of strings.
func stringsToJS(s []string) js.Value {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}

	return js.ValueOf(out)
}

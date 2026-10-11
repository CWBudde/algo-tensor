// Types and typed wrappers for the signals demo's Go functions
// (cmd/demo/signals_js.go).

import { loadAlgoTensor, unwrap, type WasmResult } from "./loader";

/** globalThis.algoTensor.signals */
export interface SignalsApi {
  /** A centered moving average of taps values through ops.Conv1D. */
  movingAverage(
    signal: Float32Array,
    taps: number,
  ): WasmResult<Float32Array<ArrayBuffer>>;
}

declare module "./loader" {
  interface AlgoTensorDemos {
    signals: SignalsApi;
  }
}

/**
 * Smooths a signal with a centered moving average of taps values (odd, at
 * most the signal's length) through ops.Conv1D.
 */
export async function signalsMovingAverage(
  signal: Float32Array,
  taps: number,
): Promise<Float32Array<ArrayBuffer>> {
  const at = await loadAlgoTensor();
  return unwrap(at.signals.movingAverage(signal, taps));
}

// Types and typed wrappers for the attention demo's Go functions
// (cmd/demo/attention_js.go).

import { loadAlgoTensor, unwrap, type WasmResult } from "./loader";

/** globalThis.algoTensor.attention */
export interface AttentionApi {
  /** The n × n attention weights of a toy sequence, row by row. */
  weights(n: number, causal: boolean): WasmResult<Float32Array<ArrayBuffer>>;
}

declare module "./loader" {
  interface AlgoTensorDemos {
    attention: AttentionApi;
  }
}

/**
 * The n × n weights softmax(q·kᵀ / √d) of a toy sequence of n positions, row
 * by row, with ops.CausalMask if causal.
 */
export async function attentionWeights(
  n: number,
  causal: boolean,
): Promise<Float32Array<ArrayBuffer>> {
  const at = await loadAlgoTensor();
  return unwrap(at.attention.weights(n, causal));
}

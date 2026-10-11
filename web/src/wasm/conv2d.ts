// Types and typed wrappers for the conv2d demo's Go functions
// (cmd/demo/conv2d_js.go).

import { loadAlgoTensor, unwrap, type WasmResult } from "./loader";

/** The 3 × 3 filters of the stub. */
export type Conv2DFilterName = "identity" | "box";

/** globalThis.algoTensor.conv2d */
export interface Conv2DApi {
  /** RGBA pixels filtered with a 3 × 3 depthwise ops.Conv2D. */
  filter(
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
    filter: Conv2DFilterName,
  ): WasmResult<Uint8ClampedArray<ArrayBuffer>>;
}

declare module "./loader" {
  interface AlgoTensorDemos {
    conv2d: Conv2DApi;
  }
}

/** Filters an image with a 3 × 3 depthwise ops.Conv2D in Go. */
export async function conv2dFilter(
  image: ImageData,
  filter: Conv2DFilterName,
): Promise<ImageData> {
  const at = await loadAlgoTensor();
  const rgba = unwrap(
    at.conv2d.filter(image.data, image.width, image.height, filter),
  );
  return new ImageData(rgba, image.width, image.height);
}

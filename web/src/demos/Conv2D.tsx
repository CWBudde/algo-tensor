// The conv2d demo: an image filtered by ops.Conv2D. Scaffold stub: a test
// pattern and its 3 × 3 box blur.

import { useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { conv2dFilter } from "@/wasm/conv2d";

const SIZE = 48;
const CELL = 6;

/** A checkerboard tinted by a color gradient. */
function testPattern(): ImageData {
  const image = new ImageData(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = 4 * (y * SIZE + x);
      const on = (Math.floor(x / CELL) + Math.floor(y / CELL)) % 2 === 0;
      image.data[i] = on ? (255 * x) / SIZE : 20;
      image.data[i + 1] = on ? (255 * y) / SIZE : 20;
      image.data[i + 2] = on ? 220 : 20;
      image.data[i + 3] = 255;
    }
  }
  return image;
}

function draw(canvas: HTMLCanvasElement | null, image: ImageData) {
  canvas?.getContext("2d")?.putImageData(image, 0, 0);
}

function Figure({
  label,
  canvasRef,
}: {
  label: string;
  canvasRef: React.Ref<HTMLCanvasElement>;
}) {
  return (
    <figure className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        className="size-48 rounded-md border [image-rendering:pixelated]"
      />
      <figcaption className="text-sm text-muted-foreground">{label}</figcaption>
    </figure>
  );
}

export function Conv2D() {
  const before = useRef<HTMLCanvasElement>(null);
  const after = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState("Filtering…");

  useEffect(() => {
    let cancelled = false;
    const input = testPattern();
    draw(before.current, input);
    conv2dFilter(input, "box").then(
      (output) => {
        if (cancelled) return;
        draw(after.current, output);
        setStatus(
          `Go returned ${output.width} × ${output.height} RGBA pixels.`,
        );
      },
      (err: unknown) => {
        if (!cancelled) setStatus(`Error: ${String(err)}`);
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Conv2D image filters <Badge variant="secondary">stub</Badge>
        </CardTitle>
        <CardDescription>
          A 3 × 3 box blur, one depthwise <code>ops.Conv2D</code> over the red,
          green and blue channels with SAME padding.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap justify-center gap-6">
          <Figure label="Input" canvasRef={before} />
          <Figure label="ops.Conv2D, box blur" canvasRef={after} />
        </div>
        <p
          className="text-sm text-muted-foreground"
          data-testid="conv2d-status"
        >
          {status}
        </p>
      </CardContent>
    </Card>
  );
}

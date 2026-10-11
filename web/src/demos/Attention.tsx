// The attention demo: attention scores as a heatmap. Scaffold stub: the
// weights of a toy sequence, with and without the causal mask.

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { attentionWeights } from "@/wasm/attention";

const N = 8;

export function Attention() {
  const [causal, setCausal] = useState(true);
  const [weights, setWeights] = useState<Float32Array>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    attentionWeights(N, causal).then(
      (w) => {
        if (!cancelled) setWeights(w);
      },
      (err: unknown) => {
        if (!cancelled) setError(String(err));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [causal]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Attention heatmap <Badge variant="secondary">stub</Badge>
        </CardTitle>
        <CardDescription>
          softmax(q·kᵀ / √d) of a toy sequence of {N} positions:{" "}
          <code>tensor.MatMul</code>, <code>ops.CausalMask</code> and{" "}
          <code>tensor.Softmax</code>. Row i is query i, column j is key j.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <Switch
            id="attention-causal"
            checked={causal}
            onCheckedChange={setCausal}
          />
          <Label htmlFor="attention-causal">Causal mask</Label>
        </div>
        {error !== undefined && (
          <p className="text-sm text-destructive">Error: {error}</p>
        )}
        {weights !== undefined && (
          <div
            className="grid w-fit gap-0.5"
            style={{ gridTemplateColumns: `repeat(${N}, 2rem)` }}
            data-testid="attention-heatmap"
          >
            {Array.from(weights, (w, i) => (
              <div
                key={i}
                title={`query ${Math.floor(i / N)}, key ${i % N}: ${w.toFixed(3)}`}
                className="size-8 rounded-sm border"
                style={{
                  backgroundColor: `color-mix(in oklab, var(--primary) ${(w * 100).toFixed(1)}%, transparent)`,
                }}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// The signals demo: ops.Conv1D/ConvTranspose1D on a waveform and RoPE
// patterns. Scaffold stub: a noisy sine smoothed by a Conv1D moving average.

import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { signalsMovingAverage } from "@/wasm/signals";

const LENGTH = 200;
const TAPS = 9;

const chartConfig = {
  input: { label: "Input", color: "var(--chart-1)" },
  smoothed: { label: `Conv1D, ${TAPS}-tap average`, color: "var(--chart-2)" },
} satisfies ChartConfig;

/** Three periods of a sine plus deterministic noise. */
function noisySine(): Float32Array<ArrayBuffer> {
  const signal = new Float32Array(LENGTH);
  let seed = 1;
  for (let i = 0; i < LENGTH; i++) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    const noise = (seed / 2147483648 - 0.5) * 0.6;
    signal[i] = Math.sin((2 * Math.PI * 3 * i) / LENGTH) + noise;
  }
  return signal;
}

const INPUT = noisySine();

export function Signals() {
  const [smoothed, setSmoothed] = useState<Float32Array>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    signalsMovingAverage(INPUT, TAPS).then(
      (out) => {
        if (!cancelled) setSmoothed(out);
      },
      (err: unknown) => {
        if (!cancelled) setError(String(err));
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const data = useMemo(
    () =>
      Array.from(INPUT, (v, i) => ({
        i,
        input: v,
        smoothed: smoothed?.[i],
      })),
    [smoothed],
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Conv1D / RoPE signals <Badge variant="secondary">stub</Badge>
        </CardTitle>
        <CardDescription>
          A noisy sine smoothed by a centered {TAPS}-tap moving average, one{" "}
          <code>ops.Conv1D</code> with zero padding.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {error !== undefined && (
          <p className="text-sm text-destructive">Error: {error}</p>
        )}
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-64 w-full"
          data-testid="signals-chart"
          data-ready={smoothed !== undefined}
        >
          <LineChart data={data} margin={{ left: 0, right: 8 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="i"
              tickLine={false}
              axisLine={false}
              minTickGap={32}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={32}
              domain={[-2, 2]}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line
              dataKey="input"
              stroke="var(--color-input)"
              strokeOpacity={0.5}
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              dataKey="smoothed"
              stroke="var(--color-smoothed)"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

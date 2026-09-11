#!/usr/bin/env node
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Worker } from 'node:worker_threads';

import { layout, type Graph, type Layout, type LayoutSettings } from '@tensorspine/ui/layout';

import { diagramsFor, spikeModels, spikeRoot, type Diagram } from './graphs.ts';
import { toSvg } from './svg.ts';
import type { LayoutRequest, LayoutResponse } from './worker.ts';

/**
 * `pnpm spike:layout` — the layout spike of feature 0.4.
 *
 * It draws the three readings of §4.7 and §4.9 for the three models the feature names, lays each
 * one out with `packages/ui/src/layout/elk.ts` — once in this thread and once in a worker thread,
 * because §5.6 puts the layout in a worker — and writes:
 *
 *     <model>.folded.svg      the document with its compositions opened as compound nodes
 *     <model>.collapsed.svg   the document with its compositions collapsed, as artboard S1 draws
 *     <model>.expanded.svg    D1, every emitted node and edge (artboard S12)
 *     measurements.json       what each layout cost, and how large the drawing came out
 *
 * The SVGs and the measurements are committed beside `NOTE.md`, which compares the reading they
 * give with `--view`'s. The expanded graphs come from the oracle (`pnpm oracle`), which is the
 * only thing here that reads the repository's tools.
 */

/** One layout, measured. */
interface Measurement {
  readonly diagram: string;
  readonly model: string;
  readonly reading: string;
  readonly boxes: number;
  readonly compound: number;
  readonly edges: number;
  readonly width: number;
  readonly height: number;
  /** Milliseconds in this thread, best of {@link RUNS}. */
  readonly milliseconds: number;
  /** Milliseconds through a worker thread, including both structured clones. */
  readonly workerMilliseconds: number;
  readonly file: string;
}

const RUNS = 3;

/**
 * One layout, asked of the worker thread.
 *
 * A worker answers in three ways and all three have to settle the promise: the reply, a throw
 * that escaped the worker's own handler (`error`), and the thread simply ending (`exit`) — an
 * out-of-memory on the largest expanded graph being the one that matters here. Waiting on
 * `message` alone would leave `pnpm spike:layout` hanging with nothing printed instead of failing
 * with the reason.
 */
function ask(worker: Worker, request: LayoutRequest): Promise<Layout> {
  return new Promise((resolve, reject) => {
    const done = (settle: () => void): void => {
      worker.off('message', onMessage);
      worker.off('error', onError);
      worker.off('exit', onExit);
      settle();
    };
    const onMessage = (response: LayoutResponse): void => {
      if (response.id !== request.id) return;
      done(() => {
        if ('failure' in response) reject(new Error(response.failure));
        else resolve(response.layout);
      });
    };
    const onError = (error: Error): void => {
      done(() => {
        reject(error);
      });
    };
    const onExit = (code: number): void => {
      done(() => {
        reject(new Error(`the layout worker exited with code ${String(code)} before it answered`));
      });
    };
    worker.on('message', onMessage);
    worker.once('error', onError);
    worker.once('exit', onExit);
    worker.postMessage(request);
  });
}

async function measure(
  graph: Graph,
  settings: Omit<LayoutSettings, 'engine'>,
): Promise<{ layout: Layout; best: number }> {
  let best = Number.POSITIVE_INFINITY;
  let last: Layout | undefined;
  for (let run = 0; run < RUNS; run += 1) {
    last = await layout(graph, settings);
    best = Math.min(best, last.milliseconds);
  }
  if (last === undefined) throw new Error('no run');
  return { layout: last, best };
}

async function main(): Promise<void> {
  const worker = new Worker(new URL('./worker.ts', import.meta.url));
  const settings: Omit<LayoutSettings, 'engine'> = {};
  const measurements: Measurement[] = [];
  try {
    for (const model of spikeModels) {
      for (const diagram of diagramsFor(model)) {
        const { layout: placed, best } = await measure(diagram.graph, settings);
        const workerStarted = performance.now();
        await ask(worker, { id: diagram.id, graph: diagram.graph, settings });
        const workerMilliseconds = performance.now() - workerStarted;
        const file = `${diagram.model}.${diagram.reading}.svg`;
        writeFileSync(join(spikeRoot, file), toSvg(diagram, placed), 'utf8');
        measurements.push(record(diagram, placed, best, workerMilliseconds, file));
        report(measurements[measurements.length - 1]);
      }
    }
  } finally {
    await worker.terminate();
  }
  writeFileSync(
    join(spikeRoot, 'measurements.json'),
    `${JSON.stringify(
      {
        generated_by: 'editor/spikes/layout/render.ts (pnpm spike:layout)',
        note: `Milliseconds are the best of ${RUNS} runs in one warm process; the worker figure is one
run through a Node worker thread, structured clone included. The expanded graphs are the
oracle's D1 (pnpm oracle).`,
        node: process.version,
        layouts: measurements,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
}

function record(
  diagram: Diagram,
  placed: Layout,
  best: number,
  workerMilliseconds: number,
  file: string,
): Measurement {
  return {
    diagram: diagram.id,
    model: diagram.model,
    reading: diagram.reading,
    boxes: placed.nodes.length,
    compound: new Set(placed.nodes.filter((node) => node.depth > 0).map((node) => node.parent)).size,
    edges: placed.edges.length,
    width: Math.round(placed.width),
    height: Math.round(placed.height),
    milliseconds: Math.round(best),
    workerMilliseconds: Math.round(workerMilliseconds),
    file,
  };
}

function report(measurement: Measurement | undefined): void {
  if (measurement === undefined) return;
  const name = `${measurement.model} · ${measurement.reading}`.padEnd(32);
  console.log(
    `${name} ${String(measurement.boxes).padStart(4)} boxes  ${String(measurement.edges).padStart(4)} edges  ` +
      `${String(measurement.milliseconds).padStart(5)} ms  (worker ${String(measurement.workerMilliseconds).padStart(5)} ms)  ` +
      `${measurement.width} × ${measurement.height} px  ${measurement.file}`,
  );
}

await main();

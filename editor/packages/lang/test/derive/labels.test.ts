import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  d2,
  derivationGraph,
  parse,
  valueGeometry,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { corpus, library, schemas } from '../describe/source.js';

// The value-type label of a diagram, against `--view`'s own (feature 1.8c).
//
// `tools/view.py` prints `bf16[tokens, model.width=4096]` over an edge, and the plan's finding F6
// decides that `view.py` will be removed and "the editor restates the conventions from the derived
// document". This is the one-time check of that restatement: the labels the tool itself printed,
// recorded in `view-labels.json` while it is still there, against the strings the core's
// `valueGeometry` renders from the D2 the core itself derives.
//
// Nothing here runs Python, and nothing reads the oracle: the fixture *is* the recording, and it
// stays true after `view.py` is gone — which is the whole reason the comparison is made once and
// written down rather than run on every commit.
//
// The comparison is over a **set**, and the set is equal, not merely included: `--view` prints at
// most two distinct labels per edge and collapses beyond that, so its labels could have been a
// subset of the document's — for these three documents they are measured to be all of them.

interface Recorded {
  note: readonly string[];
  recorded: string;
  recorded_from: string;
  repository_commit: string;
  labels: Readonly<Record<string, readonly string[]>>;
}

const fixture = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'view-labels.json'), 'utf8'),
) as Recorded;

/** The distinct labels the core renders over one corpus document's values. */
function labelsOf(name: string): string[] {
  const graph = derivationGraph(parse(corpus(name)), { schemas, library }).graph;
  const values = d2(graph, library)['values'] as readonly PyValue[];
  return [...new Set(values.map((value) => valueGeometry(value as PyRecord)))].sort();
}

describe('the labels `--view` printed', () => {
  it('names the three documents recorded, llama3-8b among them', () => {
    expect(Object.keys(fixture.labels)).toEqual([
      'llama3-8b',
      'voxtral-realtime',
      'qwen3.5-35b-a3b',
    ]);
    expect(fixture.labels['llama3-8b']).toContain('bf16[tokens, model.width=4096]');
  });

  for (const [name, recorded] of Object.entries(fixture.labels)) {
    it(`are the labels the core renders for ${name}`, { timeout: 60_000 }, () => {
      expect(labelsOf(name)).toEqual([...recorded].sort());
    });
  }

  it('reaches each form of the stream axis across the three', () => {
    // The stream axis has three forms in `view.py`: the stream's own name at one element per
    // element, `audio/8` at one per eight, and a sum where a transform inserted one stream into
    // another. The recorded labels carry all three; the fourth, a multiplier (`pixels×0.3`), is
    // in no corpus count and is the unit suite's (`d2.test.ts`).
    const all = Object.values(fixture.labels).flat();
    expect(all.some((label) => label.startsWith('i32[tokens]'))).toBe(true);
    expect(all.some((label) => label.includes('[audio/8,'))).toBe(true);
    expect(all.some((label) => label.includes('[tokens + pixels/4,'))).toBe(true);
    expect(all.every((label) => /^[a-z0-9]+\[.*\]$/.test(label))).toBe(true);
  });
});

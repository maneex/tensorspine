import { describe, expect, it } from 'vitest';

import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { languageAnchors, parse, type JsonValue, type SchemaRegistry } from '@tensorspine/lang';
import { SchemaShapes } from '@tensorspine/store';

import {
  documentResolver,
  grammarAt,
  parseAt,
  printAt,
  rowsOf,
  type LanguageContext,
} from '../../src/expressions/index.js';
import { presentation } from '../../src/presentation/index.js';
import {
  readRepositoryFile,
  registry as repositorySchemas,
  repositoryRoot,
} from '../presentation/source.js';

/**
 * What the expression editors cost — §5.6's "sheet keystroke to render < 16 ms", measured where
 * the work is.
 *
 * Four figures, and each is a place this feature could have been slow:
 *
 *  - **building the grammar** at an anchor, which walks the union, every alternative and every
 *    binding. It is memoised per registry, per bindings and per anchor, so what matters is that
 *    the first one is affordable and the rest are free.
 *  - **printing and parsing** one expression, which happens on every keystroke of the text field
 *    and once per row of the tree.
 *  - **the tree's rows**, which are rebuilt on every render of the editor.
 *  - **the resolver**, which resolves *every* quantity of the document to answer S7's `4096`; the
 *    sheet keeps one per tree for exactly that reason.
 *
 * The bounds are far above the measurements, as every timing suite of this package states: the
 * unit layer runs a hundred files at once and a guard tight enough to be interesting on an idle
 * machine fails on a loaded one. The printed line beside each is the measurement.
 */

const BOUND_MS = 400;

const registry: SchemaRegistry = repositorySchemas();
const context: LanguageContext = {
  registry,
  shapes: new SchemaShapes(registry),
  bindings: presentation(),
};
const anchors = languageAnchors(registry);

function median(times: number[]): number {
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of the grammar', () => {
  it('builds once per anchor and answers from the cache afterwards', () => {
    // A fresh context, so the first build is a cold one — the caches are keyed by the shapes.
    const cold: LanguageContext = { registry, shapes: new SchemaShapes(registry), bindings: presentation() };
    const first = performance.now();
    const built = grammarAt(cold, anchors.condition);
    const marks = [...built.marks];
    const coldMs = performance.now() - first;

    const times: number[] = [];
    for (let run = 0; run < 50; run += 1) {
      const at = performance.now();
      grammarAt(cold, anchors.condition);
      times.push(performance.now() - at);
    }
    const warm = median(times);
    console.log(
      `grammar: ${coldMs.toFixed(2)} ms cold (${String(built.productions.length)} productions, ` +
        `${String(marks.length)} marks), ${warm.toFixed(4)} ms memoised`,
    );
    expect(coldMs).toBeLessThan(BOUND_MS);
    expect(warm).toBeLessThan(1);
  });
});

describe('the cost of one expression', () => {
  const written = parse('{"op": "floor_divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}');

  it('prints, parses and rows inside the keystroke budget', () => {
    const measure = (what: () => void): number => {
      const times: number[] = [];
      for (let run = 0; run < 200; run += 1) {
        const at = performance.now();
        what();
        times.push(performance.now() - at);
      }
      return median(times);
    };
    const printing = measure(() => printAt(context, anchors.expression, written));
    const parsing = measure(() => parseAt(context, anchors.expression, 'd div heads'));
    const rows = measure(() => rowsOf(context, anchors.expression, written));
    console.log(
      `one expression: print ${printing.toFixed(4)} ms · parse ${parsing.toFixed(4)} ms · ` +
        `rows ${rows.toFixed(4)} ms`,
    );
    for (const one of [printing, parsing, rows]) expect(one).toBeLessThan(BOUND_MS);
  });
});

describe('the cost of the repository’s own expressions', () => {
  it('round-trips every one of them in a measured time', () => {
    const values: { anchor: string; value: JsonValue }[] = [];
    const collect = (value: JsonValue, anchor: string): void => {
      if (typeof value !== 'object' || value === null) return;
      if (Array.isArray(value)) {
        for (const one of value as readonly JsonValue[]) collect(one, anchor);
        return;
      }
      if (!('members' in value)) return;
      if (registry.accepts(value, anchor)) values.push({ anchor, value });
      for (const member of value.members) collect(member.value, anchor);
    };
    for (const name of readdirSync(join(repositoryRoot, 'data/models'))) {
      if (!name.endsWith('.json')) continue;
      collect(parse(readRepositoryFile(`data/models/${name}`)), anchors.expression);
    }
    const at = performance.now();
    for (const one of values) {
      const text = printAt(context, one.anchor, one.value);
      parseAt(context, one.anchor, text);
    }
    const total = performance.now() - at;
    console.log(
      `the corpus’s expressions: ${String(values.length)} printed and parsed in ` +
        `${total.toFixed(1)} ms (${(total / Math.max(values.length, 1)).toFixed(4)} ms each)`,
    );
    expect(values.length).toBeGreaterThan(500);
    expect(total / values.length).toBeLessThan(BOUND_MS);
  });
});

describe('the cost of the resolver', () => {
  for (const name of ['llama3-8b', 'deepseek-v4-pro']) {
    it(`resolves every quantity of ${name} once`, () => {
      const tree = parse(readRepositoryFile(`data/models/${name}.json`));
      const times: number[] = [];
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const resolve = documentResolver(tree);
        resolve(parse('{"quantity": "d"}'), anchors.expression);
        times.push(performance.now() - at);
      }
      const built = median(times);
      console.log(`resolver: ${name} ${built.toFixed(3)} ms per tree`);
      expect(built).toBeLessThan(BOUND_MS);
    });
  }
});

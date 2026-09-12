import { describe, expect, it } from 'vitest';

import { parse, serialize, toJsonValue, toPlain } from '@tensorspine/lang';

import { derivedBytes, derivedPathOf } from '../../src/derived/export.js';
import { freshnessOf } from '../../src/derived/freshness.js';
import { derivedReading } from '../../src/derived/products.js';
import { noReading, type Reading } from '../../src/documents/pipeline.js';
import { context, CORPUS, derivedOf, registry } from './source.js';

// The two halves of §4.18 that are not a table: the header's freshness ("`fresh` / `stale since
// <edit>` / `derivation failed: …`"), and **Export Derived Document…** ("writes the JSON,
// validated by the core against the derived schema before writing, as the tools do").

/** A reading of a document that has been derived, at the revision the caller names. */
function readingAt(revision: number, derivedAt: number): Reading {
  return {
    ...noReading(revision),
    derived: derivedOf('llama3-8b') as Reading['derived'],
    derivedAt,
    derivation: 'fresh',
  };
}

describe('the header’s freshness — §5.4’s own rule, per revision', () => {
  it('is fresh where the products were computed for the document’s own revision', () => {
    const found = freshnessOf(readingAt(4, 4), 'Set heads');
    expect(found).toEqual({
      derived: true,
      stale: false,
      since: null,
      failure: null,
      running: false,
      skipped: false,
    });
  });

  it('is stale since the command that moved the document (D13’s own name for it)', () => {
    const found = freshnessOf(readingAt(5, 4), 'Set heads');
    expect(found.stale).toBe(true);
    expect(found.since).toBe('Set heads');
  });

  it('is stale with no name where the log has none — a revert, or a fresh open', () => {
    expect(freshnessOf(readingAt(5, 4), null).since).toBeNull();
  });

  it('is not stale where nothing has been derived: undone is a different word (feature 2.6)', () => {
    const found = freshnessOf({ ...noReading(3), derivation: 'running' }, 'Set heads');
    expect(found).toMatchObject({ derived: false, stale: false, running: true });
  });

  it('says a refusal whichever revision it was computed for', () => {
    const failure = {
      code: 'derivation',
      message: 'not valid, no products: …',
      path: 'models/llama3-8b.json',
      severity: 'error' as const,
      source: 'derivation' as const,
    };
    const found = freshnessOf({ ...noReading(3), derivation: 'failed', failure }, null);
    expect(found.failure?.message).toBe('not valid, no products: …');
  });

  it('says the validation is why, where the pipeline skipped the derivation', () => {
    expect(freshnessOf({ ...noReading(3), derivation: 'skipped' }, null).skipped).toBe(true);
  });
});

describe('Export Derived Document…', () => {
  it('writes `<model>.derived.json`, which is the name §4.4 gives it', () => {
    expect(derivedPathOf('models/llama3-8b.json')).toBe('models/llama3-8b.derived.json');
    expect(derivedPathOf('a/b')).toBe('a/b.derived.json');
  });

  // The whole corpus derived, serialized, parsed back and checked: eight seconds on an idle box,
  // which the default five-second budget is no claim about (feature 2.13's own reading of it).
  it('writes the core’s own document, and it conforms to the derived schema', { timeout: 120_000 }, () => {
    for (const name of CORPUS) {
      const derived = derivedOf(name);
      const written = derivedBytes(derived, registry);
      expect(written.problems, name).toEqual([]);
      // Deep-equal to `derive`'s own answer, which is what the browser case asserts through the
      // application: the export is the product and not a second reading of it. The comparison is
      // over the *plain* readings because the two trees differ in one thing a reader cannot see —
      // a parsed number keeps its source lexeme where a computed one has none (D12) — and the
      // bytes are held beside it by the round trip below.
      expect(toPlain(parse(written.text)), name).toEqual(toPlain(toJsonValue(derived)));
      expect(serialize(parse(written.text)), name).toBe(written.text);
      expect(written.text.endsWith('}\n'), name).toBe(true);
    }
  });

  it('is the core’s serializer and not `--derive`’s layout (feature 1.8e’s finding)', () => {
    // `--derive` writes `indent=1`, the one output of the repository that does not write
    // `indent=2`; what is exported is the *document*, through D12's writer, so the bytes are the
    // editor's canonical rendering and the contract is the deep equality above.
    const written = derivedBytes(derivedOf('llama3-8b'), registry);
    expect(written.text).toBe(serialize(toJsonValue(derivedOf('llama3-8b'))));
    expect(written.text.split('\n')[1]?.startsWith('  ')).toBe(true);
  });

  it('reports a refusal rather than raising, and still answers the bytes', () => {
    // The one case the check can catch that `derive`'s own self-check cannot: a workspace whose
    // schemas are not the build's (plan §1). A document with a member the derived schema refuses
    // stands in for it.
    const written = derivedBytes({ schema: 'tensorspine-derived/2.1' }, registry);
    expect(written.problems.length).toBeGreaterThan(0);
    expect(written.text.length).toBeGreaterThan(0);
  });

  it('writes the bytes with no registry at all — the stub platform reads no schemas', () => {
    const written = derivedBytes(derivedOf('llama3-8b'), null);
    expect(written.problems).toEqual([]);
    expect(written.text.length).toBeGreaterThan(1000);
  });
});

describe('a document the panel cannot read', () => {
  it('answers no product and no header rather than raising', () => {
    expect(derivedReading(null, context)).toEqual({ products: [], header: [] });
    expect(derivedReading('not a document', context)).toEqual({ products: [], header: [] });
    expect(derivedReading({ d3: null }, context).products.map((one) => one.member)).toEqual(['d3']);
  });
});

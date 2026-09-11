import { expect, it, suite } from 'vitest';

import { d2, d3, d4, d5, pyStr, type PyRecord, type PyValue } from '../../src/index.js';
import { agrees, at, derivedCorpus, generated, inCI, libraryOf } from './derived.js';

// Parity of D5 (feature 1.8d): the Derived Logical Resource Requirements and Costs of §7, as
// `derive.d5` writes it.
//
// The fixture is `derived.ts`'s — the corpus derived once beside `--derive`'s own documents — and
// the comparison is over its `d5` member, twice: the readings deep-equal, then the two written
// with the core's serializer and compared as text. The second is not a repetition of the first.
// D5's bytes carry the integer/float distinction of every figure it hands on (a parameter byte
// total is a float wherever a sub-byte dtype is in the inventory, `deepseek-v4-pro`'s is, while
// its element count is an integer), the member order of every `bytes_per_invocation` it restates
// from D2, and the exact order of `corrections` and `sparsity`, which is the resolved nodes' own
// order and, within a node, the primitive's declaration order.
//
// D5 reads the other three products, so this suite computes them: D3 for the parameter totals, D4
// for the state totals, D2 for the split payloads, and the analysis's own counters for the
// inventory rule (`validate.analyse`'s "D5, first derivation" block, feature 1.6c). The tools'
// `products` runs them in that order and so does this.
//
// Beside the fifteen documents, this suite states **what the corpus reaches**, measured: the three
// `per` values its corrections carry and the one it does not, the two statuses its corrections
// carry and the two it does not, the one status every sparsity bound carries, the blank `units`
// and blank bound a sparsity unit can have, and the single place a correction's status moves an
// `operations` total off `exact`. A corpus that stopped exercising one of them fails here rather
// than going quiet — and the cells of §2.2 none of them reaches are proved in
// `test/derive/qualified.test.ts`, against the specification rather than against a document.

/** D5 of one case, over the three products the tools compute before it and in their order. */
function costs(one: ReturnType<typeof derivedCorpus>[number]): PyRecord {
  const library = libraryOf(one);
  const p3 = d3(one.derivation.graph, library);
  const p4 = d4(one.derivation.graph, library);
  const p2 = d2(one.derivation.graph, library);
  return d5(one.derivation.graph, p3, p4, p2, one.derivation.analysis.stats);
}

suite('D5 against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)(
    'derives the logical costs of every corpus document',
    { timeout: 300_000 },
    () => {
      const documents = derivedCorpus();
      expect(documents).toHaveLength(15);
      let corrections = 0;
      let units = 0;
      for (const one of documents) {
        const answer = costs(one);
        agrees(answer, at(one.expected, 'd5'), one.path);
        corrections += (at(answer, 'corrections') as readonly PyValue[]).length;
        units += (at(answer, 'sparsity') as readonly PyValue[]).length;
      }
      // 889 applying corrections and 446 sparsity units over the fifteen documents.
      expect([corrections, units]).toEqual([889, 446]);
    },
  );

  it.skipIf(!generated)(
    'reaches the statuses, the units and the blanks the corpus carries — and no others',
    { timeout: 300_000 },
    () => {
      const seen = new Set<string>();
      for (const one of derivedCorpus()) {
        const product = at(one.expected, 'd5');
        for (const row of at(product, 'corrections') as readonly PyValue[]) {
          const entry = row as PyRecord;
          seen.add(`correction per:${pyStr(entry['per'] as PyValue)}`);
          seen.add(`correction status:${pyStr(entry['status'] as PyValue)}`);
          if (entry['value'] === null) seen.add('correction without a value');
        }
        for (const row of at(product, 'sparsity') as readonly PyValue[]) {
          const entry = row as PyRecord;
          const bound = entry['union_per_invocation'] as PyRecord;
          seen.add(`bound status:${pyStr(bound['status'] as PyValue)}`);
          if (entry['units'] === null) seen.add('sparsity without an extent');
          else seen.add('sparsity with an extent');
          if (bound['value'] === null) seen.add('bound without a value');
          if (entry['activated_fraction'] === null) seen.add('sparsity without a fraction');
        }
        for (const [per, figure] of Object.entries(at(product, 'operations') as PyRecord)) {
          seen.add(`operations ${per}:${pyStr((figure as PyRecord)['status'] as PyValue)}`);
        }
      }
      expect([...seen].sort()).toEqual([
        // `invocation` is the `per` no cost entry of the reference base declares, so that total is
        // exact in every document by having nothing to combine with.
        'bound status:upper_bound',
        'bound without a value',
        'correction per:cached_position',
        'correction per:element',
        'correction per:sequence',
        'correction status:estimate',
        'correction status:exact',
        'operations cached_position:exact',
        'operations element:exact',
        'operations invocation:exact',
        'operations sequence:estimate',
        'operations sequence:exact',
        'sparsity with an extent',
        'sparsity without a fraction',
        'sparsity without an extent',
      ]);
    },
  );

  it.skipIf(!generated)(
    'moves an `operations` total off `exact` exactly where a correction says so',
    { timeout: 300_000 },
    () => {
      // The one place in the repository where §2.2's algebra does visible work: `embedding.time`
      // declares a `sequence` correction with status `estimate`, `voxtral-realtime` instantiates
      // it, and its `operations.sequence` is the only figure of the whole corpus that is not
      // `exact`. The suite asserts the exact set, so a second one is a finding.
      const moved = new Map<string, string[]>();
      for (const one of derivedCorpus()) {
        const operations = at(costs(one), 'operations') as PyRecord;
        const odd = Object.entries(operations)
          .filter(([, figure]) => (figure as PyRecord)['status'] !== 'exact')
          .map(([per, figure]) => `${per}:${pyStr((figure as PyRecord)['status'] as PyValue)}`);
        if (odd.length > 0) moved.set(one.name, odd);
      }
      expect(Object.fromEntries(moved)).toEqual({ 'voxtral-realtime': ['sequence:estimate'] });
    },
  );

  it.skipIf(!generated)(
    'restates D2’s split payloads and D3’s and D4’s totals without touching them',
    { timeout: 300_000 },
    () => {
      // D5 computes two things and carries five. The carried ones are asserted against the
      // products they come from, not against the recorded document, so that a D5 that recomputed
      // a figure of its own — rounded a byte total, re-summed a payload — fails here even where
      // the recomputation happened to agree with the tools.
      for (const one of derivedCorpus()) {
        const library = libraryOf(one);
        const p3 = d3(one.derivation.graph, library);
        const p4 = d4(one.derivation.graph, library);
        const p2 = d2(one.derivation.graph, library);
        const answer = d5(one.derivation.graph, p3, p4, p2, one.derivation.analysis.stats);

        const t3 = at(p3, 'totals') as PyRecord;
        const parameters = at(answer, 'parameters') as PyRecord;
        expect(parameters['elements'], one.name).toBe(t3['elements']);
        expect(parameters['bytes'], one.name).toBe(t3['bytes']);
        expect(parameters['status'], one.name).toBe('exact');

        const t4 = at(p4, 'totals') as PyRecord;
        const state = at(answer, 'state') as PyRecord;
        for (const name of ['append_bytes_per_cached_position', 'bounded_bytes', 'fixed_bytes']) {
          expect(state[name], `${one.name}.${name}`).toBe(t4[name]);
        }
        expect(state['status'], one.name).toBe('exact');

        const splits = at(p2, 'graph_splits') as readonly PyValue[];
        const carried = at(answer, 'graph_splits') as readonly PyValue[];
        expect(carried).toHaveLength(splits.length);
        for (const [index, split] of splits.entries()) {
          const source = split as PyRecord;
          const row = carried[index] as PyRecord;
          expect(Object.keys(row), one.name).toEqual([
            'graph_split',
            'bytes_per_element',
            'bytes_per_invocation',
          ]);
          expect(row['graph_split'], one.name).toBe(source['graph_split']);
          expect(row['bytes_per_element'], one.name).toBe(source['bytes_per_element']);
          expect(row['bytes_per_invocation'], one.name).toBe(source['bytes_per_invocation']);
        }

        // The four counters are `--validate`'s own, unrounded and unconverted.
        for (const [per, counter] of [
          ['element', 'ops_per_element'],
          ['cached_position', 'ops_per_cached_position'],
          ['sequence', 'ops_per_sequence'],
          ['invocation', 'ops_per_invocation'],
        ] as const) {
          const figure = (at(answer, 'operations') as PyRecord)[per] as PyRecord;
          expect(figure['value'], `${one.name}.${per}`).toBe(
            one.derivation.analysis.stats.get(counter),
          );
        }
      }
    },
  );

  it.skipIf(!generated)(
    'derives the same costs from a template instance as from the flat document',
    { timeout: 300_000 },
    () => {
      // `tests/run_costs.py`'s third claim, at the level of the whole product: "shieldstral-3b and
      // its composite derive the same figures". Everything but the identifiers is equal — the
      // composite's nodes carry the `text/` prefix its template instance supplies (§5.2 rule 2) —
      // so the comparison drops `node` from every row and keeps the rest.
      const documents = new Map(derivedCorpus().map((one) => [one.name, one]));
      const flat = documents.get('shieldstral-3b');
      const composite = documents.get('shieldstral-3b-composite');
      expect([flat, composite].every((one) => one !== undefined)).toBe(true);
      /** One row without the member the prefix renames. */
      const without = (row: PyValue, name: string): PyValue => {
        const copy = { ...(row as PyRecord) };
        delete copy[name];
        return copy;
      };
      const unnamed = (product: PyRecord): PyRecord => {
        const copy = { ...product };
        for (const name of ['corrections', 'sparsity']) {
          copy[name] = (copy[name] as readonly PyValue[]).map((row) => without(row, 'node'));
        }
        // A graph split is named after the composition or family it cuts, which the prefix moves
        // too; the figures are what this claim is about.
        copy['graph_splits'] = (copy['graph_splits'] as readonly PyValue[]).map((row) =>
          without(row, 'graph_split'),
        );
        return copy;
      };
      expect(unnamed(costs(composite as NonNullable<typeof composite>))).toEqual(
        unnamed(costs(flat as NonNullable<typeof flat>)),
      );
    },
  );
});

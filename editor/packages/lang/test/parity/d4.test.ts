import { expect, it, suite } from 'vitest';

import { d4, pyStr, type PyRecord, type PyValue } from '../../src/index.js';
import { agrees, at, derivedCorpus, generated, inCI, libraryOf } from './derived.js';

// Parity of D4 (feature 1.8b): the Derived State Inventory and Behavior of §7, as `derive.d4`
// writes it.
//
// The fixture is `derived.ts`'s — the corpus derived once beside `--derive`'s own documents — and
// the comparison is over its `d4` member, twice: the readings deep-equal, then the two written
// with the core's serializer and compared as text. The second is not a repetition of the first.
// D4's bytes carry three things a structural comparison alone would let past: the order of
// `by_evolution`'s members, which is a Python `Counter`'s first-seen order and nothing else; the
// integer/float distinction of every byte figure; and the exact wording of `visits`, which is §7's
// rule in words and therefore a contract of prose.
//
// Beside the fifteen documents, this suite states **what the corpus reaches**, measured: every
// evolution, every access geometry, every sharing granularity, the three sets of operations the
// reference base declares, both branches of `carried_across_fragments`, a source-indexed state
// that is *not* carried, a multi-component payload, a multi-member identity, and the one place the
// tools write a `writer` naming a node the expanded graph has not. A corpus that stopped
// exercising one of them fails here rather than going quiet — and the branches it never reaches
// are the unit suite's (`test/derive/d4.test.ts`).

suite('D4 against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)(
    'derives the state inventory of every corpus document',
    { timeout: 300_000 },
    () => {
      const documents = derivedCorpus();
      expect(documents).toHaveLength(15);
      let states = 0;
      for (const one of documents) {
        const answer = d4(one.derivation.graph, libraryOf(one));
        agrees(answer, at(one.expected, 'd4'), one.path);
        states += (at(answer, 'states') as readonly PyValue[]).length;
      }
      // 1 031 state identity instances over the fifteen documents, `colbert-v2`'s none included.
      expect(states).toBe(1031);
    },
  );

  it.skipIf(!generated)(
    'reaches every branch of the inventory the corpus carries',
    { timeout: 300_000 },
    () => {
      const seen = new Set<string>();
      for (const one of derivedCorpus()) {
        for (const state of at(at(one.expected, 'd4'), 'states') as readonly PyValue[]) {
          const row = state as PyRecord;
          seen.add(`evolution:${pyStr(row['evolution'] as PyValue)}`);
          seen.add(`access:${pyStr(row['access'] as PyValue)}`);
          seen.add(`sharing:${pyStr(row['sharing'] as PyValue)}`);
          const effects = (row['operations'] as readonly PyValue[]).map((one) => pyStr(one));
          seen.add(`operations:${effects.join('+')}`);
          if (row['indexed_by_source'] === true) seen.add('indexed by source');
          if (row['span'] !== null) seen.add('span');
          if (row['stride'] !== null) seen.add('stride');
          if ((row['members'] as readonly PyValue[]).length > 1) seen.add('shared identity');
          if ((row['payload'] as readonly PyValue[]).length > 1) seen.add('several components');
          if (row['carried_across_fragments'] === true) {
            seen.add(row['indexed_by_source'] === true ? 'carried by source' : 'carried by rule');
          } else if (row['indexed_by_source'] === true) {
            seen.add('source-indexed, not carried');
          }
        }
      }
      expect([...seen].sort()).toEqual([
        'access:aggregate',
        'access:logical_position',
        'access:ring',
        'access:selected',
        'carried by rule',
        'carried by source',
        'evolution:append',
        'evolution:fixed',
        'evolution:window',
        'indexed by source',
        'operations:append+evict+read',
        'operations:append+read',
        'operations:read+write',
        'several components',
        'shared identity',
        'sharing:at_fork_point',
        'sharing:by_position',
        'sharing:by_source',
        'sharing:within_span',
        'source-indexed, not carried',
        'span',
        'stride',
      ]);
    },
  );

  it.skipIf(!generated)(
    'writes a template instance’s writer without the prefix its members carry',
    { timeout: 300_000 },
    () => {
      // Feature 1.8a's finding, reproduced at the place it shows: `instances(kind)` replaces
      // `identity` and `members` and nothing else, so a state identity inside a template instance
      // names its writer by the sub-graph's own key — a `value_reference` naming a node the
      // expanded graph has not. The suite asserts the exact set of documents where that happens,
      // so a second one is a finding rather than a silent exemption, and asserts that everywhere
      // else the writer is one of the members.
      const outside = new Map<string, number>();
      for (const one of derivedCorpus()) {
        const answer = d4(one.derivation.graph, libraryOf(one));
        for (const state of at(answer, 'states') as readonly PyValue[]) {
          const row = state as PyRecord;
          const writer = row['writer'];
          expect(writer, `${one.name}: ${pyStr(row['identity'] as PyValue)}`).not.toBeNull();
          const members = row['members'] as readonly PyValue[];
          if (members.includes(writer as PyValue)) continue;
          outside.set(one.name, (outside.get(one.name) ?? 0) + 1);
        }
      }
      expect(Object.fromEntries(outside)).toEqual({ 'shieldstral-3b-composite': 26 });
    },
  );
});

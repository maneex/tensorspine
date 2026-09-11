import { expect, it, suite } from 'vitest';

import { d2, pyStr, type PyRecord, type PyValue } from '../../src/index.js';
import { agrees, at, derivedCorpus, generated, inCI, libraryOf } from './derived.js';

// Parity of D2 (feature 1.8c): the Derived Value Shapes and Lifetimes of §7, as `derive.d2`
// writes it.
//
// The fixture is `derived.ts`'s — the corpus derived once beside `--derive`'s own documents — and
// the comparison is over its `d2` member, twice: the readings deep-equal, then the two written
// with the core's serializer and compared as text. The second is not a repetition of the first.
// D2's bytes carry four things a structural comparison alone would let past: the member order of
// every `count`, which is the order the propagation built it in; the member order of every
// `bytes_per_invocation`, which is a `Counter`'s first-seen order over the values it weighed; the
// integer/float distinction of every byte figure, a count being a float and making a product of it
// one; and the order of the terms of every sum, `sum` starting at the integer zero.
//
// Beside the fifteen documents, this suite states **what the corpus reaches**, measured: the
// joining input and its alignment (Voxtral), the merge and the insert, a value no count reaches, a
// value a public output exposes, an input that is required for nothing, both kinds of graph split,
// and the peak. A corpus that stopped exercising one of them fails here rather than going quiet —
// and the branches it never reaches are the unit suite's (`test/derive/d2.test.ts`).

suite('D2 against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)(
    'derives the values, streams, graph splits and peak of every corpus document',
    { timeout: 300_000 },
    () => {
      const documents = derivedCorpus();
      expect(documents).toHaveLength(15);
      let values = 0;
      let splits = 0;
      for (const one of documents) {
        const answer = d2(one.derivation.graph, libraryOf(one));
        agrees(answer, at(one.expected, 'd2'), one.path);
        values += (at(answer, 'values') as readonly PyValue[]).length;
        splits += (at(answer, 'graph_splits') as readonly PyValue[]).length;
      }
      // 5 189 values and 895 structural graph splits over the fifteen documents.
      expect({ values, splits }).toEqual({ values: 5189, splits: 895 });
    },
  );

  it.skipIf(!generated)(
    'reaches every branch of the inventory the corpus carries',
    { timeout: 300_000 },
    () => {
      const seen = new Set<string>();
      for (const one of derivedCorpus()) {
        const product = at(one.expected, 'd2');
        for (const [name, stream] of Object.entries(at(product, 'streams') as PyRecord)) {
          const entry = stream as PyRecord;
          seen.add(`kind:${pyStr(entry['kind'] as PyValue)}`);
          if (entry['fragment_alignment'] !== undefined) seen.add('fragment alignment');
          const count = entry['count'] as PyRecord | null;
          if (count !== null && Object.keys(count).length > 1) seen.add('a stream of two counts');
          if (count !== null && !Object.hasOwn(count, name)) seen.add('a stream counted elsewhere');
        }
        for (const value of at(product, 'values') as readonly PyValue[]) {
          const row = value as PyRecord;
          const count = row['count'] as PyRecord | null;
          if (row['input'] !== undefined) seen.add('a public input’s value');
          if (row['exposed'] !== undefined) seen.add('a value a public output exposes');
          if (row['required'] === true) seen.add('an input required for an output');
          if (row['required'] === false) seen.add('an input required for nothing');
          if (count === null) seen.add('a value with no count');
          else {
            if (Object.keys(count).length > 1) seen.add('a count over two inputs');
            for (const multiplier of Object.values(count)) {
              if (multiplier === 1) seen.add('one element per element');
              else if ((multiplier as number) < 1) seen.add('a merged count');
              else seen.add('a count above one');
            }
          }
          if ((row['to'] as readonly PyValue[]).length > 1) seen.add('a value with two consumers');
          if ((row['shape'] as readonly PyValue[]).length === 0) seen.add('a value of no axes');
          if ((row['shape'] as readonly PyValue[]).length > 2) seen.add('a value of three axes');
        }
        for (const split of at(product, 'graph_splits') as readonly PyValue[]) {
          const row = split as PyRecord;
          seen.add(`split:${pyStr(row['kind'] as PyValue)}`);
          if ((row['payload'] as readonly PyValue[]).length > 1) seen.add('a payload of two values');
        }
        const peak = at(product, 'peak_live') as PyRecord;
        if (peak['node'] !== null) seen.add('a peak');
        if ((peak['values'] as readonly PyValue[]).length > 2) seen.add('a peak of three values');
      }
      // Four of the probes above are never lit, and each is a branch of the port the unit suite
      // owns: a count above one (an `insert` adds counts that are each at most one), a stream
      // whose own count is not `{itself: 1.0}`, a stream carrying two counts, and a value with no
      // count at all — every value of every corpus document is reached by some public input.
      expect([...seen].sort()).toEqual([
        'a count over two inputs',
        'a merged count',
        'a payload of two values',
        'a peak',
        'a peak of three values',
        'a public input’s value',
        'a value a public output exposes',
        'a value of no axes',
        'a value of three axes',
        'a value with two consumers',
        'an input required for an output',
        'an input required for nothing',
        'fragment alignment',
        'kind:patch',
        'kind:position',
        'kind:sequence',
        'kind:token',
        'one element per element',
        'split:family',
        'split:layer',
      ]);
    },
  );

  it.skipIf(!generated)(
    'takes a joining input’s count from the stream it joins, and aligns its fragments',
    { timeout: 300_000 },
    () => {
      // The one document of the corpus with a joining input (§5.3, V19): Voxtral's `tokens` join
      // `audio` at kind `token`, "where the projector's merge left `{"audio": 0.125}` — one
      // language-model position per eight frames". The alignment is the derived deployment
      // obligation beside it: "a stride of 2, then 4 frames per token, 8".
      const one = derivedCorpus().find((document) => document.name === 'voxtral-realtime');
      expect(one, 'voxtral-realtime is in the corpus').toBeDefined();
      const answer = d2(
        (one as NonNullable<typeof one>).derivation.graph,
        libraryOf(one as NonNullable<typeof one>),
      );
      const streams = at(answer, 'streams') as PyRecord;
      expect(Object.keys(streams)).toEqual(['audio', 'delay']);
      expect(streams['audio']).toEqual({
        kind: 'position',
        count: { audio: 1 },
        fragment_alignment: 8n,
      });
      const values = at(answer, 'values') as readonly PyValue[];
      const joining = values.find((value) => (value as PyRecord)['input'] === 'tokens') as PyRecord;
      expect(joining['count']).toEqual({ audio: 0.125 });
      expect(joining['domain']).toEqual({ kind: 'token', stream: 'audio' });
      // The introducing input's own count is one per element of the stream it introduces.
      const audio = values.find((value) => (value as PyRecord)['input'] === 'audio') as PyRecord;
      expect(audio['count']).toEqual({ audio: 1 });
    },
  );
});

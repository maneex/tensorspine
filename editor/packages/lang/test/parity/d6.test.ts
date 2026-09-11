import { expect, it, suite } from 'vitest';

import { d2, d4, d6, expand, pyStr, type PyRecord, type PyValue } from '../../src/index.js';
import { agrees, at, derivedCorpus, generated, inCI, libraryOf, type Case } from './derived.js';

// Parity of D6 (feature 1.8e): the Derived Decomposition Options of §7, as `derive.d6` writes it.
//
// The fixture is `derived.ts`'s — the corpus derived once beside `--derive`'s own documents — and
// the comparison is over its `d6` member, twice: the readings deep-equal, then the two written
// with the core's serializer and compared as text. The second is not a repetition of the first.
// D6's bytes carry the integer/float distinction of every `granularity` (an undeclared one is the
// *integer* one, an evaluated one whatever its expression answers); the order of every `block`,
// which is D1's published topological order filtered rather than sorted; the order of
// `partition_options` and `information_loss`, which is the resolved nodes' own order and, within a
// node, the primitive's declaration order; and the `sizes`, `span` and `bytes_per_cached_position`
// figures D6 restates from D2 and D4 without touching them.
//
// D6 reads two products and one order, so this suite computes all three: D2 for the splits it
// restates, D4 for the states a split separates, and D1 — re-emitted, as `derive.products`
// re-emits it — for the order a block is published in.
//
// Beside the fifteen documents, this suite reproduces `tests/run_derived.py`'s own D6 claims, each
// recomputed **from D1 alone** rather than read back from the product: a block is the ancestor
// closure of its seeds, in D1 order, of the size D2 states; no edge enters a block; the layer
// splits of one composition nest; the separated states are exactly D4's identities with members on
// both sides; and `history_needed_by` is its definition. Then the documents the script names by
// hand — gemma3n's two shared identities across ten layer splits each, llama3-8b's empty
// separations, its absent information loss and its head partition's granularity of four — and what
// the corpus reaches, measured, so that a corpus that stopped exercising one of them fails here
// rather than going quiet.

/** D6 of one case, over the two products the tools compute before it and D1's published order. */
function options(one: Case): PyRecord {
  const library = libraryOf(one);
  const p4 = d4(one.derivation.graph, library);
  const p2 = d2(one.derivation.graph, library);
  return d6(one.derivation.graph, p2, p4, orderOf(one));
}

/** D1's `topological_order` for a case, re-emitted as `products` re-emits it. */
function orderOf(one: Case): readonly PyValue[] {
  const assignment = at(one.expected, 'assignment') as PyRecord;
  const emitted = expand(one.derivation.document, libraryOf(one), { assignment });
  return at(at(emitted, 'd1'), 'topological_order') as readonly PyValue[];
}

/** Every member of a list of records, read as records. */
function rows(product: PyValue, name: string): readonly PyRecord[] {
  return at(product, name) as readonly PyRecord[];
}

/** The strings of a list. */
function names(value: PyValue): string[] {
  return (value as readonly PyValue[]).map((one) => pyStr(one));
}

/** The node a `node.state` member names, as D6 reads it. */
function nodeOf(member: string): string {
  const dot = member.lastIndexOf('.');
  return dot === -1 ? member : member.slice(0, dot);
}

/** The recorded document of one case, by name. */
function documents(): Map<string, Case> {
  return new Map(derivedCorpus().map((one) => [one.name, one]));
}

suite('D6 against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)(
    'derives the decomposition options of every corpus document',
    { timeout: 300_000 },
    () => {
      const corpus = derivedCorpus();
      expect(corpus).toHaveLength(15);
      let splits = 0;
      let partitions = 0;
      let losses = 0;
      for (const one of corpus) {
        const answer = options(one);
        agrees(answer, at(one.expected, 'd6'), one.path);
        splits += rows(answer, 'graph_splits').length;
        partitions += rows(answer, 'partition_options').length;
        losses += rows(answer, 'information_loss').length;
      }
      // 895 structural graph splits — D2's own count (feature 1.8c) — 8 082 partition options and
      // 1 208 information-loss rows over the fifteen documents.
      expect([splits, partitions, losses]).toEqual([895, 8082, 1208]);
    },
  );

  it.skipIf(!generated)(
    'states every block as the ancestor closure of its seeds, in D1 order, that no edge enters',
    { timeout: 300_000 },
    () => {
      // `tests/run_derived.py`'s first two D6 claims, recomputed from D1's own nodes and edges:
      // the seeds are read back out of the split's *name* — the layer prefix or the family — the
      // closure is walked up the edges, and the block must be that closure, in the order D1
      // publishes, of the first size D2 states. Then the edge rule that makes a split valid at
      // all: "the ancestor closure of a layer prefix or of a family is downward closed, so every
      // crossing edge points out of it".
      for (const one of derivedCorpus()) {
        const emitted = at(expand(one.derivation.document, libraryOf(one), {
          assignment: at(one.expected, 'assignment') as PyRecord,
        }), 'd1');
        const nodes = at(emitted, 'nodes') as PyRecord;
        const edges = at(emitted, 'edges') as readonly PyValue[];
        const order = names(at(emitted, 'topological_order'));
        const place = new Map(order.map((name, index) => [name, index]));

        const parents = new Map<string, string[]>();
        const endpoints = edges.map((edge) => [
          pyStr(at(at(edge, 'from'), 'node')),
          pyStr(at(at(edge, 'to'), 'node')),
        ]);
        for (const [from, to] of endpoints) {
          const held = parents.get(to as string);
          if (held === undefined) parents.set(to as string, [from as string]);
          else held.push(from as string);
        }
        const closure = (seeds: readonly string[]): Set<string> => {
          const seen = new Set<string>();
          const stack = [...seeds];
          while (stack.length > 0) {
            const node = stack.pop() as string;
            if (seen.has(node)) continue;
            seen.add(node);
            stack.push(...(parents.get(node) ?? []));
          }
          return seen;
        };
        /** The nodes a split's own name names: a layer prefix, or a family. */
        const seedsOf = (split: string): string[] => {
          const layer = /^(.*)\[([A-Za-z_][A-Za-z0-9_]*)<=(-?\d+)\]$/.exec(split);
          if (layer !== null) {
            const [, composition, index, bound] = layer as unknown as string[];
            const at_ = new RegExp(`[\\[,]${index as string}=(-?\\d+)[,\\]]`);
            return Object.keys(nodes).filter((node) => {
              if (!node.startsWith(`${composition as string}/`)) return false;
              const found = at_.exec(node);
              return found !== null && Number(found[1]) <= Number(bound);
            });
          }
          expect(split.startsWith('family:'), split).toBe(true);
          return Object.keys(nodes).filter((node) =>
            names(at(nodes[node] as PyValue, 'families')).includes(split.slice(7)),
          );
        };

        const previous = new Map<string, Set<string>>();
        for (const split of rows(options(one), 'graph_splits')) {
          const name = pyStr(split['graph_split'] as PyValue);
          const block = names(split['block'] as PyValue);
          const held = new Set(block);
          expect(held.size, `${one.name} ${name}`).toBe(block.length);
          expect([...held].sort(), `${one.name} ${name}`).toEqual(
            [...closure(seedsOf(name))].sort(),
          );
          expect(BigInt(block.length), `${one.name} ${name}`).toBe(
            (split['sizes'] as readonly PyValue[])[0],
          );
          expect(block, `${one.name} ${name}`).toEqual(
            [...block].sort((left, right) => (place.get(left) ?? -1) - (place.get(right) ?? -1)),
          );
          // No edge enters the block: every crossing edge leaves it.
          expect(
            endpoints.filter(([from, to]) => held.has(to as string) && !held.has(from as string)),
            `${one.name} ${name}`,
          ).toEqual([]);
          // The layer splits of one composition nest.
          if (split['kind'] === 'layer') {
            const composition = name.slice(0, name.indexOf('['));
            for (const node of previous.get(composition) ?? []) {
              expect(held.has(node), `${one.name} ${name} ⊇ ${composition}`).toBe(true);
            }
            previous.set(composition, held);
          }
        }
      }
    },
  );

  it.skipIf(!generated)(
    'separates exactly D4’s identities with members on both sides, with their far-side history',
    { timeout: 300_000 },
    () => {
      // `tests/run_derived.py`'s third D6 claim, recomputed from D4 and the block: an identity is
      // separated when its members fall on both sides, the writer's side decides which side is
      // "far", and `history_needed_by` is every far member but the writer — for a `window`
      // identity alone, since nothing an `append` or a `fixed` state was given is ever dropped.
      for (const one of derivedCorpus()) {
        const states = rows(d4(one.derivation.graph, libraryOf(one)), 'states');
        for (const split of rows(options(one), 'graph_splits')) {
          const block = new Set(names(split['block'] as PyValue));
          const expected: PyRecord[] = [];
          for (const state of states) {
            const members = names(state['members'] as PyValue);
            const first = members.filter((member) => block.has(nodeOf(member)));
            const second = members.filter((member) => !block.has(nodeOf(member)));
            if (first.length === 0 || second.length === 0) continue;
            const writer = state['writer'] as string | null;
            const writerFirst = writer === null || block.has(nodeOf(writer));
            const far = writerFirst ? second : first;
            expected.push({
              identity: state['identity'] as PyValue,
              evolution: state['evolution'] as PyValue,
              span: state['span'] as PyValue,
              bytes_per_cached_position: state['bytes_per_cached_position'] as PyValue,
              sharing: state['sharing'] as PyValue,
              writer,
              writer_side: writerFirst ? 'first' : 'second',
              first,
              second,
              history_needed_by:
                state['evolution'] === 'window' ? far.filter((member) => member !== writer) : [],
            });
          }
          expected.sort((left, right) =>
            (left['identity'] as string) < (right['identity'] as string) ? -1 : 1,
          );
          expect(split['separated_states'], `${one.name} ${pyStr(split['graph_split'] as PyValue)}`)
            .toEqual(expected);
        }
      }
    },
  );

  it.skipIf(!generated)(
    'reads gemma3n’s two shared identities exactly where they are separated',
    { timeout: 300_000 },
    () => {
      // `tests/run_derived.py`'s two named gemma3n claims. The sliding ring is written at layer 18
      // and read by nine later layers, so every layer split from 18 to 27 separates it, its writer
      // on the first side and every far-side reader in `history_needed_by` — and no other split of
      // the document does. The full cache is written at layer 19 and is an `append` identity, so
      // the splits from 19 to 28 separate it with nobody needing history.
      const one = documents().get('gemma3n-kvshare');
      expect(one, 'gemma3n-kvshare is in the corpus').toBeDefined();
      const separated = new Map<string, Map<string, PyRecord>>();
      for (const split of rows(options(one as Case), 'graph_splits')) {
        separated.set(
          pyStr(split['graph_split'] as PyValue),
          new Map(
            (split['separated_states'] as readonly PyRecord[]).map((state) => [
              pyStr(state['identity'] as PyValue),
              state,
            ]),
          ),
        );
      }
      const sliding = Array.from({ length: 10 }, (_, k) => `decoder[layer<=${18 + k}]`);
      const full = Array.from({ length: 10 }, (_, k) => `decoder[layer<=${19 + k}]`);

      for (const split of sliding) {
        const state = separated.get(split)?.get('shared.sliding.kv');
        expect(state, split).toBeDefined();
        expect((state as PyRecord)['writer_side'], split).toBe('first');
        expect((state as PyRecord)['history_needed_by'], split).toEqual(
          (state as PyRecord)['second'],
        );
      }
      expect(
        [...separated].filter(([, held]) => held.has('shared.sliding.kv')).map(([name]) => name),
      ).toEqual(sliding);
      for (const split of full) {
        const state = separated.get(split)?.get('shared.full.kv');
        expect(state, split).toBeDefined();
        expect((state as PyRecord)['history_needed_by'], split).toEqual([]);
      }
    },
  );

  it.skipIf(!generated)(
    'gives llama3-8b no separated state, no information loss and its KV-group granularity',
    { timeout: 300_000 },
    () => {
      // The script's llama3-8b claims: every identity has one member, so no split separates one;
      // every flattened axis of the primitives it instantiates declares its factors; the head
      // partition keeps whole KV groups — 32 / 8 = 4 — every other partition of the document has a
      // granularity of one, and each lists its communications, the embedding's two.
      const one = documents().get('llama3-8b');
      expect(one, 'llama3-8b is in the corpus').toBeDefined();
      const answer = options(one as Case);
      for (const split of rows(answer, 'graph_splits')) {
        expect(split['separated_states'], pyStr(split['graph_split'] as PyValue)).toEqual([]);
      }
      expect(rows(answer, 'information_loss')).toEqual([]);

      const partitions = rows(answer, 'partition_options');
      const heads = partitions.find(
        (row) =>
          row['node'] === 'decoder/attn[layer=0]' &&
          pyStr((row['target'] as PyRecord)['argument_axis'] as PyValue) === 'attention.heads',
      );
      expect(heads, 'the head partition of the first layer').toBeDefined();
      expect((heads as PyRecord)['granularity']).toBe(4n);
      expect((heads as PyRecord)['communication']).toEqual(['all_reduce']);
      for (const row of partitions) {
        expect(Array.isArray(row['communication']), pyStr(row['node'] as PyValue)).toBe(true);
        const target = row['target'] as PyRecord;
        const heads_ = target['argument_axis'] === 'attention.heads';
        expect(row['granularity'], pyStr(row['node'] as PyValue)).toBe(heads_ ? 4n : 1n);
      }
      const vocabulary = partitions.find(
        (row) => row['node'] === 'embed' && (row['target'] as PyRecord)['argument_axis'] === 'model.vocabulary',
      );
      expect(vocabulary, 'the embedding’s vocabulary partition').toBeDefined();
      expect((vocabulary as PyRecord)['communication']).toEqual(['all_gather', 'all_reduce']);
    },
  );

  it.skipIf(!generated)(
    'counts the partition options and losses the script names for gemma3n and qwen3.5-35b-a3b',
    { timeout: 300_000 },
    () => {
      // The two figures `tests/run_derived.py` pins beside its D3 claims, and the only place the
      // repository states a D6 total of its own.
      const held = documents();
      const gemma = options(held.get('gemma3n-kvshare') as Case);
      expect([
        rows(gemma, 'partition_options').length,
        rows(gemma, 'information_loss').length,
      ]).toEqual([575, 32]);
      expect(rows(options(held.get('qwen3.5-35b-a3b') as Case), 'partition_options')).toHaveLength(
        656,
      );
    },
  );

  it.skipIf(!generated)(
    'reaches the targets, communications, granularities and separations the corpus carries — and no others',
    { timeout: 300_000 },
    () => {
      // What the fifteen documents exercise of D6, measured off the recorded documents: all five
      // partition targets, all four communications and the one primitive that admits two, eight
      // distinct granularities and no blank, both kinds of split, two evolutions among the
      // separated identities and one writer side. What they never reach — a granularity that does
      // not resolve, a partition whose `when` fails, a writer on the second side, a `fixed` or
      // unruled identity separated, a writer that is not a member, a split D2 names and the
      // structural walk does not — is `test/derive/d6.test.ts`'s, over hand-built graphs.
      const seen = new Set<string>();
      for (const one of derivedCorpus()) {
        const product = at(one.expected, 'd6');
        for (const row of rows(product, 'partition_options')) {
          seen.add(`target:${Object.keys(row['target'] as PyRecord)[0] as string}`);
          seen.add(`communication:${names(row['communication'] as PyValue).join('+')}`);
          seen.add(row['granularity'] === null ? 'granularity blank' : 'granularity a number');
        }
        for (const row of rows(product, 'graph_splits')) {
          seen.add(`kind:${pyStr(row['kind'] as PyValue)}`);
          for (const state of row['separated_states'] as readonly PyRecord[]) {
            seen.add(`separated:${pyStr(state['evolution'] as PyValue)}`);
            seen.add(`writer_side:${pyStr(state['writer_side'] as PyValue)}`);
            seen.add(
              (state['history_needed_by'] as readonly PyValue[]).length > 0
                ? 'history needed'
                : 'no history needed',
            );
          }
        }
        if (rows(product, 'information_loss').length > 0) seen.add('information loss');
      }
      expect([...seen].sort()).toEqual([
        'communication:all_gather',
        'communication:all_gather+all_reduce',
        'communication:all_reduce',
        'communication:all_to_all',
        'communication:none',
        'granularity a number',
        'history needed',
        'information loss',
        'kind:family',
        'kind:layer',
        'no history needed',
        'separated:append',
        'separated:window',
        'target:any_axis',
        'target:argument_axis',
        'target:instance_key_axis',
        'target:none',
        'target:payload_axis',
        'writer_side:first',
      ]);
    },
  );
});

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { expect, it, suite } from 'vitest';

import {
  basesOf,
  consistent,
  d3,
  derivationGraph,
  expand,
  identOf,
  loadLibrary,
  parse,
  pyStr,
  serialize,
  toJsonValue,
  toPython,
  type Derivation,
  type ExpandedGraph,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of D3 (feature 1.8a): the Derived Parameter Tensor Inventory of §7, as `derive.d3`
// writes it.
//
// The expectation is the repository's own derived document — `out/derive/<name>.derived.json`,
// what `--derive` wrote for the corpus and for the template under its documented assignment — and
// the comparison is over its `d3` member, twice: the readings deep-equal, then the two written
// with the core's serializer and compared as text. The reading decides the structure *and* the
// integer/float distinction, because `toPython` keeps it (a `bf16` tensor's `bytes` is an integer
// and a `fp4` tensor's a float, and `1024n` is not `1024`); the text names the line that moved.
//
// The assignment is read from the derived document itself, which records it (§7), so the whole
// expectation is one file and the external quantities keep their lexemes: `3072` a whole number,
// `1e-05` a real.
//
// Beside it, the **expansion** D3 and every later product are written over. `_expand` rebuilds the
// validator's graph with every template instance expanded in place (§5.1), and four of its answers
// are D1's own, node for node: the identifiers of `resolved`, the rewired `edges`, and the public
// interfaces resolved to instance ports. D1 is already compared byte for byte against `--d1`
// (feature 1.7), so requiring the expansion to agree with it holds the parts of `_expand` that D3
// does not read to a fixture that is the tools' own output. Its `order` is *not* D1's — the
// validator's Kahn and the emitter's differ in their successors' order (feature 1.7), and
// `gemma3n-kvshare` is where that shows — so it is held to what it is: a permutation of the nodes
// in which every edge points forward.
//
// `consistent` is the third thing this suite proves: `derive.products` runs both expansions and
// requires them to agree node by node, and it is run here on every corpus document.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);
const libraries = new Map<string, Library>();

function libraryFor(bases: readonly string[]): Library {
  const key = bases.join('|');
  const held = libraries.get(key);
  if (held !== undefined) return held;
  const library = loadLibrary(bases, { schemas, source });
  libraries.set(key, library);
  return library;
}

/** The bases a corpus document declares, as `load_for` resolves them. */
function declaredBases(path: string, document: PyValue): string[] {
  return [...basesOf(path, document).bases];
}

/** A member of a record read from a document, or a failure naming it. */
function at(value: PyValue, name: string): PyValue {
  const held = (value as PyRecord)[name];
  expect(held, `the recorded document has no '${name}'`).toBeDefined();
  return held as PyValue;
}

/** One corpus document, derived by the core, beside the document the tools derived. */
interface Case {
  name: string;
  path: string;
  /** `--derive`'s own output, read as a document is read: `1e-05` a float, `4096` a whole number. */
  expected: PyValue;
  derivation: Derivation;
}

const cases: Case[] = [];

function corpus(): readonly Case[] {
  if (cases.length > 0) return cases;
  for (const one of readOracleManifest().documents) {
    const expected = toPython(parse(readFileSync(join(oracleOut, one.derived), 'utf8')));
    const assignment = at(expected, 'assignment') as PyRecord;
    const text = readFileSync(join(repositoryRoot, one.path), 'utf8');
    const tree = parse(text);
    const library = libraryFor(declaredBases(one.path, toPython(tree)));
    cases.push({
      name: one.name,
      path: one.path,
      expected,
      derivation: derivationGraph(tree, { schemas, library, assignment }),
    });
  }
  return cases;
}

/** The library the document was derived under, by name. */
function libraryOf(one: Case): Library {
  return libraryFor(declaredBases(one.path, one.derivation.document));
}

/** The two comparisons, in the order that makes a failure readable. */
function agrees(answer: PyValue, expected: PyValue, where: string): void {
  expect(answer, where).toEqual(expected);
  expect(serialize(toJsonValue(answer)), where).toBe(serialize(toJsonValue(expected)));
}

/** D1, as the derived document carries it. */
function d1Of(one: Case): PyValue {
  return at(one.expected, 'd1');
}

/** `node.port`, as D1 writes an endpoint. */
function endpoint(value: PyValue): string {
  return `${pyStr(at(value, 'node'))}.${pyStr(at(value, 'port'))}`;
}

/** The expanded graph's edges as D1 lists them: `(rule, from, to)`, sorted. */
function edgesOf(graph: ExpandedGraph): string[] {
  return graph.edges
    .map(
      (edge) =>
        `${edge.binding} ${identOf(edge.from)}.${pyStr(edge.fromPort)} -> ` +
        `${identOf(edge.to)}.${pyStr(edge.toPort)}`,
    )
    .sort();
}

suite('D3 against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)(
    'derives the parameter tensor inventory of every corpus document',
    { timeout: 300_000 },
    () => {
      const documents = corpus();
      expect(documents).toHaveLength(15);
      let tensors = 0;
      for (const one of documents) {
        const answer = d3(one.derivation.graph, libraryOf(one));
        agrees(answer, at(one.expected, 'd3'), one.path);
        tensors += (at(answer, 'tensors') as readonly PyValue[]).length;
      }
      // 11 838 identity instances over the fifteen documents, which is what the corpus holds.
      expect(tensors).toBeGreaterThan(10_000);
    },
  );

  it.skipIf(!generated)(
    'reaches every branch of the inventory the corpus carries',
    { timeout: 300_000 },
    () => {
      // Stated so that a corpus that stopped exercising one of them is a failure here rather than
      // a silent hole: the three dtype selectors, a tied tensor, a declared multiplicity and its
      // storage axis, the three location forms the corpus writes, a sparsity unit, and a declared
      // decomposition (O5.10). The figures are the recorded documents' own.
      const seen = new Set<string>();
      for (const one of corpus()) {
        for (const row of at(at(one.expected, 'd3'), 'tensors') as readonly PyValue[]) {
          if ((row as PyRecord)['tied'] === true) seen.add('tied');
          if ((row as PyRecord)['sparsity'] !== undefined) seen.add('sparsity');
          const location = (row as PyRecord)['location'];
          if (location !== undefined) seen.add(`location:${Object.keys(location as PyRecord)[0]}`);
          for (const axis of at(row, 'shape') as readonly PyValue[]) {
            if ((axis as PyRecord)['factors'] !== undefined) seen.add('factors');
            if ((axis as PyRecord)['axis'] === 'storage.multiplicity') seen.add('storage axis');
          }
          if ((row as PyRecord)['multiplicity'] !== 1n) seen.add('multiplicity');
        }
      }
      expect([...seen].sort()).toEqual([
        'factors',
        'location:slice',
        'location:stack',
        'location:tensor',
        'multiplicity',
        'sparsity',
        'storage axis',
        'tied',
      ]);
    },
  );

  it.skipIf(!generated)(
    'expands the analysis onto D1’s own nodes, edges and interfaces',
    { timeout: 300_000 },
    () => {
      for (const one of corpus()) {
        const graph = one.derivation.graph;
        const d1 = d1Of(one);
        const nodes = at(d1, 'nodes') as PyRecord;
        expect([...graph.resolved.values()].map((node) => identOf(node.site)).sort(), one.name)
          .toEqual(Object.keys(nodes).sort());

        const edges = (at(d1, 'edges') as readonly PyValue[])
          .map((edge) => {
            const from = endpoint(at(edge, 'from'));
            const to = endpoint(at(edge, 'to'));
            return `${pyStr(at(edge, 'rule'))} ${from} -> ${to}`;
          })
          .sort();
        expect(edgesOf(graph), one.name).toEqual(edges);

        const interfaces = at(d1, 'interfaces');
        const inputs = at(interfaces, 'inputs') as PyRecord;
        for (const [name, declared] of Object.entries(inputs)) {
          expect(
            (graph.inputsAt.get(name) ?? []).map(
              (port) => `${identOf(port.site)}.${pyStr(port.port)}`,
            ),
            `${one.name}: input ${name}`,
          ).toEqual((at(declared, 'to') as readonly PyValue[]).map(endpoint));
        }
        const outputs = at(interfaces, 'outputs') as PyRecord;
        for (const [name, declared] of Object.entries(outputs)) {
          const port = graph.outputsAt.get(name);
          expect(
            port === undefined ? null : `${identOf(port.site)}.${pyStr(port.port)}`,
            `${one.name}: output ${name}`,
          ).toBe(endpoint(declared));
        }

        // The families D1 lists per node are the expansion's `meta`, sorted; `meta` and
        // `resolved` are keyed alike, so the join is the key itself.
        for (const [key, node] of graph.resolved) {
          const meta = graph.meta.get(key);
          expect(meta, `${one.name}: meta of ${identOf(node.site)}`).toBeDefined();
          expect([...(meta as { families: ReadonlySet<string> }).families].sort(), identOf(node.site))
            .toEqual(
              (at(nodes[identOf(node.site)] as PyValue, 'families') as readonly PyValue[])
                .map(String)
                .sort(),
            );
        }
      }
    },
  );

  it.skipIf(!generated)(
    'orders the expanded graph so that every edge points forward',
    { timeout: 300_000 },
    () => {
      for (const one of corpus()) {
        const graph = one.derivation.graph;
        const order = graph.order.map((site) => identOf(site));
        expect(new Set(order).size, one.name).toBe(order.length);
        expect([...order].sort(), one.name).toEqual(
          [...graph.resolved.values()].map((node) => identOf(node.site)).sort(),
        );
        const at_ = new Map(order.map((name, index) => [name, index]));
        for (const edge of graph.edges) {
          expect(
            (at_.get(identOf(edge.from)) as number) < (at_.get(identOf(edge.to)) as number),
            `${one.name}: ${edge.binding}`,
          ).toBe(true);
        }
      }
    },
  );

  it.skipIf(!generated)(
    'requires D1 and the validator to resolve every node identically',
    { timeout: 300_000 },
    () => {
      for (const one of corpus()) {
        const emitted = expand(one.derivation.document, libraryOf(one), {
          assignment: at(one.expected, 'assignment') as PyRecord,
        });
        consistent(at(at(emitted, 'd1'), 'nodes'), one.derivation.graph);
      }
    },
  );
});

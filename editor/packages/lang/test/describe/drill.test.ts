import { describe, expect, it } from 'vitest';

import {
  alternationCount,
  complementOf,
  compositionMove,
  drillGraph,
  drillPresence,
  expand,
  foldedGraph,
  isJsonObject,
  parse,
  previousIteration,
  pyStr,
  reachedIteration,
  scopedEndpoint,
  serialize,
  toPython,
  whereOfSite,
  generatedSite,
  derive,
  describe as describeDocument,
  type DrillGraph,
  type DrillPresence,
  type JsonObject,
  type JsonValue,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { corpus, library, schemas } from './source.js';

/**
 * The drill-in reading of §4.8, held to the documents it draws and to D1 beside it.
 *
 * Four things are asserted, and they are the four a component cannot check for itself:
 *
 *  - **the drawing is the document's** — six sites, eight scoped edges, one ghost column with the
 *    two carry rules through it and two pinned terminals on `llama3-8b`'s `decoder`, which is what
 *    artboard S4 draws;
 *  - **presence is D1's** — `attn` 24 of 30 and `attn_full` 6 of 30 on `gemma3n-kvshare`, counted
 *    over the expanded graph and not over a guard the strip evaluated for itself;
 *  - **the two readings agree** — a guarded site is in D1 exactly where its guard is true, and a
 *    scoped edge is present exactly where D1 holds an edge of that rule between the two nodes the
 *    rule names there. That is the cross-check the design turns on: the strip reads D1, the badge
 *    reads the evaluator, and neither may say what the other denies;
 *  - **§4.20's move keeps the document valid** — a three-instance chain extracted into a
 *    composition validates and derives, with D3's totals unmoved.
 */

const MODELS = [
  'llama3-8b',
  'qwen3.5-4b-text',
  'gemma3n-kvshare',
  'deepseek-v4-pro',
  'whisper-large-v3',
  'colbert-v2',
  'llama4-scout',
  'qwen3.5-35b-a3b',
] as const;

/** One corpus document's drill-in of one composition. */
function drillOf(model: string, composition: string): DrillGraph {
  const graph = drillGraph(parse(corpus(model)), { composition });
  if (graph === null) throw new Error(`${model} declares no composition ${composition}`);
  return graph;
}

/** D1 of one corpus document, through the core's own expansion (parity holds it to `d1.py`). */
function expansionOf(model: string): PyRecord {
  return expand(toPython(parse(corpus(model))), library);
}

function presenceOf(model: string, composition: string): DrillPresence {
  return drillPresence(drillOf(model, composition), expansionOf(model));
}

describe('the drill-in of llama3-8b › decoder (S4)', () => {
  const drill = drillOf('llama3-8b', 'decoder');

  it('draws the composition’s own six sites, in the document’s order', () => {
    expect(drill.sites.map((site) => site.name)).toEqual([
      'attn_n',
      'attn',
      'attn_r',
      'ffn_n',
      'ffn',
      'ffn_r',
    ]);
    expect(drill.name).toBe('decoder');
    expect(drill.pointer).toBe('/compositions/decoder');
    expect(drill.families).toEqual(['decoder']);
  });

  it('carries the index strip’s bounds as the document writes them, with the scrubber’s track', () => {
    expect(drill.ranges).toHaveLength(1);
    const [layer] = drill.ranges;
    expect(layer?.name).toBe('layer');
    // Erratum E10: `stop` is the literal 32, and only the template names a quantity there.
    expect(serialize(layer?.stop as JsonValue)).toBe('{\n  "literal": 32\n}\n');
    expect(layer?.values).toHaveLength(32);
    expect(layer?.values?.[0]).toBe(0n);
    expect(layer?.values?.[31]).toBe(31n);
    expect(drill.points).toHaveLength(32);
    expect(drill.points[0]?.label).toBe('layer=0');
    expect(drill.count).toBe(32n);
  });

  it('draws the eight scoped edges, two of them guarded, with the written place of each', () => {
    const scoped = drill.edges.filter((edge) => edge.scoped);
    expect(scoped.map((edge) => edge.rule)).toEqual([
      'attn_n.carry',
      'attn_r.a_carry',
      'attn.norm_in',
      'attn_r.b',
      'ffn_n.in',
      'ffn_r.a',
      'ffn.norm_in',
      'ffn_r.b',
    ]);
    const carry = scoped[0];
    // The place is the **written** one: §5.2 rule 7 calls the rule `decoder.attn_n.carry`, and the
    // file has no such member (feature 2.8's hoisting, read the other way round).
    expect(carry?.pointer).toBe('/compositions/decoder/bindings/values/attn_n.carry');
    expect(carry?.hoisted).toBe('decoder.attn_n.carry');
    expect(scoped.filter((edge) => edge.guard !== null).map((edge) => edge.rule)).toEqual([
      'attn_n.carry',
      'attn_r.a_carry',
    ]);
  });

  it('draws one ghost column, on the left, with the two carry rules through it (§4.8)', () => {
    expect(drill.ghosts).toHaveLength(1);
    const [ghost] = drill.ghosts;
    expect(ghost?.name).toBe('ffn_r');
    expect(ghost?.side).toBe('left');
    expect(ghost?.site).toBe('/compositions/decoder/instances/ffn_r');
    expect(ghost?.edges).toEqual([
      '/compositions/decoder/bindings/values/attn_n.carry',
      '/compositions/decoder/bindings/values/attn_r.a_carry',
    ]);
    // Ghosts are per (site, override expression): both rules override `layer` the same way, so
    // there is one column and not two.
    expect(ghost?.indices.map((one) => one.name)).toEqual(['layer']);
  });

  it('pins two boundary terminals, the entry carrying both rules that enter (S4)', () => {
    expect(drill.terminals.map((one) => [one.side, one.outside])).toEqual([
      ['left', 'embed.output'],
      ['right', 'final_n.input'],
    ]);
    const [entry, exit] = drill.terminals;
    expect(entry?.links.map((link) => `${link.rule} → ${link.name}.${link.port}`)).toEqual([
      'decoder.entry → attn_n.input',
      'decoder.entry.a → attn_r.a',
    ]);
    expect(exit?.links.map((link) => `${link.rule} ← ${link.name}.${link.port}`)).toEqual([
      'final_n.in ← ffn_r.output',
    ]);
  });

  it('dims the carry edges and their guard at layer 0, and keeps them at layer 1', () => {
    const presence = presenceOf('llama3-8b', 'decoder');
    const carry = '/compositions/decoder/bindings/values/attn_n.carry';
    expect(presence.edges.get(carry)?.has('layer=0')).toBe(false);
    expect(presence.edges.get(carry)?.has('layer=1')).toBe(true);
    expect(presence.edges.get(carry)?.size).toBe(31);
    expect(presence.guards.get(carry)?.get('layer=0')).toBe(false);
    expect(presence.guards.get(carry)?.get('layer=1')).toBe(true);
    // No site of this composition is guarded, so every one of the six is present at every index.
    for (const site of drill.sites) {
      expect(presence.sites.get(site.name)?.size).toBe(32);
      expect(presence.guards.has(site.name)).toBe(false);
    }
    // The boundary rules belong to the iteration their inside end names.
    expect([...(presence.edges.get('/bindings/values/decoder.entry') ?? [])]).toEqual(['layer=0']);
    expect([...(presence.edges.get('/bindings/values/final_n.in') ?? [])]).toEqual(['layer=31']);
  });
});

describe('the alternation strip of gemma3n-kvshare › decoder (S5)', () => {
  const drill = drillOf('gemma3n-kvshare', 'decoder');
  const presence = presenceOf('gemma3n-kvshare', 'decoder');

  it('counts what D1 emitted, site by site', () => {
    expect(presence.points).toHaveLength(30);
    expect(alternationCount(presence, 'attn')).toEqual({ at: 24, of: 30 });
    expect(alternationCount(presence, 'attn_full')).toEqual({ at: 6, of: 30 });
    expect(alternationCount(presence, 'ffn_sparse')).toEqual({ at: 10, of: 30 });
    expect(alternationCount(presence, 'ffn')).toEqual({ at: 20, of: 30 });
  });

  it('lists four guarded sites of seventeen, the other thirteen present at every index', () => {
    const guarded = drill.sites.filter((site) => site.guard !== null).map((site) => site.name);
    expect(guarded).toEqual(['attn', 'ffn_sparse', 'attn_full', 'ffn']);
    expect(drill.sites).toHaveLength(17);
    for (const site of drill.sites) {
      if (guarded.includes(site.name)) continue;
      expect(alternationCount(presence, site.name)).toEqual({ at: 30, of: 30 });
    }
  });

  it('says at which index each guard holds, and it is the index D1 emitted the node at', () => {
    for (const name of ['attn', 'attn_full', 'ffn_sparse', 'ffn']) {
      const truths = presence.guards.get(name);
      const present = presence.sites.get(name);
      expect(truths).toBeDefined();
      for (const point of presence.points) {
        expect(truths?.get(point.label)).toBe(present?.has(point.label));
      }
    }
  });
});

describe('the two readings, over the corpus', () => {
  // The strip reads D1 and the badge reads the evaluator; this is what says they agree. A site
  // with no guard is present at every point, a guarded one exactly where its guard holds — which
  // is §5.2 rule 3 read from both ends.
  it.each(MODELS)('%s: a site is in D1 exactly where its guard is true', (model) => {
    const tree = parse(corpus(model));
    const folded = foldedGraph(tree);
    const expansion = expand(toPython(tree), library);
    for (const node of folded.nodes) {
      if (node.kind !== 'composition') continue;
      const drill = drillGraph(tree, { composition: node.name, folded });
      expect(drill).not.toBeNull();
      const presence = drillPresence(drill as DrillGraph, expansion);
      for (const site of (drill as DrillGraph).sites) {
        const present = presence.sites.get(site.name);
        expect(present).toBeDefined();
        if (site.guard === null) {
          expect(present?.size).toBe(presence.points.length);
          continue;
        }
        const truths = presence.guards.get(site.name);
        for (const point of presence.points) {
          expect(truths?.get(point.label)).toBe(present?.has(point.label));
        }
      }
    }
  });

  it.each(MODELS)('%s: a scoped edge is present exactly where D1 emitted it', (model) => {
    const tree = parse(corpus(model));
    const folded = foldedGraph(tree);
    const expansion = expand(toPython(tree), library);
    const emitted = edgeKeys(expansion);
    for (const node of folded.nodes) {
      if (node.kind !== 'composition') continue;
      const drill = drillGraph(tree, { composition: node.name, folded }) as DrillGraph;
      const presence = drillPresence(drill, expansion);
      for (const edge of drill.edges) {
        if (!edge.scoped) continue;
        const present = presence.edges.get(edge.pointer);
        expect(present).toBeDefined();
        // D1 lists the rule under the name §5.2 rule 7 hoists it to; every edge it emitted for
        // that rule belongs to one point of the grid, and that set must be the set the drill-in
        // dims at. The point of an emitted edge is read off the end that writes no override.
        const fromD1 = pointsOfRule(drill, emitted, edge.hoisted);
        expect([...(present ?? [])].sort()).toEqual([...fromD1].sort());
      }
    }
  });
});

/** Every edge D1 emitted, as `rule` → the pairs of node identifiers. */
function edgeKeys(expansion: PyValue): Map<string, { from: string; to: string }[]> {
  const found = new Map<string, { from: string; to: string }[]>();
  const graph = (expansion as Record<string, PyValue>)['d1'] as Record<string, PyValue>;
  for (const edge of graph['edges'] as PyValue[]) {
    const one = edge as Record<string, PyValue>;
    const rule = pyStr(one['rule'] as PyValue);
    const ends = {
      from: pyStr((one['from'] as Record<string, PyValue>)['node'] as PyValue),
      to: pyStr((one['to'] as Record<string, PyValue>)['node'] as PyValue),
    };
    found.set(rule, [...(found.get(rule) ?? []), ends]);
  }
  return found;
}

/**
 * The points of the grid one rule's emitted edges belong to.
 *
 * An emitted edge carries no environment, so the point is read off the endpoint that writes no
 * override: a site endpoint with no `indices` is the current iteration, which is the very
 * environment `d1.emit` unrolled the rule in. Both ends overridden is admitted by the grammar and
 * written by no corpus document; the reading falls back to every index the edge touches, and the
 * suite says so rather than pretending it decided.
 */
function pointsOfRule(
  drill: DrillGraph,
  emitted: Map<string, { from: string; to: string }[]>,
  rule: string,
): Set<string> {
  const edge = drill.edges.find((one) => one.hoisted === rule && one.scoped);
  const points = new Set<string>();
  if (edge === undefined) return points;
  const plain = [edge.from, edge.to].find((end) => end !== null && end.site !== null && !end.overridden);
  for (const ends of emitted.get(rule) ?? []) {
    const node = plain === edge.from ? ends.from : ends.to;
    const at = node.slice(node.indexOf('[') + 1, node.lastIndexOf(']'));
    points.add(at);
  }
  return points;
}

describe('§4.20’s move: Extract to Composition', () => {
  /** A three-instance chain: `a → b → c`, each a `norm.rms`, with an input and an output. */
  function chain(): JsonObject {
    return parse(chainText) as JsonObject;
  }

  it('turns the selection into a composition whose sites are the instances', () => {
    const move = compositionMove(chain(), { instances: ['a', 'b', 'c'], composition: 'block', index: 'layer' });
    expect(move).not.toBeNull();
    expect(move?.created).toBe(true);
    expect(move?.index).toBe('layer');
    expect(move?.moved.map((one) => one.name)).toEqual(['a', 'b', 'c']);
    // The two edges among the selection move inside as scoped rules; the parameter rules follow
    // their members.
    expect(move?.absorbed.map((one) => `${one.map}/${one.name}`)).toEqual([
      'values/b.in',
      'values/c.in',
      'parameters/a.weight',
      'parameters/b.weight',
      'parameters/c.weight',
    ]);
    // The interfaces cross the boundary: the input enters at the first index, the output leaves at
    // the last.
    expect(move?.rewritten.map((one) => one.path.join('/'))).toEqual([
      'interfaces/inputs/in/to/0',
      'interfaces/outputs/out/from',
    ]);
  });

  it('writes a scoped endpoint where the rule named a root instance', () => {
    const move = compositionMove(chain(), { instances: ['a', 'b', 'c'], composition: 'block', index: 'layer' });
    const rule = move?.absorbed.find((one) => one.name === 'b.in');
    expect(serialize(rule?.value as JsonValue)).toBe(
      ['{', '  "from": {', '    "site": "a",', '    "port": "output"', '  },', '  "to": {', '    "site": "b",', '    "port": "input"', '  }', '}', ''].join('\n'),
    );
  });

  it('the extracted document validates and derives, with D3’s totals unmoved', () => {
    const before = chain();
    const after = applied(before, compositionMove(before, {
      instances: ['a', 'b', 'c'],
      composition: 'block',
      index: 'layer',
    }));
    const answer = describeDocument(after, { schemas, library });
    expect(answer.structural.map((one) => one.message)).toEqual([]);
    expect(answer.conforms).toBe(true);
    expect(answer.analysis?.problems ?? []).toEqual([]);
    const one = derive(before, { schemas, library });
    const other = derive(after, { schemas, library });
    expect(totals(other)).toEqual(totals(one));
  });
});

describe('the writers of §4.8 and §4.20', () => {
  it('writes the carry override §4.8 names, and the guard it proposes', () => {
    expect(serialize(previousIteration('layer'))).toBe(
      ['{', '  "op": "subtract",', '  "args": [', '    {', '      "index": "layer"', '    },', '    {', '      "literal": 1', '    }', '  ]', '}', ''].join('\n'),
    );
    expect(serialize(reachedIteration('layer'))).toBe(
      [
        '{',
        '  "compare": {',
        '    "operator": "greater_or_equal",',
        '    "left": {',
        '      "index": "layer"',
        '    },',
        '    "right": {',
        '      "literal": 1',
        '    }',
        '  }',
        '}',
        '',
      ].join('\n'),
    );
  });

  it('writes a site endpoint with the override, as the corpus writes one', () => {
    const written = scopedEndpoint('ffn_r', 'output', [
      { name: 'layer', written: previousIteration('layer') },
    ]);
    const corpusRule = parse(corpus('llama3-8b'));
    const rule = nodeOf(corpusRule, ['compositions', 'decoder', 'bindings', 'values', 'attn_n.carry', 'from']);
    expect(serialize(written)).toBe(serialize(rule));
  });

  it('complements a guard by negating it, never by rewriting the comparison', () => {
    const guard = nodeOf(parse(corpus('gemma3n-kvshare')), [
      'compositions',
      'decoder',
      'instances',
      'ffn_sparse',
      'when',
    ]);
    expect(serialize(complementOf(guard))).toContain('"not"');
    // The negation denotes the complement of the guard: `ffn_sparse` fires below ten, its
    // complement at ten and above, which is what `ffn` itself carries in the document.
    const drill = drillOf('gemma3n-kvshare', 'decoder');
    const presence = presenceOf('gemma3n-kvshare', 'decoder');
    const sparse = presence.guards.get('ffn_sparse');
    const dense = presence.guards.get('ffn');
    expect(drill.points).toHaveLength(30);
    for (const point of presence.points) {
      expect(dense?.get(point.label)).toBe(!(sparse?.get(point.label) ?? false));
    }
  });
});

/** The node at a path of a parsed document. */
function nodeOf(tree: JsonValue, path: readonly string[]): JsonValue {
  let node: JsonValue = tree;
  for (const step of path) {
    if (!isJsonObject(node)) throw new Error(`no node at ${path.join('/')}`);
    const found = node.members.find((one) => one.name === step);
    if (found === undefined) throw new Error(`no node at ${path.join('/')}`);
    node = found.value;
  }
  return node;
}

/** D3's and D4's own totals, which an extraction must not move. */
function totals(document: PyRecord): { parameters: PyValue; states: PyValue } {
  const read = (product: string): PyValue =>
    ((document[product] as Record<string, PyValue>)['totals'] ?? null);
  return { parameters: read('d3'), states: read('d4') };
}

/**
 * The document a move produces, applied here the way the store applies it (2.1's commands).
 *
 * The move answers *what* to write; a test that re-decided any of it would be testing itself. So
 * this is the mechanical half and nothing else: the composition is written, the instances move
 * into it, the absorbed rules move with them, and the rewritten places take their new value.
 */
function applied(tree: JsonObject, move: ReturnType<typeof compositionMove>): JsonObject {
  if (move === null) throw new Error('no move');
  const json = JSON.parse(JSON.stringify(tree)) as JsonObject;
  const object = (node: JsonValue, name: string): JsonObject => {
    const found = (node as JsonObject).members.find((one) => one.name === name);
    return found?.value as JsonObject;
  };
  const drop = (node: JsonObject, name: string): JsonObject => ({
    kind: 'object',
    members: node.members.filter((one) => one.name !== name),
  });
  const put = (node: JsonObject, name: string, value: JsonValue): JsonObject => ({
    kind: 'object',
    members: [...node.members.filter((one) => one.name !== name), { name, value }],
  });

  const sites: JsonObject = {
    kind: 'object',
    members: move.moved.map((one) => ({ name: one.name, value: one.value })),
  };
  const byMap = new Map<string, { name: string; value: JsonValue }[]>();
  for (const rule of move.absorbed) {
    byMap.set(rule.map, [...(byMap.get(rule.map) ?? []), { name: rule.name, value: rule.value }]);
  }
  const definition: JsonObject = {
    kind: 'object',
    members: [
      { name: 'indices', value: (move.definition as Record<string, JsonValue>)['indices'] as JsonValue },
      { name: 'families', value: (move.definition as Record<string, JsonValue>)['families'] as JsonValue },
      { name: 'instances', value: sites },
      {
        name: 'bindings',
        value: {
          kind: 'object',
          members: [...byMap].map(([name, rules]) => ({
            name,
            value: { kind: 'object', members: rules },
          })),
        },
      },
    ],
  };

  let root = json;
  // The instances leave the top level, the absorbed rules leave their maps.
  let instances = object(root, 'instances');
  for (const one of move.moved) instances = drop(instances, one.from);
  root = put(root, 'instances', instances);
  const bindings = object(root, 'bindings');
  let nextBindings = bindings;
  for (const rule of move.absorbed) {
    const map = rule.from[1] as string;
    nextBindings = put(nextBindings, map, drop(object(nextBindings, map), rule.from[2] as string));
  }
  root = put(root, 'bindings', nextBindings);
  root = put(root, 'compositions', put(object(root, 'compositions'), move.composition, definition));

  // The rewritten places take their new value, by path.
  for (const place of move.rewritten) {
    root = write(root, place.path, place.value);
  }
  return root;
}

/** Write one value at a path of a copied tree. */
function write(node: JsonValue, path: readonly (string | number)[], value: JsonValue): JsonObject {
  const [step, ...rest] = path;
  if (!isJsonObject(node) || typeof step !== 'string') throw new Error('no place to write at');
  const found = node.members.find((one) => one.name === step);
  const next: JsonValue =
    rest.length === 0
      ? value
      : Array.isArray(found?.value) && typeof rest[0] === 'number'
        ? found.value.map((item: JsonValue, at: number) =>
            at === rest[0] ? (rest.length === 1 ? value : write(item, rest.slice(1), value)) : item,
          )
        : write(found?.value ?? null, rest, value);
  return {
    kind: 'object',
    members: node.members.map((one) => (one.name === step ? { name: one.name, value: next } : one)),
  };
}

/** The synthetic chain: three `norm.rms` in a row, with an input, an output and three weights. */
const chainText = JSON.stringify(
  {
    schema: 'tensorspine/2.0',
    model: 'drill-chain',
    primitive_libraries: [{ base: '../../data/primitive-library/' }],
    quantities: {
      d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
    },
    constants: {},
    instances: Object.fromEntries(
      ['a', 'b', 'c'].map((name) => [
        name,
        {
          primitive: { name: 'norm.rms', version: '1.0.0' },
          arguments: { width: { quantity: 'd' }, eps: { literal: 1e-5 } },
          families: ['block'],
        },
      ]),
    ),
    compositions: {},
    bindings: {
      values: {
        'b.in': {
          from: { instance: { kind: 'root', instance: 'a' }, port: 'output' },
          to: { instance: { kind: 'root', instance: 'b' }, port: 'input' },
        },
        'c.in': {
          from: { instance: { kind: 'root', instance: 'b' }, port: 'output' },
          to: { instance: { kind: 'root', instance: 'c' }, port: 'input' },
        },
      },
      parameters: Object.fromEntries(
        ['a', 'b', 'c'].map((name) => [
          `${name}.weight`,
          {
            tensor: { name: `${name}.weight` },
            members: [{ instance: { kind: 'root', instance: name }, parameter: 'weight' }],
            dtype: 'bf16',
            location: { tensor: [`${name}.weight`] },
          },
        ]),
      ),
      constants: {},
      states: {},
    },
    interfaces: {
      inputs: {
        in: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' },
      },
      outputs: {
        out: {
          from: { instance: { kind: 'root', instance: 'c' }, port: 'output' },
          generative: false,
        },
      },
    },
  },
  null,
  2,
);

// The site identifier a drill-in names is the validator's own, so a suite that builds one by hand
// would be a second reading of §5.2 rule 2. This is the one place the test names it, to say that
// the reading above is keyed by what D1 lists.
it('keys presence by the identifier D1 lists a node under', () => {
  const drill = drillOf('llama3-8b', 'decoder');
  const point = drill.points[3];
  expect(point).toBeDefined();
  expect(whereOfSite(generatedSite('decoder', 'attn', point?.indices ?? []))).toBe(
    'decoder/attn[layer=3]',
  );
});

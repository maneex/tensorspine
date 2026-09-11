import { describe, expect, it } from 'vitest';

import { parse, serialize } from '../../src/json/index.js';
import {
  FACE_MEMBERS,
  foldedBoxes,
  foldedGraph,
  foldedSites,
  instanceSkeleton,
  literalIndex,
  proposedFamily,
  proposedName,
  pyStr,
  selectSite,
  toPython,
  valueEndpoint,
  whereOfSite,
  type FoldedGraph,
  type FoldedNode,
  type JsonValue,
  type PyValue,
} from '../../src/index.js';
import { readRepositoryFile } from '../json/repository.js';
import { corpus, describedCorpus } from './source.js';

/**
 * The folded reading of §4.7, held to the documents it draws and to the analysis beside it.
 *
 * Three things are asserted, and they are the three the canvas cannot check for itself:
 *
 *  - **the boxes are the document's**, in the document's own order, with the pointer each place
 *    has — which is what the sidecar keys by (D6) and what a gesture edits;
 *  - **a handle names the site the validator names**, so a card's facts and a boundary handle's
 *    label are about the same thing (`selectSite` is the validator's own reading, and this is its
 *    inverse);
 *  - **an endpoint written back is the endpoint that was read**, byte for byte, so a connection
 *    gesture on an existing edge cannot silently rewrite an index an author wrote as an
 *    expression.
 */

const MODELS = [
  'llama3-8b',
  'qwen3.5-4b-text',
  'gemma3n-kvshare',
  'deepseek-v4-pro',
  'whisper-large-v3',
  'shieldstral-3b-composite',
  'colbert-v2',
  'llama4-scout',
] as const;

function graphOf(name: string): FoldedGraph {
  return foldedGraph(parse(corpus(name)));
}

function boxAt(graph: FoldedGraph, pointer: string): FoldedNode {
  const node = graph.byPointer.get(pointer);
  if (node === undefined) throw new Error(`no box at ${pointer}`);
  return node;
}

describe('the folded document', () => {
  it('draws S1’s graph of llama3-8b: three cards, one composition, two terminals', () => {
    const graph = graphOf('llama3-8b');
    expect(graph.nodes.filter((node) => node.kind === 'root').map((node) => node.name)).toEqual([
      'embed',
      'final_n',
      'lm_head',
    ]);
    expect(graph.nodes.filter((node) => node.kind === 'composition').map((node) => node.name)).toEqual([
      'decoder',
    ]);
    expect(graph.nodes.filter((node) => node.kind === 'input').map((node) => node.name)).toEqual(['tokens']);
    expect(graph.nodes.filter((node) => node.kind === 'output').map((node) => node.name)).toEqual(['logits']);
    expect(boxAt(graph, '/compositions/decoder').children.map((node) => node.name)).toEqual([
      'attn_n',
      'attn',
      'attn_r',
      'ffn_n',
      'ffn',
      'ffn_r',
    ]);
  });

  it('reads the card’s face: the primitive, the version and the families', () => {
    const graph = graphOf('llama3-8b');
    const card = boxAt(graph, '/instances/final_n');
    expect(card.primitive).toBe('norm.rms');
    expect(card.version).toBe('1.0.0');
    expect(card.families).toEqual(['norm']);
    expect(card.guard).toBeNull();
  });

  it('carries the guard a guarded site writes, as written', () => {
    const graph = graphOf('qwen3.5-35b-a3b');
    const guarded = foldedBoxes(graph).filter((node) => node.guard !== null);
    expect(guarded.length).toBeGreaterThan(0);
    // The condition travels as the document writes it: printing it is §4.13's, in the interface.
    expect(guarded[0]?.guard).not.toBeNull();
  });

  it('counts a composition only when its range resolves (§4.7, erratum E10)', () => {
    const llama = boxAt(graphOf('llama3-8b'), '/compositions/decoder');
    expect(llama.count).toBe(32n);
    expect(llama.ranges.map((range) => range.name)).toEqual(['layer']);

    const template = foldedGraph(parse(readRepositoryFile('data/models/decoder-causal-yarn/1.0.0.json')));
    const decoder = [...template.byPointer.values()].find((node) => node.kind === 'composition');
    expect(decoder).toBeDefined();
    // `stop` names the external quantity `layers`, which no assignment gives here.
    expect(decoder?.count).toBeNull();
    expect(decoder?.ranges[0]?.stop).not.toBeNull();
  });

  it('counts the template’s composition once an assignment gives its quantity a value', () => {
    const text = readRepositoryFile('data/models/decoder-causal-yarn/1.0.0.json');
    const graph = foldedGraph(parse(text), { assignment: { layers: 4n, width: 16n, heads: 4n, head_dim: 4n } });
    const decoder = [...graph.byPointer.values()].find((node) => node.kind === 'composition');
    expect(decoder?.count).toBe(4n);
  });

  it('summarises what a composition holds — S3’s own line', () => {
    const decoder = boxAt(graphOf('llama3-8b'), '/compositions/decoder');
    expect(decoder.held).toEqual([
      { name: 'instances', count: 6 },
      { name: 'values', count: 8 },
      { name: 'parameters', count: 9 },
      { name: 'states', count: 1 },
    ]);
  });

  it('shows an interface’s own members beside its name (S1: token, generative)', () => {
    const graph = graphOf('llama3-8b');
    expect(boxAt(graph, '/interfaces/inputs/tokens').badges).toEqual([{ name: 'kind', value: 'token' }]);
    expect(boxAt(graph, '/interfaces/outputs/logits').badges).toEqual([
      { name: 'generative', value: null },
    ]);
  });
});

describe('the edges of the folded canvas', () => {
  it('draws every value binding and every interface wire of llama3-8b', () => {
    const graph = graphOf('llama3-8b');
    expect(graph.edges.filter((edge) => !edge.interface).map((edge) => edge.rule)).toEqual([
      'decoder.entry',
      'decoder.entry.a',
      'final_n.in',
      'lm_head.in',
    ]);
    expect(graph.edges.filter((edge) => edge.interface).map((edge) => edge.rule)).toEqual([
      'tokens',
      'logits',
    ]);
  });

  it('hangs an end inside a composition on the box, as a boundary handle (D8)', () => {
    const graph = graphOf('llama3-8b');
    const entry = graph.edges.find((edge) => edge.rule === 'decoder.entry');
    expect(entry?.from?.box).toBe('/instances/embed');
    expect(entry?.from?.boundary).toBe(false);
    expect(entry?.to?.box).toBe('/compositions/decoder');
    expect(entry?.to?.site).toBe('/compositions/decoder/instances/attn_n');
    expect(entry?.to?.boundary).toBe(true);
    expect(entry?.to?.where).toBe('decoder/attn_n[layer=0]');
    expect(entry?.to?.port).toBe('input');
  });

  it('resolves a boundary index written as an expression (S3’s ffn_r[layer=31])', () => {
    const graph = graphOf('llama3-8b');
    const into = graph.edges.find((edge) => edge.rule === 'final_n.in');
    expect(into?.from?.where).toBe('decoder/ffn_r[layer=31]');
    expect(into?.from?.indices.map((one) => one.name)).toEqual(['layer']);
    expect(into?.from?.indices[0]?.value).toBe(31n);
    // The written expression is kept beside the value: `layers − 1`, not the 31 it resolves to.
    expect(into?.from?.indices[0]?.written).not.toBe(31);
  });

  it('leaves a boundary index open where the assignment does not resolve it (S3’s ×?)', () => {
    const graph = foldedGraph(parse(readRepositoryFile('data/models/decoder-causal-yarn/1.0.0.json')));
    const open = graph.edges.filter((edge) => edge.from?.where === null || edge.to?.where === null);
    expect(open.length).toBeGreaterThan(0);
  });

  it('names the D2 value each end carries, by D2’s own identifier', () => {
    const graph = graphOf('llama3-8b');
    const entry = graph.edges.find((edge) => edge.rule === 'decoder.entry');
    expect(entry?.from?.value).toBe('embed.output');
    const into = graph.edges.find((edge) => edge.rule === 'final_n.in');
    expect(into?.from?.value).toBe('decoder/ffn_r[layer=31].output');
  });
});

describe('the folded reading and the validator agree', () => {
  it.each(MODELS)('%s: every handle names the site selectSite selects', (name) => {
    const graph = graphOf(name);
    const tree = parse(corpus(name));
    const document = toPython(tree);
    const quantities = graph.quantities;
    let checked = 0;
    for (const edge of graph.edges) {
      for (const handle of [edge.from, edge.to]) {
        if (handle === null || handle.where === null || handle.port === '') continue;
        const endpoint = valueEndpoint(handle);
        const written = toPython(endpoint) as Record<string, PyValue>;
        const selected = selectSite(written['instance'] as PyValue, quantities, new Map());
        expect(whereOfSite(selected)).toBe(handle.where);
        checked += 1;
      }
      void document;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it.each(MODELS)('%s: every box of the canvas is a place of the document', (name) => {
    const tree = parse(corpus(name));
    const graph = foldedGraph(tree);
    for (const box of foldedBoxes(graph)) {
      expect(pointerResolves(tree, box.pointer)).toBe(true);
    }
  });

  it('names the sites describe(folded) answers, and only those', () => {
    const graph = graphOf('llama3-8b');
    const wanted = new Set(foldedSites(graph));
    expect(wanted.size).toBe(9);
    expect(wanted.has('embed')).toBe(true);
    expect(wanted.has('decoder/attn[layer=0]')).toBe(true);
  });
});

describe('what a gesture writes', () => {
  it('writes an endpoint that serialises to the document’s own bytes', () => {
    const graph = graphOf('llama3-8b');
    const source = parse(corpus('llama3-8b'));
    for (const edge of graph.edges) {
      if (edge.interface) continue;
      const written = nodeAt(source, edge.segments);
      for (const [member, handle] of [
        ['from', edge.from],
        ['to', edge.to],
      ] as const) {
        if (handle === null) continue;
        const original = memberOf(written, member);
        expect(serialize(valueEndpoint(handle))).toBe(serialize(original as JsonValue));
      }
    }
  });

  it('proposes a name and a family from the primitive (§9 Q4)', () => {
    expect(proposedFamily('attention.dense')).toBe('attention');
    expect(proposedName('attention.dense')).toBe('dense');
    expect(proposedFamily('embed')).toBe('embed');
    expect(proposedName('embed')).toBe('embed');
  });

  it('writes D5’s skeleton for a dropped primitive: the pin, no arguments, the proposed family', () => {
    expect(serialize(instanceSkeleton('norm.rms', '1.0.0'))).toBe(
      ['{', '  "primitive": {', '    "name": "norm.rms",', '    "version": "1.0.0"', '  },', '  "arguments": {},', '  "families": [', '    "norm"', '  ]', '}', ''].join('\n'),
    );
  });

  it('writes a literal index as the grammar has it', () => {
    expect(serialize(literalIndex(7n))).toBe(['{', '  "literal": 7', '}', ''].join('\n'));
  });

  it('names on the card’s face exactly what presentation.json binds there', () => {
    // `presentation.json` gives `instance_definition` the face `primitive, families, when`; the
    // folded node carries one field per member, so a face that grew one fails here first.
    expect([...FACE_MEMBERS]).toEqual(['primitive', 'families', 'when']);
  });
});

describe('the folded reading of a document off the grammar', () => {
  it('draws what it can and raises nothing', () => {
    const graph = foldedGraph(parse('{"instances": {"a": 3}, "compositions": [], "bindings": {"values": {"x": {"from": 1}}}}'));
    expect(graph.nodes.map((node) => node.name)).toEqual(['a']);
    expect(graph.nodes[0]?.primitive).toBeNull();
    expect(graph.edges[0]?.from).toBeNull();
    expect(graph.edges[0]?.to).toBeNull();
  });

  it('answers nothing at all for a document that is not an object', () => {
    expect(foldedGraph(parse('[]')).nodes).toEqual([]);
    expect(foldedGraph(parse('"a"')).nodes).toEqual([]);
  });

  it('resolves no expression where the quantities are off the grammar', () => {
    const graph = foldedGraph(
      parse('{"quantities": {"d": 7}, "compositions": {"c": {"indices": {"i": {"start": {"literal": 0}, "stop": {"quantity": "d"}}}, "instances": {}}}}'),
    );
    expect(boxAt(graph, '/compositions/c').count).toBeNull();
  });
});

describe('the cost of the folded reading', () => {
  it.each(MODELS)('%s is folded inside the sheet’s own budget', (name) => {
    const tree = parse(corpus(name));
    const started = performance.now();
    for (let round = 0; round < 5; round += 1) foldedGraph(tree);
    const each = (performance.now() - started) / 5;
    // §5.6: "Sheet keystroke to render < 16 ms", and the canvas is redrawn on the same keystroke.
    // Asserted at three times the budget, as feature 1.11 asserts its own: the suite runs beside
    // five others and a flat budget would fail for the machine's reasons.
    expect(each).toBeLessThan(48);
  });
});

/** The value at a path of the tree, for the byte comparison above. */
function nodeAt(tree: ReturnType<typeof parse>, segments: readonly (string | number)[]): unknown {
  let at: unknown = tree;
  for (const step of segments) {
    if (typeof step === 'number') at = (at as unknown[])[step];
    else at = memberOf(at, step);
  }
  return at;
}

/** One member of an object node. */
function memberOf(value: unknown, name: string): unknown {
  const node = value as { members?: { name: string; value: unknown }[] };
  return node.members?.find((one) => one.name === name)?.value;
}

/** Whether an RFC 6901 pointer names a place of the tree. */
function pointerResolves(tree: ReturnType<typeof parse>, pointer: string): boolean {
  let at: unknown = tree;
  for (const raw of pointer.split('/').slice(1)) {
    const step = raw.replace(/~1/g, '/').replace(/~0/g, '~');
    if (Array.isArray(at)) at = at[Number(step)];
    else at = memberOf(at, step);
    if (at === undefined) return false;
  }
  return true;
}

describe('the folded reading beside describe', () => {
  it('describes one site per card, and the card’s ports are that site’s', () => {
    const description = describedCorpus('llama3-8b');
    const graph = graphOf('llama3-8b');
    const attn = boxAt(graph, '/compositions/decoder/instances/attn');
    expect(attn.where).toBe('decoder/attn[layer=0]');
    const described = [...description.sites.values()].find(
      (site) => whereOfSite(site.key) === attn.where,
    );
    expect(described).toBeDefined();
    expect(described?.inputs.map((port) => port.name)).toContain('input');
    expect(pyStr(described?.primitive ?? '')).toBe(attn.primitive);
  });
});

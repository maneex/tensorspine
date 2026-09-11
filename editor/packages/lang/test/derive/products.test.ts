import { describe, expect, it } from 'vitest';

import {
  consistent,
  d3,
  derivationGraph,
  expand,
  parse,
  serialize,
  toPython,
  type ExpandedGraph,
  type JsonValue,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { PyValueError } from '../../src/expr/errors.js';
import { corpus, library, schemas } from '../describe/source.js';

// What every derivation crosses (feature 1.8a): `derive.products`' two refusals, and the agreement
// between the two expansions the repository carries.
//
// "A document off the schema has no products, whatever asked for them" — so both stages of
// `--validate` are crossed here, in their order, and the refusal carries the validator's own first
// line. `_consistent` is the second: D1 resolves an instance's arguments *as written* where
// `analyse` resolves them typed (feature 1.7), and `products` requires the two to agree node by
// node rather than assuming it.

/** A corpus document with a few members changed, as a tree the grammar stage can read. */
function edited(name: string, edit: (document: Record<string, unknown>) => void): JsonValue {
  const document = JSON.parse(corpus(name)) as Record<string, unknown>;
  edit(document);
  return parse(JSON.stringify(document));
}

/** llama3-8b, derived. */
function llama(): { graph: ExpandedGraph; nodes: PyValue } {
  const tree = parse(corpus('llama3-8b'));
  const { document, graph } = derivationGraph(tree, { schemas, library });
  const emitted = expand(document, library);
  return { graph, nodes: (emitted['d1'] as PyRecord)['nodes'] as PyValue };
}

describe('the gate every derivation is behind', () => {
  it('derives a corpus document', () => {
    const { analysis, graph } = derivationGraph(parse(corpus('llama3-8b')), { schemas, library });
    expect(analysis.problems).toEqual([]);
    expect(graph.resolved.size).toBe(195);
  });

  it('refuses a document off the grammar with the schema stage’s own line', () => {
    const tree = edited('llama3-8b', (document) => {
      delete document['quantities'];
    });
    expect(() => derivationGraph(tree, { schemas, library })).toThrowError(PyValueError);
    expect(() => derivationGraph(tree, { schemas, library })).toThrowError(
      "not valid, no products: <root>: 'quantities' is a required property",
    );
  });

  it('refuses a document the meaning stage refuses, with its first line', () => {
    const tree = edited('llama3-8b', (document) => {
      const instances = document['instances'] as Record<string, Record<string, unknown>>;
      (instances['embed']?.['primitive'] as Record<string, unknown>)['name'] = 'embedz';
    });
    expect(() => derivationGraph(tree, { schemas, library })).toThrowError(
      'not valid, no products: [V1] primitive absent from primitive library: embedz',
    );
  });
});

describe('D1 and the validator, node by node', () => {
  it('agree on a corpus document', () => {
    const { graph, nodes } = llama();
    expect(() => consistent(nodes, graph)).not.toThrow();
  });

  it('name a node only one of them knows', () => {
    const { graph, nodes } = llama();
    const extra = { ...(nodes as PyRecord), zzz: { arguments: {} } };
    expect(() => consistent(extra, graph)).toThrowError(
      'zzz: emitted by D1, unknown to the validator',
    );
    const fewer = { ...(nodes as PyRecord) };
    delete (fewer as Record<string, unknown>)['decoder/attn[layer=0]'];
    expect(() => consistent(fewer, graph)).toThrowError(
      'decoder/attn[layer=0]: resolved by the validator, absent from D1',
    );
  });

  it('name the first argument they resolve differently, in code-point order', () => {
    const { graph, nodes } = llama();
    const node = (nodes as PyRecord)['decoder/attn[layer=0]'] as PyRecord;
    const changed = {
      ...(nodes as PyRecord),
      'decoder/attn[layer=0]': {
        ...node,
        arguments: { ...(node['arguments'] as PyRecord), cross: 'WRONG' },
      },
    };
    expect(() => consistent(changed, graph)).toThrowError(
      "decoder/attn[layer=0]: D1 and the validator resolve argument 'cross' differently — " +
        "'WRONG' against False",
    );
  });

  it('name an argument only one of them carries', () => {
    const { graph, nodes } = llama();
    const node = (nodes as PyRecord)['decoder/attn[layer=0]'] as PyRecord;
    const dropped: Record<string, PyValue> = { ...(node['arguments'] as PyRecord) };
    delete dropped['cross'];
    const changed = {
      ...(nodes as PyRecord),
      'decoder/attn[layer=0]': { ...node, arguments: dropped },
    };
    // `node['arguments'].get(k)` answers `None` for a name it has not, and `repr(None)` is `None`.
    expect(() => consistent(changed, graph)).toThrowError(
      "decoder/attn[layer=0]: D1 and the validator resolve argument 'cross' differently — " +
        'None against False',
    );
  });
});

describe('the reading the derivation takes', () => {
  it('answers the document as a value and writes into neither it nor the tree', () => {
    // The store holds one tree (plan D1) and the editor derives from it on every keystroke; a
    // derivation that edited it would edit what the author is looking at.
    const text = corpus('llama3-8b');
    const tree = parse(text);
    const { document, graph } = derivationGraph(tree, { schemas, library });
    expect(document).toEqual(toPython(tree));
    d3(graph, library);
    expect(serialize(tree)).toBe(text);
  });
});

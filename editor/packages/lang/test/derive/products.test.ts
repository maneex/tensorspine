import { readFileSync, readdirSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  consistent,
  d3,
  derivationGraph,
  derive,
  DerivedSchemaError,
  expand,
  formatProblems,
  loadLibrary,
  loadSchemas,
  parse,
  serialize,
  toJsonValue,
  toPython,
  type ExpandedGraph,
  type JsonValue,
  type Library,
  type PyRecord,
  type PyValue,
  type SchemaRegistry,
} from '../../src/index.js';
import { PyTypeError, PyValueError } from '../../src/expr/errors.js';
import { corpus, library, schemas } from '../describe/source.js';
import { BLANK_MODEL, FED_MODEL, SCRATCH_BASE } from './source.js';
import { repositoryRoot } from '../json/repository.js';
import { nodeSource } from '../library/source.js';

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

describe('the whole derived document', () => {
  it('runs the six products and appends the five to D1’s envelope', () => {
    // `products` builds no envelope of its own: it is `d1.emit`'s — `schema`, `model`,
    // `primitive_libraries`, `assignment`, `d1` — and the five products are appended in numerical
    // order, which is what `document.update` does. The order they are *computed* in is another
    // matter and is not visible in the answer; the parity suite is what holds it, since D5 reads
    // three inventories and D6 reads two products and D1's own published order.
    const answer = derive(parse(corpus('llama3-8b')), { schemas, library });
    expect(Object.keys(answer)).toEqual([
      'schema',
      'model',
      'primitive_libraries',
      'assignment',
      'd1',
      'd2',
      'd3',
      'd4',
      'd5',
      'd6',
    ]);
    expect(answer['schema']).toBe('tensorspine-derived/2.1');
    expect(answer['assignment']).toEqual({});
  });

  it('crosses the two stages of `--validate` before any product', () => {
    const tree = edited('llama3-8b', (document) => {
      delete document['interfaces'];
    });
    expect(() => derive(tree, { schemas, library })).toThrowError(
      "not valid, no products: <root>: 'interfaces' is a required property",
    );
  });

  it('derives from the tree without writing into it', () => {
    const text = corpus('colbert-v2');
    const tree = parse(text);
    derive(tree, { schemas, library });
    expect(serialize(tree)).toBe(text);
  });
});

describe('the emitter’s own reading of its own output', () => {
  // `d1.self_check`: "an emitter validates its own output against the derived schema before
  // writing it: a document it cannot vouch for is not written". The core writes nothing, so its
  // counterpart is to return nothing — `DerivedSchemaError`, carrying every line the tools would
  // have printed (they print the first five) and the document itself, for a log that has to say
  // what was wrong with it.

  it('vouches for what it returns', () => {
    const answer = derive(parse(corpus('llama3-8b')), { schemas, library });
    expect(schemas.structural(toJsonValue(answer), 'derived')).toEqual([]);
  });

  it('is where a `writer` nobody writes would surface', () => {
    // Feature 1.8b's finding, settled here. The tools write `"writer": null` for a state identity
    // no member writes, and the derived schema's `value_reference` is a bare *required string*: the
    // document they emit would be refused by their own self-check. V20 refuses such a model before
    // any derivation, so no emitter ever meets it — and the port writes what the tools write rather
    // than inventing a value, because the guard is what would name the disagreement. This is that
    // guard, asked directly: a D4 state whose writer is blanked is off the derived schema, at the
    // pointer that names it.
    const answer = derive(parse(corpus('llama3-8b')), { schemas, library });
    const states = (answer['d4'] as PyRecord)['states'] as PyRecord[];
    const first = states[0] as PyRecord;
    expect(first['writer']).not.toBeNull();
    states[0] = { ...first, writer: null };
    const problems = schemas.structural(toJsonValue(answer), 'derived');
    expect(formatProblems(problems)).toEqual([
      "d4/states/0/writer: None is not of type 'string'",
    ]);
  });

  it('refuses to answer, naming every line and carrying the document it refused', () => {
    // The registry has no schema of the `derived` role, which is `self_check`'s own first refusal
    // ("no schema with $id ending in /derived.json under …") and the one a workspace can produce:
    // §1's rule is that the editor reads the repository's schemas, and a workspace may override
    // them with its own (plan §1, "the schemas are the repository's files").
    let raised: unknown;
    try {
      derive(parse(corpus('colbert-v2')), { schemas: withoutDerived(), library });
    } catch (error) {
      raised = error;
    }
    expect(raised).toBeInstanceOf(DerivedSchemaError);
    const refusal = raised as DerivedSchemaError;
    expect(refusal.message).toBe(
      'the emitted document is off the derived schema: ' +
        'no schema with $id ending in /derived.json under schemas/',
    );
    expect(refusal.problems).toHaveLength(1);
    // The document travels with the refusal, for the log alone: it is not a product.
    expect(Object.keys(refusal.document)).toContain('d6');
  });
});

describe('a port shape a primitive leaves undetermined', () => {
  // Feature 1.8c's finding, settled here. A *produced* value's byte size is guarded
  // (`n * BYTES[dtype] if n is not None else None`) and a *public input's* is not, so a port shape
  // that does not resolve raises `TypeError` out of the whole derivation. No port shape of the
  // reference base cites an argument that may be absent, so no document of the repository reaches
  // it — and a primitive declared in the editor (D15, feature 3.3) can, which is why it is reached
  // here through a scratch base and a document that goes all the way through `derive`.
  //
  // The decision: reproduced, not corrected. The exception is the tools' own, in Python's words,
  // and `derive` neither catches it nor turns it into a blank — the products would be a fiction
  // either way. A finding for the tools; the editor's worker (feature 1.11) is what catches it.

  // The base and the two documents are acceptance fixtures since feature 1.13
  // (`editor/tests/fixtures/`), so the tools read the same files: the oracle records the raise and
  // the blank, and `test/parity/fixtures.test.ts` compares them.
  const scratch = (): Library =>
    loadLibrary(['data/primitive-library', SCRATCH_BASE], {
      schemas,
      source: nodeSource(repositoryRoot),
    });

  it('raises out of the derivation in Python’s own words', () => {
    expect(() => derive(parse(BLANK_MODEL), { schemas, library: scratch() })).toThrow(
      new PyTypeError("unsupported operand type(s) for *: 'NoneType' and 'int'"),
    );
  });

  it('is the *input*’s reading alone: the produced value is blank rather than a refusal', () => {
    // The same shape on the output port alone derives, with `elements` and `bytes_per_element`
    // blank — which is what the schema's nullable figures are for, and what the input's reading
    // would answer if it were guarded too.
    const answer = derive(parse(FED_MODEL), { schemas, library: scratch() });
    const values = (answer['d2'] as PyRecord)['values'] as PyRecord[];
    const produced = values.find((one) => one['value'] === 'only.output') as PyRecord;
    expect([produced['elements'], produced['bytes_per_element']]).toEqual([null, null]);
  });
});

// --- what the repository has not ---------------------------------------------

/** The repository's schemas without the one of the `derived` role. */
function withoutDerived(): SchemaRegistry {
  const directory = `${repositoryRoot}/schemas`;
  const files = readdirSync(directory)
    .filter((name) => name.endsWith('.json'))
    .map((name) => ({ path: `schemas/${name}`, text: readFileSync(`${directory}/${name}`, 'utf8') }))
    .filter((file) => !`${(JSON.parse(file.text) as { $id: string }).$id}`.endsWith('/derived.json'));
  return loadSchemas(files, { origin: 'schemas' });
}

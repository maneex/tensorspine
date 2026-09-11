import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  externalNames,
  indexGrid,
  QUANTITIES,
  quantityReadings,
  missingAssignment,
  PyTypeError,
  PyValueError,
  resolveQuantities,
  staticArgument,
  toPython,
  Unassigned,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../../src/expr/index.js';
import { readRepositoryFile } from '../json/repository.js';

// `resolve_quantities`, `external_names`, `missing_assignment`, `index_grid` and
// `static_argument`: what the validator, D1 and the derivation all start from. The document is
// read here as the editor reads one — through the lexeme-preserving parser and the value model —
// so that `1e-05` stays a real and `4096` stays a whole number all the way into the map.

/** A document with the quantities given, read as a document is read. */
function document(quantities: string): PyRecord {
  return toPython(parse(`{"quantities": ${quantities}}`)) as PyRecord;
}

const literal = (value: string): string => `{"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": ${value}}}`;
const derived = (expression: string): string =>
  `{"type": {"kind": "cardinality"}, "source": {"kind": "derived", "expression": ${expression}}}`;

describe('quantityReadings', () => {
  // What a *reader* of a document is told about its quantities (the editor's outline, §4.5):
  // where each is declared, what it resolves to, and whether the document computes the value or
  // simply writes it. The last of the three is the one `resolveQuantities` cannot answer, and
  // `llama3-8b`'s `head_dim` is why: a literal of 128 that declares its own derivation.
  it('answers the place each quantity is declared at, in the document’s own order', () => {
    const model = document(`{
      "d": ${literal('4096')},
      "heads": ${literal('32')}
    }`);
    expect(quantityReadings(model).map((one) => one.pointer)).toEqual([
      `/${QUANTITIES}/d`,
      `/${QUANTITIES}/heads`,
    ]);
  });

  it('calls a literal that declares its derivation computed, and a plain literal not', () => {
    const model = document(`{
      "d": ${literal('4096')},
      "heads": ${literal('32')},
      "head_dim": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 128,
        "derivation": {"op": "floor_divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}}}
    }`);
    const readings = new Map(quantityReadings(model).map((one) => [one.name, one]));
    expect(readings.get('d')?.computed).toBe(false);
    expect(readings.get('d')?.value).toBe(4096n);
    expect(readings.get('head_dim')?.computed).toBe(true);
    expect(readings.get('head_dim')?.value).toBe(128n);
  });

  it('calls a derived source and an external with a default computed, and leaves what nothing resolves without a value', () => {
    const model = document(`{
      "heads": {"type": {"kind": "cardinality"}, "domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}},
                "source": {"kind": "external"}},
      "kv_heads": {"type": {"kind": "cardinality"}, "domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}},
                   "source": {"kind": "external", "default": {"literal": 8}}},
      "inner": ${derived('{"op": "multiply", "args": [{"literal": 4}, {"quantity": "kv_heads"}]}')}
    }`);
    const readings = new Map(quantityReadings(model).map((one) => [one.name, one]));
    expect(readings.get('heads')?.computed).toBe(false);
    expect(readings.get('heads')?.value).toBeUndefined();
    expect(readings.get('kv_heads')?.computed).toBe(true);
    expect(readings.get('kv_heads')?.value).toBe(8n);
    expect(readings.get('inner')?.computed).toBe(true);
    // Under an assignment, what the assignment says — the same resolution the validator reads.
    const assigned = new Map(
      quantityReadings(model, { heads: 12n, kv_heads: 3n }).map((one) => [one.name, one]),
    );
    expect(assigned.get('heads')?.value).toBe(12n);
    expect(assigned.get('inner')?.value).toBe(12n);
  });

  it('resolves exactly as `resolveQuantities` does, being the same resolution', () => {
    const model = toPython(parse(readRepositoryFile('data/models/llama3-8b.json'))) as PyRecord;
    const resolved = resolveQuantities(model);
    for (const reading of quantityReadings(model)) {
      expect(reading.value).toBe(resolved.get(reading.name));
    }
    expect(quantityReadings(model)).toHaveLength(resolved.size);
  });
});

describe('resolveQuantities', () => {
  it('resolves a derived chain in any declaration order', () => {
    const model = document(`{
      "inner": ${derived('{"op": "multiply", "args": [{"literal": 4}, {"quantity": "width"}]}')},
      "width": ${derived('{"op": "multiply", "args": [{"quantity": "heads"}, {"quantity": "head_dim"}]}')},
      "heads": ${literal('32')},
      "head_dim": ${literal('128')}
    }`);
    const resolved = resolveQuantities(model);
    expect(resolved.get('width')).toBe(4096n);
    expect(resolved.get('inner')).toBe(16384n);
  });

  it('leaves a cyclic derivation absent rather than looping', () => {
    const model = document(`{
      "a": ${derived('{"op": "add", "args": [{"quantity": "b"}, {"literal": 1}]}')},
      "b": ${derived('{"op": "add", "args": [{"quantity": "a"}, {"literal": 1}]}')}
    }`);
    const resolved = resolveQuantities(model);
    expect(resolved.has('a')).toBe(false);
    expect(resolved.has('b')).toBe(false);
  });

  it('takes an external from the assignment, and its declared default otherwise', () => {
    const model = document(`{
      "layers": {"type": {"kind": "cardinality"}, "domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}},
                 "source": {"kind": "external"}},
      "heads": {"type": {"kind": "cardinality"}, "domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}},
                "source": {"kind": "external", "default": {"literal": 32}}},
      "kv_heads": {"type": {"kind": "cardinality"}, "domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}},
                   "source": {"kind": "external", "default": {"op": "floor_divide", "args": [{"quantity": "heads"}, {"literal": 4}]}}}
    }`);
    expect(resolveQuantities(model).has('layers')).toBe(false);
    expect(resolveQuantities(model).get('heads')).toBe(32n);
    expect(resolveQuantities(model).get('kv_heads')).toBe(8n);

    const assigned = resolveQuantities(model, { layers: 26n, heads: 8n });
    expect(assigned.get('layers')).toBe(26n);
    expect(assigned.get('heads')).toBe(8n);
    expect(assigned.get('kv_heads')).toBe(2n);
  });

  it('keeps a literal real a real and a literal whole number whole', () => {
    const model = document(`{
      "eps": {"type": {"kind": "real"}, "source": {"kind": "literal", "value": 1e-05}},
      "d": ${literal('4096')}
    }`);
    const resolved = resolveQuantities(model);
    expect(resolved.get('eps')).toBe(1e-5);
    expect(typeof resolved.get('eps')).toBe('number');
    expect(resolved.get('d')).toBe(4096n);
    expect(typeof resolved.get('d')).toBe('bigint');
  });

  it('resolves every quantity of a corpus document', () => {
    const model = toPython(parse(readRepositoryFile('data/models/llama3-8b.json'))) as PyRecord;
    const resolved = resolveQuantities(model);
    expect(resolved.get('d')).toBe(4096n);
    expect(resolved.get('head_dim')).toBe(128n);
    expect(resolved.get('vocab')).toBe(128256n);
    expect(resolved.get('eps')).toBe(1e-5);
    expect(resolved.get('precision')).toBe('bf16');
    expect(missingAssignment(model)).toEqual([]);
  });

  it('resolves the template under the assignment the repository’s suites use', () => {
    const model = toPython(
      parse(readRepositoryFile('data/models/decoder-causal-yarn/1.0.0.json')),
    ) as PyRecord;
    expect(missingAssignment(model)).toEqual([
      'eps',
      'head_dim',
      'heads',
      'inner',
      'kv_heads',
      'layers',
      'precision',
      'width',
    ]);
    const resolved = resolveQuantities(model, {
      width: 3072n,
      layers: 26n,
      heads: 32n,
      kv_heads: 8n,
      head_dim: 128n,
      inner: 9216n,
      eps: 1e-5,
      precision: 'bf16',
    });
    expect(resolved.get('width')).toBe(3072n);
    expect(resolved.get('layers')).toBe(26n);
    expect(resolved.get('eps')).toBe(1e-5);
  });
});

describe('externalNames and missingAssignment', () => {
  const model = document(`{
    "b": {"type": {"kind": "cardinality"}, "source": {"kind": "external"}},
    "a": {"type": {"kind": "cardinality"}, "source": {"kind": "external", "default": {"literal": 1}}},
    "c": ${literal('3')}
  }`);

  it('lists every external, and only those a default cannot stand for when asked', () => {
    expect([...externalNames(model)].sort()).toEqual(['a', 'b']);
    expect([...externalNames(model, false)].sort()).toEqual(['b']);
  });

  it('answers what the assignment leaves unset, in order', () => {
    expect(missingAssignment(model)).toEqual(['b']);
    expect(missingAssignment(model, { b: 1n })).toEqual([]);
  });
});

describe('indexGrid', () => {
  const quantities = new Map<string, PyValue>([['layers', 4n]]);
  const range = (start: string, stop: string, step: string): string =>
    `{"start": ${start}, "stop": ${stop}, "step": ${step}}`;
  const indices = (text: string): PyValue => toPython(parse(text));

  it('unrolls a range in lexicographic order of the index names', () => {
    const grid = indexGrid(
      indices(`{"stage": ${range('{"literal": 0}', '{"literal": 2}', '{"literal": 1}')},
                "layer": ${range('{"literal": 0}', '{"quantity": "layers"}', '{"literal": 1}')}}`),
      quantities,
    );
    expect(grid.names).toEqual(['layer', 'stage']);
    expect(grid.ranges).toEqual([
      [0n, 1n, 2n, 3n],
      [0n, 1n],
    ]);
  });

  it('walks a step of more than one, and answers nothing for an empty range', () => {
    expect(
      indexGrid(indices(`{"i": ${range('{"literal": 1}', '{"literal": 8}', '{"literal": 3}')}}`), quantities)
        .ranges[0],
    ).toEqual([1n, 4n, 7n]);
    expect(
      indexGrid(indices(`{"i": ${range('{"literal": 4}', '{"literal": 0}', '{"literal": 1}')}}`), quantities)
        .ranges[0],
    ).toEqual([]);
    expect(
      indexGrid(indices(`{"i": ${range('{"literal": 3}', '{"literal": 0}', '{"literal": -1}')}}`), quantities)
        .ranges[0],
    ).toEqual([3n, 2n, 1n]);
  });

  it('refuses a bound that does not resolve, naming the index and the edge', () => {
    expect(() =>
      indexGrid(
        indices(`{"layer": ${range('{"literal": 0}', '{"quantity": "unknown"}', '{"literal": 1}')}}`),
        quantities,
      ),
    ).toThrow(new Unassigned("index 'layer': stop does not resolve to a value"));
  });

  it('refuses a real bound and a step of zero, as `range` does', () => {
    expect(() =>
      indexGrid(indices(`{"i": ${range('{"literal": 0}', '{"literal": 4.0}', '{"literal": 1}')}}`), quantities),
    ).toThrow(PyTypeError);
    expect(() =>
      indexGrid(indices(`{"i": ${range('{"literal": 0}', '{"literal": 4}', '{"literal": 0}')}}`), quantities),
    ).toThrow(PyValueError);
  });

  it('unrolls a corpus composition’s indices', () => {
    const model = toPython(parse(readRepositoryFile('data/models/llama3-8b.json'))) as PyRecord;
    const compositions = model['compositions'] as PyRecord;
    const decoder = compositions['decoder'] as PyRecord;
    const grid = indexGrid(decoder['indices'] as PyValue, resolveQuantities(model));
    expect(grid.names).toEqual(['layer']);
    expect(grid.ranges[0]).toHaveLength(32);
    expect(grid.ranges[0]?.[31]).toBe(31n);
  });
});

describe('staticArgument', () => {
  const quantities = new Map<string, PyValue>([
    ['heads', 32n],
    ['eps', 1e-5],
  ]);

  it('evaluates a scalar argument as an expression', () => {
    expect(staticArgument(toPython(parse('{"quantity": "heads"}')), quantities)).toBe(32n);
    expect(staticArgument(toPython(parse('{"literal": true}')), quantities)).toBe(true);
  });

  it('evaluates a record argument field by field, recursively', () => {
    const value = toPython(
      parse(`{"record": {"theta": {"literal": 500000.0},
                         "scaling": {"record": {"kind": {"literal": "yarn"},
                                                "factor": {"quantity": "eps"}}}}}`),
    );
    expect(staticArgument(value, quantities)).toEqual({
      theta: 500000,
      scaling: { kind: 'yarn', factor: 1e-5 },
    });
  });

  it('leaves an unresolved field in the record rather than dropping it', () => {
    const value = toPython(parse('{"record": {"a": {"quantity": "unknown"}, "b": {"literal": 1}}}'));
    const resolved = staticArgument(value, quantities) as PyRecord;
    expect(resolved['a']).toBe(UNRESOLVED);
    expect(resolved['b']).toBe(1n);
  });

  it('reads an index in scope, as a composition-scoped argument does', () => {
    const value = toPython(parse('{"op": "add", "args": [{"index": "layer"}, {"literal": 1}]}'));
    expect(staticArgument(value, quantities, new Map([['layer', 3n]]))).toBe(4n);
    expect(staticArgument(value, quantities)).toBe(UNRESOLVED);
  });
});

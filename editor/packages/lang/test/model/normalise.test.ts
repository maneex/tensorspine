import { describe, expect, it } from 'vitest';

import { toJsonValue, toPython, type PyRecord, type PyValue } from '../../src/expr/index.js';
import { parse, serialize, toPlain } from '../../src/json/index.js';
import { loadModel, normalise } from '../../src/model/index.js';
import { PyKeyError } from '../../src/expr/index.js';
import { readRepositoryFile } from '../json/repository.js';
import { document } from './document.js';

// The expansion of §5.2 rule 7, stated one branch at a time: "each such rule `R` of composition
// `C` denotes exactly the top-level rule `C.R` whose `for_each` is `C`'s index ranges and whose
// site endpoints select the generated instance at the current indices, overridden index by index
// where the endpoint says so; a scoped parameter or state rule without a declared identity names
// it `C.R`, indexed by `C`'s indices".
//
// Every expectation here is a *text*, because member order is part of the answer: the normalised
// document is what the serializer writes and what D1 and the derivation read, and the order the
// tools' assignments make is the order the parity fixture carries. `compact` is that text without
// the indentation, so a rule fits on a line.

const RANGE = '{"start":{"literal":0},"stop":{"literal":2},"step":{"literal":1}}';
const ONE_INDEX = `{"i":${RANGE}}`;
const CURRENT = '{"i":{"index":"i"}}';

/** A value as JSON, member order kept: what `normalise` answered, spelled out. */
function compact(value: PyValue): string {
  return JSON.stringify(toPlain(toJsonValue(value)));
}

/** One top-level rule of a normalised document. */
function rule(text: string, kind: string, name: string): PyValue {
  const model = loadModel(text);
  const bindings = model['bindings'] as PyRecord;
  const map = bindings[kind] as PyRecord;
  const found = map[name];
  expect(found, `no ${kind} rule named '${name}'`).toBeDefined();
  return found as PyValue;
}

/** A document with one scoped value rule of the shape a test names. */
function values(rules: string, top = ''): string {
  return document({ bindings: `{"values": ${rules}}` }, top);
}

describe('a scoped value rule is the top-level rule it denotes', () => {
  it('carries the composition’s ranges and selects the site at the current indices', () => {
    const text = values('{"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}');
    expect(compact(rule(text, 'values', 'c.link'))).toBe(
      `{"for_each":${ONE_INDEX},` +
        `"from":{"instance":{"kind":"generated","composition":"c","instance":"a","indices":${CURRENT}},"port":"out"},` +
        `"to":{"instance":{"kind":"generated","composition":"c","instance":"b","indices":${CURRENT}},"port":"in"}}`,
    );
  });

  it('overrides the indices the endpoint names, and only those', () => {
    const text = values(
      '{"carry": {"from": {"site": "a", "indices": {"i": {"op": "subtract", ' +
        '"args": [{"index": "i"}, {"literal": 1}]}}, "port": "out"}, ' +
        '"to": {"site": "b", "port": "in"}}}',
    );
    expect(compact(rule(text, 'values', 'c.carry'))).toContain(
      '"instance":"a","indices":{"i":{"op":"subtract","args":[{"index":"i"},{"literal":1}]}}',
    );
    expect(compact(rule(text, 'values', 'c.carry'))).toContain(`"instance":"b","indices":${CURRENT}`);
  });

  it('puts a guard after the ranges, whatever order the rule was written in', () => {
    const written = values(
      '{"link": {"when": {"boolean": true}, "to": {"site": "b", "port": "in"}, ' +
        '"from": {"site": "a", "port": "out"}}}',
    );
    const other = values(
      '{"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}, ' +
        '"when": {"boolean": true}}}',
    );
    expect(Object.keys(rule(written, 'values', 'c.link') as PyRecord)).toEqual([
      'for_each',
      'when',
      'from',
      'to',
    ]);
    expect(compact(rule(written, 'values', 'c.link'))).toBe(compact(rule(other, 'values', 'c.link')));
  });

  it('passes an explicit selector through untouched', () => {
    const text = values(
      '{"link": {"from": {"instance": {"kind": "root", "instance": "root"}, "port": "out"}, ' +
        '"to": {"site": "b", "port": "in"}}}',
    );
    expect(compact(rule(text, 'values', 'c.link'))).toContain(
      '"from":{"instance":{"kind":"root","instance":"root"},"port":"out"}',
    );
  });
});

describe('a scoped parameter, state or constant rule', () => {
  it('names an undeclared tensor after the rule, indexed by the composition’s indices', () => {
    const text = document({
      bindings: '{"parameters": {"a.w": {"members": [{"site": "a", "parameter": "w"}]}}}',
    });
    expect(compact(rule(text, 'parameters', 'c.a.w'))).toBe(
      `{"for_each":${ONE_INDEX},` +
        `"members":[{"instance":{"kind":"generated","composition":"c","instance":"a","indices":${CURRENT}},"parameter":"w"}],` +
        `"tensor":{"name":"c.a.w","indices":${CURRENT}}}`,
    );
  });

  it('keeps a declared tensor, and writes dtype and location where the tools write them', () => {
    const text = document({
      bindings:
        '{"parameters": {"a.w": {"location": {"tensor": ["t"]}, "dtype": {"quantity": "p"}, ' +
        '"tensor": {"name": "shared.w"}, "members": [{"site": "a", "parameter": "w"}]}}}',
    });
    expect(Object.keys(rule(text, 'parameters', 'c.a.w') as PyRecord)).toEqual([
      'for_each',
      'members',
      'dtype',
      'tensor',
      'location',
    ]);
    expect(compact(rule(text, 'parameters', 'c.a.w'))).toContain('"tensor":{"name":"shared.w"}');
  });

  it('names an undeclared state identity after the rule, and keeps a declared one', () => {
    const implied = document({
      bindings: '{"states": {"a.kv": {"members": [{"site": "a", "state": "kv"}]}}}',
    });
    expect(compact(rule(implied, 'states', 'c.a.kv'))).toBe(
      `{"for_each":${ONE_INDEX},` +
        `"members":[{"instance":{"kind":"generated","composition":"c","instance":"a","indices":${CURRENT}},"state":"kv"}],` +
        `"identity":{"name":"c.a.kv","indices":${CURRENT}}}`,
    );
    const declared = document({
      bindings:
        '{"states": {"a.kv": {"identity": {"name": "shared.kv"}, "dtype": "bf16", ' +
        '"members": [{"site": "a", "state": "kv"}]}}}',
    });
    expect(Object.keys(rule(declared, 'states', 'c.a.kv') as PyRecord)).toEqual([
      'for_each',
      'members',
      'dtype',
      'identity',
    ]);
  });

  it('writes a constant rule’s constant last — and copies a dtype the grammar does not admit', () => {
    // `scoped_constant_binding` has no `dtype`; `_hoist` copies one for every kind but `values`
    // all the same, so a document off the grammar that carries one is normalised with it. The
    // line is the tools', and the port reproduces the line, not the schema.
    const text = document({
      bindings:
        '{"constants": {"a.m": {"dtype": "u8", "constant": "mask", ' +
        '"members": [{"site": "a", "constant": "mask"}]}}}',
    });
    expect(compact(rule(text, 'constants', 'c.a.m'))).toBe(
      `{"for_each":${ONE_INDEX},` +
        `"members":[{"instance":{"kind":"generated","composition":"c","instance":"a","indices":${CURRENT}},"constant":"mask"}],` +
        '"dtype":"u8","constant":"mask"}',
    );
  });
});

describe('the composition the rules came from', () => {
  it('loses its bindings member, and loses it even when it holds nothing', () => {
    for (const scoped of ['{"values": {}}', '{}']) {
      const model = loadModel(document({ bindings: scoped }));
      const compositions = model['compositions'] as PyRecord;
      expect(Object.keys(compositions['c'] as PyRecord)).toEqual([
        'indices',
        'families',
        'instances',
      ]);
      const bindings = model['bindings'] as PyRecord;
      expect(Object.keys(bindings['values'] as PyRecord)).toEqual([]);
    }
  });

  it('is left exactly as it stands when it carries no bindings at all', () => {
    const text = document({});
    expect(serialize(toJsonValue(loadModel(text)))).toBe(
      serialize(toJsonValue(toPython(parse(text)))),
    );
  });

  it('is left as it stands when the document declares no composition', () => {
    const text = `{"schema": "tensorspine/2.0", "bindings": {"values": {}}}\n`;
    expect(serialize(toJsonValue(loadModel(text)))).toBe(
      '{\n  "schema": "tensorspine/2.0",\n  "bindings": {\n    "values": {}\n  }\n}\n',
    );
  });
});

describe('several indices, several rules', () => {
  const TWO = `{"i":${RANGE},"j":${RANGE}}`;
  const BOTH = '{"i":{"index":"i"},"j":{"index":"j"}}';

  it('carries every index of the composition, in the composition’s order', () => {
    const text = document({
      indices: TWO,
      bindings: '{"values": {"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
    });
    expect(compact(rule(text, 'values', 'c.link'))).toBe(
      `{"for_each":${TWO},` +
        `"from":{"instance":{"kind":"generated","composition":"c","instance":"a","indices":${BOTH}},"port":"out"},` +
        `"to":{"instance":{"kind":"generated","composition":"c","instance":"b","indices":${BOTH}},"port":"in"}}`,
    );
  });

  it('an override keeps the index in its place, not at the end', () => {
    const text = document({
      indices: TWO,
      bindings:
        '{"values": {"link": {"from": {"site": "a", "indices": {"i": {"literal": 0}}, "port": "out"}, ' +
        '"to": {"site": "b", "port": "in"}}}}',
    });
    expect(compact(rule(text, 'values', 'c.link'))).toContain(
      '"instance":"a","indices":{"i":{"literal":0},"j":{"index":"j"}}',
    );
  });

  it('appends the hoisted rules in the order the document writes them', () => {
    const text = document({
      bindings:
        '{"values": {"second": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}, ' +
        '"third": {"from": {"site": "b", "port": "out"}, "to": {"site": "a", "port": "in"}}}}',
      sites: ['a', 'b'],
    }, '{"values": {"first": {"from": {"instance": {"kind": "root", "instance": "root"}, "port": "out"}, ' +
      '"to": {"instance": {"kind": "root", "instance": "root"}, "port": "in"}}}, ' +
      '"parameters": {}, "constants": {}, "states": {}}');
    const bindings = loadModel(text)['bindings'] as PyRecord;
    expect(Object.keys(bindings['values'] as PyRecord)).toEqual(['first', 'c.second', 'c.third']);
  });

  it('keeps a member named __proto__ a member', () => {
    // The identifier grammar admits it, CPython's `json` reads it as a name like any other, and a
    // plain assignment in JavaScript would set the prototype instead (feature 1.1 found the same
    // trap in the reading Ajv validates).
    const text = document({
      indices: `{"__proto__": ${RANGE}}`,
      bindings: '{"parameters": {"a.w": {"members": [{"site": "a", "parameter": "w"}]}}}',
    });
    expect(compact(rule(text, 'parameters', 'c.a.w'))).toContain(
      '"tensor":{"name":"c.a.w","indices":{"__proto__":{"index":"__proto__"}}}',
    );
  });
});

describe('normalise itself', () => {
  it('is idempotent: the normalised document normalises to itself', () => {
    const text = document({
      bindings:
        '{"values": {"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}, ' +
        '"parameters": {"a.w": {"members": [{"site": "a", "parameter": "w"}]}}, ' +
        '"states": {"a.kv": {"members": [{"site": "a", "state": "kv"}]}}}',
    });
    const once = loadModel(text);
    expect(serialize(toJsonValue(normalise(once)))).toBe(serialize(toJsonValue(once)));
  });

  it('does not modify the document it is given', () => {
    const text = document({
      bindings: '{"values": {"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
    });
    const read = toPython(parse(text));
    const before = serialize(toJsonValue(read));
    normalise(read);
    expect(serialize(toJsonValue(read))).toBe(before);
  });
});

describe('the kinds of scoped binding the grammar admits', () => {
  it('are exactly the ones hoisted, and a kind outside them is a KeyError', () => {
    // `_hoist` reads `{'parameters': 'parameter', 'states': 'state', 'constants': 'constant'}[kind]`
    // and treats `values` before it, so the four members of `scoped_bindings` are the four the port
    // covers. The schema is what says which four they are (plan §1); a fifth would be a `KeyError`
    // here and in the tools alike, which is the failure both would have.
    const schema = JSON.parse(
      readRepositoryFile('schemas/tensorspine.schema.json'),
    ) as { $defs: Record<string, { properties: Record<string, unknown> }> };
    const kinds = Object.keys(schema.$defs['scoped_bindings']?.properties ?? {});
    expect(kinds.sort()).toEqual(['constants', 'parameters', 'states', 'values']);
    expect(Object.keys(schema.$defs['bindings']?.properties ?? {}).sort()).toEqual(kinds.sort());
    for (const kind of kinds) {
      const rules =
        kind === 'values'
          ? '{"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}'
          : kind === 'constants'
            ? '{"link": {"constant": "m", "members": [{"site": "a", "constant": "m"}]}}'
            : `{"link": {"members": [{"site": "a", "${kind.slice(0, -1)}": "s"}]}}`;
      const model = loadModel(document({ bindings: `{"${kind}": ${rules}}` }));
      expect(Object.keys((model['bindings'] as PyRecord)[kind] as PyRecord), kind).toEqual(['c.link']);
    }
    expect(() => loadModel(document({ bindings: '{"effects": {"link": {}}}' }))).toThrowError(
      PyKeyError,
    );
  });
});

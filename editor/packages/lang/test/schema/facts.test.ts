import { describe, expect, it } from 'vitest';

import { factsOf, mergeFacts, NO_FACTS } from '../../src/schema/facts.js';
import { alternativeLabel } from '../../src/schema/vocabulary.js';
import { parse } from '../../src/json/parse.js';
import { repositorySchemas } from './repository.js';

import type { SchemaRegistry } from '../../src/schema/index.js';

let once: SchemaRegistry | null = null;

/** The repository's schemas, loaded once for the suite (the registry costs ~200 ms). */
function registry(): SchemaRegistry {
  once ??= repositorySchemas();
  return once;
}

// `factsOf` is where draft 2020-12's own words are named — `type`, `enum`, `const`, `minimum`,
// `propertyNames` — so that no source of the interface has to name them. Two of them collide with
// the language: the keyword `enum` and the type `boolean` are also values of the unit schema's
// `argument_type.kind`, and catching rule §1 (b) is a whole-literal scan that cannot tell the
// meta-schema's vocabulary from the language's. This module is the one place both are written.
//
// `accepts` is the other half: Ajv on one *place* of one schema. A generated form needs it to say
// which alternative of a union a value is — the operators separate `unary` from `nary`, not the
// keys — and whether a conditional branch applies.

const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const UNIT = 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json';

describe('what one schema node asserts', () => {
  it('says nothing for a boolean schema, and nothing for a node with no keyword it reads', () => {
    expect(factsOf(true)).toEqual(NO_FACTS);
    expect(factsOf(false)).toEqual(NO_FACTS);
    expect(factsOf({ $comment: 'prose' }).states).toBe(false);
    expect(factsOf({ type: 'string' }).states).toBe(true);
  });

  it('separates the six shapes a form has a widget for', () => {
    expect(factsOf({ type: 'object' })).toMatchObject({ holdsObject: true, holdsArray: false });
    expect(factsOf({ type: 'array' }).holdsArray).toBe(true);
    expect(factsOf({ type: 'string' }).holdsText).toBe(true);
    expect(factsOf({ type: 'integer' })).toMatchObject({
      holdsWholeNumber: true,
      holdsNumber: false,
    });
    expect(factsOf({ type: 'number' })).toMatchObject({
      holdsWholeNumber: false,
      holdsNumber: true,
    });
    expect(factsOf({ type: 'boolean' }).holdsTruth).toBe(true);
  });

  it('reads a nullable type as its type, and a bare null as nothing at all', () => {
    const nullable = factsOf({ type: ['number', 'null'] });
    expect(nullable).toMatchObject({ nullable: true, holdsNumber: true, holdsOnlyNothing: false });
    expect(factsOf({ type: 'null' })).toMatchObject({ nullable: false, holdsOnlyNothing: true });
  });

  it('keeps a bound’s exclusivity beside the bound', () => {
    expect(factsOf({ minimum: 1 })).toMatchObject({ minimum: 1, minimumExcluded: false });
    expect(factsOf({ exclusiveMinimum: 0 })).toMatchObject({ minimum: 0, minimumExcluded: true });
    expect(factsOf({ maximum: 1 })).toMatchObject({ maximum: 1, maximumExcluded: false });
    expect(factsOf({ exclusiveMaximum: 1 })).toMatchObject({ maximum: 1, maximumExcluded: true });
  });

  it('tells a `const` of `false` from no `const` at all', () => {
    expect(factsOf({ const: false })).toMatchObject({ fixed: true, constant: false });
    expect(factsOf({ type: 'boolean' })).toMatchObject({ fixed: false, constant: undefined });
  });

  it('reads the members, the requirement, the closure and the map-ness of an object', () => {
    const node = {
      type: 'object',
      additionalProperties: false,
      required: ['b'],
      properties: { a: { type: 'string' }, b: { type: 'string' } },
    };
    expect(factsOf(node)).toMatchObject({
      members: ['a', 'b'],
      required: ['b'],
      closed: true,
      keyed: false,
    });
    expect(factsOf({ additionalProperties: { type: 'string' } })).toMatchObject({
      keyed: true,
      closed: false,
    });
  });
});

describe('the facts of a chain', () => {
  it('takes each field from the first node that states it', () => {
    const merged = mergeFacts([
      { description: 'the member’s own' },
      { type: 'string', description: 'the definition’s', minLength: 1 },
    ]);
    expect(merged).toMatchObject({
      description: 'the member’s own',
      holdsText: true,
      minLength: 1,
    });
  });

  it('accumulates the members and the requirements, in the chain’s order', () => {
    const merged = mergeFacts([
      { properties: { a: true }, required: ['a'] },
      { properties: { b: true, a: true }, required: ['b'] },
    ]);
    expect(merged.members).toEqual(['a', 'b']);
    expect(merged.required).toEqual(['a', 'b']);
  });

  it('answers nothing for an empty chain', () => {
    expect(mergeFacts([])).toEqual(NO_FACTS);
  });
});

describe('what a chooser labels an alternative by', () => {
  it('is the tags, then the declared type, then what the `$ref` names', () => {
    const vocabulary = registry().vocabulary();
    const location = vocabulary.unionAt(`${MODEL}#/$defs/location`);
    expect(location?.alternatives.map((one) => alternativeLabel(one))).toEqual([
      'tensor',
      'stack',
      'concat',
      'slice',
    ]);
    // The string form of a `physical_name` item requires no key, so its type is the only label.
    const item = vocabulary.unionAt(`${MODEL}#/$defs/physical_name/items`);
    expect(item?.alternatives.map((one) => alternativeLabel(one))).toEqual([
      'string',
      'index',
      'coordinate',
    ]);
  });
});

describe('Ajv on one place of one schema', () => {
  it('accepts what that place admits and refuses what it does not', () => {
    const at = `${MODEL}#/$defs/location`;
    expect(registry().accepts(parse('{"tensor": ["w"]}'), at)).toBe(true);
    expect(registry().accepts(parse('{"nope": 1}'), at)).toBe(false);
  });

  it('separates two alternatives no required key tells apart', () => {
    // The one case feature 1.1 recorded: `unary` and `nary` both require `op` and `args`, and
    // only the operators their `op` admits separate them.
    const unary = `${MODEL}#/$defs/unary_operation_expression`;
    const nary = `${MODEL}#/$defs/nary_operation_expression`;
    const add = parse('{"op": "add", "args": [{"literal": 1}, {"literal": 2}]}');
    const negate = parse('{"op": "negate", "args": [{"literal": 1}]}');
    expect([registry().accepts(add, unary), registry().accepts(add, nary)]).toEqual([false, true]);
    expect([registry().accepts(negate, unary), registry().accepts(negate, nary)]).toEqual([
      true,
      false,
    ]);
  });

  it('answers a conditional branch’s own test', () => {
    const test = `${MODEL}#/$defs/quantity_definition/if`;
    expect(
      registry().accepts(parse('{"type": {"kind": "cardinality"}, "source": {"kind": "external"}}'), test),
    ).toBe(true);
    expect(
      registry().accepts(
        parse('{"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 4}}'),
        test,
      ),
    ).toBe(false);
  });

  it('resolves a `$ref` that crosses a schema file', () => {
    const at = `${UNIT}#/$defs/argument_type`;
    expect(registry().accepts(parse('{"kind": "physical", "unit": "tokens"}'), at)).toBe(true);
    expect(registry().accepts(parse('{"kind": "physical"}'), at)).toBe(false);
  });

  it('refuses an anchor that names nothing, rather than answering false', () => {
    // A silent `false` would make a mistyped anchor look like a value that does not conform.
    expect(() => registry().accepts(parse('1'), 'not-an-anchor')).toThrow(/not a schema anchor/);
    expect(() => registry().accepts(parse('1'), 'https://elsewhere.test/a.json#/x')).toThrow(
      /does not hold/,
    );
    expect(() => registry().accepts(parse('1'), `${MODEL}#/$defs/no_such_definition`)).toThrow(
      /names nothing/,
    );
  });
});

describe('the vocabulary is read once and kept', () => {
  it('answers the same object to every caller', () => {
    // Feature 2.2 measured the walk at 1–2.5 ms over the five schemas: nothing at startup, and a
    // trap for a generated form that asks per render.
    expect(registry().vocabulary()).toBe(registry().vocabulary());
  });
});

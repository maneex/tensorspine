import { factsOf, type SchemaNode } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import {
  CHOOSER,
  editsOneValue,
  FIXED,
  JSON_EDITOR,
  LIST,
  MAP,
  NOTHING,
  NUMBER,
  SCALAR,
  SECTION,
  SELECT,
  TEXT,
  TOGGLE,
  WHOLE,
  widgetOf,
  WIDGETS,
} from '../../src/forms/widget.js';

// Plan §4.12's type-to-widget table is keyed by `argument_type.kind` — cardinality, real,
// physical, boolean, enum, record — and those six names are an enumeration of the unit schema,
// which catching rule §1 (b) forbids the interface to write. It does not have to: the tools
// publish each primitive's arguments as JSON Schema (`--document primitive-schema`, vendored with
// the build and consumed as it is, F5), where a cardinality is `{"type": "integer"}`, a real and
// a physical in seconds `{"type": "number"}`, a whole-number physical `{"type": "integer"}` with
// its unit beside it, a boolean a boolean, an enum an `enum` and a record an object. So one rule
// over what a schema asserts *is* that table, and it is the same rule that renders the four
// schemas themselves.
//
// What each row below asserts is the rule, not a fixture: the shapes are written here as JSON
// Schema, and `factsOf` reads them exactly as it reads a definition of `schemas/`.

const widget = (node: SchemaNode): string => widgetOf(factsOf(node), false);

describe('the widget a place gets from its schema', () => {
  it('gives a whole number a stepper and a fractional one a number field', () => {
    expect(widget({ type: 'integer', minimum: 1 })).toBe(WHOLE);
    expect(widget({ type: 'number', exclusiveMinimum: 0 })).toBe(NUMBER);
  });

  it('gives a truth value a toggle and an enumeration a select', () => {
    expect(widget({ type: 'boolean' })).toBe(TOGGLE);
    expect(widget({ enum: ['causal', 'chunked', 'none'] })).toBe(SELECT);
    // An enumeration wins over the type it happens to declare: the options are the choice.
    expect(widget({ type: 'string', enum: ['a'] })).toBe(SELECT);
  });

  it('gives an object with declared members a section and one with open names a map', () => {
    expect(widget({ type: 'object', properties: { span: { type: 'integer' } } })).toBe(SECTION);
    expect(widget({ type: 'object', additionalProperties: { type: 'integer' } })).toBe(MAP);
    // A place that closes or names its members is an object whether or not it says so.
    expect(widget({ additionalProperties: false, properties: { a: true } })).toBe(SECTION);
    expect(widget({ propertyNames: { pattern: '^a' }, additionalProperties: true })).toBe(SECTION);
  });

  it('gives an array a list and a fixed value a read-only row', () => {
    expect(widget({ type: 'array', items: { type: 'string' } })).toBe(LIST);
    expect(widget({ items: { type: 'string' } })).toBe(LIST);
    expect(widget({ const: 'tensorspine/2.0' })).toBe(FIXED);
  });

  it('gives text a text field, several scalar types one scalar row, and no type at all the JSON row', () => {
    expect(widget({ type: 'string', pattern: '^[A-Za-z_]' })).toBe(TEXT);
    expect(widget({ type: ['string', 'number', 'boolean'] })).toBe(SCALAR);
    expect(widget({ type: 'null' })).toBe(NOTHING);
    expect(widget({ $comment: 'nothing a form can read' })).toBe(JSON_EDITOR);
    expect(widget(true)).toBe(JSON_EDITOR);
  });

  it('reads a nullable figure as its own type, which is what a derived row shows', () => {
    expect(widget({ type: ['number', 'null'] })).toBe(NUMBER);
    expect(widget({ type: ['object', 'null'], additionalProperties: { type: 'number' } })).toBe(MAP);
  });

  it('is a chooser wherever a union is, whatever else the place says', () => {
    expect(widgetOf(factsOf({ type: 'object' }), true)).toBe(CHOOSER);
    expect(widgetOf(factsOf({}), true)).toBe(CHOOSER);
    // …except a fixed value and an enumeration, which are the choice themselves.
    expect(widgetOf(factsOf({ const: 1 }), true)).toBe(FIXED);
  });
});

describe('which widgets edit one value on the row itself', () => {
  it('is every scalar one, and no shaped one', () => {
    for (const one of [TEXT, WHOLE, NUMBER, TOGGLE, SELECT, FIXED, SCALAR, NOTHING]) {
      expect(editsOneValue(one), one).toBe(true);
    }
    for (const one of [SECTION, MAP, LIST, CHOOSER, JSON_EDITOR]) {
      expect(editsOneValue(one), one).toBe(false);
    }
    // A widget a presentation binding names is never one: the editor owns its subtree.
    expect(editsOneValue('expression')).toBe(false);
    expect(editsOneValue('token-list')).toBe(false);
  });

  it('covers every widget this module decides, so a new one forces the decision', () => {
    for (const one of WIDGETS) {
      expect(typeof editsOneValue(one), one).toBe('boolean');
    }
    expect(new Set(WIDGETS).size).toBe(WIDGETS.length);
    expect(WIDGETS).toHaveLength(13);
  });
});

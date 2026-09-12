import { describe, expect, it } from 'vitest';

import {
  languageAnchors,
  parse,
  serialize,
  toPlain,
  type JsonValue,
  type SchemaRegistry,
} from '@tensorspine/lang';
import { SchemaShapes } from '@tensorspine/store';

import {
  blankAt,
  blankFor,
  grammarAt,
  parseAt,
  nodeAt,
  printAt,
  productionLabel,
  replaceAt,
  replacementsAt,
  rowsOf,
  unwrapsTo,
  unwrapTo,
  withMoreOperands,
  withOperator,
  withoutOperand,
  withPayload,
  UNNAMED,
  wrapIn,
  wrappersAt,
  type LanguageContext,
  type Production,
} from '../../src/expressions/index.js';
import { presentation } from '../../src/presentation/index.js';
import { registry as repositorySchemas } from '../presentation/source.js';

/**
 * The tree view of §4.13 — the rows S7 draws and the four gestures it names: a node "can be
 * wrapped ('wrap in floor_divide by …'), unwrapped, or replaced", with "operator selects from the
 * schema enums and arity from the `args` bounds".
 *
 * Every assertion below is about something the schemas or `presentation.json` say: the operators a
 * select offers are an enumeration, the arity is `minItems`/`maxItems`, the referent a picker is
 * for is the `references` binding, and the blank a wrap writes is the first value the grammar
 * accepts. Nothing here is a table of this feature's own.
 */

const registry: SchemaRegistry = repositorySchemas();
const context: LanguageContext = {
  registry,
  shapes: new SchemaShapes(registry),
  bindings: presentation(),
};
const anchors = languageAnchors(registry);

/** The one production at an anchor with that name, or a failure naming what there was. */
function production(anchor: string, name: string): Production {
  const found = replacementsAt(context, anchor).find((one) => productionLabel(one) === name);
  if (found === undefined) {
    throw new Error(`${name}: not a production of ${anchor}`);
  }
  return found;
}

/** S7's own expression: `head_dim`'s derivation in `llama3-8b`. */
const HEAD_DIM = parse('{"op": "floor_divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}');

describe('the rows of S7’s tree', () => {
  const rows = rowsOf(context, anchors.expression, HEAD_DIM);

  it('is one row per node, indented by its depth', () => {
    expect(rows.map((row) => [row.path.join('.'), row.depth, row.label])).toEqual([
      ['', 0, 'floor_divide'],
      ['0', 1, 'quantity'],
      ['1', 1, 'quantity'],
    ]);
  });

  it('offers the operator select the schema’s own enumeration fills', () => {
    // `binary_operation_expression`'s five, and not the eleven of the three alternatives: the
    // arity is what separates them, and a select that offered `add` would be offering a value
    // this alternative does not admit.
    expect(rows[0]?.operators).toEqual(['subtract', 'divide', 'ceil_divide', 'floor_divide', 'modulo']);
    expect(rows[0]?.operator).toBe('floor_divide');
    expect(rows[1]?.operators).toEqual([]);
  });

  it('says what the schema admits of the arity, which is S7’s own note', () => {
    expect([rows[0]?.least, rows[0]?.most, rows[0]?.extensible, rows[0]?.reducible]).toEqual([
      2,
      2,
      false,
      false,
    ]);
    // An n-ary operator is where `+ arg` appears: `minItems` 2 and no maximum at all.
    const nary = rowsOf(context, anchors.expression, parse('{"op": "add", "args": [{"literal": 1}, {"literal": 2}]}'));
    expect([nary[0]?.least, nary[0]?.most, nary[0]?.extensible, nary[0]?.reducible]).toEqual([
      2,
      Infinity,
      true,
      false,
    ]);
  });

  it('says which rows hold a name the editor picks, and what kind', () => {
    expect(rows[1]).toMatchObject({ payload: true, named: true, referent: 'quantity', written: 'd', prefix: '' });
    const indexed = rowsOf(context, anchors.expression, parse('{"index": "layer"}'));
    expect(indexed[0]).toMatchObject({ named: true, referent: 'index', prefix: '$' });
    const literal = rowsOf(context, anchors.expression, parse('{"literal": 4096}'));
    expect(literal[0]).toMatchObject({ payload: true, named: false, written: 4096 });
  });

  it('walks a condition into its expressions, each at its own anchor', () => {
    const guard = parse(
      '{"all": [{"compare": {"operator": "equal", "left": {"op": "modulo", "args": [{"index": "layer"}, {"literal": 5}]}, "right": {"literal": 4}}}, {"boolean": true}]}',
    );
    const walked = rowsOf(context, anchors.condition, guard);
    expect(walked.map((row) => [row.path.join('.'), row.label, row.anchor === anchors.expression])).toEqual([
      ['', 'all', false],
      ['0', 'equal', false],
      ['0.0', 'modulo', true],
      ['0.0.0', 'index', true],
      ['0.0.1', 'literal', true],
      ['0.1', 'literal', true],
      ['1', 'boolean', false],
    ]);
  });

  it('names the keyword each operand of a conditional is introduced by', () => {
    const written = parse('{"if": {"boolean": true}, "then": {"literal": 1}, "else": {"literal": 2}}');
    expect(rowsOf(context, anchors.expression, written).map((row) => row.keyword)).toEqual([
      undefined,
      'if',
      'then',
      'else',
    ]);
  });

  it('carries the unit side’s present as an application over a path', () => {
    const walked = rowsOf(context, anchors.unitCondition, parse('{"present": "rope.scaling.kind"}'));
    expect(walked).toHaveLength(1);
    expect(walked[0]).toMatchObject({ payload: true, named: true, referent: 'present', written: 'rope.scaling.kind' });
  });
});

describe('the gestures §4.13 names', () => {
  const write = (value: JsonValue): unknown => toPlain(value);

  it('replaces a node at a path and leaves the rest as it was written', () => {
    const next = replaceAt(context, anchors.expression, HEAD_DIM, [1], parse('{"literal": 8}'));
    expect(write(next)).toEqual({
      op: 'floor_divide',
      args: [{ quantity: 'd' }, { literal: 8 }],
    });
    expect(nodeAt(context, anchors.expression, next, [0])).toEqual(parse('{"quantity": "d"}'));
  });

  it('changes the operator and keeps every operand the new arity admits', () => {
    expect(write(withOperator(context, anchors.expression, HEAD_DIM, 'subtract'))).toEqual({
      op: 'subtract',
      args: [{ quantity: 'd' }, { quantity: 'heads' }],
    });
    // A name the alternative does not admit changes nothing: the select never offers one.
    expect(write(withOperator(context, anchors.expression, HEAD_DIM, 'add'))).toEqual(write(HEAD_DIM));
  });

  it('wraps a node, keeping it as the first operand and blanking the rest', () => {
    const wrapped = wrapIn(
      context,
      anchors.expression,
      parse('{"quantity": "d"}'),
      production(anchors.expression, 'floor_divide'),
    );
    expect(write(wrapped)).toEqual({
      op: 'floor_divide',
      args: [{ quantity: 'd' }, { literal: 0 }],
    });
    // The blank is the first thing the *grammar* accepts there and no proposal of the editor's:
    // `scalar_literal` admits a string, a number and a boolean, and the number is what a literal
    // of an expression is in this repository.
    expect(printAt(context, anchors.expression, wrapped)).toBe('d div 0');
  });

  it('unwraps an application to the operand written where it stands', () => {
    expect(write(unwrapTo(context, anchors.expression, HEAD_DIM))).toEqual({ quantity: 'd' });
    // A comparison's sides are expressions where the comparison is a condition, so there is no
    // operand of it a condition could be replaced by, and the gesture is not offered (D5).
    const compared = parse(
      '{"compare": {"operator": "equal", "left": {"quantity": "a"}, "right": {"literal": 1}}}',
    );
    expect(unwrapsTo(context, anchors.condition, compared)).toBeNull();
    expect(write(unwrapTo(context, anchors.condition, compared))).toEqual(write(compared));
  });

  it('adds and drops an operand only where the schema admits it', () => {
    const nary = parse('{"op": "multiply", "args": [{"quantity": "a"}, {"quantity": "b"}]}');
    const wider = withMoreOperands(context, anchors.expression, nary);
    expect(write(wider)).toEqual({
      op: 'multiply',
      args: [{ quantity: 'a' }, { quantity: 'b' }, { literal: 0 }],
    });
    expect(write(withoutOperand(context, anchors.expression, wider, 2))).toEqual(write(nary));
    // `minItems` 2 is the floor, and `maxItems` 2 is the ceiling of a binary operator.
    expect(write(withoutOperand(context, anchors.expression, nary, 1))).toEqual(write(nary));
    expect(write(withMoreOperands(context, anchors.expression, HEAD_DIM))).toEqual(write(HEAD_DIM));
  });

  it('writes a picked name into the payload the production holds', () => {
    expect(write(withPayload(context, anchors.expression, parse('{"quantity": "d"}'), 'heads'))).toEqual({
      quantity: 'heads',
    });
    expect(write(withPayload(context, anchors.unitCondition, parse('{"present": "a"}'), 'b.c'))).toEqual({
      present: 'b.c',
    });
  });

  it('starts a place nothing is written at from the first production of its anchor', () => {
    expect(write(blankAt(context, anchors.expression))).toEqual({ literal: 0 });
    expect(write(blankAt(context, anchors.condition))).toEqual({ boolean: false });
  });

  it('offers every production of the anchor as a replacement, one per operator', () => {
    const offered = replacementsAt(context, anchors.expression).map(productionLabel);
    expect(offered).toEqual([
      'literal',
      'quantity',
      'index',
      'negate',
      'absolute',
      'subtract',
      'divide',
      'ceil_divide',
      'floor_divide',
      'modulo',
      'add',
      'multiply',
      'min',
      'max',
      'if | then | else',
    ]);
  });
});

describe('what an edit leaves behind', () => {
  it('is on the grammar, whatever the gesture', () => {
    const value = HEAD_DIM;
    const made: JsonValue[] = [
      withOperator(context, anchors.expression, value, 'modulo'),
      wrapIn(context, anchors.expression, value, production(anchors.expression, 'add')),
      unwrapTo(context, anchors.expression, value),
      replaceAt(context, anchors.expression, value, [0], blankAt(context, anchors.expression)),
      withMoreOperands(context, anchors.expression, parse('{"op": "add", "args": [{"literal": 1}, {"literal": 2}]}')),
    ];
    for (const one of made) {
      expect(registry.accepts(one, anchors.expression), printAt(context, anchors.expression, one)).toBe(true);
    }
  });

  it('prints and parses back, so the tree and the text cannot disagree', () => {
    const wrapped = wrapIn(context, anchors.expression, HEAD_DIM, production(anchors.expression, 'add'));
    const printed = printAt(context, anchors.expression, wrapped);
    expect(printed).toBe('d div heads + 0');
    expect(grammarAt(context, anchors.expression).chosen(wrapped)?.production.operator).toBe('add');
  });
});

describe('the shapes the grammar admits and no repository file writes', () => {
  it('keeps a one-element connective apart from a chain of the same one', () => {
    // `all` takes `minItems: 1`, so a one-element one is on the grammar and the blank a chooser
    // writes; a chain of `and` would swallow it if the text did not say otherwise.
    const one = parse('{"all": [{"all": [{"boolean": true}]}, {"boolean": true}]}');
    const text = printAt(context, anchors.condition, one);
    expect(text).toBe('(and(true)) and true');
    expect(serialize(parseAt(context, anchors.condition, text).value ?? null)).toBe(serialize(one));
    // And without the parentheses the reading is the author's own text, not a flattening.
    expect(toPlain(parseAt(context, anchors.condition, 'and(true) and true').value ?? null)).toEqual({
      all: [{ all: [{ boolean: true }] }, { boolean: true }],
    });
  });

  it('writes every blank of every anchor on the grammar (D5)', () => {
    for (const anchor of [anchors.expression, anchors.condition, anchors.unitExpression, anchors.unitCondition]) {
      for (const one of replacementsAt(context, anchor)) {
        const blank = blankFor(context, anchor, one);
        expect(registry.accepts(blank, anchor), `${productionLabel(one)} at ${anchor}`).toBe(true);
        // And it prints as something the parser reads back, so the text view is never stuck.
        const text = printAt(context, anchor, blank);
        expect(serialize(parseAt(context, anchor, text).value ?? null), text).toBe(serialize(blank));
      }
    }
  });

  it('wraps and unwraps only where the operand is written at the node’s own anchor (D5)', () => {
    for (const anchor of [anchors.expression, anchors.condition, anchors.unitExpression, anchors.unitCondition]) {
      const value = blankAt(context, anchor);
      for (const one of wrappersAt(context, anchor)) {
        const wrapped = wrapIn(context, anchor, value, one);
        expect(registry.accepts(wrapped, anchor), `${productionLabel(one)} at ${anchor}`).toBe(true);
        const back = unwrapTo(context, anchor, wrapped);
        expect(registry.accepts(back, anchor), `unwrap ${productionLabel(one)}`).toBe(true);
      }
    }
    // A conditional holds a *condition* in its `if`, so an expression wrapped in one becomes its
    // `then`; a comparison holds two expressions, so no condition can be wrapped in one at all.
    const conditional = wrapIn(
      context,
      anchors.expression,
      parse('{"quantity": "d"}'),
      production(anchors.expression, 'if | then | else'),
    );
    expect(printAt(context, anchors.expression, conditional)).toBe('if false then d else 0');
    expect(wrappersAt(context, anchors.condition).map(productionLabel)).toEqual(['not', 'all', 'any']);
  });

  it('names a reference the grammar accepts before one is picked', () => {
    // The one blank a schema cannot answer: `identifier` is a pattern, and `""` does not match it.
    expect(toPlain(blankFor(context, anchors.expression, production(anchors.expression, 'quantity')))).toEqual({
      quantity: UNNAMED,
    });
    expect(toPlain(blankFor(context, anchors.unitCondition, production(anchors.unitCondition, 'present')))).toEqual({
      present: UNNAMED,
    });
  });
});

describe('a name spelled like a word the text form claims', () => {
  it('is written in the tree and read back as the literal, which is the rule stated', () => {
    // `identifier` admits `true`; the text form has no escape for a name, so `{"quantity":"true"}`
    // prints `true` and reads back as `{"literal": true}` — the value moves. No name of the corpus
    // or of the reference base is one of the fifteen words (measured), and the tree view writes
    // and shows it either way; this pins the reading rather than leaving it to be discovered.
    const named = parse('{"quantity": "true"}');
    expect(printAt(context, anchors.expression, named)).toBe('true');
    expect(toPlain(parseAt(context, anchors.expression, 'true').value ?? null)).toEqual({ literal: true });
    // A keyword that begins a form of its own refuses instead, which is the other outcome.
    expect(printAt(context, anchors.expression, parse('{"quantity": "if"}'))).toBe('if');
    expect(parseAt(context, anchors.expression, 'if').value).toBeUndefined();
    // An index is safe whatever it is called: its prefix is what the text form reads it by.
    expect(toPlain(parseAt(context, anchors.expression, '$true').value ?? null)).toEqual({ index: 'true' });
  });
});

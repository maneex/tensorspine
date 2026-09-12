import { describe, expect, it } from 'vitest';

import { languageAnchors, parse, serialize, toPlain, type SchemaRegistry } from '@tensorspine/lang';
import { SchemaShapes } from '@tensorspine/store';

import { grammarAt, parseAt, printAt, type PrintContext } from '../../src/expressions/index.js';
import { presentation, readPresentation } from '../../src/presentation/index.js';
import shipped from '../../src/presentation.json';
import { registry as repositorySchemas } from '../presentation/source.js';

/**
 * The strict parser of §4.13, held to the four things the feature's own block names and to the
 * readings that depend on where a token stands.
 *
 * Everything asserted here is a consequence of `presentation.json` and the schemas, never of a
 * table in the parser: `/` is `divide` because that is the symbol bound to `divide`, `$layer` is
 * an index because `$` is the prefix bound at the index expression's own member, and `min(a, b)`
 * is `min` applied because `min` carries no symbol at all.
 */

const registry: SchemaRegistry = repositorySchemas();
const context: PrintContext = {
  registry,
  shapes: new SchemaShapes(registry),
  bindings: presentation(),
};
const anchors = languageAnchors(registry);

/** What a text parses to, as plain JSON — what the assertions below are written against. */
function read(anchor: string, text: string): unknown {
  const parsed = parseAt(context, anchor, text);
  if (parsed.value === undefined) {
    throw new Error(`${text}: ${parsed.problems.map((one) => one.message).join('; ')}`);
  }
  return toPlain(parsed.value);
}

const expression = (text: string): unknown => read(anchors.expression, text);
const condition = (text: string): unknown => read(anchors.condition, text);
const unitExpression = (text: string): unknown => read(anchors.unitExpression, text);
const unitCondition = (text: string): unknown => read(anchors.unitCondition, text);

/** The refusals a text produces, in the parser's own words. */
function refusals(anchor: string, text: string): readonly string[] {
  return parseAt(context, anchor, text).problems.map((one) => one.message);
}

describe('what the feature’s block names', () => {
  it('reads a / as divide, and never as a rounding', () => {
    // §4.13: "The parser is strict: `a / b` is `divide` (a real), never a floor — the language's
    // explicit-rounding rule is the core's to enforce (V3)".
    expect(expression('d / heads')).toEqual({
      op: 'divide',
      args: [{ quantity: 'd' }, { quantity: 'heads' }],
    });
    expect(expression('d div heads')).toEqual({
      op: 'floor_divide',
      args: [{ quantity: 'd' }, { quantity: 'heads' }],
    });
    expect(expression('d cdiv heads')).toEqual({
      op: 'ceil_divide',
      args: [{ quantity: 'd' }, { quantity: 'heads' }],
    });
  });

  it('reads $layer as an index and layer as a quantity', () => {
    expect(expression('$layer')).toEqual({ index: 'layer' });
    expect(expression('layer')).toEqual({ quantity: 'layer' });
    // And on the unit side, where the one referent is an argument path.
    expect(unitExpression('rope.scaling.kind')).toEqual({ argument: 'rope.scaling.kind' });
  });

  it('reads an operator with no symbol as name(args), which is §4.13’s own fallback', () => {
    expect(expression('min(a, b)')).toEqual({
      op: 'min',
      args: [{ quantity: 'a' }, { quantity: 'b' }],
    });
    expect(expression('max(1, 2, 3)')).toEqual({
      op: 'max',
      args: [{ literal: 1 }, { literal: 2 }, { literal: 3 }],
    });
    // It round-trips, which is the half of the sentence that makes it a fallback and not a hole.
    for (const text of ['min(a, b)', 'max(1, 2, 3)', 'abs(d)']) {
      expect(printAt(context, anchors.expression, parseAt(context, anchors.expression, text).value ?? null)).toBe(text);
    }
  });
});

describe('literals', () => {
  it('keeps a number’s own lexeme, so 1.0 is not 1 and 1e-05 is not 0.00001', () => {
    const parsed = parseAt(context, anchors.expression, '1e-05');
    expect(serialize(parsed.value ?? null)).toBe('{\n  "literal": 1e-05\n}\n');
    expect(serialize(parseAt(context, anchors.expression, '1.0').value ?? null)).toBe(
      '{\n  "literal": 1.0\n}\n',
    );
    expect(serialize(parseAt(context, anchors.expression, '1').value ?? null)).toBe(
      '{\n  "literal": 1\n}\n',
    );
  });

  it('reads a quoted string as a literal and a bare name as a reference', () => {
    expect(expression('"causal"')).toEqual({ literal: 'causal' });
    expect(expression('causal')).toEqual({ quantity: 'causal' });
    expect(unitCondition('mask = "causal"')).toEqual({
      compare: { operator: 'equal', left: { argument: 'mask' }, right: { literal: 'causal' } },
    });
    expect(unitCondition('causal = true')).toEqual({
      compare: { operator: 'equal', left: { argument: 'causal' }, right: { literal: true } },
    });
  });

  it('reads true and false as the literal an expression admits and the boolean a condition is', () => {
    expect(expression('true')).toEqual({ literal: true });
    expect(condition('true')).toEqual({ boolean: true });
    expect(condition('false')).toEqual({ boolean: false });
  });

  it('reads an escape as JSON reads it', () => {
    expect(expression('"a\\"b"')).toEqual({ literal: 'a"b' });
    expect(expression('"\\u00e9"')).toEqual({ literal: 'é' });
    expect(printAt(context, anchors.expression, parseAt(context, anchors.expression, '"a\\"b"').value ?? null)).toBe(
      '"a\\"b"',
    );
  });
});

describe('where a token stands decides what it is', () => {
  it('reads a sign against a number as the literal, and after an operand as a subtraction', () => {
    expect(expression('-1')).toEqual({ literal: -1 });
    expect(expression('- 1')).toEqual({ op: 'negate', args: [{ literal: 1 }] });
    expect(expression('d -1')).toEqual({ op: 'subtract', args: [{ quantity: 'd' }, { literal: 1 }] });
    expect(expression('d - 1')).toEqual({ op: 'subtract', args: [{ quantity: 'd' }, { literal: 1 }] });
    expect(expression('d - -1')).toEqual({ op: 'subtract', args: [{ quantity: 'd' }, { literal: -1 }] });
  });

  it('prints the two apart, which is what makes that reading exact in both directions', () => {
    // The space is not typography: the printer asks the lexer whether the symbol and its operand
    // would run into one token, and `-1` is one number where `-d` is two tokens.
    const negation = printAt(context, anchors.expression, parse('{"op": "negate", "args": [{"literal": 1}]}'));
    const minusOne = printAt(context, anchors.expression, parse('{"literal": -1}'));
    expect([negation, minusOne]).toEqual(['- 1', '-1']);
    expect(expression(negation)).toEqual({ op: 'negate', args: [{ literal: 1 }] });
    expect(expression(minusOne)).toEqual({ literal: -1 });
    expect(printAt(context, anchors.expression, parse('{"op": "negate", "args": [{"quantity": "d"}]}'))).toBe('-d');
  });

  it('reads a word-shaped symbol as a symbol only where one can stand', () => {
    // `div` between two operands is the operator; nothing else in the corpus or the base is
    // spelled like one, and a hyphen is a character an identifier admits, so `a-b` is one name.
    expect(expression('a-b')).toEqual({ quantity: 'a-b' });
    expect(expression('a - b')).toEqual({ op: 'subtract', args: [{ quantity: 'a' }, { quantity: 'b' }] });
  });

  it('reads the operand of a prefix symbol as a primary and no further', () => {
    expect(condition('not true')).toEqual({ not: { boolean: true } });
    expect(condition('not (a = b)')).toEqual({
      not: { compare: { operator: 'equal', left: { quantity: 'a' }, right: { quantity: 'b' } } },
    });
    // What the printer never writes, the parser does not read: `not` carries no precedence, so
    // its operand stops at the primary and the comparison's left side is never reached.
    expect(refusals(anchors.condition, 'not a = b')).toEqual([`'a' begins nothing this place admits`]);
  });
});

describe('precedence, associativity and the n-ary forms', () => {
  it('groups by the precedences presentation.json carries', () => {
    expect(expression('a + b * c')).toEqual({
      op: 'add',
      args: [{ quantity: 'a' }, { op: 'multiply', args: [{ quantity: 'b' }, { quantity: 'c' }] }],
    });
    expect(expression('(a + b) * c')).toEqual({
      op: 'multiply',
      args: [{ op: 'add', args: [{ quantity: 'a' }, { quantity: 'b' }] }, { quantity: 'c' }],
    });
    expect(condition('$layer mod 5 = 4 and $layer >= 4')).toEqual({
      all: [
        {
          compare: {
            operator: 'equal',
            left: { op: 'modulo', args: [{ index: 'layer' }, { literal: 5 }] },
            right: { literal: 4 },
          },
        },
        { compare: { operator: 'greater_or_equal', left: { index: 'layer' }, right: { literal: 4 } } },
      ],
    });
  });

  it('reads a chain of one n-ary operator as one application, flat', () => {
    expect(expression('a * b * c')).toEqual({
      op: 'multiply',
      args: [{ quantity: 'a' }, { quantity: 'b' }, { quantity: 'c' }],
    });
    expect(condition('true and false and true')).toEqual({
      all: [{ boolean: true }, { boolean: false }, { boolean: true }],
    });
  });

  it('reads parentheses as the nesting they are, which is what keeps the three apart', () => {
    expect(expression('a * (b * c)')).toEqual({
      op: 'multiply',
      args: [{ quantity: 'a' }, { op: 'multiply', args: [{ quantity: 'b' }, { quantity: 'c' }] }],
    });
    expect(expression('(a * b) * c')).toEqual({
      op: 'multiply',
      args: [{ op: 'multiply', args: [{ quantity: 'a' }, { quantity: 'b' }] }, { quantity: 'c' }],
    });
  });

  it('reads a chain of a binary operator left to right, which is all the language has', () => {
    expect(expression('a - b - c')).toEqual({
      op: 'subtract',
      args: [{ op: 'subtract', args: [{ quantity: 'a' }, { quantity: 'b' }] }, { quantity: 'c' }],
    });
    expect(expression('a * b div c')).toEqual({
      op: 'floor_divide',
      args: [{ op: 'multiply', args: [{ quantity: 'a' }, { quantity: 'b' }] }, { quantity: 'c' }],
    });
  });
});

describe('the forms only a binding could give the text', () => {
  it('reads if … then … else … , which §4.13’s symbol list carries', () => {
    expect(expression('if $layer >= 20 then "shared" else "own"')).toEqual({
      if: { compare: { operator: 'greater_or_equal', left: { index: 'layer' }, right: { literal: 20 } } },
      then: { literal: 'shared' },
      else: { literal: 'own' },
    });
    expect(expression('if true then 1 else 2 + 3')).toEqual({
      if: { boolean: true },
      then: { literal: 1 },
      else: { op: 'add', args: [{ literal: 2 }, { literal: 3 }] },
    });
    expect(
      printAt(
        context,
        anchors.expression,
        parseAt(context, anchors.expression, '(if true then 1 else 2) + 3').value ?? null,
      ),
    ).toBe('(if true then 1 else 2) + 3');
  });

  it('reads present(path) on the unit side, where the payload is a path and not an expression', () => {
    expect(unitCondition('present(chunk)')).toEqual({ present: 'chunk' });
    expect(unitCondition('present(rope.scaling.kind)')).toEqual({ present: 'rope.scaling.kind' });
    expect(unitCondition('not present(window)')).toEqual({ not: { present: 'window' } });
    // The model side has no such form; the text is a quantity followed by a parenthesis.
    expect(refusals(anchors.condition, 'present(chunk)').length).toBeGreaterThan(0);
  });
});

describe('a refusal is a row, never a blocked keystroke (Q5)', () => {
  it('answers no value and says where it stopped', () => {
    const parsed = parseAt(context, anchors.expression, 'd div');
    expect(parsed.value).toBeUndefined();
    expect(parsed.problems).toEqual([
      { message: 'the expression stops before it says anything', from: 5, to: 5 },
    ]);
  });

  it('names the character it could not cut', () => {
    expect(refusals(anchors.expression, 'a ; b')).toEqual([`';' begins nothing the text form writes`]);
  });

  it('names an arity the schema does not admit', () => {
    expect(refusals(anchors.expression, 'abs(a, b)')).toEqual([
      'absolute takes 1 operands, not 2',
    ]);
  });

  it('refuses a text that says more than the place admits', () => {
    expect(refusals(anchors.expression, 'd heads')).toEqual([
      `'heads' is more than the expression needs`,
    ]);
  });
});

describe('the grammar is the schemas and the bindings, and nothing else', () => {
  it('cuts exactly the marks those two carry, at each of the four anchors', () => {
    const marks = (anchor: string): readonly string[] => [...grammarAt(context, anchor).marks].sort();
    // The closure is over the *operand* anchors, so an expression carries the condition's marks
    // too — a conditional's `if` is a condition and a comparison's sides are expressions, and one
    // text can hold both. A model expression's `$` is what the unit side has not, and the unit
    // side's `present` is what the model side has not; the rest is one set.
    const shared = ['!=', '*', '+', '-', '/', '<', '<=', '=', '>', '>=', 'abs', 'and', 'cdiv', 'div', 'else', 'if', 'mod', 'not', 'or', 'then'];
    expect(marks(anchors.expression)).toEqual([...shared, '$'].sort());
    expect(marks(anchors.condition)).toEqual([...shared, '$'].sort());
    expect(marks(anchors.unitExpression)).toEqual([...shared, 'present'].sort());
    expect(marks(anchors.unitCondition)).toEqual([...shared, 'present'].sort());
  });

  it('offers one production per operator the enumerations admit', () => {
    const operators = (anchor: string): readonly string[] =>
      grammarAt(context, anchor)
        .productions.map((one) => one.operator)
        .filter((one): one is string => one !== undefined)
        .sort();
    // The model's eleven operators, from the three enumerations, and nothing invented.
    expect(operators(anchors.expression)).toEqual(
      ['negate', 'absolute', 'subtract', 'divide', 'ceil_divide', 'floor_divide', 'modulo', 'add', 'multiply', 'min', 'max'].sort(),
    );
  });
});

describe('a binding neither half of §4.13 understands', () => {
  it('prints and parses as name(args), which is §1’s answer for an unknown rendering', () => {
    // A `form` the interface has no rendering for: the loader admits it ("unknown constructs get
    // the generic widget" one level up, feature 2.2), and the two halves have to agree about what
    // it looks like. They agree on the fallback every unbound operator already takes.
    // The shipped file with one symbol's `form` replaced, so that everything else about the
    // grammar — the references, the prefixes, the other symbols — is the one under test.
    const model = anchors.expression.split('#')[0] ?? '';
    const file = readPresentation({
      ...(shipped as Record<string, unknown>),
      [`${model}#/$defs/binary_operation_expression/properties/op`]: {
        symbols: { floor_divide: { text: '⌊/⌋', form: 'circumfix' } },
      },
    });
    const odd = { registry, shapes: context.shapes, bindings: file };
    const written = parse('{"op": "floor_divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}');
    const text = printAt(odd, anchors.expression, written);
    expect(text).toBe('floor_divide(d, heads)');
    expect(serialize(parseAt(odd, anchors.expression, text).value ?? null)).toBe(serialize(written));
  });
});

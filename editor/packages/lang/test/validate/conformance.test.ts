import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  PyKeyError,
  PyOverflowError,
  PyTypeError,
  PyValueError,
  toPython,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../../src/expr/index.js';
import {
  checkDomain,
  checkType,
  formatSemanticProblems,
  type SemanticProblem,
} from '../../src/validate/index.js';

// `_check_type` and `_check_domain`: V3's type table and the model quantity's domain, the two
// leaves both the quantity side (feature 1.5) and the argument side (1.6a) call.
//
// Every value is written as JSON and read as a document is read, so `4096` is Python's `int` and
// `4096.0` its `float` — the distinction the table is there to enforce. The messages say
// `argument '…'`, which is the wording of the table itself; `checkQuantities` rewrites the first
// occurrence, and its own suite checks that.

/** A value as a document holds it: whole and real numbers kept apart (D12, feature 1.2). */
function value(json: string): PyValue {
  return toPython(parse(json));
}

/** The lines a check produced, in order. */
function lines(run: (problems: SemanticProblem[]) => void): string[] {
  const problems: SemanticProblem[] = [];
  run(problems);
  return formatSemanticProblems(problems);
}

/** The lines `checkType` produced for one value against one declared type. */
function typed(json: string, declared: string): string[] {
  return typedValue(value(json), declared);
}

/** The same, for a value that is built rather than read: one no JSON layer can hand over. */
function typedValue(given: PyValue, declared: string): string[] {
  return lines((problems) => {
    checkType(given, value(declared), 'x', problems);
  });
}

/** The lines `checkDomain` produced for one value against one declared domain. */
function inDomain(json: string, domain: string): string[] {
  return lines((problems) => {
    checkDomain(value(json), value(domain), 'x', problems);
  });
}

const CARDINALITY = '{"kind": "cardinality"}';
const REAL = '{"kind": "real"}';
const BOOLEAN = '{"kind": "boolean"}';
const ENUM = '{"kind": "enum", "values": ["bf16", "f16"]}';
const tokens = (unit = 'tokens'): string => `{"kind": "physical", "unit": "${unit}"}`;

describe('checkType', () => {
  it('accepts a whole number as a cardinality and refuses everything else', () => {
    expect(typed('4096', CARDINALITY)).toEqual([]);
    expect(typed('0', CARDINALITY)).toEqual([]);
    expect(typed('4096.0', CARDINALITY)).toEqual([
      "[V3] argument 'x' = 4096.0 is not a cardinality (non-negative integer)",
    ]);
    expect(typed('-1', CARDINALITY)).toEqual([
      "[V3] argument 'x' = -1 is not a cardinality (non-negative integer)",
    ]);
    expect(typed('"thirty-two"', CARDINALITY)).toEqual([
      "[V3] argument 'x' = 'thirty-two' is not a cardinality (non-negative integer)",
    ]);
    expect(typed('null', CARDINALITY)).toEqual([
      "[V3] argument 'x' = None is not a cardinality (non-negative integer)",
    ]);
  });

  it('decides a boolean before any numeric kind, because Python’s bool is an int', () => {
    // `isinstance(True, int)` is true in Python, so a boolean would pass the cardinality test;
    // the table tests for it first and names the declared kind in the refusal.
    expect(typed('true', CARDINALITY)).toEqual([
      "[V3] argument 'x' = True is a boolean, not cardinality",
    ]);
    expect(typed('false', REAL)).toEqual(["[V3] argument 'x' = False is a boolean, not real"]);
    expect(typed('true', BOOLEAN)).toEqual([]);
    expect(typed('"yes"', BOOLEAN)).toEqual(["[V3] argument 'x' = 'yes' is not a boolean"]);
    expect(typed('1', BOOLEAN)).toEqual(["[V3] argument 'x' = 1 is not a boolean"]);
  });

  it('takes a whole number for a real, and refuses text', () => {
    expect(typed('1e-05', REAL)).toEqual([]);
    expect(typed('1', REAL)).toEqual([]);
    expect(typed('"banana"', REAL)).toEqual(["[V3] argument 'x' = 'banana' is not a number"]);
  });

  it('requires a whole number of every physical unit but seconds', () => {
    expect(typed('4096', tokens())).toEqual([]);
    expect(typed('4096.0', tokens())).toEqual([]);
    expect(typed('2.5', tokens())).toEqual([
      "[V3] argument 'x' = 2.5 is not a whole number of tokens (only seconds is real)",
    ]);
    expect(typed('0.5', tokens('seconds'))).toEqual([]);
    expect(typed('"plenty"', tokens('bytes'))).toEqual([
      "[V3] argument 'x' = 'plenty' is not a number of bytes",
    ]);
  });

  it('compares `float(v)` with `int(v)`, so an integer past 2^53 is not a whole number', () => {
    // The tools' own arithmetic: `float(2**53 + 1)` rounds down and `int(2**53 + 1)` does not, so
    // Python's own test refuses it. Absurd for a document, and reproduced rather than corrected.
    expect(typed('9007199254740992', tokens())).toEqual([]);
    expect(typed('9007199254740993', tokens())).toEqual([
      "[V3] argument 'x' = 9007199254740993 is not a whole number of tokens (only seconds is real)",
    ]);
  });

  it('raises where `int()` and `float()` raise', () => {
    // CPython's `json` accepts `Infinity` and `NaN` as bare tokens (feature 0.3), so a document
    // can carry one; `int()` of either raises, and the tools do not catch it.
    expect(() => typed('Infinity', tokens())).toThrow(PyOverflowError);
    expect(() => typed('NaN', tokens())).toThrow(PyValueError);
    expect(() => typed('NaN', tokens())).toThrow('cannot convert float NaN to integer');
    // An integer past the double range is the other refusal, and it cannot be *read*: an integer
    // lexeme whose `Number()` is not finite is a `number` here and an exact `int` in CPython (the
    // JSON layer's limit, features 0.3 and 1.2). The value the algebra computes is exact, so this
    // is where it comes from — a derived quantity multiplying its way past 1e308.
    expect(() => typedValue(10n ** 400n, tokens())).toThrow(PyOverflowError);
    expect(() => typedValue(10n ** 400n, tokens())).toThrow('int too large to convert to float');
    // A `seconds` value never reaches the conversion, whatever it is.
    expect(typed('Infinity', tokens('seconds'))).toEqual([]);
    expect(typedValue(10n ** 400n, tokens('seconds'))).toEqual([]);
  });

  it('refuses an enum value outside its set, comparing as Python’s `in` compares', () => {
    expect(typed('"bf16"', ENUM)).toEqual([]);
    expect(typed('"fp4"', ENUM)).toEqual([
      "[V3] argument 'x' = 'fp4' is not among ['bf16', 'f16']",
    ]);
    expect(typed('1', ENUM)).toEqual(["[V3] argument 'x' = 1 is not among ['bf16', 'f16']"]);
    // `1 in [1.0]` is true in Python: `in` compares with `==`, which crosses int and float.
    expect(typed('1', '{"kind": "enum", "values": [1.0]}')).toEqual([]);
    expect(typed('1.0', '{"kind": "enum", "values": [1]}')).toEqual([]);
  });

  it('refuses an unresolved value, and a kind it does not know', () => {
    expect(
      lines((problems) => {
        checkType(UNRESOLVED, value(CARDINALITY), 'x', problems);
      }),
    ).toEqual(["[V3] argument 'x' does not resolve to a value"]);
    expect(typed('1', '{"kind": "colour"}')).toEqual([
      "[V3] argument 'x': type 'colour' is unknown to this validator",
    ]);
  });

  it('refuses a scalar where a record is declared, without needing the record check', () => {
    expect(typed('0.1', '{"kind": "record", "fields": {}}')).toEqual([
      "[V3] argument 'x' = 0.1 is not a record",
    ]);
    expect(typed('[1]', '{"kind": "record", "fields": {}}')).toEqual([
      "[V3] argument 'x' = [1] is not a record",
    ]);
  });

  it('hands a record value to the record check, and refuses to guess without one', () => {
    // The recursion is the argument side's (feature 1.6a): a quantity's type is a scalar one, so
    // reaching it here would be a defect of the caller, not a refusal of the document.
    const seen: { value: PyRecord; label: string }[] = [];
    const declared = value('{"kind": "record", "fields": {"theta": {"kind": "real"}}}');
    const problems: SemanticProblem[] = [];
    checkType(value('{"theta": 1.0}'), declared, 'rope', problems, (given, type, label) => {
      seen.push({ value: given, label });
      expect(type).toBe(declared);
    });
    expect(seen).toEqual([{ value: { theta: 1 }, label: 'rope' }]);
    expect(problems).toEqual([]);
    expect(() => {
      checkType(value('{"theta": 1.0}'), declared, 'rope', []);
    }).toThrow('no record check was supplied');
  });

  it('raises for a declaration the grammar would have refused', () => {
    expect(() => typed('1', '{}')).toThrow(PyKeyError);
    expect(() => typed('1', '{"kind": "physical"}')).toThrow(PyKeyError);
    expect(() => typed('1', '{"kind": "enum"}')).toThrow(PyKeyError);
  });
});

describe('checkDomain', () => {
  const lower = (bound: string, inclusive = true): string =>
    `{"kind": "interval", "lower": {"value": ${bound}, "inclusive": ${String(inclusive)}}}`;
  const upper = (bound: string, inclusive = true): string =>
    `{"kind": "interval", "upper": {"value": ${bound}, "inclusive": ${String(inclusive)}}}`;

  it('reads both edges, inclusive and exclusive', () => {
    expect(inDomain('4096', lower('{"literal": 4096}'))).toEqual([]);
    expect(inDomain('4096', lower('{"literal": 4096}', false))).toEqual([
      "[V3] argument 'x' = 4096 is below the domain bound 4096 (exclusive)",
    ]);
    expect(inDomain('4096', lower('{"literal": 8192}'))).toEqual([
      "[V3] argument 'x' = 4096 is below the domain bound 8192 (inclusive)",
    ]);
    expect(inDomain('4096', upper('{"literal": 4096}'))).toEqual([]);
    expect(inDomain('4096', upper('{"literal": 4096}', false))).toEqual([
      "[V3] argument 'x' = 4096 is above the domain bound 4096 (exclusive)",
    ]);
    expect(inDomain('1.0', upper('{"literal": 1}', false))).toEqual([
      "[V3] argument 'x' = 1.0 is above the domain bound 1 (exclusive)",
    ]);
  });

  it('reports the lower edge before the upper one, and an interval with no edge at all', () => {
    expect(
      inDomain(
        '4096',
        '{"kind": "interval", "lower": {"value": {"literal": 8192}, "inclusive": true},' +
          ' "upper": {"value": {"literal": 1024}, "inclusive": true}}',
      ),
    ).toEqual([
      "[V3] argument 'x' = 4096 is below the domain bound 8192 (inclusive)",
      "[V3] argument 'x' = 4096 is above the domain bound 1024 (inclusive)",
    ]);
    expect(inDomain('4096', '{"kind": "interval"}')).toEqual([]);
  });

  it('evaluates a bound in an empty scope, so one reading a quantity is skipped', () => {
    // "only a literal bound is checked here (the empty scope leaves a quantity-referencing bound
    // undecidable)" — a bound that does not resolve is not read as a limit (I7).
    expect(inDomain('4096', lower('{"quantity": "d"}'))).toEqual([]);
    expect(
      inDomain('4096', lower('{"op": "multiply", "args": [{"literal": 2}, {"literal": 4096}]}')),
    ).toEqual(["[V3] argument 'x' = 4096 is below the domain bound 8192 (inclusive)"]);
  });

  it('compares a set with `==`, and writes it as Python writes a list', () => {
    expect(inDomain('4096', '{"kind": "set", "values": [4096]}')).toEqual([]);
    expect(inDomain('4096', '{"kind": "set", "values": [4096.0]}')).toEqual([]);
    expect(inDomain('4096', '{"kind": "set", "values": [1024, 2048]}')).toEqual([
      "[V3] argument 'x' = 4096 is outside the set [1024, 2048]",
    ]);
    expect(inDomain('"fp4"', '{"kind": "set", "values": ["bf16", "f16"]}')).toEqual([
      "[V3] argument 'x' = 'fp4' is outside the set ['bf16', 'f16']",
    ]);
  });

  it('reads a `nan` as outside every bound, as every Python comparison with one is false', () => {
    expect(inDomain('NaN', lower('{"literal": 1}'))).toEqual([
      "[V3] argument 'x' = nan is below the domain bound 1 (inclusive)",
    ]);
  });

  it('raises where Python cannot order the two, naming the operator the line was written with', () => {
    expect(() => inDomain('"bf16"', lower('{"literal": 1}'))).toThrow(PyTypeError);
    expect(() => inDomain('"bf16"', lower('{"literal": 1}'))).toThrow(
      "'>=' not supported between instances of 'str' and 'int'",
    );
    expect(() => inDomain('"bf16"', lower('{"literal": 1}', false))).toThrow(
      "'>' not supported between instances of 'str' and 'int'",
    );
    expect(() => inDomain('"bf16"', upper('{"literal": 1}'))).toThrow(
      "'<=' not supported between instances of 'str' and 'int'",
    );
    expect(() => inDomain('"bf16"', upper('{"literal": 1}', false))).toThrow(
      "'<' not supported between instances of 'str' and 'int'",
    );
  });

  it('raises for a domain the grammar would have refused', () => {
    expect(() => inDomain('1', '{}')).toThrow(PyKeyError);
    expect(() => inDomain('1', '{"kind": "set"}')).toThrow(PyKeyError);
    expect(() => inDomain('1', '{"kind": "interval", "lower": {"inclusive": true}}')).toThrow(
      PyKeyError,
    );
  });
});

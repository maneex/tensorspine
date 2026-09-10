import { describe, expect, it } from 'vitest';

import { JsonParseError, parse } from '../../src/json/parse.js';
import {
  isJsonArray,
  isJsonNumber,
  isJsonObject,
  memberNames,
  toPlain,
  type JsonObject,
  type JsonValue,
} from '../../src/json/tree.js';
import { readRepositoryFile } from './repository.js';

// The parser reads what the tools read: `tools/model.py`'s `load` and
// `tools/primitive_library.py`'s `read_json` both call CPython's `json.load` with an
// `object_pairs_hook` that refuses a duplicate member name (V12), and both keep nothing else of
// the source. The core keeps two things more — member order and every number's lexeme and
// float-ness — because D12 and V3 need them.

function number(value: JsonValue | undefined): { value: number; real: boolean; lexeme: string | undefined } {
  if (value === undefined || !isJsonNumber(value)) throw new Error('not a number node');
  return { value: value.value, real: value.real, lexeme: value.lexeme };
}

function object(value: JsonValue): JsonObject {
  if (!isJsonObject(value)) throw new Error('not an object node');
  return value;
}

function item(value: JsonValue, position: number): JsonValue | undefined {
  if (!isJsonArray(value)) throw new Error('not an array node');
  return value[position];
}

function refusal(text: string): JsonParseError {
  let caught: unknown;
  try {
    parse(text);
  } catch (error: unknown) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(JsonParseError);
  return caught as JsonParseError;
}

describe('the tree a text parses to', () => {
  it('is an ordered object whose members carry their values', () => {
    expect(parse('{"a": 1, "b": "x"}')).toEqual({
      kind: 'object',
      members: [
        { name: 'a', value: { kind: 'number', value: 1, real: false, lexeme: '1' } },
        { name: 'b', value: 'x' },
      ],
    });
  });

  it('reads arrays, booleans, null and the empty containers', () => {
    expect(parse('[]')).toEqual([]);
    expect(parse('{}')).toEqual({ kind: 'object', members: [] });
    expect(parse('[true, false, null, {}, []]')).toEqual([true, false, null, { kind: 'object', members: [] }, []]);
  });

  it('reads a scalar on its own, as `json.loads` does', () => {
    expect(parse('  "text"  ')).toBe('text');
    expect(parse('true')).toBe(true);
    expect(parse('null')).toBeNull();
    expect(number(parse('12'))).toEqual({ value: 12, real: false, lexeme: '12' });
  });

  it('decodes the escapes JSON has, and the one the corpus uses', () => {
    expect(parse(String.raw`"a\nb"`)).toBe('a\nb');
    expect(parse(String.raw`"\"\\\/\b\f\n\r\t"`)).toBe('"\\/\b\f\n\r\t');
    expect(parse(String.raw`"é"`)).toBe('é');
    // A surrogate pair, written as two escapes, is one character, as it is in Python.
    expect(parse(String.raw`"😀"`)).toBe('\u{1f600}');
  });

  it('keeps text written as it stands, which is what `ensure_ascii=False` writes', () => {
    expect(parse('"§ → θ −"')).toBe('§ → θ −');
  });
});

describe('a number keeps its lexeme and the float-ness V3 reads in it', () => {
  it('reads a whole number as a cardinality and a fraction or an exponent as a real', () => {
    expect(number(parse('32'))).toEqual({ value: 32, real: false, lexeme: '32' });
    // V3: `32.0` is not a cardinality — the distinction `JSON.parse` loses.
    expect(number(parse('32.0'))).toEqual({ value: 32, real: true, lexeme: '32.0' });
    expect(number(parse('1e-05'))).toEqual({ value: 1e-5, real: true, lexeme: '1e-05' });
    expect(number(parse('1E5'))).toEqual({ value: 1e5, real: true, lexeme: '1E5' });
    expect(number(parse('-0'))).toEqual({ value: -0, real: false, lexeme: '-0' });
    expect(number(parse('0.25'))).toEqual({ value: 0.25, real: true, lexeme: '0.25' });
  });

  it('keeps the digits of a number no double holds exactly', () => {
    // The value is the nearest double, as Python's float would be; the lexeme is the source, so
    // that writing the document back writes what it said.
    expect(number(parse('1.2772588722239782'))).toEqual({
      value: 1.2772588722239782,
      real: true,
      lexeme: '1.2772588722239782',
    });
    expect(number(parse('10000000000000000001')).lexeme).toBe('10000000000000000001');
  });

  it('reads the three non-finite names `json.load` reads, keeping their text', () => {
    expect(number(parse('NaN')).lexeme).toBe('NaN');
    expect(number(parse('NaN')).value).toBeNaN();
    const both = parse('[Infinity, -Infinity]');
    expect(number(item(both, 0))).toEqual({ value: Number.POSITIVE_INFINITY, real: true, lexeme: 'Infinity' });
    expect(number(item(both, 1))).toEqual({ value: Number.NEGATIVE_INFINITY, real: true, lexeme: '-Infinity' });
  });
});

describe('member order is the document own order', () => {
  it('keeps an order a plain JavaScript object would rearrange', () => {
    const text = '{"10": 1, "2": 2, "1": 3, "b": 4, "a": 5}';
    expect(memberNames(object(parse(text)))).toEqual(['10', '2', '1', 'b', 'a']);
    // What the tree exists to avoid: a plain object enumerates the integer-like keys first, in
    // numeric order, whatever the source said.
    expect(Object.keys(JSON.parse(text) as object)).toEqual(['1', '2', '10', 'b', 'a']);
  });
});

describe('a duplicate member name is refused, as V12 requires', () => {
  interface RejectionManifest {
    cases: { document: string; expect: string; match: string }[];
  }

  it('refuses the rejection suite case with the wording that suite matches', () => {
    const manifest = JSON.parse(readRepositoryFile('tests/rejections/models.json')) as RejectionManifest;
    const rejection = manifest.cases.find((entry) => entry.document === 'models/v12-duplicate-member-name.json');
    expect(rejection).toBeDefined();
    const text = readRepositoryFile('tests/rejections/models/v12-duplicate-member-name.json');
    const error = refusal(text);

    // `tools/model.py`: `duplicate member name '<name>' (V12)`, which `validate.structural`
    // prints as `[V12] <message>` and the rejection suite matches a substring of.
    expect(error.message).toBe("duplicate member name 'model' (V12)");
    expect(error.message).toContain(rejection?.match ?? '');
    // `tools/primitive_library.py` wraps the reason instead: `<path>: <reason> (V12)`.
    expect(error.reason).toBe("duplicate member name 'model'");
    expect(error.duplicateMember).toBe('model');
    // The position of the second member, which the tools do not carry and the editor shows: the
    // fixture repeats `"model"` on its fourth line.
    expect(error.line).toBe(4);
    expect(error.column).toBe(3);
    expect(text.slice(error.offset, error.offset + 7)).toBe('"model"');
  });

  it('names the member the hook of `json.load` would name, the innermost object first', () => {
    // The hook runs when an object closes, so the inner `x` is refused before the outer `a`,
    // though `a` repeats earlier in the text.
    expect(() => parse('{"a": 1, "a": 2, "b": {"x": 1, "x": 2}}')).toThrow("duplicate member name 'x' (V12)");
    expect(() => parse('{"a": 1, "a": 2}')).toThrow("duplicate member name 'a' (V12)");
  });

  it('accepts the same name in two different objects', () => {
    expect(toPlain(parse('{"a": {"x": 1}, "b": {"x": 2}}'))).toEqual({ a: { x: 1 }, b: { x: 2 } });
  });
});

describe('a syntax error reads as CPython prints it', () => {
  const cases: [string, string, string][] = [
    ['', 'Expecting value', 'Expecting value: line 1 column 1 (char 0)'],
    ['  ', 'Expecting value', 'Expecting value: line 1 column 3 (char 2)'],
    [
      '{',
      'Expecting property name enclosed in double quotes',
      'Expecting property name enclosed in double quotes: line 1 column 2 (char 1)',
    ],
    ['{"a"', "Expecting ':' delimiter", "Expecting ':' delimiter: line 1 column 5 (char 4)"],
    ['{"a":1', "Expecting ',' delimiter", "Expecting ',' delimiter: line 1 column 7 (char 6)"],
    [
      '{"a":1,}',
      'Expecting property name enclosed in double quotes',
      'Expecting property name enclosed in double quotes: line 1 column 8 (char 7)',
    ],
    ['{"a":1 "b":2}', "Expecting ',' delimiter", "Expecting ',' delimiter: line 1 column 8 (char 7)"],
    [
      '{a:1}',
      'Expecting property name enclosed in double quotes',
      'Expecting property name enclosed in double quotes: line 1 column 2 (char 1)',
    ],
    ['[1,]', 'Expecting value', 'Expecting value: line 1 column 4 (char 3)'],
    ['[1 2]', "Expecting ',' delimiter", "Expecting ',' delimiter: line 1 column 4 (char 3)"],
    ['[01]', "Expecting ',' delimiter", "Expecting ',' delimiter: line 1 column 3 (char 2)"],
    ['[.5]', 'Expecting value', 'Expecting value: line 1 column 2 (char 1)'],
    ['[5.]', "Expecting ',' delimiter", "Expecting ',' delimiter: line 1 column 3 (char 2)"],
    ['[+1]', 'Expecting value', 'Expecting value: line 1 column 2 (char 1)'],
    ['[1e]', "Expecting ',' delimiter", "Expecting ',' delimiter: line 1 column 3 (char 2)"],
    ['[-]', 'Expecting value', 'Expecting value: line 1 column 2 (char 1)'],
    ['["a]', 'Unterminated string starting at', 'Unterminated string starting at: line 1 column 2 (char 1)'],
    ['nul', 'Expecting value', 'Expecting value: line 1 column 1 (char 0)'],
    ['tru', 'Expecting value', 'Expecting value: line 1 column 1 (char 0)'],
    ['{} {}', 'Extra data', 'Extra data: line 1 column 4 (char 3)'],
  ];

  for (const [text, reason, message] of cases) {
    it(`refuses ${JSON.stringify(text)} with ${reason}`, () => {
      const error = refusal(text);
      expect(error.message).toBe(message);
      expect(error.reason).toBe(reason);
      expect(error.duplicateMember).toBeNull();
    });
  }

  it('refuses a raw control character in a string, as a strict reading does', () => {
    expect(refusal('["a\tb"]').message).toBe('Invalid control character at: line 1 column 4 (char 3)');
  });

  it('refuses an escape JSON does not have', () => {
    expect(refusal(String.raw`["\x41"]`).message).toBe('Invalid \\escape: line 1 column 3 (char 2)');
    expect(refusal(String.raw`["\u12"]`).message).toBe('Invalid \\uXXXX escape: line 1 column 4 (char 3)');
  });

  it('counts lines and columns from where the text broke', () => {
    const error = refusal('{\n  "a": 1,\n  "b" 2\n}\n');
    expect(error.line).toBe(3);
    expect(error.column).toBe(7);
    expect(error.message).toBe("Expecting ':' delimiter: line 3 column 7 (char 18)");
  });
});

describe('the reading a plain `json.load` gives, for the one caller that needs it', () => {
  // `tools/schema.py`'s `check` reads a file with no `object_pairs_hook`, and the library loader
  // runs the schema stage before `read_json` — so a unit that is both off-schema and holds a
  // duplicate is refused for being off-schema (feature 1.3). This is that reading.
  it('keeps the last value of a repeated name, at the place of the first', () => {
    const tree = parse('{"b": 1, "a": 2, "b": 3}', { duplicates: 'last' }) as JsonObject;
    expect(memberNames(tree)).toEqual(['b', 'a']);
    expect(toPlain(tree)).toEqual({ b: 3, a: 2 });
  });

  it('folds a duplicate at every depth, and leaves the strict reading alone', () => {
    const text = '{"a": {"x": 1, "x": 2}}';
    expect(toPlain(parse(text, { duplicates: 'last' }))).toEqual({ a: { x: 2 } });
    expect(() => parse(text)).toThrow(JsonParseError);
    expect(() => parse(text, { duplicates: 'refuse' })).toThrow(JsonParseError);
  });
});

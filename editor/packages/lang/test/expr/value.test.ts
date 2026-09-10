import { describe, expect, it } from 'vitest';

import { parse, serialize } from '../../src/json/index.js';
import {
  isInteger,
  isResolved,
  pyIterate,
  pythonTypeName,
  PyTypeError,
  toJsonValue,
  toPython,
  truthy,
  UNRESOLVED,
  type PyRecord,
} from '../../src/expr/index.js';
import { interchangeFiles, readRepositoryFile } from '../json/repository.js';

// The value model: the reading CPython's `json` gives a document, in JavaScript. What it has to
// keep is the integer/float distinction the language reads (V3: `32.0` is not a cardinality) and
// the derived documents are written with — so the round trip through it is asserted over every
// file the editor loads, not over a sample.

describe('the reading of a document', () => {
  it('reads a whole number as an integer and a real as a float', () => {
    const read = toPython(parse('{"a": 32, "b": 32.0, "c": 1e-05, "d": -7, "e": 1e16}')) as PyRecord;
    expect(read['a']).toBe(32n);
    expect(read['b']).toBe(32);
    expect(typeof read['b']).toBe('number');
    expect(read['c']).toBe(1e-5);
    expect(read['d']).toBe(-7n);
    expect(read['e']).toBe(1e16);
    expect(typeof read['e']).toBe('number');
  });

  it('keeps an integer exact past the doubles, as Python’s `int` is', () => {
    const read = toPython(parse('{"n": 100000000000000000000000001}')) as PyRecord;
    expect(read['n']).toBe(100000000000000000000000001n);
  });

  it('reads strings, booleans, null, arrays and records as themselves', () => {
    expect(toPython(parse('{"a": "x", "b": true, "c": null, "d": [1, 2.0]}'))).toEqual({
      a: 'x',
      b: true,
      c: null,
      d: [1n, 2],
    });
  });

  it('keeps a member named `__proto__` a member', () => {
    const read = toPython(parse('{"__proto__": 1}')) as PyRecord;
    expect(Object.hasOwn(read, '__proto__')).toBe(true);
    expect(read['__proto__']).toBe(1n);
  });

  it('reads a number whose value no integer text denotes as a float', () => {
    // A tree the parser cannot make, but a command can: the float-ness follows the value, so V3
    // is left a `1.5` to refuse rather than an exception to raise.
    expect(toPython({ kind: 'number', value: 1.5, real: false })).toBe(1.5);
    expect(toPython({ kind: 'number', value: Number.POSITIVE_INFINITY, real: false })).toBe(
      Number.POSITIVE_INFINITY,
    );
  });

  it('reads a number from its value when its text no longer denotes it', () => {
    expect(toPython({ kind: 'number', value: 7, real: false, lexeme: '4' })).toBe(7n);
  });
});

describe('the writing back', () => {
  it('writes an integer bare and a float with its point', () => {
    expect(serialize(toJsonValue(4096n))).toBe('4096\n');
    expect(serialize(toJsonValue(4096))).toBe('4096.0\n');
    expect(serialize(toJsonValue(1e-5))).toBe('1e-05\n');
    expect(serialize(toJsonValue({ a: 1n, b: [2, 'x', true, null] }))).toBe(
      '{\n  "a": 1,\n  "b": [\n    2.0,\n    "x",\n    true,\n    null\n  ]\n}\n',
    );
  });

  it('refuses to write the sentinel: a document holds decided values only', () => {
    expect(() => toJsonValue(UNRESOLVED)).toThrow(PyTypeError);
  });

  it('round-trips every file the editor loads, byte for byte', () => {
    // tree → value → tree → bytes: the distinction the value model carries is the one D12's
    // writer needs, so every corpus document and every reference unit must come back as written.
    for (const path of interchangeFiles().filter((name) => name.endsWith('.json'))) {
      const text = readRepositoryFile(path);
      expect(serialize(toJsonValue(toPython(parse(text)))), path).toBe(text);
    }
  });
});

describe('Python’s reading of a value', () => {
  it('names a type as Python names it', () => {
    expect(pythonTypeName(1n)).toBe('int');
    expect(pythonTypeName(1)).toBe('float');
    expect(pythonTypeName(true)).toBe('bool');
    expect(pythonTypeName('x')).toBe('str');
    expect(pythonTypeName(null)).toBe('NoneType');
    expect(pythonTypeName([])).toBe('list');
    expect(pythonTypeName({})).toBe('dict');
    expect(pythonTypeName(UNRESOLVED)).toBe('object');
  });

  it('takes Python’s truth, which is not JavaScript’s for an empty container', () => {
    expect(truthy([])).toBe(false);
    expect(truthy({})).toBe(false);
    expect(truthy('')).toBe(false);
    expect(truthy(0n)).toBe(false);
    expect(truthy(0)).toBe(false);
    expect(truthy(null)).toBe(false);
    expect(truthy(false)).toBe(false);
    expect(truthy(Number.NaN)).toBe(true);
    expect(truthy(UNRESOLVED)).toBe(true);
    expect(truthy([null])).toBe(true);
  });

  it('walks a list, a record’s names and a string’s characters, and refuses the rest', () => {
    expect(pyIterate([1n, 2n])).toEqual([1n, 2n]);
    expect(pyIterate({ a: 1n, b: 2n })).toEqual(['a', 'b']);
    expect(pyIterate('ab')).toEqual(['a', 'b']);
    expect(() => pyIterate(1n)).toThrow(PyTypeError);
  });

  it('tells an integer and the sentinel apart', () => {
    expect(isInteger(1n)).toBe(true);
    expect(isInteger(true)).toBe(true);
    expect(isInteger(1)).toBe(false);
    expect(isResolved(1n)).toBe(true);
    expect(isResolved(UNRESOLVED)).toBe(false);
  });
});

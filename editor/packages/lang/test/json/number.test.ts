import { describe, expect, it } from 'vitest';

import { formatNumber, isNumberLexeme, lexemeDenotes, lexemeIsReal } from '../../src/json/number.js';

// The rendering rule of the plan's §7 F7, which D12 makes the serializer's: shortest round-trip
// digits, the exponent form below 1e-4 and from 1e16, a fraction or an exponent on every real,
// integers bare. Every expectation below is CPython 3.11's `repr` of the same double, read off
// the interpreter this repository's tools run on; the corpus's own lexemes (`1e-05` six times,
// `1e-06` seven, `1e-12` once, `1.0` twice, `0.25` five, `1.2772588722239782` three,
// `0.7071067811865476`, `0.95`, `0.1`, `2.5`) are among them.

describe('formatNumber, the real cases', () => {
  const cases: [number, string][] = [
    [1e-5, '1e-05'],
    [1e-6, '1e-06'],
    [1e-12, '1e-12'],
    [1, '1.0'],
    [0.25, '0.25'],
    [1e16, '1e+16'],
    [1.2345678901234568e17, '1.2345678901234568e+17'],
    [16, '16.0'],
    [0.1, '0.1'],
    [0.95, '0.95'],
    [2.5, '2.5'],
    [1.2772588722239782, '1.2772588722239782'],
    [0.7071067811865476, '0.7071067811865476'],
    // The two thresholds, from either side: the exponent form starts below 1e-4 and from 1e16.
    [0.0001, '0.0001'],
    [0.00012, '0.00012'],
    [1e15, '1000000000000000.0'],
    [1234567890123456, '1234567890123456.0'],
    [9999999999999998, '9999999999999998.0'],
    // 12345678901234567 as a double: the first value past the sixteenth place, so the first the
    // exponent form takes.
    [12345678901234568, '1.2345678901234568e+16'],
    [1e21, '1e+21'],
    // The exponent is signed and at least two digits, and grows past them where it must.
    [-1.5e-7, '-1.5e-07'],
    [1e100, '1e+100'],
    [1e-100, '1e-100'],
    [5e-324, '5e-324'],
    [1.7976931348623157e308, '1.7976931348623157e+308'],
    // Zero, and the negative zero a repr keeps.
    [0, '0.0'],
    [-0, '-0.0'],
    [-2.5, '-2.5'],
  ];

  for (const [value, expected] of cases) {
    it(`writes ${expected}`, () => {
      expect(formatNumber(value, true)).toBe(expected);
    });
  }
});

describe('formatNumber, the integer cases', () => {
  const cases: [number, string][] = [
    [16, '16'],
    [0, '0'],
    // A negative zero is an integer zero, as Python's `int` is.
    [-0, '0'],
    [-8, '-8'],
    [32, '32'],
    [128256, '128256'],
    [10000000, '10000000'],
    [2 ** 53, '9007199254740992'],
    // Beyond 1e21 `String(value)` would write an exponent; an integer is written in digits.
    [1e21, '1000000000000000000000'],
  ];

  for (const [value, expected] of cases) {
    it(`writes ${expected}`, () => {
      expect(formatNumber(value, false)).toBe(expected);
    });
  }
});

describe('formatNumber refuses what no document can hold', () => {
  it('refuses a fraction asked for as an integer', () => {
    expect(() => formatNumber(1.5, false)).toThrow(RangeError);
    expect(() => formatNumber(1.5, false)).toThrow('1.5 is not a whole number');
  });

  it('refuses the non-finite values, which have no JSON form', () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      expect(() => formatNumber(value, true)).toThrow(RangeError);
      expect(() => formatNumber(value, false)).toThrow(RangeError);
    }
  });
});

describe('the rule holds for any double', () => {
  // A seeded generator, so that a failure is reproducible: an xorshift over 64 bits, read as the
  // bit pattern of a double, which covers the normals, the denormals and the huge exponents.
  function* doubles(count: number): Generator<number> {
    const buffer = new DataView(new ArrayBuffer(8));
    let state = 0x9e3779b97f4a7c15n;
    for (let i = 0; i < count; i += 1) {
      state ^= (state << 13n) & 0xffffffffffffffffn;
      state ^= state >> 7n;
      state ^= (state << 17n) & 0xffffffffffffffffn;
      buffer.setBigUint64(0, state);
      const value = buffer.getFloat64(0);
      if (Number.isFinite(value)) yield value;
    }
  }

  it('round-trips every value it writes, in the form the rule states', () => {
    const sample = [...doubles(20000)];
    expect(sample.length).toBeGreaterThan(15000);
    for (const value of sample) {
      const lexeme = formatNumber(value, true);
      // Shortest round-trip digits: the text denotes exactly the double it was written from.
      expect(Object.is(Number(lexeme), value)).toBe(true);
      // A fraction or an exponent on every real, so that V3 reads it as one.
      expect(lexemeIsReal(lexeme)).toBe(true);
      expect(isNumberLexeme(lexeme)).toBe(true);
      // The exponent form below 1e-4 and from 1e16, the fixed form between them.
      const magnitude = Math.abs(value);
      const exponential = magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16);
      expect(lexeme.includes('e')).toBe(exponential);
      if (exponential) {
        expect(/e[-+][0-9]{2,}$/.test(lexeme)).toBe(true);
      }
      // The digits are the fewest that denote the value: `toExponential()`'s own.
      const digits = (text: string): string =>
        text.replace(/^-/, '').replace(/e.*$/, '').replace(/[.]/, '').replace(/0+$/, '').replace(/^0+/, '');
      expect(digits(lexeme)).toBe(digits(magnitude.toExponential()));
    }
  });

  it('writes every whole number in digits', () => {
    for (const value of [...doubles(4000)].map(Math.round).filter(Number.isFinite)) {
      const lexeme = formatNumber(value, false);
      expect(/^-?[0-9]+$/.test(lexeme)).toBe(true);
      expect(lexemeIsReal(lexeme)).toBe(false);
    }
  });
});

describe('a number token, read as V3 reads it', () => {
  it('is a real when it carries a fraction or an exponent', () => {
    for (const lexeme of ['1.0', '0.25', '1e-05', '1E5', '32.0', '-0.0', '1e+16']) {
      expect(lexemeIsReal(lexeme)).toBe(true);
    }
  });

  it('is a whole number when it carries neither', () => {
    for (const lexeme of ['0', '1', '-8', '128256']) expect(lexemeIsReal(lexeme)).toBe(false);
  });

  it('reads the three names Python writes floats with as reals', () => {
    for (const lexeme of ['NaN', 'Infinity', '-Infinity']) {
      expect(isNumberLexeme(lexeme)).toBe(true);
      expect(lexemeIsReal(lexeme)).toBe(true);
    }
  });

  it('refuses what is not a JSON number token', () => {
    for (const lexeme of ['', '+1', '01', '.5', '5.', '0x10', '1e', ' 1', 'nan']) {
      expect(isNumberLexeme(lexeme)).toBe(false);
    }
  });
});

describe('a lexeme may be written only while it still means the value', () => {
  it('holds when the text denotes the value with the same float-ness', () => {
    expect(lexemeDenotes('1e-05', 1e-5, true)).toBe(true);
    expect(lexemeDenotes('16', 16, false)).toBe(true);
    expect(lexemeDenotes('16.0', 16, true)).toBe(true);
    expect(lexemeDenotes('-0.0', -0, true)).toBe(true);
    expect(lexemeDenotes('NaN', Number.NaN, true)).toBe(true);
  });

  it('fails when the value moved, when the float-ness moved, and on a negative zero', () => {
    expect(lexemeDenotes('16.0', 32, true)).toBe(false);
    expect(lexemeDenotes('16.0', 16, false)).toBe(false);
    expect(lexemeDenotes('16', 16, true)).toBe(false);
    expect(lexemeDenotes('0.0', -0, true)).toBe(false);
    expect(lexemeDenotes('0x10', 16, false)).toBe(false);
  });
});

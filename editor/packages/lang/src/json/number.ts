/**
 * How a number is written, and how a number token is read.
 *
 * The corpus is written by Python's `json.dumps(indent=2)`, so the numbers in it are Python's
 * `repr` of a float and of an int. The plan's D12 makes the core reproduce that, and §7 F7 states
 * the rule the documentation change set will carry into the guides:
 *
 * > shortest round-trip digits, the exponent form below 1e-4 and from 1e16, a fraction or an
 * > exponent on every real, integers bare.
 *
 * Both halves of that are available in JavaScript. `Number.prototype.toExponential()` called
 * without an argument yields the shortest digit string that denotes the value exactly — the same
 * digits Python's repr produces — and the layout rules below are CPython's: the exponent form
 * when the decimal point falls at or before the fourth place after it (`decpt <= -4`) or beyond
 * the sixteenth before it (`decpt > 16`), `1e-05` with a signed, two-digit-minimum exponent, and
 * a `.0` on a real that would otherwise read as a whole number.
 *
 * The float-ness is never guessed: `formatNumber` is told whether the value is a real, because
 * `16` and `16.0` are the same double and only the declared type says which one a document means
 * (V3: `32.0` is not a cardinality).
 */

/**
 * A JSON number token, as RFC 8259 and CPython's scanner both read it: an optional minus, an
 * integer part with no leading zero, an optional fraction, an optional exponent.
 */
const NUMBER_TOKEN = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?$/;

/**
 * The three tokens CPython's `json` reads as floats beyond the JSON grammar. `tools/model.py`
 * and `tools/primitive_library.py` read documents with the default `json.load`, which accepts
 * them, so the core accepts them too and writes them back as they were written; no repository
 * file contains one, and `formatNumber` refuses to produce one.
 */
export const NON_FINITE_LEXEMES: readonly string[] = ['NaN', 'Infinity', '-Infinity'];

/** Whether the text is a number token the parser would accept, non-finite names included. */
export function isNumberLexeme(lexeme: string): boolean {
  return NUMBER_TOKEN.test(lexeme) || NON_FINITE_LEXEMES.includes(lexeme);
}

/**
 * V3's lexical rule applied to a number token: a token with a fraction or an exponent is a real,
 * one without is a whole number. The three non-finite names are floats, as they are in Python.
 */
export function lexemeIsReal(lexeme: string): boolean {
  if (NON_FINITE_LEXEMES.includes(lexeme)) return true;
  return /[.eE]/.test(lexeme);
}

/**
 * Whether a lexeme may be written for this value: it is a number token, it denotes exactly this
 * double, and its spelling agrees with the value's float-ness.
 *
 * The serializer asks before re-emitting a lexeme, so that a number whose value was changed
 * without its text — a patch applied to `value` alone — is written from the value rather than
 * from a text that no longer means it. `Object.is` is the comparison: it separates `0` from `-0`
 * and holds `NaN` equal to itself, which `===` does neither of.
 */
export function lexemeDenotes(lexeme: string, value: number, real: boolean): boolean {
  if (!isNumberLexeme(lexeme)) return false;
  if (lexemeIsReal(lexeme) !== real) return false;
  return Object.is(Number(lexeme), value);
}

/**
 * The text for a number the editor writes: Python's `repr`, chosen by the float-ness the caller
 * supplies.
 *
 * A real that is not finite has no JSON form and no repository file holds one, so it is refused
 * rather than written as `NaN`; a value that is not whole cannot be a cardinality, and asking for
 * one is refused too. Both are mistakes of the caller, never of a document.
 */
export function formatNumber(value: number, real: boolean): string {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${String(value)} has no JSON form: a document holds finite numbers only`);
  }
  if (real) return formatReal(value);
  if (!Number.isInteger(value)) {
    throw new RangeError(`${String(value)} is not a whole number: it cannot be written as an integer`);
  }
  // `String(value)` writes an exponent from 1e21 and `-0` as `0`; BigInt writes the double's
  // exact integer value, in digits, as Python's `int` repr does.
  return BigInt(value).toString();
}

/** Python's `repr` of a finite float. */
function formatReal(value: number): string {
  const sign = value < 0 || Object.is(value, -0) ? '-' : '';
  const magnitude = Math.abs(value);
  // `toExponential()` without an argument: the fewest digits that denote the value exactly.
  const [mantissa = '', exponent = '0'] = magnitude.toExponential().split('e');
  const digits = mantissa.replace('.', '');
  // The decimal point's place in the digit string: `decpt` digits stand before it.
  const decpt = Number(exponent) + 1;

  if (decpt <= -4 || decpt > 16) {
    const power = decpt - 1;
    const body = digits.length > 1 ? `${digits.slice(0, 1)}.${digits.slice(1)}` : digits;
    const magnitudeOfPower = Math.abs(power).toString().padStart(2, '0');
    return `${sign}${body}e${power < 0 ? '-' : '+'}${magnitudeOfPower}`;
  }
  if (decpt <= 0) return `${sign}0.${'0'.repeat(-decpt)}${digits}`;
  if (decpt >= digits.length) return `${sign}${digits}${'0'.repeat(decpt - digits.length)}.0`;
  return `${sign}${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
}

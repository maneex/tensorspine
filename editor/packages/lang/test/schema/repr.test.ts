import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import { pythonRepr, pythonReprString } from '../../src/schema/index.js';

// `repr`, as CPython writes it: every message of the structural stage interpolates a value with
// it, so the wording contract (plan D2, §7 F1) rests on writing a value exactly as the tools do.
// The table below is the output of `repr(json.loads(text))` for each text, read off CPython 3.11
// and pinned here; a document is parsed with the core's own parser first, so a number's
// float-ness is the one the text states (D12) and not a guess.

/** `[the JSON text, what `repr` writes for the value it denotes]`. */
const VALUES: readonly (readonly [string, string])[] = [
  ["1", "1"],
  ["1.0", "1.0"],
  ["-1", "-1"],
  ["-1.0", "-1.0"],
  ["0.25", "0.25"],
  ["1e-05", "1e-05"],
  ["1e+16", "1e+16"],
  ["1e15", "1000000000000000.0"],
  ["0.0001", "0.0001"],
  ["123456789012", "123456789012"],
  ["0.0", "0.0"],
  ["-0.0", "-0.0"],
  ["3.0e2", "300.0"],
  ["2.5", "2.5"],
  ["true", "True"],
  ["false", "False"],
  ["null", "None"],
  ["\"plain\"", "'plain'"],
  ["\"it's\"", "\"it's\""],
  ["\"say \\\"hi\\\"\"", "'say \"hi\"'"],
  ["\"both ' and \\\"\"", "'both \\' and \"'"],
  ["\"tab\\there\"", "'tab\\there'"],
  ["\"line\\nfeed\"", "'line\\nfeed'"],
  ["\"carriage\\rreturn\"", "'carriage\\rreturn'"],
  ["\"back\\\\slash\"", "'back\\\\slash'"],
  ["\"bell\\u0007\"", "'bell\\x07'"],
  ["\"del\\u007f\"", "'del\\x7f'"],
  ["\"\\u00e9\"", "'\u00e9'"],
  ["\"\\u4e2d\\u6587\"", "'\u4e2d\u6587'"],
  ["\"\\ud83d\\ude00\"", "'\ud83d\ude00'"],
  ["\"\\u2028\"", "'\\u2028'"],
  ["\"\\u00a0\"", "'\\xa0'"],
  ["\"\\ud800\"", "'\\ud800'"],
  ["\"\"", "''"],
  ["[]", "[]"],
  ["[1, 2.0, true, null, \"x\"]", "[1, 2.0, True, None, 'x']"],
  ["{}", "{}"],
  ["{\"a\": 1, \"b\": [true, null]}", "{'a': 1, 'b': [True, None]}"],
  ["{\"z\": {\"y\": {\"x\": 1.0}}}", "{'z': {'y': {'x': 1.0}}}"],
  ["{\"quote's\": 1}", "{\"quote's\": 1}"],
];

describe('pythonRepr', () => {
  it.each(VALUES.map(([text, expected]) => [text, expected] as const))(
    '%s reads as %s',
    (text, expected) => {
      expect(pythonRepr(parse(text))).toBe(expected);
    },
  );

  it('writes a whole plain number whole and a fractional one as a float', () => {
    // A value taken from a schema file is plain data with no float-ness of its own; a whole one
    // is written whole, which is what every numeric literal of the five schemas is.
    expect(pythonRepr(0)).toBe('0');
    expect(pythonRepr(512)).toBe('512');
    expect(pythonRepr(0.5)).toBe('0.5');
    expect(pythonRepr(-1)).toBe('-1');
  });

  it('writes a plain list and a plain object as Python writes them', () => {
    expect(pythonRepr(['causal', 'chunked', 'none'])).toBe("['causal', 'chunked', 'none']");
    expect(pythonRepr({ type: 'string' })).toBe("{'type': 'string'}");
    expect(pythonRepr([])).toBe('[]');
    expect(pythonRepr({})).toBe('{}');
  });

  it('names the three values JSON has no literal for', () => {
    expect(pythonRepr(true)).toBe('True');
    expect(pythonRepr(false)).toBe('False');
    expect(pythonRepr(null)).toBe('None');
  });

  it('chooses the quote the way `repr` chooses it', () => {
    expect(pythonReprString('plain')).toBe("'plain'");
    expect(pythonReprString("it's")).toBe('"it\'s"');
    expect(pythonReprString('say "hi"')).toBe('\'say "hi"\'');
    expect(pythonReprString('both \' and "')).toBe('\'both \\\' and "\'');
  });

  it('refuses a value that has no JSON form', () => {
    expect(() => pythonRepr(undefined)).toThrow(TypeError);
  });
});

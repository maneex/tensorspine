import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/parse.js';
import { serialize } from '../../src/json/serialize.js';
import {
  isJsonObject,
  jsonInteger,
  jsonObject,
  jsonReal,
  memberNames,
  withMember,
  withoutMember,
  type JsonObject,
  type JsonValue,
} from '../../src/json/tree.js';

// D12 and §5.5: the serializer is the writer of record. Its output is what
// `json.dumps(document, indent=2, ensure_ascii=False)` writes, with a trailing newline — the form
// every file of the corpus and of the reference base is written in.

function object(value: JsonValue): JsonObject {
  if (!isJsonObject(value)) throw new Error('not an object node');
  return value;
}

describe('the layout is the one the corpus is written in', () => {
  it('indents by two, puts one member to a line, and ends with a newline', () => {
    const compact = '{"a": 1, "b": [], "c": {}, "d": [1, [2]], "e": "x", "f": ""}';
    // Character for character, `json.dumps(…, indent=2, ensure_ascii=False) + "\n"`.
    expect(serialize(parse(compact))).toBe(
      [
        '{',
        '  "a": 1,',
        '  "b": [],',
        '  "c": {},',
        '  "d": [',
        '    1,',
        '    [',
        '      2',
        '    ]',
        '  ],',
        '  "e": "x",',
        '  "f": ""',
        '}',
        '',
      ].join('\n'),
    );
  });

  it('writes an empty document as the empty object, and a lone scalar as itself', () => {
    expect(serialize(jsonObject())).toBe('{}\n');
    expect(serialize([])).toBe('[]\n');
    expect(serialize('text')).toBe('"text"\n');
    expect(serialize(true)).toBe('true\n');
    expect(serialize(null)).toBe('null\n');
  });

  it('ends with exactly one newline', () => {
    const text = serialize(parse('{"a": [1, 2]}'));
    expect(text.endsWith('}\n')).toBe(true);
    expect(text.endsWith('\n\n')).toBe(false);
  });
});

describe('text is written as `ensure_ascii=False` writes it', () => {
  it('leaves every non-ASCII character as it stands', () => {
    // The characters the reference base is written with, none of them escaped.
    expect(serialize('§ ± · ¹ ½ × Φ θ — … ⁻ → − √ ⊙')).toBe('"§ ± · ¹ ½ × Φ θ — … ⁻ → − √ ⊙"\n');
    expect(serialize(parse('{"θ": "√2"}'))).toBe('{\n  "θ": "√2"\n}\n');
  });

  it('escapes what JSON escapes, and nothing else', () => {
    expect(serialize('a\nb')).toBe(String.raw`"a\nb"` + '\n');
    expect(serialize('"\\/\b\f\n\r\t')).toBe(String.raw`"\"\\/\b\f\n\r\t"` + '\n');
    // A control character with no name of its own: `\u00xx`, in lower case, as Python writes it.
    expect(serialize(String.fromCharCode(1))).toBe('"' + String.raw`\u0001` + '"' + String.fromCharCode(10));
    // The delete character is not a control character JSON escapes: Python leaves it, so do we.
    expect(serialize(String.fromCharCode(0x7f))).toBe(`"${String.fromCharCode(0x7f)}"\n`);
  });
});

describe('a number is written with the text it was read with', () => {
  it('re-emits the corpus lexemes untouched', () => {
    for (const lexeme of ['1e-05', '1e-06', '1e-12', '1.0', '0.25', '1.2772588722239782', '32', '0', '-0']) {
      expect(serialize(parse(lexeme))).toBe(`${lexeme}\n`);
    }
  });

  it('re-emits what a plain writer would change', () => {
    expect(serialize(parse('[1e-05, 1.0, 1e+16]'))).toBe('[\n  1e-05,\n  1.0,\n  1e+16\n]\n');
    // What `JSON.stringify` makes of the same three: JavaScript keeps the fixed form from 1e-7
    // to 1e21, where Python's repr leaves it below 1e-4 and from 1e16, and it writes no `.0`.
    expect(JSON.stringify([1e-5, 1.0, 1e16])).toBe('[0.00001,1,10000000000000000]');
  });

  it('writes a number the editor makes from the float-ness its caller states', () => {
    expect(serialize(jsonReal(16))).toBe('16.0\n');
    expect(serialize(jsonInteger(16))).toBe('16\n');
    expect(serialize(jsonReal(1e-5))).toBe('1e-05\n');
    expect(serialize(jsonReal(-0))).toBe('-0.0\n');
    expect(serialize(jsonInteger(-0))).toBe('0\n');
  });

  it('writes the value, not the text, when the two no longer agree', () => {
    // A patch that moved the value and left the lexeme behind would otherwise write a document
    // that means something else than the tree the editor validated.
    expect(serialize({ kind: 'number', value: 32, real: true, lexeme: '16.0' })).toBe('32.0\n');
    expect(serialize({ kind: 'number', value: 16, real: false, lexeme: '16.0' })).toBe('16\n');
    expect(serialize({ kind: 'number', value: 16, real: true, lexeme: '16' })).toBe('16.0\n');
    // A lexeme that is not a number token at all is never copied out.
    expect(serialize({ kind: 'number', value: 16, real: false, lexeme: '0x10' })).toBe('16\n');
  });

  it('keeps a non-finite number as it was written, and refuses to invent one', () => {
    expect(serialize(parse('[NaN, Infinity, -Infinity]'))).toBe('[\n  NaN,\n  Infinity,\n  -Infinity\n]\n');
    expect(() => serialize(jsonReal(Number.NaN))).toThrow(RangeError);
  });
});

describe('what the writer refuses', () => {
  it('refuses a bare JavaScript number, whose float-ness nothing states', () => {
    expect(() => serialize(16 as unknown as JsonValue)).toThrow(TypeError);
    expect(() => serialize(16 as unknown as JsonValue)).toThrow('jsonInteger or jsonReal');
    const tree = withMember(object(parse('{"a": 1}')), 'b', 2 as unknown as JsonValue);
    expect(() => serialize(tree)).toThrow('/b');
  });

  it('refuses a value that is not of the tree, naming where it is', () => {
    expect(() => serialize(undefined as unknown as JsonValue)).toThrow('undefined at the root');
    const tree = withMember(object(parse('{"a": 1}')), 'b/c', { note: 'plain' } as unknown as JsonValue);
    expect(() => serialize(tree)).toThrow('an object of no kind at /b~1c');
  });
});

describe('an edit keeps the order of everything it did not touch', () => {
  const document = '{\n  "schema": "tensorspine/2.0",\n  "model": "llama3-8b",\n  "quantities": {}\n}\n';

  it('puts a new member where the gesture put it', () => {
    const tree = withMember(object(parse(document)), 'primitive_libraries', [], 2);
    expect(memberNames(tree)).toEqual(['schema', 'model', 'primitive_libraries', 'quantities']);
    expect(serialize(tree)).toBe(
      [
        '{',
        '  "schema": "tensorspine/2.0",',
        '  "model": "llama3-8b",',
        '  "primitive_libraries": [],',
        '  "quantities": {}',
        '}',
        '',
      ].join('\n'),
    );
  });

  it('appends a new member when no position is given', () => {
    const tree = withMember(object(parse(document)), 'interfaces', jsonObject());
    expect(memberNames(tree)).toEqual(['schema', 'model', 'quantities', 'interfaces']);
  });

  it('leaves an edited member where it stands, and writes the rest as it was', () => {
    const tree = withMember(object(parse(document)), 'model', 'llama3-8b-2');
    expect(memberNames(tree)).toEqual(['schema', 'model', 'quantities']);
    expect(serialize(tree)).toBe(document.replace('llama3-8b', 'llama3-8b-2'));
  });

  it('writes the document it read when the edit is undone', () => {
    const tree = object(parse(document));
    const edited = withMember(tree, 'model', 'other');
    const reverted = withMember(edited, 'model', 'llama3-8b');
    expect(serialize(reverted)).toBe(document);
    expect(serialize(withoutMember(withMember(tree, 'note', 'x'), 'note'))).toBe(document);
  });
});

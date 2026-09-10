import { describe, expect, it } from 'vitest';

import {
  layoutVocabulary,
  legacyLayout,
  loadLibrary,
  memorySource,
  pyRepr,
  pyStr,
  readRefusal,
  readText,
} from '../../src/library/index.js';
import { UNRESOLVED } from '../../src/expr/index.js';

// `primitive_library.read_json`, which carries four refusals and wraps every one of them as
// `<path>: <reason> (V12)`: a text that is not JSON, a duplicate member name, the earlier untagged
// field layout, and a `schema` revision this reading does not support.
//
// Two of the four cannot be reached through the loader as the schemas stand: both the unit schema
// and the model schema fix `schema` as a `const`, so the structural stage refuses an unsupported
// revision before `read_json` sees it, and a legacy document fails that same `const`. They are
// tested here directly, because the refusals exist and a workspace may hold such a file.

/** The reading of a text that parses, with no refusal wanted. */
function read(text: string) {
  const reading = readText('u.json', text);
  expect(reading.problem).toBeNull();
  return reading.read!;
}

describe('the four refusals of read_json', () => {
  it('states a text that is not JSON in CPython’s words, under V12', () => {
    const reading = readText('u.json', '{');
    expect(reading.read).toBeNull();
    expect(reading.problem?.code).toBe('V12');
    expect(reading.problem?.message).toBe(
      'u.json: Expecting property name enclosed in double quotes: line 1 column 2 (char 1) (V12)',
    );
  });

  it('states a duplicate member name once the caller has had its plain reading', () => {
    const reading = readText('u.json', '{"a": 1, "a": 2}');
    expect(reading.problem).toBeNull();
    expect(reading.read?.duplicate?.duplicateMember).toBe('a');
    expect(readRefusal('u.json', reading.read!)?.message).toBe(
      "u.json: duplicate member name 'a' (V12)",
    );
  });

  it('refuses the earlier untagged field layout, naming the conversion', () => {
    const reading = read('{"schema": "tensorspine/2.0", "instances": {"a": {"contract": {}}}}');
    expect(readRefusal('u.json', reading)?.message).toBe(
      'u.json: legacy field layout; convert it with python3 tools/migrate.py INPUT -o OUTPUT (V12)',
    );
  });

  it('refuses a tensorspine revision this reading does not support', () => {
    const reading = read('{"schema": "tensorspine-catalog-unit/2.0", "kind": "contract"}');
    expect(readRefusal('u.json', reading)?.message).toBe(
      "u.json: unsupported revision 'tensorspine-catalog-unit/2.0'; convert supported legacy " +
        'input with python3 tools/migrate.py INPUT -o OUTPUT (V12)',
    );
  });

  it('judges no revision that is not ours at all', () => {
    expect(readRefusal('u.json', read('{"schema": "something/1"}'))).toBeNull();
    expect(readRefusal('u.json', read('{"schema": 3}'))).toBeNull();
    expect(readRefusal('u.json', read('[]'))).toBeNull();
  });

  it('accepts the four revisions the loader reads, and the fixture format', () => {
    for (const revision of [
      'tensorspine/2.0',
      'tensorspine-primitive-library-unit/2.0',
      'tensorspine-derived/2.1',
      'tensorspine-capabilities/1',
      'tensorspine-fixture/1',
    ]) {
      expect(readRefusal('u.json', read(`{"schema": ${JSON.stringify(revision)}}`))).toBeNull();
    }
  });
});

describe('the field layout a document is written in', () => {
  it('is legacy when a renamed key stands where a format name stands', () => {
    expect(legacyLayout({ instances: { a: { contract: { name: 'n' } } } })).toBe(true);
    expect(layoutVocabulary({ instances: { a: { contract: { name: 'n' } } } })).toEqual(
      new Set(['current', 'legacy']),
    );
  });

  it('is current when every key is', () => {
    expect(legacyLayout({ primitive_libraries: [], instances: { a: { primitive: {} } } })).toBe(false);
  });

  it('says nothing about a name an author chose', () => {
    // `instances` and `arguments` are keyed by authored identities, so a site called `contract`
    // or an argument called `cut` is not a legacy field name.
    expect(legacyLayout({ instances: { contract: { primitive: {} } } })).toBe(false);
    expect(layoutVocabulary({ arguments: { contract: 1 } })).toEqual(new Set());
  });

  it('reads `instances` under `d1` as the expansion provenance it is', () => {
    expect(layoutVocabulary({ d1: { instances: {} } })).toEqual(new Set());
  });
});

describe('a value written as a message writes it', () => {
  it('is `repr` for a value of the algebra, the int/float distinction kept', () => {
    expect(pyRepr(4n)).toBe('4');
    expect(pyRepr(4)).toBe('4.0');
    expect(pyRepr(1e-5)).toBe('1e-05');
    expect(pyRepr(true)).toBe('True');
    expect(pyRepr(null)).toBe('None');
    expect(pyRepr("it's")).toBe('"it\'s"');
    expect(pyRepr(['a', 1n])).toBe("['a', 1]");
    expect(pyRepr({ kind: 'window', span: 4096n })).toBe("{'kind': 'window', 'span': 4096}");
    expect(pyRepr(UNRESOLVED)).toBe('<object object>');
  });

  it('is `str` for a string interpolated bare, which is what `_provide` writes', () => {
    expect(pyStr('model.width')).toBe('model.width');
    expect(pyStr(4n)).toBe('4');
  });
});

describe('the loader on a text it cannot read', () => {
  it('names the file and goes on to the next unit', () => {
    const library = loadLibrary(['base'], {
      schemas: null,
      source: memorySource({ 'base/axes/a.json': '{', 'base/axes/b.json': '{' }),
    });
    // The tools raise at the first; the port continues and marks what follows (plan §3).
    expect(library.problems).toHaveLength(2);
    expect(library.problems[0]?.afterRefusal).toBe(false);
    expect(library.problems[1]?.afterRefusal).toBe(true);
  });
});

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import { pythonPattern, pythonRegExp } from '../../src/schema/index.js';
import { repositorySchemas } from './repository.js';

// A schema `pattern` is `re.search` on the Python side, and Python's `$` — without
// `re.MULTILINE` — matches at the end of the string **and just before a single trailing newline**.
// JavaScript's matches only at the end. Nothing in the repository carries such a value, which is
// exactly why neither reader of the editor could find it on its own: Ajv and the walk compile the
// same text and would have agreed with each other and both refused what `--validate` accepts.
//
// The rows below are `bool(re.search(pattern, text))` read off CPython 3.11 and pinned here, as
// `repr.test.ts`'s table is.

const registry = repositorySchemas();

/** `[the pattern, the text, what Python's `re.search` answers]`. */
const SEARCHES: readonly (readonly [string, string, boolean])[] = [
  ['^[A-Za-z_][A-Za-z0-9_-]*$', 'embed', true],
  ['^[A-Za-z_][A-Za-z0-9_-]*$', 'embed\n', true],
  ['^[A-Za-z_][A-Za-z0-9_-]*$', 'embed\n\n', false],
  ['^[A-Za-z_][A-Za-z0-9_-]*$', 'embed\nx', false],
  ['^[A-Za-z_][A-Za-z0-9_-]*$', '\nembed', false],
  ['^\\S+$', 'x', true],
  ['^\\S+$', 'x\n', true],
  ['^[^\\r\\n]*\\S$', 'a b', true],
  ['^[^\\r\\n]*\\S$', 'a b\n', true],
  ['^[^\\r\\n]*\\S$', 'a b \n', false],
  ['\\S', ' x ', true],
  ['\\S', '  ', false],
  ['^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$', '1.0.0', true],
  ['^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$', '1.0.0\n', true],
];

describe('a schema pattern, as Python’s `re` reads it', () => {
  it.each(SEARCHES.map((row) => row))('%s matches %j: %s', (pattern, text, expected) => {
    expect(pythonRegExp(pattern).test(text)).toBe(expected);
  });

  it('translates every `$` that is neither escaped nor inside a class', () => {
    expect(pythonPattern('^a$')).toBe('^a(?=\\n?$)');
    expect(pythonPattern('^a$|^b$')).toBe('^a(?=\\n?$)|^b(?=\\n?$)');
    // A `$` that stands for the character itself is not a position and is left alone.
    expect(pythonPattern('^a\\$$')).toBe('^a\\$(?=\\n?$)');
    expect(pythonPattern('^[a$b]$')).toBe('^[a$b](?=\\n?$)');
    // And a pattern with no `$` comes back as it was written.
    expect(pythonPattern('\\S')).toBe('\\S');
    expect(pythonPattern('^[A-Za-z_][A-Za-z0-9_-]*')).toBe('^[A-Za-z_][A-Za-z0-9_-]*');
  });

  it('leaves every pattern of the five schemas matching what it matched', () => {
    // The translation must not change any verdict the repository's own files get: the corpus and
    // the reference base are structurally valid before it and after it, which the suite beside
    // this one asserts over every file. Here is the narrower statement it rests on — the fourteen
    // patterns still accept the strings they are written for.
    for (const [text, pattern] of [
      ['embed', '^[A-Za-z_][A-Za-z0-9_-]*$'],
      ['decoder/attn', '^[A-Za-z_][A-Za-z0-9_-]*(/[A-Za-z0-9_-]+)*$'],
      ['attention.dense', '^[A-Za-z_][A-Za-z0-9_-]*(\\.[A-Za-z_][A-Za-z0-9_-]*)*$'],
      ['1.0.0', '^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$'],
      ['sequence-operator', '^[a-z][a-z0-9_-]*$'],
    ] as const) {
      expect(pythonRegExp(pattern).test(text), `${pattern} on ${text}`).toBe(true);
    }
  });

  it('is the rule Ajv is compiled with, not only the walk’s', () => {
    // Ajv pronounces the verdict (D4) and the walk explains it, so a translation applied to one
    // and not the other would make the two disagree about the same document. A member name with a
    // newline on it — what a paste can produce — is the case: the tools take it at the schema
    // stage, so the editor must too.
    const document = {
      schema: 'tensorspine/2.0',
      model: 'm',
      primitive_libraries: [{ base: 'b' }],
      quantities: { d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } } },
      constants: {},
      instances: {
        'embed\n': {
          primitive: { name: 'norm.rms', version: '1.0.0' },
          arguments: {},
          families: ['norm'],
        },
      },
      compositions: {},
      bindings: { values: {}, parameters: {}, constants: {}, states: {} },
      interfaces: { inputs: {}, outputs: {} },
    };
    const problems = registry.structural(parse(JSON.stringify(document)), 'model');
    expect(problems.map((one) => one.message).filter((one) => one.includes('does not match'))).toEqual([]);
  });
});

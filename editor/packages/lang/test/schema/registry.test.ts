import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  comparePaths,
  comparePythonStrings,
  formatProblem,
  formatProblems,
  isType,
  loadSchemas,
  pointerOf,
  SchemaLoadError,
  whereOf,
} from '../../src/schema/index.js';
import { readRepositoryFile } from '../json/repository.js';
import { repositorySchemas } from './repository.js';

// `tools/schema.py`'s `discover`, `locate` and `format_error`, and the JSON layer that runs before
// the grammar. The parity suites hold the messages to account; this one holds the reading of a
// set of schema files, and the shapes the interface will address a problem by.

const PROBE = 'https://tensorspine.dev/schema/test/probe.json';

function probe(body: Record<string, unknown>, id: string = PROBE): { path: string; text: string } {
  return { path: 'probe.json', text: JSON.stringify({ $id: id, ...body }) };
}

describe('loadSchemas', () => {
  it('indexes a schema under its own $id and reads its role off the last segment', () => {
    const registry = loadSchemas([probe({ type: 'object' })]);
    expect(registry.schemas).toHaveLength(1);
    expect(registry.byId(PROBE)?.role).toBe('probe');
    expect(registry.locate('probe')?.id).toBe(PROBE);
  });

  it('skips a file that declares no $id, as `discover` skips it', () => {
    const registry = loadSchemas([
      { path: 'a.json', text: JSON.stringify({ type: 'object' }) },
      probe({ type: 'object' }),
    ]);
    expect(registry.schemas.map((one) => one.id)).toEqual([PROBE]);
  });

  it('keeps one entry per identity, the last file read', () => {
    const registry = loadSchemas([
      { path: 'a.json', text: JSON.stringify({ $id: PROBE, type: 'object' }) },
      { path: 'b.json', text: JSON.stringify({ $id: PROBE, type: 'array' }) },
    ]);
    expect(registry.schemas).toHaveLength(1);
    expect(registry.byId(PROBE)?.path).toBe('b.json');
    expect(registry.conforms(parse('[]'), 'probe')).toBe(true);
  });

  it('reads the files in the order a sorted directory listing gives them', () => {
    const registry = loadSchemas([
      { path: 'z.json', text: JSON.stringify({ $id: `${PROBE}#z`.replace('#z', 'z.json') }) },
      probe({ type: 'object' }),
    ]);
    expect(registry.schemas.map((one) => one.path)).toEqual(['probe.json', 'z.json']);
  });

  it('refuses a file that is not a JSON object', () => {
    expect(() => loadSchemas([{ path: 'a.json', text: '[1]' }])).toThrow(SchemaLoadError);
    expect(() => loadSchemas([{ path: 'a.json', text: 'not json' }])).toThrow(SchemaLoadError);
  });

  it('names the directory the schemas came from when a role is missing', () => {
    const registry = loadSchemas([probe({ type: 'object' })], { origin: 'vendor/schemas' });
    expect(formatProblems(registry.structural(parse('{}'), 'model'))).toEqual([
      'no schema with $id ending in /model.json under vendor/schemas/',
    ]);
    expect(registry.conforms(parse('{}'), 'model')).toBe(false);
  });

  it('resolves a $ref into another schema of the registry', () => {
    const other = 'https://tensorspine.dev/schema/test/other.json';
    const registry = loadSchemas([
      { path: 'a-other.json', text: JSON.stringify({ $id: other, $defs: { word: { type: 'string' } } }) },
      probe({ type: 'object', properties: { w: { $ref: `${other}#/$defs/word` } } }),
    ]);
    expect(formatProblems(registry.structural(parse('{"w": 1}'), 'probe'))).toEqual([
      "w: 1 is not of type 'string'",
    ]);
  });

  it('refuses a $ref the registry cannot follow', () => {
    const registry = loadSchemas([probe({ $ref: 'https://elsewhere.example/x.json' })]);
    // Ajv compiles the schema before anything is validated against it, so its own refusal is the
    // one a reader sees; the walk refuses the same reference in its own words when it is reached
    // without a compiled validator.
    expect(() => registry.structural(parse('{}'), 'probe')).toThrow(
      /can't resolve reference https:\/\/elsewhere\.example\/x\.json/,
    );
    expect(() => registry.explain(parse('{}'), 'probe')).toThrow(SchemaLoadError);
  });
});

describe('the JSON layer, before the grammar', () => {
  const registry = repositorySchemas();

  it('refuses a duplicate member name in the tools words (V12)', () => {
    const problems = registry.structuralText(
      readRepositoryFile('tests/rejections/models/v12-duplicate-member-name.json'),
    );
    expect(problems).toHaveLength(1);
    expect(problems[0]?.code).toBe('V12');
    expect(formatProblem(problems[0]!)).toBe("[V12] duplicate member name 'model' (V12)");
  });

  it('states a text that is not JSON at all rather than crossing the stage', () => {
    const problems = registry.structuralText('{"model": ');
    expect(problems).toHaveLength(1);
    expect(problems[0]?.code).toBe('V12');
    expect(problems[0]?.message).toMatch(/^Expecting value: line 1 column 11 \(char 10\)$/);
  });
});

describe('a problem names its place', () => {
  const registry = repositorySchemas();

  it('carries the pointer, the segments and the tools rendering of one place', () => {
    const document = JSON.parse(
      readRepositoryFile('tests/rejections/models/structural-interface-old-form.json'),
    ) as Record<string, unknown>;
    const problems = registry.structural(parse(JSON.stringify(document)));
    const deepest = problems[problems.length - 1];
    expect(deepest?.segments).toEqual(['interfaces', 'inputs', 'tokens', 'to']);
    expect(deepest?.path).toBe('/interfaces/inputs/tokens/to');
    expect(deepest?.keyword).toBe('type');
    expect(formatProblem(deepest!)).toBe(
      "interfaces/inputs/tokens/to: {'instance': {'kind': 'root', 'instance': 'embed'}, 'port': 'tokens'} is not of type 'array'",
    );
  });

  it('writes a pointer as RFC 6901 writes one', () => {
    expect(pointerOf([])).toBe('');
    expect(pointerOf(['instances', 'embed'])).toBe('/instances/embed');
    expect(pointerOf(['a', 0, 'b'])).toBe('/a/0/b');
    expect(pointerOf(['a/b', 'c~d'])).toBe('/a~1b/c~0d');
    expect(whereOf([])).toBe('<root>');
    expect(whereOf(['definition', 'state_ports', 'kv', 'rules', 3])).toBe(
      'definition/state_ports/kv/rules/3',
    );
  });
});

describe('the order two places are reported in', () => {
  it('is Python`s, element by element, the shorter path first', () => {
    expect(comparePaths(['a'], ['b'])).toBeLessThan(0);
    expect(comparePaths(['a'], ['a', 'b'])).toBeLessThan(0);
    expect(comparePaths([1], [2])).toBeLessThan(0);
    expect(comparePaths([10], [2])).toBeGreaterThan(0);
    expect(comparePaths(['a', 'b'], ['a', 'b'])).toBe(0);
  });

  it('compares two names by code point, as Python does and JavaScript does not', () => {
    // `'\uFFFD'` is one code unit and `'\u{10000}'` is two, the first of which is below it; the
    // order of the two therefore differs between a code-unit comparison and a code-point one.
    expect(comparePythonStrings('\uFFFD', '\u{10000}')).toBeLessThan(0);
    expect('\uFFFD' < '\u{10000}').toBe(false);
    expect(comparePaths(['\uFFFD'], ['\u{10000}'])).toBeLessThan(0);
    expect(comparePythonStrings('a', 'ab')).toBeLessThan(0);
    expect(comparePythonStrings('b', 'a')).toBeGreaterThan(0);
    expect(comparePythonStrings('a', 'a')).toBe(0);
  });

  it('orders an index before a name, where Python raises instead', () => {
    // `sorted(errors, key=lambda e: list(e.absolute_path))` compares an int with a str and raises
    // `TypeError`; nothing in the repository produces such a pair, and raising is not an answer
    // the editor can give, so the order is stated rather than left to chance.
    expect(comparePaths(['a', 0], ['a', 'b'])).toBeLessThan(0);
    expect(comparePaths(['a', 'b'], ['a', 0])).toBeGreaterThan(0);
  });
});

describe('the type checker the error ranking asks', () => {
  it('reads 2020-12`s types, a whole float among the integers', () => {
    expect(isType(parse('1.0'), 'integer')).toBe(true);
    expect(isType(parse('1.5'), 'integer')).toBe(false);
    expect(isType(parse('1'), 'number')).toBe(true);
    expect(isType(true, 'boolean')).toBe(true);
    expect(isType(true, 'number')).toBe(false);
    expect(isType(parse('[]'), 'array')).toBe(true);
    expect(isType(parse('{}'), 'object')).toBe(true);
    expect(isType(null, 'null')).toBe(true);
    expect(isType('x', 'string')).toBe(true);
    expect(isType('x', 'nothing')).toBe(false);
  });
});

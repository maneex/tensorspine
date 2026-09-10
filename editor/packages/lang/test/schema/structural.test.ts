import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import { formatProblem, formatProblems } from '../../src/schema/index.js';
import { filesUnder, readRepositoryFile } from '../json/repository.js';
import { baseFiles, rejectionCases, repositorySchemas } from './repository.js';

// The structural stage of `tools/validate.py`, ported (feature 1.1). Three things are held to
// account here, and they are the feature's own tests:
//
//   * every corpus document and every unit of the reference base is structurally valid;
//   * every structural case of `tests/rejections` is refused, with the code and the message
//     substring the manifest names — the wording contract of the plan's D2 and §7 F1;
//   * the walk that phrases a refusal and the Ajv validator that pronounces it agree, everywhere,
//     which is what keeps Ajv the validator (D4) and the walk its explanation.

const registry = repositorySchemas();

/** The unit schema is applied to a unit as `primitive_library` applies it: the deepest leaves. */
const DEEPEST = { deepest: true } as const;

/** Which role a repository file is read under: `data/models/` are documents, the base's units. */
function roleOf(path: string): 'model' | 'primitive-library-unit' {
  return path.startsWith('data/models/') ? 'model' : 'primitive-library-unit';
}

describe('the schema registry', () => {
  it('indexes every schema of the repository under its own $id', () => {
    const files = filesUnder('schemas').filter((path) => path.endsWith('.json'));
    expect(registry.schemas).toHaveLength(files.length);
    for (const schema of registry.schemas) {
      expect(schema.id).toMatch(/^https:\/\/tensorspine\.dev\/schema\//);
      expect(registry.byId(schema.id)).toBe(schema);
    }
  });

  it('locates a schema by the role its $id ends with, as `schema.locate` does', () => {
    for (const role of ['model', 'primitive-library-unit', 'documentation', 'derived']) {
      const found = registry.locate(role);
      expect(found, role).toBeDefined();
      expect(found?.id.endsWith(`/${role}.json`), role).toBe(true);
    }
    expect(registry.locate('nowhere')).toBeUndefined();
  });

  it('answers a role it does not hold in the words `validate.structural` uses', () => {
    const problems = registry.structural(parse('{}'), 'nowhere');
    expect(problems).toHaveLength(1);
    expect(formatProblem(problems[0]!)).toBe(
      'no schema with $id ending in /nowhere.json under schemas/',
    );
  });
});

describe('the corpus and the reference base', () => {
  const corpus = filesUnder('data/models').filter((path) => path.endsWith('.json'));
  const base = filesUnder('data/primitive-library').filter((path) => path.endsWith('.json'));

  it('has documents and units to read', () => {
    expect(corpus.length).toBeGreaterThan(10);
    expect(base.length).toBeGreaterThan(100);
  });

  it.each([...corpus, ...base])('%s is on its schema', (path) => {
    const role = roleOf(path);
    const tree = parse(readRepositoryFile(path));
    expect(registry.conforms(tree, role)).toBe(true);
    expect(formatProblems(registry.structural(tree, role))).toEqual([]);
    // The walk without the shortcut: Ajv's verdict and its explanation must agree.
    expect(formatProblems(registry.explain(tree, role))).toEqual([]);
  });
});

describe('the structural cases of tests/rejections/models.json', () => {
  const cases = rejectionCases('models').filter((one) => one.expect === 'schema');

  it('are the seven the manifest carries', () => {
    expect(cases).toHaveLength(7);
  });

  it.each(cases.map((one) => [one.document as string, one.match] as const))(
    '%s is refused for %s',
    (document, match) => {
      const problems = registry.structuralText(readRepositoryFile(`tests/rejections/${document}`));
      expect(problems.length).toBeGreaterThan(0);
      // `expect: "schema"` names the stage, and the stage has two codes: the grammar's own, and
      // the JSON layer's V12, which `validate.structural` runs before it and returns from.
      for (const problem of problems) expect(['schema', 'V12']).toContain(problem.code);
      expect(formatProblems(problems).filter((line) => line.includes(match))).not.toEqual([]);
    },
  );
});

describe('the cases of tests/rejections/models.json the grammar accepts', () => {
  // Every other case is refused by the semantic stage, which assumes the grammar; a document the
  // core refused here would hide the rule the case is about.
  const cases = rejectionCases('models').filter((one) => one.expect !== 'schema');

  it.each(cases.map((one) => one.document as string))('%s crosses the grammar', (document) => {
    const tree = parse(readRepositoryFile(`tests/rejections/${document}`));
    expect(formatProblems(registry.structural(tree))).toEqual([]);
  });
});

describe('the schema refusals of tests/rejections/primitive-library.json', () => {
  // A case whose base holds a unit off the unit schema is this feature's; the others are the
  // loader's cross-reference checks (feature 1.3) and only have to cross the grammar here.
  const cases = rejectionCases('primitive-library').map((one) => {
    const base = one.base as string;
    const units = baseFiles(base);
    const refused = units.flatMap((path) => {
      const tree = parse(readRepositoryFile(path));
      const problems = registry.structural(tree, 'primitive-library-unit', DEEPEST);
      for (const problem of problems) expect(problem.code, path).toBe('schema');
      return formatProblems(problems).map((line) => `${path}: ${line}`);
    });
    return { base, match: one.match, units, refused };
  });

  it('names a base with units for every case', () => {
    for (const one of cases) expect(one.units.length, one.base).toBeGreaterThan(0);
  });

  it('refuses twelve of the thirty-three at the schema stage', () => {
    expect(cases.filter((one) => one.refused.length > 0)).toHaveLength(12);
  });

  it.each(cases.map((one) => [one.base, one.match] as const))(
    '%s: what the schema stage says about it',
    (base, match) => {
      const one = cases.find((each) => each.base === base);
      if (one === undefined || one.refused.length === 0) {
        // A cross-reference case: its unit is on the schema, and the loader refuses it later.
        expect(one?.refused).toEqual([]);
        return;
      }
      expect(one.refused.filter((line) => line.includes(match))).not.toEqual([]);
    },
  );
});

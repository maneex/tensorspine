import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  ASSERTION_KEYWORDS,
  formatProblems,
  KNOWN_KEYWORDS,
  loadSchemas,
  type SchemaObject,
} from '../../src/schema/index.js';

// The one mapping from a keyword to the tools' wording, held to one case per keyword.
//
// The corpus and the rejection suite reach five or six of the twenty-odd message forms
// `jsonschema` writes; the rest — `uniqueItems`, `multipleOf`, `dependentRequired`, the `False`
// schema, "is valid under each of" — cannot occur against the repository's schemas at all, and
// would otherwise rest on a reading of `_keywords.py` that nothing checks. `messages.json`
// records what the tools print for a small schema and a small instance per keyword, taken from
// `tools/schema.py` itself; this suite requires the core to print the same, both ways a document
// is read.

const here = dirname(fileURLToPath(import.meta.url));

interface MessageCase {
  name: string;
  schema: SchemaObject;
  /** The instance as JSON text, so that the float-ness of its numbers survives (D12). */
  instance: string;
  /** What `validate.structural` prints. */
  lines: string[];
  /** What `schema.deepest` prints. */
  deepest: string[];
}

const recorded = JSON.parse(readFileSync(join(here, 'messages.json'), 'utf8')) as {
  recorded_from: { jsonschema: string };
  schema_id: string;
  cases: MessageCase[];
};

/** A registry holding the one schema a case is written against, under the role `probe`. */
function registryFor(one: MessageCase): ReturnType<typeof loadSchemas> {
  return loadSchemas([{ path: 'probe.json', text: JSON.stringify(one.schema) }], {
    origin: 'probe',
  });
}

describe('the keyword-to-wording mapping', () => {
  it('was recorded from the jsonschema the oracle runs', () => {
    expect(recorded.recorded_from.jsonschema).toBe('4.25.0');
    expect(recorded.cases.length).toBeGreaterThan(50);
  });

  it.each(recorded.cases.map((one) => [one.name, one] as const))(
    '%s reads as the tools read it',
    (_name, one) => {
      const registry = registryFor(one);
      const tree = parse(one.instance);
      expect(formatProblems(registry.structural(tree, 'probe'))).toEqual(one.lines);
      expect(formatProblems(registry.structural(tree, 'probe', { deepest: true }))).toEqual(
        one.deepest,
      );
      // The walk and Ajv's own verdict agree: no problems exactly when Ajv conforms.
      expect(registry.conforms(tree, 'probe')).toBe(one.lines.length === 0);
      expect(formatProblems(registry.explain(tree, 'probe'))).toEqual(one.lines);
    },
  );

  it('covers every keyword of the vocabulary that a schema of the repository can carry', () => {
    const used = new Set<string>();
    const visit = (node: unknown): void => {
      if (Array.isArray(node)) {
        for (const child of node) visit(child);
        return;
      }
      if (node === null || typeof node !== 'object') return;
      for (const [keyword, value] of Object.entries(node)) {
        used.add(keyword);
        visit(value);
      }
    };
    for (const one of recorded.cases) visit(one.schema);
    // Every keyword the walk knows is exercised, save the two that address another schema
    // document — `$ref` is exercised by the corpus suite, `$dynamicRef` by nothing, since no
    // schema of the repository writes one.
    const untested = KNOWN_KEYWORDS.filter(
      (keyword) => !used.has(keyword) && keyword !== '$ref' && keyword !== '$dynamicRef',
    );
    expect(untested).toEqual(['contains', 'format', 'unevaluatedItems', 'unevaluatedProperties']);
  });

  it('states the assertions and the applicators as one partition of the vocabulary', () => {
    for (const keyword of ASSERTION_KEYWORDS) expect(KNOWN_KEYWORDS).toContain(keyword);
    expect(new Set(ASSERTION_KEYWORDS).size).toBe(ASSERTION_KEYWORDS.length);
  });
});

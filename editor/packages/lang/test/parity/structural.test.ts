import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import { formatProblems } from '../../src/schema/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of the structural stage (feature 1.1): the core's schema problems, line for line, are
// the lines `tools/` prints — `validate.structural` for a model document, `schema.deepest` for a
// library unit, which are the two readings the tools themselves take.
//
// The material is the oracle's (§0.5): every corpus document, every unit of the reference base,
// every rejection document and every unit of every rejection base, and beside them a spread of
// mutations — one document with one value changed, deleted or added — so that the mapping from a
// keyword to `jsonschema`'s wording is held to several hundred cases rather than to the twenty
// the rejection suite happens to carry.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** One case as `structural/index.json` records it. */
interface StructuralCase {
  /** A repository-relative path, for a document the repository carries. */
  document: string | null;
  /** A path under the oracle's output, for a mutated document the oracle wrote. */
  file: string | null;
  role: 'model' | 'primitive-library-unit' | 'derived';
  /** The document a mutation was made from, and where. */
  source?: string;
  place?: string;
  mutation?: string;
  /** What the tools printed. */
  lines: string[];
}

/** The cases the oracle recorded, empty when it recorded none. */
function recordedCases(): StructuralCase[] {
  const manifest = readOracleManifest() as unknown as { structural?: { index: string } };
  const index = manifest.structural?.index;
  if (index === undefined) return [];
  const read = JSON.parse(readFileSync(join(oracleOut, index), 'utf8')) as {
    cases: StructuralCase[];
  };
  return read.cases;
}

/** What the core says about one case, read the way the tools read that role. */
function coreLines(one: StructuralCase, registry: ReturnType<typeof repositorySchemas>): string[] {
  const path =
    one.document === null ? join(oracleOut, one.file ?? '') : join(repositoryRoot, one.document);
  const text = readFileSync(path, 'utf8');
  if (one.role !== 'primitive-library-unit') {
    return formatProblems(registry.structuralText(text, one.role));
  }
  return formatProblems(registry.structural(parse(text), one.role, { deepest: true }));
}

/** How a case is named when it fails: a path, or the mutation that made it. */
function nameOf(one: StructuralCase): string {
  if (one.document !== null) return `${one.document} [${one.role}]`;
  return `${one.source ?? '?'} ${one.place ?? '?'} (${one.mutation ?? '?'}) [${one.role}]`;
}

describe('the structural stage against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records both readings, over the repository and over mutations', () => {
    const cases = recordedCases();
    expect(cases.length).toBeGreaterThan(400);
    expect(cases.filter((one) => one.document !== null).length).toBeGreaterThan(200);
    expect(cases.filter((one) => one.file !== null).length).toBeGreaterThan(100);
    expect(cases.filter((one) => one.lines.length > 0).length).toBeGreaterThan(100);
  });

  it.skipIf(!generated)('reproduces every line of every case', () => {
    const registry = repositorySchemas();
    const wrong: string[] = [];
    for (const one of recordedCases()) {
      const actual = coreLines(one, registry);
      if (JSON.stringify(actual) === JSON.stringify(one.lines)) continue;
      wrong.push(
        [
          nameOf(one),
          `  tools: ${JSON.stringify(one.lines)}`,
          `  core:  ${JSON.stringify(actual)}`,
        ].join('\n'),
      );
    }
    expect(wrong.join('\n\n')).toBe('');
  });

  it.skipIf(!generated)('agrees with Ajv on every case, verdict and explanation', () => {
    // D4's catching rule: Ajv is the validator, the walk explains what Ajv refused, and the two
    // are held to each other on every document the oracle read. `structural` takes Ajv's verdict
    // as a shortcut; `explain` always walks, so comparing the two is comparing the readings.
    const registry = repositorySchemas();
    const wrong: string[] = [];
    for (const one of recordedCases()) {
      const path =
        one.document === null ? join(oracleOut, one.file ?? '') : join(repositoryRoot, one.document);
      const text = readFileSync(path, 'utf8');
      let tree;
      try {
        tree = parse(text);
      } catch {
        // The JSON layer refused it before the grammar; there is no tree to hand to Ajv.
        continue;
      }
      const options = one.role === 'primitive-library-unit' ? { deepest: true } : {};
      const conforms = registry.conforms(tree, one.role);
      const walked = formatProblems(registry.explain(tree, one.role, options));
      const shortcut = formatProblems(registry.structural(tree, one.role, options));
      if (conforms !== (walked.length === 0)) {
        wrong.push(`${nameOf(one)}: Ajv says ${String(conforms)}, the walk says ${String(walked.length === 0)}`);
      }
      if (JSON.stringify(walked) !== JSON.stringify(shortcut)) {
        wrong.push(`${nameOf(one)}: the walk and the shortcut differ`);
      }
    }
    expect(wrong.join('\n')).toBe('');
  }, 120_000);
});

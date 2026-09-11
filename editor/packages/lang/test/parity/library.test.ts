import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import { isRecord, member, toPython, type PyRecord, type PyValue } from '../../src/expr/index.js';
import {
  formatLibraryProblem,
  formatLibraryProblems,
  identityKey,
  loadLibrary,
  primitiveReferences,
  templateInterface,
  type Library,
} from '../../src/library/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { encode } from './encoding.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';
import { movedMessage } from './wording.js';

// Parity of the primitive library loader (feature 1.3): the refusals `tools/primitive_library.py`
// raises, word for word, and the cross-references it resolves, case by case.
//
// The material is the oracle's (§0.5), and it is of three kinds:
//
//   * the reference base gathered — what `load` answers for `data/primitive-library`, which every
//     corpus document pins;
//   * the 33 cases of `tests/rejections/primitive-library.json`, each with the whole text of the
//     `PrimitiveLibraryError` the tools raised, which is what the suite's `match` is matched
//     against;
//   * `primitive_references` over one-value mutations of every unit of the reference base — one
//     name suffixed, one name replaced by an axis the base does hold, one boolean flipped — some
//     six thousand of them, because the 33 cases reach about fifteen of the checker's forty
//     message sites and the mutations reach ninety-six distinct message shapes.
//
// Each mutation is recorded as a *pointer into a repository file*: both implementations read the
// same bytes and apply the same change, so neither side owns the input.
//
// The paths are repository-relative on both sides — the oracle strips the root, the source here
// is rooted at it — so a refusal's text is compared exactly as it was raised.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** One rejection case as `library/index.json` records it. */
interface RejectionCase {
  base: string;
  match: string;
  /** The whole `str(PrimitiveLibraryError)` the tools raised, paths made repository-relative. */
  error: string;
}

/** One cross-reference case: a unit of the reference base, with one value changed. */
interface MutationCase {
  /** The unit's file, relative to the repository root. */
  unit: string;
  /** A JSON pointer into the *definition*; `null` for the unit as the repository writes it. */
  pointer: string | null;
  kind: 'none' | 'suffix' | 'swap' | 'flip';
  value: string | boolean | null;
  /** What `primitive_references` answered, line for line. */
  lines: string[];
}

/** What the oracle recorded for the reference base. */
interface Reference {
  base: string;
  by_id: [string, string][];
  axes: string[];
  precision: string[];
  primitives: Record<string, string>;
  templates: Record<string, string>;
  files: Record<string, string>;
}

interface Recorded {
  reference: Reference;
  cases: RejectionCase[];
  mutations: MutationCase[];
  template_interfaces: Record<string, unknown>;
}

function recorded(): Recorded {
  return JSON.parse(readFileSync(join(oracleOut, 'library', 'index.json'), 'utf8')) as Recorded;
}

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);

/** The reference base, gathered as the tools gather it. */
function reference(base: string): Library {
  return loadLibrary([base], { schemas, source });
}

/** The value at a JSON pointer inside a plain reading of a document. */
function at(root: PyValue, pointer: string): { holder: PyValue; key: string } {
  const parts = pointer.split('/').slice(1);
  let cursor = root;
  for (const step of parts.slice(0, -1)) {
    cursor = Array.isArray(cursor)
      ? ((cursor as readonly PyValue[])[Number(step)] as PyValue)
      : (member(cursor as PyRecord, step) as PyValue);
  }
  return { holder: cursor, key: parts[parts.length - 1] as string };
}

/** The value at a pointer, replaced; the value that was there is returned so it can be put back. */
function replace(root: PyValue, pointer: string, value: PyValue): PyValue {
  const { holder, key } = at(root, pointer);
  if (Array.isArray(holder)) {
    const list = holder as PyValue[];
    const previous = list[Number(key)] as PyValue;
    list[Number(key)] = value;
    return previous;
  }
  const record = holder as Record<string, PyValue>;
  const previous = record[key] as PyValue;
  record[key] = value;
  return previous;
}

describe('the library loader against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records the reference base, the 33 cases and the mutations', () => {
    const material = recorded();
    const manifest = readOracleManifest() as unknown as {
      library?: { cases: number; mutations: number };
    };
    expect(manifest.library?.cases).toBe(33);
    expect(material.cases).toHaveLength(33);
    expect(material.mutations.length).toBeGreaterThan(5000);
    expect(material.mutations.filter((one) => one.pointer === null)).toHaveLength(36);
    expect(material.mutations.filter((one) => one.lines.length > 0).length).toBeGreaterThan(1000);
  });

  it.skipIf(!generated)('gathers the reference base exactly, with no problem', () => {
    const material = recorded();
    const library = reference(material.reference.base);
    expect(formatLibraryProblems(library.problems)).toEqual([]);
    expect([...library.byId.values()].map((one) => [one.name, one.version]).sort()).toEqual(
      [...material.reference.by_id].sort(),
    );
    expect([...library.axes.keys()]).toEqual(material.reference.axes);
    expect([...library.precision.keys()]).toEqual(material.reference.precision);
    const chosen: Record<string, string> = {};
    for (const [name, definition] of library.primitives) {
      chosen[name] = member(definition as PyRecord, 'version') as string;
    }
    expect(chosen).toEqual(material.reference.primitives);
    const templates: Record<string, string> = {};
    for (const [key, pin] of library.templates) templates[key] = pin.path;
    expect(templates).toEqual(material.reference.templates);
  });

  it.skipIf(!generated)('refuses every one of the 33 rejection cases in the tools’ words', () => {
    const material = recorded();
    const wrong: string[] = [];
    for (const one of material.cases) {
      const library = loadLibrary([`tests/rejections/${one.base}`, material.reference.base], {
        schemas,
        source,
      });
      const first = library.problems[0];
      const text = first === undefined ? '(accepted)' : formatLibraryProblem(first);
      if (text !== one.error) {
        wrong.push([one.base, `  tools: ${one.error}`, `  core:  ${text}`].join('\n'));
        continue;
      }
      // The suite's own contract: the refusal carries the substring the manifest matches on.
      if (!text.includes(one.match)) wrong.push(movedMessage(one.base, one.match, [text]));
      // A refusal names the file it is about (`PrimitiveLibraryError` "carries the file path").
      if (first !== undefined && !text.startsWith(first.file)) {
        wrong.push(`${one.base}: the text does not begin with the file it names`);
      }
    }
    expect(wrong.join('\n\n')).toBe('');
  }, 60_000);

  it.skipIf(!generated)('resolves every cross-reference case as the tools resolve it', () => {
    const material = recorded();
    const library = reference(material.reference.base);
    const definitions = new Map<string, PyValue>();
    for (const one of material.mutations) {
      if (definitions.has(one.unit)) continue;
      const unit = toPython(parse(readFileSync(join(repositoryRoot, one.unit), 'utf8')));
      definitions.set(one.unit, member(unit as PyRecord, 'definition') as PyValue);
    }
    const wrong: string[] = [];
    for (const one of material.mutations) {
      const definition = definitions.get(one.unit) as PyValue;
      const previous =
        one.pointer === null ? null : replace(definition, one.pointer, one.value);
      let lines: string[];
      try {
        lines = primitiveReferences(definition, library).map((problem) => problem.message);
      } finally {
        if (one.pointer !== null) replace(definition, one.pointer, previous);
      }
      if (JSON.stringify(lines) === JSON.stringify(one.lines)) continue;
      wrong.push(
        [
          `${one.unit} ${one.pointer ?? '(unmutated)'} [${one.kind}]`,
          `  tools: ${JSON.stringify(one.lines)}`,
          `  core:  ${JSON.stringify(lines)}`,
        ].join('\n'),
      );
      if (wrong.length > 20) break;
    }
    expect(wrong.join('\n\n')).toBe('');
  }, 120_000);

  it.skipIf(!generated)('places every problem it reports inside the unit it is about', () => {
    // The pointer beside the tools' wording is the port's own (plan §3): a Problems row with
    // somewhere to click. It is held to one thing here — that it resolves in the definition.
    const material = recorded();
    const library = reference(material.reference.base);
    const wrong: string[] = [];
    for (const one of material.mutations) {
      if (one.pointer === null || one.lines.length === 0) continue;
      const unit = toPython(parse(readFileSync(join(repositoryRoot, one.unit), 'utf8')));
      const definition = member(unit as PyRecord, 'definition') as PyValue;
      replace(definition, one.pointer, one.value);
      for (const problem of primitiveReferences(definition, library)) {
        if (resolves(definition, problem.segments)) continue;
        wrong.push(`${one.unit} ${one.pointer}: ${problem.path} resolves nowhere`);
      }
      if (wrong.length > 10) break;
    }
    expect(wrong.join('\n')).toBe('');
  }, 120_000);

  it.skipIf(!generated)('computes the template interface the validator computes', () => {
    const material = recorded();
    const library = reference(material.reference.base);
    for (const [key, expected] of Object.entries(material.template_interfaces)) {
      const pin = library.templates.get(key);
      expect(pin, `${key} was not pinned`).toBeDefined();
      const identity = library.byId.get(key);
      expect(identity, `${key} is not an identity of the base`).toBeDefined();
      const answered = templateInterface(
        (identity as { definition: PyValue }).definition,
        (pin as { document: PyValue }).document,
      );
      expect(encode(answered)).toEqual(expected);
    }
    expect(Object.keys(material.template_interfaces)).toEqual([
      identityKey('decoder.causal_yarn', '1.0.0'),
    ]);
  });
});

/** Whether a path of segments names something inside a value. */
function resolves(root: PyValue, segments: readonly (string | number)[]): boolean {
  let cursor: PyValue = root;
  for (const segment of segments) {
    if (typeof segment === 'number') {
      if (!Array.isArray(cursor)) return false;
      const list = cursor as readonly PyValue[];
      if (segment >= list.length) return false;
      cursor = list[segment] as PyValue;
      continue;
    }
    if (!isRecord(cursor) || !Object.hasOwn(cursor, segment)) return false;
    cursor = member(cursor, segment) as PyValue;
  }
  return true;
}

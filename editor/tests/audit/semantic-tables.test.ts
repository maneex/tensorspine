import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { COMPARISONS, OPERATORS, toPython, type PyValue } from '../../packages/lang/src/expr/index.js';
import { loadSchemas, type Vocabulary } from '../../packages/lang/src/schema/index.js';
import { parse } from '../../packages/lang/src/json/index.js';
import {
  checkArgumentDomain,
  checkType,
  formatSemanticProblems,
  type SemanticProblem,
} from '../../packages/lang/src/validate/index.js';
import { editorRoot } from './tree.js';

// The catching rule (d) of the implementation plan's §1: "The set-equality test of the core's
// semantic tables against the schema enums."
//
// The plan admits a table keyed by a vocabulary item in `packages/lang` and nowhere else, and
// only to attach semantics to the name: "an evaluator must say what `floor_divide` does". This
// audit is what keeps the admission honest for the two tables feature 1.2 carries — the
// operators of the algebra (§2.2 O0.1) and the comparison operators of the condition language
// (§4.3) — on **both** sides of the language, since one implementation serves the model
// expressions and the primitive expressions alike.
//
// The enumerations are not named here by their values. They are found by walking the grammar:
// every alternative of the expression union that carries an `op`, and every alternative of the
// condition union that carries a `compare`. A schema that gains an operator, or an alternative
// that carries one, therefore reaches this audit without anyone remembering to add it — and a
// second check requires that no enum written at a place named `op` or `operator`, anywhere in
// the five schemas, has escaped the walk.
//
// The third table is V3's, in `packages/lang/src/validate/conformance.ts`: the dimensional types
// of §2.1, which it decides by name, and the one unit the language treats apart — "a physical
// value in `tokens`, `elements`, `bytes` or `operations` is a whole number, only `seconds` being
// real". It is written as the tools write it, an `if`/`else if` chain ending in "type '…' is
// unknown to this validator", so the audit reads it the way a table cannot be read: by *asking*
// it about every kind the grammar declares, and requiring that none falls through.
//
// The fourth is the *domain*'s, beside it: `domain['kind'] == 'set'` and everything else read as
// an interval — the tools' own two-branch reading, which has no fall-through at all, so a third
// shape would be silently read as an interval. The audit asks each shape the grammar declares
// which branch decides it, and requires the shapes and the branches to be the same two.

const repositoryRoot = resolve(editorRoot, '..');

/** The repository's own schemas, as the registry loads them at startup. */
function vocabularyOfRepository(): Vocabulary {
  const directory = join(repositoryRoot, 'schemas');
  const files = readdirSync(directory)
    .filter((name) => name.endsWith('.json'))
    .map((name) => ({
      path: `schemas/${name}`,
      text: readFileSync(join(directory, name), 'utf8'),
    }));
  return loadSchemas(files, { origin: 'schemas' }).vocabulary();
}

const vocabulary = vocabularyOfRepository();

const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const UNIT = 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json';

/** The union at that pointer, or a failure naming it: the anchor is part of the grammar. */
function union(pointer: string): { alternatives: readonly { target: string | null }[] } {
  const found = vocabulary.unionAt(pointer);
  expect(found, `${pointer} is not a union of the loaded schemas`).toBeDefined();
  return found as { alternatives: readonly { target: string | null }[] };
}

/** Every value of the enum at `<alternative>/<suffix>`, over the alternatives of one union. */
function namesUnder(pointer: string, suffix: string): string[] {
  const names = new Set<string>();
  for (const alternative of union(pointer).alternatives) {
    if (alternative.target === null) continue;
    const found = vocabulary.enumAt(`${alternative.target}${suffix}`);
    if (found === undefined) continue;
    for (const value of found.values) names.add(String(value));
  }
  return [...names].sort();
}

/** The operator names an expression union admits, over every alternative that carries an `op`. */
const operatorsOf = (pointer: string): string[] => namesUnder(pointer, '/properties/op');

/** The comparison operators a condition union admits. */
const comparisonsOf = (pointer: string): string[] =>
  namesUnder(pointer, '/properties/compare/properties/operator');

describe('the operator table of packages/lang/src/expr', () => {
  const implemented = Object.keys(OPERATORS).sort();

  it('has exactly one implementation per operator of the model grammar', () => {
    const declared = operatorsOf(`${MODEL}#/$defs/scalar_expression`);
    expect(declared.length).toBeGreaterThan(0);
    expect(implemented).toEqual(declared);
  });

  it('has exactly one implementation per operator of the primitive grammar', () => {
    const declared = operatorsOf(`${UNIT}#/$defs/expression`);
    expect(declared.length).toBeGreaterThan(0);
    expect(implemented).toEqual(declared);
  });
});

describe('the comparison table of packages/lang/src/expr', () => {
  const implemented = Object.keys(COMPARISONS).sort();

  it('has exactly one implementation per comparison of the model condition language', () => {
    const declared = comparisonsOf(`${MODEL}#/$defs/condition`);
    expect(declared.length).toBeGreaterThan(0);
    expect(implemented).toEqual(declared);
  });

  it('has exactly one implementation per comparison of the primitive condition language', () => {
    const declared = comparisonsOf(`${UNIT}#/$defs/condition`);
    expect(declared.length).toBeGreaterThan(0);
    expect(implemented).toEqual(declared);
  });
});

describe('the walk that finds them', () => {
  it('leaves no `op` or `operator` enumeration of any schema unaccounted for', () => {
    const walked = new Set([
      ...operatorsOf(`${MODEL}#/$defs/scalar_expression`),
      ...operatorsOf(`${UNIT}#/$defs/expression`),
      ...comparisonsOf(`${MODEL}#/$defs/condition`),
      ...comparisonsOf(`${UNIT}#/$defs/condition`),
    ]);
    const missed: string[] = [];
    for (const one of vocabulary.enums) {
      if (!/\/(op|operator)$/.test(one.place)) continue;
      for (const value of one.values) {
        if (!walked.has(String(value))) missed.push(`${one.pointer}: ${String(value)}`);
      }
    }
    expect(missed).toEqual([]);
  });
});

/** The schema of that file, as JSON: what the vocabulary cannot answer is read here. */
function schemaFile(name: string): Record<string, unknown> {
  return JSON.parse(
    readFileSync(join(repositoryRoot, 'schemas', name), 'utf8'),
  ) as Record<string, unknown>;
}

describe('the type table of packages/lang/src/validate', () => {
  const KINDS = `${UNIT}#/$defs/argument_type/properties/kind`;
  const UNITS = `${MODEL}#/$defs/physical_type/properties/unit`;

  /** The lines `checkType` answered for one value against one declared type. */
  function typed(value: PyValue, declared: string): string[] {
    const problems: SemanticProblem[] = [];
    checkType(value, toPython(parse(declared)), 'x', problems, () => undefined);
    return formatSemanticProblems(problems);
  }

  /** Every kind the language declares an argument or a quantity may have. */
  function kinds(): string[] {
    const found = vocabulary.enumAt(KINDS);
    expect(found, `${KINDS} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly unknown[] }).values.map(String).sort();
  }

  it('decides every dimensional type the grammar declares', () => {
    // A kind the table does not carry falls through to "unknown to this validator", which is the
    // tools' own fall-through and the one answer no conforming document may ever get.
    expect(kinds().length).toBeGreaterThan(0);
    for (const kind of kinds()) {
      const declared = `{"kind": "${kind}", "unit": "seconds", "values": [1], "fields": {}}`;
      const lines = typed(1n, declared).join(' ');
      expect(lines, kind).not.toContain('is unknown to this validator');
    }
  });

  it('names `seconds`, and no other unit, as the one that admits a real', () => {
    const found = vocabulary.enumAt(UNITS);
    expect(found, `${UNITS} is not an enumeration of the loaded schemas`).toBeDefined();
    const units = (found as { values: readonly unknown[] }).values.map(String);
    expect(units).toContain('seconds');
    for (const unit of units) {
      const declared = `{"kind": "physical", "unit": "${unit}"}`;
      const refused = typed(2.5, declared).length > 0;
      expect(refused, unit).toBe(unit !== 'seconds');
    }
  });
});

describe('the domain table of packages/lang/src/validate', () => {
  const DOMAIN = `${UNIT}#/$defs/argument_domain`;

  /** The `kind` each alternative of a domain fixes, read from the schema's own `const`s. */
  function shapes(): string[] {
    const unit = schemaFile('tensorspine-primitive-library-unit.schema.json');
    const defs = unit['$defs'] as Record<string, Record<string, unknown>>;
    return union(DOMAIN).alternatives.map((alternative) => {
      const target = alternative.target as string;
      const name = target.slice(target.lastIndexOf('/') + 1);
      const properties = defs[name]?.['properties'] as Record<string, Record<string, unknown>>;
      const kind = properties['kind']?.['const'];
      expect(kind, `${target} fixes no kind`).toBeTypeOf('string');
      return kind as string;
    });
  }

  /**
   * Which branch decides a domain of that shape: the set's, or the interval's.
   *
   * The domain carries both a set and a bound, and the value is inside the set and below the
   * bound — so the message says which one was read.
   */
  function branchOf(kind: string): string {
    const problems: SemanticProblem[] = [];
    const domain = `{"kind": "${kind}", "values": [1], ` +
      '"lower": {"value": {"literal": 2}, "inclusive": true}}';
    checkArgumentDomain(1n, toPython(parse(domain)), 'x', problems, {});
    const lines = formatSemanticProblems(problems);
    if (lines.length === 0) return 'set';
    expect(lines.join(' '), kind).toContain('domain bound');
    return 'interval';
  }

  it('decides each shape the grammar declares by its own branch, and has no third', () => {
    const declared = shapes();
    expect(declared.length).toBeGreaterThan(0);
    const branches = declared.map((kind) => branchOf(kind));
    // One shape per branch, and one branch per shape: a third alternative in the schema would be
    // read as an interval without anyone saying so, and a branch with no shape would be dead.
    expect([...new Set(branches)].sort()).toEqual([...new Set(declared)].sort());
    expect(branches.length).toBe(new Set(branches).size);
  });
});

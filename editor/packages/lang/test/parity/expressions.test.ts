import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { formatNumber, parse } from '../../src/json/index.js';
import {
  argumentReferences,
  indexGrid,
  isRecord,
  member,
  missingAssignment,
  modelCondition,
  modelValue,
  primitiveCondition,
  primitiveValue,
  PyTypeError,
  resolveQuantities,
  staticArgument,
  toPython,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../../src/expr/index.js';
import { comparePythonStrings } from '../../src/schema/index.js';
import { oracleGenerated, oracleOut, repositoryRoot } from './oracle.js';

// Parity of the evaluators (feature 1.2): every expression and condition of the corpus and of
// the reference base, evaluated where it stands, against what `tools/expr.py` answered.
//
// The oracle records *cases*, not a walk: each one carries the expression, the environment it
// was evaluated in — a document's quantities and an index scope, or an instance's resolved
// arguments — and the answer, tagged and written as text so that the integer/float distinction
// the language reads survives the fixture. Beside the corpus it records a synthetic table over
// every operator and every comparison, because the corpus writes five of the eleven operators
// and neither `divide`, `min`, `max`, `negate` nor `absolute`.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

interface Case {
  where: string;
  form: string;
  expression: PyValue;
  result: PyValue;
  /** The document whose quantities the case was evaluated against; `null` for the algebra. */
  document?: string | null;
  env?: PyRecord;
  arguments?: bigint;
}

interface Recorded {
  documents: {
    name: string;
    path: string;
    assignment: PyRecord | null;
    quantities: PyRecord;
    missing: string[];
    grids: PyRecord;
  }[];
  environments: PyRecord[];
  cases: Case[];
}

/** The fixture, read as a document is read: `1e-05` a float, `4096` a whole number. */
function recorded(): Recorded {
  const text = readFileSync(join(oracleOut, 'expressions', 'index.json'), 'utf8');
  return toPython(parse(text)) as unknown as Recorded;
}

/** `repr(float)`, which is what the fixture writes; the three non-finite names included. */
function pythonRepr(value: number): string {
  if (Number.isFinite(value)) return formatNumber(value, true);
  if (Number.isNaN(value)) return 'nan';
  return value > 0 ? 'inf' : '-inf';
}

/** A value as the fixture encodes one — the comparison is over these, never over the values. */
function encode(value: PyValue): unknown {
  if (value === UNRESOLVED) return { unresolved: true };
  if (value === null) return { none: true };
  if (typeof value === 'boolean') return { bool: value };
  if (typeof value === 'bigint') return { int: value.toString() };
  if (typeof value === 'number') return { float: pythonRepr(value) };
  if (typeof value === 'string') return { str: value };
  if (Array.isArray(value)) return { list: value.map(encode) };
  const record: Record<string, unknown> = {};
  for (const [name, one] of Object.entries(value as PyRecord)) record[name] = encode(one);
  return { record };
}

/** The value an encoding stands for. */
function decode(encoded: PyValue): PyValue {
  if (!isRecord(encoded)) throw new Error(`not an encoded value: ${JSON.stringify(encoded)}`);
  const text = (name: string): string => member(encoded, name) as string;
  if (member(encoded, 'unresolved') !== undefined) return UNRESOLVED;
  if (member(encoded, 'none') !== undefined) return null;
  if (member(encoded, 'bool') !== undefined) return member(encoded, 'bool') as boolean;
  if (member(encoded, 'int') !== undefined) return BigInt(text('int'));
  if (member(encoded, 'float') !== undefined) {
    const written = text('float');
    if (written === 'nan') return Number.NaN;
    if (written === 'inf') return Number.POSITIVE_INFINITY;
    if (written === '-inf') return Number.NEGATIVE_INFINITY;
    return Number(written);
  }
  if (member(encoded, 'str') !== undefined) return text('str');
  if (member(encoded, 'list') !== undefined) {
    return (member(encoded, 'list') as readonly PyValue[]).map(decode);
  }
  const values = member(encoded, 'record') as PyRecord;
  const record: Record<string, PyValue> = {};
  for (const [name, one] of Object.entries(values)) record[name] = decode(one);
  return record;
}

/** An encoded map as the evaluators take one. */
function decodeMap(encoded: PyRecord): Map<string, PyValue> {
  return new Map(Object.entries(encoded).map(([name, one]) => [name, decode(one)]));
}

/** An encoded map as a record, which is the shape an assignment and an argument map have. */
function decodeRecord(encoded: PyRecord): PyRecord {
  const record: Record<string, PyValue> = {};
  for (const [name, one] of Object.entries(encoded)) record[name] = decode(one);
  return record;
}

describe('the evaluators against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records the corpus and the whole algebra', () => {
    const fixture = recorded();
    expect(fixture.documents).toHaveLength(15);
    expect(fixture.cases.length).toBeGreaterThan(10_000);
    const forms = new Set(fixture.cases.map((one) => one.form));
    expect([...forms].sort()).toEqual([
      'argument_references',
      'model_condition',
      'model_value',
      'primitive_condition',
      'primitive_value',
      'static_argument',
    ]);
    // Every operator and every comparison is exercised somewhere, whatever the corpus writes.
    const algebra = fixture.cases.filter((one) => one.where.startsWith('algebra/'));
    expect(algebra.length).toBeGreaterThan(4_000);
    for (const operator of [
      'add',
      'subtract',
      'multiply',
      'divide',
      'floor_divide',
      'ceil_divide',
      'modulo',
      'min',
      'max',
      'negate',
      'absolute',
      'equal',
      'not_equal',
      'less',
      'less_or_equal',
      'greater',
      'greater_or_equal',
    ]) {
      expect(
        algebra.some((one) => one.where.startsWith(`algebra/${operator}/`)),
        operator,
      ).toBe(true);
    }
  });

  it.skipIf(!generated)('answers what the tools answered, case for case', () => {
    const fixture = recorded();
    const quantities = new Map(
      fixture.documents.map((one) => [one.name, decodeMap(one.quantities)]),
    );
    const environments = fixture.environments.map(decodeRecord);
    const wrong: string[] = [];

    for (const one of fixture.cases) {
      const scope = one.env === undefined ? new Map<string, PyValue>() : decodeMap(one.env);
      const known =
        typeof one.document === 'string' ? quantities.get(one.document) : undefined;
      let answer: unknown;
      try {
        switch (one.form) {
          case 'model_value':
            answer = encode(modelValue(one.expression, known ?? new Map(), scope));
            break;
          case 'model_condition':
            answer = encode(modelCondition(one.expression, known ?? new Map(), scope));
            break;
          case 'static_argument':
            answer = encode(staticArgument(one.expression, known ?? new Map(), scope));
            break;
          case 'primitive_value':
            answer = encode(
              primitiveValue(one.expression, environments[Number(one.arguments)] as PyRecord),
            );
            break;
          case 'primitive_condition':
            answer = encode(
              primitiveCondition(one.expression, environments[Number(one.arguments)] as PyRecord),
            );
            break;
          case 'argument_references':
            answer = [...argumentReferences(one.expression)].sort(comparePythonStrings);
            break;
          default:
            wrong.push(`${one.where}: the fixture carries an unknown form '${one.form}'`);
            continue;
        }
      } catch (error) {
        answer =
          error instanceof PyTypeError
            ? { error: `TypeError: ${error.message}` }
            : { raised: String(error) };
      }
      // The comparison is over the encodings, never over the values: `1` and `1.0` are the
      // same double and different documents, and only the encoding tells them apart.
      const expected = one.result;
      if (JSON.stringify(answer) === JSON.stringify(expected)) continue;
      wrong.push(
        [
          `${one.where} [${one.form}]`,
          `  expression: ${JSON.stringify(one.expression, replacer)}`,
          `  tools: ${JSON.stringify(expected)}`,
          `  core:  ${JSON.stringify(answer)}`,
        ].join('\n'),
      );
    }
    expect(wrong.slice(0, 20).join('\n\n')).toBe('');
    expect(wrong).toHaveLength(0);
  });

  it.skipIf(!generated)('resolves the quantities of every document as the tools do', () => {
    const fixture = recorded();
    const wrong: string[] = [];
    for (const one of fixture.documents) {
      const model = toPython(
        parse(readFileSync(join(repositoryRoot, one.path), 'utf8')),
      ) as PyRecord;
      const assignment = one.assignment === null ? undefined : decodeRecord(one.assignment);
      const resolved = resolveQuantities(model, assignment);
      const answer: Record<string, unknown> = {};
      for (const [name, value] of resolved) answer[name] = encode(value);
      if (JSON.stringify(answer) !== JSON.stringify(one.quantities)) {
        wrong.push(
          `${one.name}: quantities\n  tools: ${JSON.stringify(one.quantities)}\n  core:  ${JSON.stringify(answer)}`,
        );
      }
      const missing = missingAssignment(model, assignment);
      if (JSON.stringify(missing) !== JSON.stringify(one.missing)) {
        wrong.push(`${one.name}: missing assignment ${JSON.stringify(missing)}`);
      }
      const compositions = member(model, 'compositions');
      for (const [composition, grid] of Object.entries(one.grids)) {
        const declared = member(
          member(compositions as PyRecord, composition) as PyRecord,
          'indices',
        ) as PyValue;
        const unrolled = indexGrid(declared, resolved);
        const answered = {
          names: unrolled.names,
          ranges: unrolled.ranges.map((range) => range.map((value) => encode(value))),
        };
        if (JSON.stringify(answered) !== JSON.stringify(grid)) {
          wrong.push(`${one.name}: the grid of '${composition}' differs`);
        }
      }
    }
    expect(wrong.join('\n\n')).toBe('');
  });
});

/** `JSON.stringify` cannot write a `bigint`; a diagnostic line may hold one. */
function replacer(_name: string, value: unknown): unknown {
  return typeof value === 'bigint' ? `${value.toString()}n` : value;
}

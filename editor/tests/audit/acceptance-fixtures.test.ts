import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parse, serialize } from '@tensorspine/lang';

import { editorRoot, readEditorFile } from './tree.js';

// The acceptance fixtures of feature 1.13 (finding F8), audited without Python.
//
// `editor/tests/fixtures/` is the one directory of this workspace holding *inputs* rather than
// recorded answers: documents and primitive-library bases, written because the grammar they use is
// written nowhere in `data/models/`, `data/primitive-library/` or `tests/rejections/`. Their
// expectations are the oracle's — `pnpm oracle` runs `--validate`, `--d1`, `--derive` and
// `artifact.check` over every one of them, and `packages/lang/test/parity/fixtures.test.ts`
// compares — which is exactly why they need an audit that does not need the oracle:
//
//   * a fixture that stopped carrying the construct it exists for would still validate, still
//     derive and still compare equal, and would prove nothing at all;
//   * a fixture added without a step to read it would sit unread;
//   * and the constructs themselves are a list the plan's §3 and the ledger's findings wrote down
//     once, which is worth holding the directory to rather than remembering.
//
// So this file reads the fixture documents as data and asserts that each construct is *there*.
// What each of them then *means* is the parity suite's business, and the core's.

const fixtures = join(editorRoot, 'tests', 'fixtures');
const models = join(fixtures, 'models');

/** One fixture document, parsed. */
function document(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(models, `${name}.json`), 'utf8')) as Record<string, unknown>;
}

/** Every `.json` under a directory of the fixtures, relative to it, sorted. */
function filesIn(directory: string): string[] {
  const out: string[] = [];
  const walk = (relative: string): void => {
    for (const entry of readdirSync(join(fixtures, relative), { withFileTypes: true })) {
      const path = relative === '' ? entry.name : `${relative}/${entry.name}`;
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith('.json')) out.push(path);
    }
  };
  walk(directory);
  return out.sort();
}

/** Whether any value inside a document satisfies a test — the document read as a tree. */
function somewhere(node: unknown, holds: (value: Record<string, unknown>) => boolean): boolean {
  if (Array.isArray(node)) return node.some((one) => somewhere(one, holds));
  if (node === null || typeof node !== 'object') return false;
  const record = node as Record<string, unknown>;
  return holds(record) || Object.values(record).some((one) => somewhere(one, holds));
}

/** Every `op` an expression of the document writes. */
function operators(node: unknown, out = new Set<string>()): Set<string> {
  somewhere(node, (record) => {
    if (typeof record['op'] === 'string') out.add(record['op']);
    return false;
  });
  return out;
}

/** The bindings of one kind, at the top level and in every composition. */
function bindings(model: Record<string, unknown>, kind: string): Record<string, unknown>[] {
  const out = Object.values(
    (model['bindings'] as Record<string, Record<string, unknown>>)[kind] ?? {},
  ) as Record<string, unknown>[];
  for (const composition of Object.values(
    model['compositions'] as Record<string, Record<string, unknown>>,
  )) {
    const scoped = composition['bindings'] as Record<string, Record<string, unknown>> | undefined;
    out.push(...(Object.values(scoped?.[kind] ?? {}) as Record<string, unknown>[]));
  }
  return out;
}

describe('the fixtures directory', () => {
  it('holds documents and bases, and nothing that records an answer', () => {
    expect(existsSync(models), 'editor/tests/fixtures/models').toBe(true);
    // Two bases of the fixtures' own, five units between them. A third base, or a unit added to
    // one of them, is a decision somebody takes rather than a file somebody drops.
    expect(filesIn('base')).toEqual([
      'base/primitive-library.json',
      'base/primitives/fixture/position_bias/1.0.0.json',
    ]);
    expect(filesIn('scratch')).toEqual([
      'scratch/primitive-library.json',
      'scratch/primitives/scratch/blank/1.0.0.json',
      'scratch/primitives/scratch/fed/1.0.0.json',
    ]);
    expect(filesIn('')).toEqual([...filesIn('base'), ...filesIn('models'), ...filesIn('scratch')]);
  });

  it('is a workspace the corpus’s own layout admits: every document names its bases', () => {
    for (const name of filesIn('models')) {
      const model = JSON.parse(readFileSync(join(fixtures, name), 'utf8')) as Record<string, unknown>;
      expect(model['schema'], name).toBe('tensorspine/2.0');
      const bases = (model['primitive_libraries'] as { base: string }[]).map((one) => one.base);
      expect(bases, name).toContain('../../../../data/primitive-library/');
      for (const base of bases) {
        expect(existsSync(join(models, base)), `${name}: ${base}`).toBe(true);
      }
    }
  });

  it('is written as the corpus is written, so the editor opens and saves it unchanged', () => {
    // D12 over material the round-trip suite does not see: these files are not under `data/`, and
    // a fixture written by a plain JSON writer would be the one workspace document the editor
    // could not save back byte for byte. Every one of them is
    // `json.dumps(document, indent=2, ensure_ascii=False) + "\n"`, which is what the corpus is
    // (feature 0.3 measured all 146 of its files).
    for (const name of filesIn('')) {
      const text = readFileSync(join(fixtures, name), 'utf8');
      expect(serialize(parse(text)), name).toBe(text);
    }
  });

  it('is read by the oracle and compared by a parity suite', () => {
    const generator = readEditorFile('tests/oracle/generate.py');
    expect(generator).toContain("FIXTURES = os.path.join(EDITOR, 'tests', 'fixtures')");
    expect(generator).toMatch(/^def fixtures\(out\):$/m);
    expect(generator).toMatch(/fixture_cases = fixtures\(out\)/);
    const suite = readEditorFile('packages/lang/test/parity/fixtures.test.ts');
    expect(suite).toContain("join(oracleOut, 'fixtures', 'index.json')");
  });
});

/** The `axis` of the `stack` a parameter rule locates with, or `null` when it locates otherwise. */
function stackOf(rule: Record<string, unknown>): string | null {
  const location = rule['location'] as Record<string, unknown> | undefined;
  const stack = location?.['stack'] as Record<string, unknown> | undefined;
  return stack === undefined ? null : (stack['axis'] as string);
}

/** The slot every member of a parameter rule names, when they agree; `null` otherwise. */
function slotNameOf(rule: Record<string, unknown>): string | null {
  const names = new Set(
    (rule['members'] as Record<string, unknown>[]).map((member) => member['parameter'] as string),
  );
  const [only] = names;
  return names.size === 1 && only !== undefined ? only : null;
}

/** Where a shape axis of `fixture.position_bias`'s slot sits, by its local name; -1 when absent. */
function axisPosition(slot: string, axis: string): number {
  const definition = JSON.parse(
    readFileSync(join(fixtures, 'base/primitives/fixture/position_bias/1.0.0.json'), 'utf8'),
  ) as { definition: { parameters: Record<string, { shape: { axes: { name: string }[] } }> } };
  const declared = definition.definition.parameters[slot];
  return declared === undefined ? -1 : declared.shape.axes.findIndex((one) => one.name === axis);
}

/**
 * The constructs the fixtures exist for: what the corpus, the reference base and the rejection
 * suite never write, measured by the features before this one and named by the plan's §3.
 *
 * A row names the fixture that carries it and the test that finds it in the file. Nothing here
 * asks what a construct *means* — that is the core's answer and the parity suite's comparison.
 */
interface Construct {
  readonly construct: string;
  readonly measured: string;
  readonly fixture: string;
  readonly holds: (model: Record<string, unknown>) => boolean;
}

const CONSTRUCTS: readonly Construct[] = [
  {
    construct: 'a composition of several indices',
    measured: 'plan §3; feature 1.4: no composition of the corpus has two',
    fixture: 'grid-composition',
    holds: (model) =>
      Object.values(model['compositions'] as Record<string, Record<string, unknown>>).some(
        (composition) => Object.keys(composition['indices'] as object).length > 1,
      ),
  },
  {
    construct: 'a `concat` location',
    measured: 'features 1.6c, 1.8a, 1.9: the corpus locates no concat at all',
    fixture: 'grid-composition',
    holds: (model) => bindings(model, 'parameters').some((rule) => somewhere(rule, (r) => 'concat' in r)),
  },
  {
    construct: 'the operators `divide`, `min`, `max`, `negate` and `absolute`',
    measured: 'feature 1.8d: the corpus and the reference base write none of the five',
    fixture: 'grid-composition',
    holds: (model) => {
      const written = operators(model['quantities']);
      return ['divide', 'min', 'max', 'negate', 'absolute'].every((op) => written.has(op));
    },
  },
  {
    construct: 'a top-level `for_each`, written as such',
    measured: 'plan §3; feature 1.4: the corpus writes none, its 706 being hoisted scoped rules',
    fixture: 'top-level-for-each',
    holds: (model) =>
      ['values', 'parameters'].every((kind) =>
        Object.values(
          (model['bindings'] as Record<string, Record<string, Record<string, unknown>>>)[kind] ?? {},
        ).some((rule) => 'for_each' in rule),
      ),
  },
  {
    construct: 'an index range whose start is not 0 and whose step is not 1',
    measured: 'feature 1.13: all 23 ranges of the corpus are `0 … stop step 1`',
    fixture: 'top-level-for-each',
    holds: (model) =>
      somewhere(
        model['compositions'],
        (record) =>
          'start' in record &&
          'step' in record &&
          JSON.stringify(record['start']) !== '{"literal":0}' &&
          JSON.stringify(record['step']) !== '{"literal":1}',
      ),
  },
  {
    construct: 'a `constants` map, a `constants` binding and its scoped form',
    measured: 'plan §3; feature 1.4: the corpus writes no constants at all',
    fixture: 'declared-constant',
    holds: (model) =>
      Object.keys(model['constants'] as object).length > 0 &&
      Object.keys((model['bindings'] as Record<string, object>)['constants'] as object).length > 0 &&
      bindings(model, 'constants').length > 1,
  },
  {
    construct: 'a `stack` along an axis that is not the first',
    measured: 'feature 1.9: every stack of the corpus is at dimension 0',
    fixture: 'declared-constant',
    holds: (model) =>
      bindings(model, 'parameters').some((rule) => {
        const stack = stackOf(rule);
        if (stack === null) return false;
        // The position of the stacked axis in the slot the rule binds: the axis is named by its
        // local name in the *unit*, so the unit is where the position is read.
        const slot = slotNameOf(rule);
        return slot !== null && axisPosition(slot, stack) > 0;
      }),
  },
  {
    construct: 'a root-level `when`, on an instance and on a binding',
    measured: 'plan §3; feature 1.7: no root instance of the corpus carries a guard',
    fixture: 'root-level-when',
    holds: (model) =>
      Object.values(model['instances'] as Record<string, Record<string, unknown>>).some(
        (instance) => 'when' in instance,
      ) &&
      Object.values(
        (model['bindings'] as Record<string, Record<string, Record<string, unknown>>>)['values'] ??
          {},
      ).some((rule) => 'when' in rule),
  },
  {
    construct: 'a boolean and a physical quantity, and an upper-inclusive domain bound',
    measured: 'feature 1.5: of the corpus’s 215 quantities, none is one of the three',
    fixture: 'root-level-when',
    holds: (model) => {
      const quantities = Object.values(model['quantities'] as Record<string, Record<string, unknown>>);
      const kinds = quantities.map((one) => (one['type'] as Record<string, unknown>)['kind']);
      // Upper *and* inclusive on the same bound: the corpus writes one upper bound, the template's
      // `eps < 1`, and it is exclusive.
      const upperInclusive = quantities.some((one) => {
        const domain = one['domain'] as Record<string, unknown> | undefined;
        const upper = domain?.['upper'] as Record<string, unknown> | undefined;
        return upper?.['inclusive'] === true;
      });
      return kinds.includes('boolean') && kinds.includes('physical') && upperInclusive;
    },
  },
  {
    construct: 'every form of the model condition language',
    measured: 'feature 1.13: the corpus’s 107 model conditions are `compare` and `all` alone',
    fixture: 'root-level-when',
    holds: (model) => {
      const forms = new Set<string>();
      somewhere(model, (record) => {
        for (const form of ['boolean', 'not', 'all', 'any', 'compare']) {
          if (Object.keys(record).length === 1 && form in record) forms.add(form);
        }
        return false;
      });
      return ['boolean', 'not', 'all', 'any', 'compare'].every((form) => forms.has(form));
    },
  },
  {
    construct: 'public interfaces that name a template instance',
    measured: 'features 1.8a, 1.8c: the composite’s interfaces name `embed` and `lm_head`',
    fixture: 'one-template-instance',
    holds: (model) => {
      const instances = model['instances'] as Record<string, Record<string, unknown>>;
      const templated = Object.entries(instances)
        .filter(([, one]) => ((one['primitive'] as Record<string, string>)['name'] ?? '').includes('.'))
        .map(([name]) => name);
      return somewhere(
        model['interfaces'],
        (record) => record['kind'] === 'root' && templated.includes(record['instance'] as string),
      );
    },
  },
  {
    construct: 'the caller’s quantities read by a dtype selector inside the instance',
    measured: 'feature 1.8a: no corpus document tells the three readings apart',
    fixture: 'one-template-instance-dtype',
    holds: (model) => {
      const precision = (model['quantities'] as Record<string, Record<string, unknown>>)['precision'];
      const declared = (precision?.['source'] as Record<string, unknown> | undefined)?.['value'];
      // The caller declares `f32`, the instance's argument supplies `f16` as a literal, and the
      // role's default is `bf16`: three readings a document can be made to tell apart.
      const supplied = Object.values(model['instances'] as Record<string, Record<string, unknown>>)
        .map((instance) => (instance['arguments'] as Record<string, unknown>)['precision'])
        .filter((value): value is Record<string, unknown> => typeof value === 'object' && value !== null);
      return declared === 'f32' && supplied.some((value) => value['literal'] === 'f16');
    },
  },
];

describe('every construct the fixtures exist for is in the file that carries it', () => {
  it('names a fixture that exists, and finds the construct in it', () => {
    for (const row of CONSTRUCTS) {
      expect(existsSync(join(models, `${row.fixture}.json`)), row.construct).toBe(true);
      expect(row.holds(document(row.fixture)), `${row.fixture}: ${row.construct}`).toBe(true);
      expect(row.measured.length, row.construct).toBeGreaterThan(20);
    }
    expect(CONSTRUCTS).toHaveLength(12);
  });

  it('reaches every document of the directory, so none sits unaccounted for', () => {
    // `scratch-blank` and `scratch-fed` are the two the list does not name by construct: what they
    // carry is a *unit*'s doing, a port shape citing an argument that may be absent, which the
    // section below states over the base rather than over the document.
    const named = new Set(CONSTRUCTS.map((row) => row.fixture));
    const missing = filesIn('models')
      .map((path) => path.slice('models/'.length, -'.json'.length))
      .filter((name) => !named.has(name) && !name.startsWith('scratch-'));
    expect(missing).toEqual([]);
  });
});

describe('the fixtures’ own bases', () => {
  /** One unit of a fixture base, parsed. */
  function unit(path: string): Record<string, unknown> {
    return JSON.parse(readFileSync(join(fixtures, path), 'utf8')) as Record<string, unknown>;
  }

  it('declares the constant slot no unit of the reference base declares (feature 1.6c)', () => {
    const definition = unit('base/primitives/fixture/position_bias/1.0.0.json')['definition'] as Record<
      string,
      unknown
    >;
    expect(Object.keys(definition['constants'] as object)).toEqual(['table']);
    // And the cost entry per invocation the reference base's eight entries never declare
    // (feature 1.8d: three `cached_position`, three `element`, two `sequence`).
    const cost = definition['logical_cost'] as { per: string }[];
    expect(cost.map((entry) => entry.per)).toContain('invocation');
  });

  it('leaves a port’s extent to an optional argument, on each side of the tools’ guard', () => {
    // Feature 1.8c's finding: a produced value's byte size is guarded and a public input's is not.
    // No port shape of the reference base cites an argument that may be absent; these two do, one
    // on its input port and one on its output port.
    for (const [file, side] of [
      ['scratch/primitives/scratch/blank/1.0.0.json', 'inputs'],
      ['scratch/primitives/scratch/fed/1.0.0.json', 'outputs'],
    ] as const) {
      const definition = unit(file)['definition'] as Record<string, unknown>;
      const optional = Object.entries(definition['arguments'] as Record<string, { required: boolean }>)
        .filter(([, declaration]) => !declaration.required)
        .map(([name]) => name);
      expect(optional, file).toEqual(['hidden']);
      const ports = (definition['ports'] as Record<string, Record<string, unknown>>)[side];
      const extents = Object.values(ports as Record<string, { shape: { axes: { extent: unknown }[] } }>)
        .flatMap((port) => port.shape.axes.map((axis) => axis.extent));
      expect(extents, file).toContainEqual({ argument: 'hidden' });
    }
  });
});

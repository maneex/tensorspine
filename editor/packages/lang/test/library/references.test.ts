import { describe, expect, it } from 'vitest';

import {
  absentComparisons,
  conditionPaths,
  declaredPaths,
  expressionPaths,
  loadLibrary,
  optionalPaths,
  primitiveReferences,
  type Library,
} from '../../src/library/index.js';
import type { PyValue } from '../../src/expr/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource, REFERENCE_BASE } from './source.js';

// `primitive_references`, on the checks the mutation harness of the parity suite cannot reach —
// a member removed rather than changed, a `none` target beside another option, an axis name that
// is reserved — and on the two decidability rules of §4.3, whose whole point is that a guard the
// primitive-side evaluator answers `false` to is never false by accident.
//
// The wording of every expectation below was read off `tools/primitive_library.py` on the same
// definitions; the parity suite is what keeps it honest at scale.

const schemas = repositorySchemas();
const library: Library = loadLibrary([REFERENCE_BASE], { schemas, source: nodeSource() });

/** A primitive definition with the required objects empty, and whatever a case adds. */
function definition(over: Record<string, PyValue> = {}): PyValue {
  return {
    version: '1.0.0',
    arguments: {},
    ports: { inputs: {}, outputs: {} },
    parameters: {},
    constants: {},
    state_ports: {},
    effects: { reads: [], writes: [] },
    partition_options: [{ target: { any_axis: true }, communication: 'none' }],
    ...over,
  };
}

/** What the checker says about one definition, line by line. */
function lines(over: Record<string, PyValue> = {}): string[] {
  return primitiveReferences(definition(over), library).map((problem) => problem.message);
}

const cardinality = { kind: 'cardinality' } as const;
const required = { type: cardinality, required: true, structural: false };
const optional = { type: cardinality, required: false, structural: false };
const port = (kind: string): PyValue => ({
  role: 'activation.hidden',
  domain: { kind, from: { self: true } },
});

describe('a template primitive', () => {
  it('has nothing to resolve: its arguments are the template’s external quantities', () => {
    expect(
      primitiveReferences(
        { version: '1.0.0', template: { name: 'x', version: '1.0.0', id: 'x' } },
        library,
      ),
    ).toEqual([]);
  });
});

describe('what a mutation cannot reach', () => {
  it('refuses a merge that declares no factor', () => {
    expect(
      lines({
        ports: { inputs: { i: port('token') }, outputs: { o: port('inherit') } },
        domain_transforms: [{ from_port: 'i', to_port: 'o', relation: 'merge' }],
      }),
    ).toEqual(['domain_transform 0: a merge declares its factor']);
  });

  it('refuses a `none` target beside another option, and with a communication', () => {
    expect(
      lines({
        partition_options: [
          { target: { none: true }, communication: 'all_reduce' },
          { target: { any_axis: true }, communication: 'none' },
        ],
      }),
    ).toEqual([
      'partition 0: a `none` target implies no communication',
      'partition 0: `none` cannot stand beside other partition_options',
    ]);
  });

  it('accepts a `none` target that stands alone with no communication', () => {
    expect(lines({ partition_options: [{ target: { none: true }, communication: 'none' }] })).toEqual(
      [],
    );
  });

  it('refuses value descriptions on a type that has no values', () => {
    expect(lines({ arguments: { w: { ...required, value_descriptions: { a: 'x' } } } })).toEqual([
      "argument 'w': value_descriptions on a non-enum type",
    ]);
  });

  it('refuses a lower domain bound that names an argument nobody declares', () => {
    expect(
      lines({
        arguments: {
          w: {
            ...required,
            domain: {
              kind: 'interval',
              lower: { value: { argument: 'nope' }, inclusive: true },
            },
          },
        },
      }),
    ).toEqual(["argument 'w': domain lower bound reads undeclared argument 'nope'"]);
  });

  it('refuses `multiplicity` as the name of a declared axis (§3.4)', () => {
    expect(
      lines({
        arguments: { w: required },
        parameters: {
          p: {
            role: 'activation.hidden',
            sharing: { kind: 'private' },
            shape: {
              axes: [
                {
                  name: 'multiplicity',
                  axis: 'model.width',
                  nature: 'feature',
                  extent: { argument: 'w' },
                },
              ],
            },
          },
        },
      }),
    ).toEqual([
      "parameter 'p': axis name 'multiplicity' is reserved — the storage axis of a slot that " +
        'declares a multiplicity (§3.4)',
    ]);
  });

  it('refuses a domain inherited from a port that itself inherits, and from no port at all', () => {
    expect(
      lines({
        ports: {
          inputs: {
            i: port('token'),
            j: { role: 'activation.hidden', domain: { kind: 'inherit', from: { port: 'i' } } },
          },
          outputs: {
            o: { role: 'activation.hidden', domain: { kind: 'inherit', from: { port: 'j' } } },
            p: { role: 'activation.hidden', domain: { kind: 'inherit', from: { port: 'zz' } } },
          },
        },
      }),
    ).toEqual([
      "port 'o': domain inherited from 'j', which itself inherits from a port",
      "port 'p': domain inherited from unknown input port 'zz'",
    ]);
  });

  it('refuses a key axis that is a value axis', () => {
    expect(
      lines({
        arguments: { w: required },
        state_ports: {
          s: {
            present_when: { boolean: true },
            key_axes: ['model.width'],
            payload: {
              c: {
                role: 'state.kv',
                shape: {
                  axes: [
                    {
                      name: 'w',
                      axis: 'model.width',
                      nature: 'feature',
                      extent: { argument: 'w' },
                    },
                  ],
                },
              },
            },
            operations: [{ effect: 'read' }],
            rules: [
              {
                when: { boolean: true },
                evolution: 'fixed',
                access: 'logical_position',
                sharing: 'per_instance',
                indexed_by: { self: true },
              },
            ],
          },
        },
      }),
    ).toEqual(["state 's': key axis 'model.width' is a value axis, not an instance axis"]);
  });

  it('checks a constant slot, which no unit of the reference base declares (F8)', () => {
    // The grammar admits a constant slot and the corpus has none, so this branch of the checker is
    // reached by no fixture and by no mutation; the editor's own case is what holds it.
    expect(
      lines({
        arguments: { w: required },
        constants: {
          table: {
            role: 'nope.role',
            shape: {
              axes: [
                { name: 'w', axis: 'nope.axis', nature: 'feature', extent: { argument: 'gone' } },
              ],
            },
            multiplicity: { argument: 'alsogone' },
          },
        },
      }),
    ).toEqual([
      "constant 'table': role 'nope.role' has no precision rule",
      "constant 'table': axis 'nope.axis' is not in the primitive_library",
      "constant 'table': extent reads undeclared argument 'gone'",
      "constant 'table': multiplicity reads undeclared argument 'alsogone'",
    ]);
  });

  it('says the same thing twice when one expression reads one undeclared path twice', () => {
    // `_expression_paths` is a generator, not a set: two occurrences are two lines, and the
    // refusal's whole text is what the rejection suite matches against.
    expect(
      lines({
        arguments: { w: required },
        logical_cost: [
          {
            expression: { op: 'add', args: [{ argument: 'gone' }, { argument: 'gone' }] },
            status: 'exact',
            per: 'element',
          },
        ],
      }),
    ).toEqual([
      "logical_cost 0: reads undeclared argument 'gone'",
      "logical_cost 0: reads undeclared argument 'gone'",
    ]);
  });
});

describe('a condition that reads an argument which may be absent (§4.3)', () => {
  const compare = {
    compare: { operator: 'less', left: { argument: 'opt' }, right: { literal: 4 } },
  };
  const invariant = (holds: PyValue, description: string): PyValue => ({ holds, description });

  it('is admitted under `all[present X, …]` and under `any[not present X, …]`, and refused bare', () => {
    expect(
      lines({
        arguments: { w: required, opt: optional },
        invariants: [
          invariant({ all: [{ present: 'opt' }, compare] }, 'guarded by all'),
          invariant({ any: [{ not: { present: 'opt' } }, compare] }, 'guarded by any'),
          invariant(compare, 'unguarded'),
        ],
      }),
    ).toEqual([
      "invariant 2 ('unguarded'): compares 'opt', which may be absent, outside a present test of " +
        'it (§4.3)',
    ]);
  });

  it('is admitted when the argument has a default, since it is then never absent', () => {
    expect(
      lines({
        arguments: { opt: { ...optional, default: { literal: 4 } } },
        invariants: [invariant(compare, 'defaulted')],
      }),
    ).toEqual([]);
  });

  it('is admitted for a field of a record whose own guard names the record', () => {
    // "Inside a record, the record itself is present by construction: a field's condition is only
    // ever evaluated when its record is."
    expect(
      lines({
        arguments: {
          rope: {
            type: {
              kind: 'record',
              fields: {
                kind: { type: { kind: 'enum', values: ['none', 'yarn'] }, required: true, structural: true },
                beta: {
                  ...optional,
                  present_when: {
                    compare: {
                      operator: 'equal',
                      left: { argument: 'rope.kind' },
                      right: { literal: 'yarn' },
                    },
                  },
                },
              },
            },
            required: false,
            structural: true,
          },
        },
      }),
    ).toEqual([]);
  });
});

describe('the walks the checker is made of', () => {
  it('finds every argument path an expression reads, conditionals included', () => {
    // The order is the generator's own, and the branches come before the test: `then`, `else`,
    // then `if`. It is not cosmetic — it is the order the lines of one refusal are printed in.
    expect(
      expressionPaths({
        op: 'add',
        args: [
          { argument: 'a' },
          { if: { present: 'b' }, then: { argument: 'c' }, else: { literal: 0 } },
        ],
      }),
    ).toEqual(['a', 'c', 'b']);
  });

  it('finds every argument path a condition tests, through not, all and any', () => {
    expect(
      conditionPaths({
        all: [
          { not: { present: 'a' } },
          { any: [{ compare: { operator: 'equal', left: { argument: 'b' }, right: { literal: 1 } } }] },
        ],
      }),
    ).toEqual(['a', 'b']);
  });

  it('flattens a record’s fields into dotted paths, and marks the maybe-absent ones', () => {
    const args = {
      w: required,
      rope: {
        type: {
          kind: 'record',
          fields: { kind: required, beta: optional },
        },
        required: false,
        structural: true,
      },
    };
    expect(declaredPaths(args)).toEqual(['w', 'rope', 'rope.kind', 'rope.beta']);
    expect(optionalPaths(args)).toEqual(['rope', 'rope.beta']);
  });

  it('reports a path under a maybe-absent record, not only the record itself', () => {
    const optionalPathSet = new Set(['rope']);
    expect(
      absentComparisons(
        { compare: { operator: 'equal', left: { argument: 'rope.kind' }, right: { literal: 1 } } },
        new Set(),
        optionalPathSet,
      ),
    ).toEqual(['rope.kind']);
  });
});

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  modelValue,
  toPython,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../../src/expr/index.js';
import { loadLibrary, memorySource, type Library, type LibrarySource } from '../../src/library/index.js';
import {
  analyse,
  declaredMultiplicity,
  dtypeValues,
  evaluateLocation,
  formatSemanticProblems,
  locationNames,
  rootSite,
  slotKeyOf,
  storageShape,
  wholeCount,
  type Analysis,
  type EvaluatedLocation,
} from '../../src/validate/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { repositoryRoot } from '../json/repository.js';

// The bindings (feature 1.6c): the second half of `analyse` — parameter and state slots with their
// presence and their multiplicity, the identities they form, the locations their tensors are stored
// at, what is carried across the fragments of a stream, and the two counters D5 reads.
//
// What the parity suite compares against the tools is in `test/parity/bindings.test.ts`, over every
// model document of the repository and over 26 edited ones. What is checked here is what the tools
// do not answer, and what no document of the repository can reach:
//
//   - the **pointer** each refusal carries beside its words (plan §3, §4.10): the tools collect
//     strings, the panel wants a place to click;
//   - the **location machinery** on its own — `_storage_shape`, `evaluate_location`,
//     `location_names` — including the storage axis a declared multiplicity puts before the slot's
//     own axes (§3.4, finding 30);
//   - the branches the **reference base cannot declare**: all eleven of its state ports are keyed
//     `instance.session, instance.branch` and both ports whose last rule is conditional are
//     exhaustive, so V9's "members keyed on … cannot share one allocation" and "no rule of … applies"
//     have no document; the one template of the repository locates all its identities, so V17's
//     "the template locates none of its identities" has none either; and no unit declares a constant
//     slot, so the whole `bindings.constants` map is read by nothing (finding F8). Those are checked
//     against a scratch base built here for them.
//
// The documents are written as JSON and read as a document is read (`4096` a whole number, `1e-05`
// a float), against the repository's own reference base wherever one will do: the primitives are
// real ones, so a shape or a role is the base's and not an invention of the test.

const schemas = repositorySchemas();
const repository = nodeSource(repositoryRoot);
const library: Library = loadLibrary(['data/primitive-library'], { schemas, source: repository });

/** A value read as a document is read (D12, feature 1.2). */
function read(json: string): PyValue {
  return toPython(parse(json));
}

/** One instance of a primitive, with its arguments written as expressions. */
function instance(name: string, args: Record<string, unknown>, extra: object = {}): object {
  return {
    primitive: { name, version: '1.0.0' },
    arguments: args,
    families: [name.split('.')[0]],
    ...extra,
  };
}

/** A root endpoint, `{kind: "root", instance: name}`. */
function root(name: string): object {
  return { kind: 'root', instance: name };
}

/** A whole document: the members the grammar requires, with the parts a case names. */
function document(parts: {
  quantities?: object;
  instances?: object;
  compositions?: object;
  values?: object;
  parameters?: object;
  constants?: object;
  states?: object;
  inputs?: object;
  outputs?: object;
}): PyValue {
  return read(
    JSON.stringify({
      schema: 'tensorspine/2.0',
      model: 'unit_bindings',
      primitive_libraries: [{ base: '../primitive-library/' }],
      quantities: parts.quantities ?? {
        d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
      },
      constants: {},
      instances: parts.instances ?? {},
      compositions: parts.compositions ?? {},
      bindings: {
        values: parts.values ?? {},
        parameters: parts.parameters ?? {},
        constants: parts.constants ?? {},
        states: parts.states ?? {},
      },
      interfaces: { inputs: parts.inputs ?? {}, outputs: parts.outputs ?? {} },
    }),
  );
}

/** A normalising instance, `norm.rms` of the reference base: one input, one output, one slot. */
function norm(): object {
  return instance('norm.rms', { width: { quantity: 'd' }, eps: { literal: 0.00001 } });
}

/** A document whose one `norm.rms` is fed by a public input and read by a public output. */
function oneNorm(parts: Parameters<typeof document>[0] = {}): Parameters<typeof document>[0] {
  return {
    instances: { n: norm() },
    inputs: { tokens: { to: [{ instance: root('n'), port: 'input' }], kind: 'token' } },
    outputs: { out: { from: { instance: root('n'), port: 'output' }, generative: false } },
    ...parts,
  };
}

/** The whole analysis of a document, under the reference base. */
function analysed(parts: Parameters<typeof document>[0], where: Library = library): Analysis {
  return analyse(document(parts), where);
}

/** The lines a document is refused with, as `analyse` prints them. */
function lines(parts: Parameters<typeof document>[0], where: Library = library): string[] {
  return formatSemanticProblems([...analysed(parts, where).problems]);
}

/** One parameter binding naming one slot of one root instance. */
function tensor(
  name: string,
  site: string,
  slot: string,
  extra: object = {},
): Record<string, object> {
  return {
    [name]: {
      tensor: { name },
      members: [{ instance: root(site), parameter: slot }],
      ...extra,
    },
  };
}

describe('the parameter bindings', () => {
  it('binds every present slot exactly once, and says where a refusal is', () => {
    const answer = analysed(oneNorm());
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      '[V7] unbound parameter slot: norm.rms@n.weight',
    ]);
    // The tools collect a string; the panel wants the place. An unbound slot is about the site.
    expect(answer.problems[0]?.path).toBe('/instances/n');
    expect(answer.problems[0]?.segments).toEqual(['instances', 'n']);
  });

  it('names the binding a refusal is about, not only the instance', () => {
    const answer = analysed(oneNorm({ parameters: tensor('w', 'n', 'nowhere') }));
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V7] parameter w: norm.rms has no parameter 'nowhere'",
      '[V7] unbound parameter slot: norm.rms@n.weight',
    ]);
    expect(answer.problems[0]?.path).toBe('/bindings/parameters/w');
    expect(answer.problems[1]?.path).toBe('/instances/n');
  });

  it('records one identity instance per rule, with its members and its dtype', () => {
    const answer = analysed(oneNorm({ parameters: tensor('w', 'n', 'weight', { dtype: 'bf16' }) }));
    expect(formatSemanticProblems([...answer.problems])).toEqual([]);
    expect(answer.bindings.tensorInstances).toHaveLength(1);
    const one = answer.bindings.tensorInstances[0];
    expect(one?.identity).toBe('w');
    expect(one?.rule).toBe('w');
    expect(one?.dtype).toBe('bf16');
    expect(one?.members.map((member) => member.name)).toEqual(['weight']);
    expect(one?.location).toBeUndefined();
    // The counters D5 reads: `norm.rms`'s one slot is `width` elements, and two operations each.
    expect(answer.stats.get('parameter_slots')).toBe(1n);
    expect(answer.stats.get('tensors')).toBe(1n);
    expect(answer.stats.get('shared')).toBe(0n);
    expect(answer.stats.get('located')).toBe(0n);
    expect(answer.stats.get('parameter_elements')).toBe(8n);
    expect(answer.stats.get('ops_per_element')).toBe(16n);
    expect(answer.stats.get('precisions_checked')).toBe(1n);
    // `slots` is keyed by the `(site, slot)` pair the tools key it by.
    expect(answer.bindings.slots.get(slotKeyOf({ site: rootSite('n'), name: 'weight' }))?.rule).toBe(
      'w',
    );
  });

  it('refuses a slot bound twice, naming both rules', () => {
    expect(
      lines(
        oneNorm({
          parameters: { ...tensor('w', 'n', 'weight'), ...tensor('again', 'n', 'weight') },
        }),
      ),
    ).toEqual(['[V7] slot norm.rms.weight bound twice (w, again)']);
  });

  it('locates a tensor, and refuses a physical name bound twice', () => {
    const answer = analysed(
      oneNorm({
        instances: { n: norm(), m: norm() },
        inputs: {
          tokens: {
            to: [
              { instance: root('n'), port: 'input' },
              { instance: root('m'), port: 'input' },
            ],
            kind: 'token',
          },
        },
        outputs: {
          out: { from: { instance: root('n'), port: 'output' }, generative: false },
          out2: { from: { instance: root('m'), port: 'output' }, generative: false },
        },
        parameters: {
          ...tensor('w', 'n', 'weight', { location: { tensor: ['w.weight'] } }),
          ...tensor('v', 'm', 'weight', { location: { tensor: ['w.weight'] } }),
        },
      }),
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V17] v: physical tensor 'w.weight' already bound by w",
    ]);
    expect(answer.problems[0]?.path).toBe('/bindings/parameters/v/location');
    // "the last writer wins": the map carries the identity that bound it last, as the tools do.
    expect(answer.bindings.physical.whole.get('w.weight')).toBe('v');
    expect(answer.stats.get('located')).toBe(2n);
  });

  it('refuses an unlocated identity in a document that locates its weights (V17)', () => {
    expect(
      lines(
        oneNorm({
          instances: { n: norm(), m: norm() },
          inputs: {
            tokens: {
              to: [
                { instance: root('n'), port: 'input' },
                { instance: root('m'), port: 'input' },
              ],
              kind: 'token',
            },
          },
          outputs: {
            out: { from: { instance: root('n'), port: 'output' }, generative: false },
            out2: { from: { instance: root('m'), port: 'output' }, generative: false },
          },
          parameters: {
            ...tensor('w', 'n', 'weight', { location: { tensor: ['w.weight'] } }),
            ...tensor('v', 'm', 'weight'),
          },
        }),
      ),
    ).toEqual(['[V17] v: no location, while the document locates its weights']);
  });
});

describe('the state bindings', () => {
  /** A causal `attention.dense`: one input, one output, one `kv` state port. */
  function attention(args: Record<string, unknown> = {}): object {
    return instance('attention.dense', {
      width: { quantity: 'd' },
      heads: { literal: 2 },
      head_dim: { literal: 4 },
      mask: { literal: 'causal' },
      ...args,
    });
  }

  /** A document whose one attention is fed by a public input and read by a public output. */
  function attending(parts: Parameters<typeof document>[0] = {}): Parameters<typeof document>[0] {
    return {
      quantities: { d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } } },
      instances: { a: attention() },
      inputs: { tokens: { to: [{ instance: root('a'), port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: root('a'), port: 'output' }, generative: false } },
      parameters: {
        ...tensor('q', 'a', 'q'),
        ...tensor('k', 'a', 'k'),
        ...tensor('v', 'a', 'v'),
        ...tensor('o', 'a', 'out'),
      },
      states: {
        kv: { identity: { name: 'kv' }, members: [{ instance: root('a'), state: 'kv' }] },
      },
      ...parts,
    };
  }

  it('derives the instance key from the identity`s indices and the port`s key axes (§4.4)', () => {
    const answer = analysed(attending());
    expect(formatSemanticProblems([...answer.problems])).toEqual([]);
    expect([...answer.bindings.instanceKeys]).toEqual([
      ['kv', ['instance.session', 'instance.branch']],
    ]);
    expect(answer.stats.get('state_slots')).toBe(1n);
    expect(answer.stats.get('state_identities')).toBe(1n);
    // One member, and it writes: `kv_source` defaults to `own`, so `written_when` holds (V20).
    const one = answer.bindings.stateInstances[0];
    expect(one?.identity).toBe('kv');
    expect(one?.indices).toEqual([]);
    expect(one?.writer?.name).toBe('kv');
  });

  it('refuses an unbound state port, and says where the instance is', () => {
    const answer = analysed(attending({ states: {} }));
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      '[V7] unbound state port: attention.dense@a.kv',
    ]);
    expect(answer.problems[0]?.path).toBe('/instances/a');
    // Nothing is carried, and nothing is advised: the stream is not fragmented.
    expect([...answer.bindings.carried]).toEqual([]);
    expect(answer.advisories).toEqual([]);
  });

  it('refuses a state binding naming a port the primitive has not, at the binding', () => {
    const answer = analysed(
      attending({
        states: {
          kv: { identity: { name: 'kv' }, members: [{ instance: root('a'), state: 'kv' }] },
          nope: { identity: { name: 'nope' }, members: [{ instance: root('a'), state: 'nowhere' }] },
        },
      }),
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V1] state nope: attention.dense has no state port 'nowhere'",
      // A member is counted when its *site* resolved, so the identity has one member and no
      // writer: the port name is what did not resolve, and V20 counts the members it was given.
      '[V20] state nope: 0 writer(s) among 1 member(s) — exactly one member writes an identity, ' +
        'the others read it',
    ]);
    expect(answer.problems[0]?.path).toBe('/bindings/states/nope');
    expect(answer.problems[1]?.path).toBe('/bindings/states/nope');
  });

  it('carries a state across the fragments of the stream it grows along (§5.3)', () => {
    // `streaming` turns on `attention.dense`'s carrying condition, and the public input is
    // fragmented: the state is carried, V16 holds, and V18 is satisfied.
    const answer = analysed(
      attending({
        instances: { a: attention({ streaming: { literal: true } }) },
        inputs: {
          tokens: {
            to: [{ instance: root('a'), port: 'input' }],
            kind: 'token',
            fragmented: true,
          },
        },
      }),
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([]);
    expect([...answer.bindings.carried]).toEqual([['kv', ['token', 'tokens']]]);
    expect(answer.advisories).toEqual([]);
  });

  it('refuses a carried state whose stream is not fragmented (V16)', () => {
    const answer = analysed(
      attending({ instances: { a: attention({ streaming: { literal: true } }) } }),
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V16] attention.dense@a.kv: carried across fragments, but its stream ('token', 'tokens') " +
        'is not a fragmented input',
    ]);
    expect(answer.problems[0]?.path).toBe('/instances/a');
    // The state is still recorded as carried: the refusal is about the stream, not the state.
    expect([...answer.bindings.carried]).toEqual([['kv', ['token', 'tokens']]]);
  });

  it('advises a self-indexed state on a fragmented stream that is not carried (§5.3)', () => {
    // Without `streaming` the carrying condition does not hold, so the cache is reset at every
    // fragment — valid, and worth a second look: `--lint` prints it, `--validate` does not. V18
    // refuses beside it, because `attention.dense` reads across positions.
    const answer = analysed(
      attending({
        inputs: {
          tokens: {
            to: [{ instance: root('a'), port: 'input' }],
            kind: 'token',
            fragmented: true,
          },
        },
      }),
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V18] attention.dense@a reads across positions of the fragmented stream 'tokens' " +
        "(port 'input') and carries no state across its fragments: the fragments would not " +
        'compute what the whole stream does',
    ]);
    expect(answer.advisories).toEqual([
      "attention.dense@a.kv: a self-indexed state on the fragmented stream 'tokens' that is not " +
        'carried — reset at every fragment',
    ]);
    expect([...answer.bindings.carried]).toEqual([]);
  });
});

describe('the location machinery', () => {
  /** The evaluated form of a location against a slot's stored shape. */
  function evaluated(
    location: string,
    slot: string,
    args: Record<string, unknown> = { width: 4 },
  ): { evaluated: EvaluatedLocation | null; problems: readonly string[] } {
    return evaluateLocation(
      read(location),
      new Map(),
      storageShape(read(slot)),
      read(JSON.stringify(args)) as PyRecord,
      (expression, env) => modelValue(expression, new Map(), env),
    );
  }

  /** A slot of one axis, `feature`, whose extent is the argument `width`. */
  const SLOT =
    '{"role": "norm.scale", "sharing": {"kind": "exclusive"}, "shape": {"axes": [' +
    '{"name": "feature", "axis": "model.width", "nature": "feature", ' +
    '"extent": {"argument": "width"}}]}}';

  /** The same slot with two copies: `_storage_shape` puts the storage axis before its own. */
  const COUNTED = `${SLOT.slice(0, -1)}, "multiplicity": {"argument": "copies"}}`;

  it('puts a declared multiplicity before the slot`s own axes (§3.4)', () => {
    const plain = storageShape(read(SLOT));
    expect((plain as PyRecord)['axes']).toHaveLength(1);
    const stored = storageShape(read(COUNTED)) as PyRecord;
    const axes = stored['axes'] as PyValue[];
    expect(axes).toHaveLength(2);
    expect((axes[0] as PyRecord)['axis']).toBe('storage.multiplicity');
    expect((axes[0] as PyRecord)['name']).toBe('multiplicity');
    expect((axes[0] as PyRecord)['nature']).toBe('storage');
    expect((axes[0] as PyRecord)['extent']).toEqual({ argument: 'copies' });
    // The slot's own shape is untouched: the storage axis "is in no primitive shape".
    expect((read(COUNTED) as PyRecord)['shape']).toEqual((read(SLOT) as PyRecord)['shape']);
    expect(declaredMultiplicity(read(COUNTED))).toEqual({ argument: 'copies' });
    expect(declaredMultiplicity(read(SLOT))).toBeNull();
  });

  it('addresses the storage axis as it addresses a shape axis', () => {
    const answer = evaluated(
      '{"stack": {"axis": "multiplicity", "part": {"tensor": ["e.", {"coordinate": "multiplicity"}]}}}',
      COUNTED,
      { width: 4, copies: 3 },
    );
    expect(answer.problems).toEqual([]);
    expect(answer.evaluated).toEqual({
      stack: {
        axis: 'multiplicity',
        dim: 0n,
        parts: [{ tensor: 'e.0' }, { tensor: 'e.1' }, { tensor: 'e.2' }],
      },
    });
    expect(locationNames(answer.evaluated as EvaluatedLocation)).toEqual({
      whole: ['e.0', 'e.1', 'e.2'],
      slices: [],
    });
  });

  it('names the axes of the slot when a form names one it has not', () => {
    expect(evaluated('{"stack": {"axis": "nowhere", "part": {"tensor": ["e"]}}}', COUNTED).problems)
      .toEqual([
        "stack: 'nowhere' is not an axis of the slot (axes: multiplicity, feature)",
      ]);
  });

  it('refuses a stack over an axis whose extent is not a count', () => {
    for (const [extent, written] of [
      [0, '0'],
      [-1, '-1'],
      [2.5, '2.5'],
      [true, 'True'],
    ] as const) {
      const answer = evaluated(
        '{"stack": {"axis": "feature", "part": {"tensor": ["e"]}}}',
        SLOT,
        { width: extent },
      );
      expect(answer.problems, String(extent)).toEqual([
        `stack: axis 'feature' has no coordinates — its extent resolves to ${written}`,
      ]);
      expect(answer.evaluated).toBeNull();
    }
    // A whole number written as a real is a count all the same: `n != int(n)` is arithmetic.
    expect(
      evaluated('{"stack": {"axis": "feature", "part": {"tensor": ["e"]}}}', SLOT, {
        width: 2.0,
      }).problems,
    ).toEqual([]);
  });

  it('reads a slice back as a region, and refuses one inside a concat', () => {
    const answer = evaluated(
      '{"slice": {"tensor": ["qkv"], "axis": "feature", "offset": {"literal": 8}}}',
      SLOT,
    );
    expect(answer.problems).toEqual([]);
    expect(answer.evaluated).toEqual({
      slice: { tensor: 'qkv', axis: 'feature', dim: 0n, offset: 8n, extent: 4n },
    });
    expect(locationNames(answer.evaluated as EvaluatedLocation)).toEqual({
      whole: [],
      slices: [{ name: 'qkv', offset: 8n, extent: 4n }],
    });
    expect(
      evaluated(
        '{"concat": {"axis": "feature", "parts": [{"tensor": ["a"]}, ' +
          '{"slice": {"tensor": ["b"], "axis": "feature", "offset": {"literal": 0}}}]}}',
        SLOT,
      ).problems,
    ).toEqual(['slice inside a concat: its extent along the axis would be unknown']);
  });

  it('refuses an offset that is not a non-negative integer', () => {
    for (const offset of ['{"literal": -1}', '{"literal": 1.5}', '{"literal": true}']) {
      expect(
        evaluated(`{"slice": {"tensor": ["q"], "axis": "feature", "offset": ${offset}}}`, SLOT)
          .problems,
      ).toEqual(['slice: the offset does not resolve to a non-negative integer']);
    }
  });

  it('refuses a location form the grammar does not carry', () => {
    expect(evaluated('{"nowhere": true}', SLOT).problems).toEqual(['unknown location form']);
  });

  it('reads a count as Python reads one', () => {
    expect(wholeCount(4n)).toBe(4n);
    expect(wholeCount(4)).toBe(4n);
    expect(wholeCount(0n)).toBeNull();
    expect(wholeCount(true)).toBeNull(); // `isinstance(n, bool)` is refused before anything else
    expect(wholeCount('4')).toBeNull();
    expect(wholeCount(null)).toBeNull();
    expect(wholeCount(UNRESOLVED)).toBeNull();
    expect(() => wholeCount(Number.POSITIVE_INFINITY)).toThrow('cannot convert float infinity');
  });
});

describe('the dtype selector', () => {
  const MODEL = read(
    JSON.stringify({
      quantities: {
        d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
        p: { type: { kind: 'enum', values: ['bf16'] }, source: { kind: 'literal', value: 'bf16' } },
        q: {
          type: { kind: 'enum', values: ['bf16', 'f16'] },
          source: { kind: 'derived', expression: { quantity: 'p' } },
        },
      },
    }),
  ) as PyRecord;

  it('answers the values a selector can take, or why it cannot', () => {
    expect(dtypeValues(MODEL, null)).toBeNull();
    expect(dtypeValues(MODEL, 'bf16')).toEqual(['bf16']);
    // A literal quantity is its one value; anything else is the enum's whole set.
    expect(dtypeValues(MODEL, { quantity: 'p' })).toEqual(['bf16']);
    expect(dtypeValues(MODEL, { quantity: 'q' })).toEqual(['bf16', 'f16']);
    expect(dtypeValues(MODEL, { quantity: 'nowhere' })).toBe('UNKNOWN');
    expect(dtypeValues(MODEL, { quantity: 'd' })).toBe('NOT_AN_ENUM');
  });
});

// --- what the reference base cannot declare ---------------------------------

/**
 * A base of the editor's own, loaded beside the repository's.
 *
 * Five branches of these blocks have no document behind them because the reference base declares
 * nothing that would reach them (see the header). A base is the smallest thing that can, and it is
 * built here rather than in `data/`, which no feature may touch.
 */
function scratchLibrary(files: Readonly<Record<string, string>>): Library {
  const memory = memorySource(files);
  const combined: LibrarySource = {
    isDirectory: (path) => memory.isDirectory(path) || repository.isDirectory(path),
    isFile: (path) => memory.isFile(path) || repository.isFile(path),
    exists: (path) => memory.exists(path) || repository.exists(path),
    find: (directory) =>
      memory.isDirectory(directory) ? memory.find(directory) : repository.find(directory),
    read: (path) => (memory.isFile(path) ? memory.read(path) : repository.read(path)),
  };
  return loadLibrary(['data/primitive-library', 'scratch/base'], {
    schemas,
    source: combined,
  });
}

/** A unit file's text, as a base holds one. */
function unit(name: string, definition: object): string {
  return JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'primitive',
    name,
    definition: { version: '1.0.0', ...definition },
  });
}

/** One axis of a shape, over the reference base's `model.width`. */
const FEATURE = {
  name: 'feature',
  axis: 'model.width',
  nature: 'feature',
  extent: { argument: 'width' },
};

/** A port that inherits its domain from the instance: what `norm.rms` declares. */
const PORT = {
  shape: { axes: [FEATURE] },
  domain: { kind: 'inherit', from: { self: true } },
  role: 'activation.hidden',
};

/** `width`, required and structural, as every shape here reads it. */
const WIDTH = { type: { kind: 'cardinality' }, required: true, structural: true };

/** One evolution rule, the simplest the grammar admits. */
function rule(when: object, indexedBy: object = { self: true }): object {
  return {
    when,
    evolution: 'append',
    access: 'logical_position',
    sharing: 'by_position',
    indexed_by: indexedBy,
  };
}

/** A condition comparing a boolean argument with `true`. */
function isTrue(argument: string): object {
  return { compare: { operator: 'equal', left: { argument }, right: { literal: true } } };
}

const TRUE = { boolean: true };

/** The effects a state port admits, as `attention.dense` declares them. */
const OPERATIONS = { read: { effect: 'read' }, append: { effect: 'append' } };

/** The scratch base's manifest and its units. */
const SCRATCH: Record<string, string> = {
  'scratch/base/primitive-library.json': JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'base',
    name: 'tensorspine.scratch',
    definition: {
      primitive_library: 'tensorspine/scratch',
      title: 'A base of the editor own tests',
      templates: '../templates/',
    },
  }),
  // Two state ports keyed differently, and one whose rules do not cover its own presence.
  'scratch/base/primitives/scratch/state/1.0.0.json': unit('scratch.state', {
    arguments: {
      width: WIDTH,
      covered: { type: { kind: 'boolean' }, required: true, structural: true },
    },
    ports: { inputs: { input: PORT }, outputs: { output: PORT } },
    parameters: {},
    constants: {},
    state_ports: {
      both: {
        present_when: TRUE,
        payload: { c: { shape: { axes: [FEATURE] }, role: 'state.kv' } },
        key_axes: ['instance.session', 'instance.branch'],
        operations: OPERATIONS,
        rules: [rule(TRUE)],
      },
      session_only: {
        present_when: TRUE,
        written_when: { not: TRUE },
        payload: { c: { shape: { axes: [FEATURE] }, role: 'state.kv' } },
        key_axes: ['instance.session'],
        operations: OPERATIONS,
        rules: [rule(TRUE)],
      },
      uncovered: {
        present_when: TRUE,
        payload: { c: { shape: { axes: [FEATURE] }, role: 'state.kv' } },
        key_axes: ['instance.session', 'instance.branch'],
        operations: OPERATIONS,
        rules: [rule(isTrue('covered'))],
      },
    },
    effects: { reads: ['input'], writes: ['output'] },
    partition_options: [{ target: { argument_axis: 'model.width' }, communication: 'all_reduce' }],
  }),
  // A slot whose extent is an argument with no domain, and a constant slot beside it: the two
  // places nothing of the reference base reaches.
  'scratch/base/primitives/scratch/counted/1.0.0.json': unit('scratch.counted', {
    arguments: { width: WIDTH },
    ports: { inputs: { input: PORT }, outputs: { output: PORT } },
    parameters: {
      weight: {
        role: 'norm.scale',
        sharing: { kind: 'exclusive' },
        shape: { axes: [FEATURE] },
      },
    },
    constants: {
      table: { role: 'norm.scale', shape: { axes: [FEATURE] } },
    },
    state_ports: {},
    effects: { reads: ['input'], writes: ['output'] },
    partition_options: [{ target: { argument_axis: 'model.width' }, communication: 'all_reduce' }],
  }),
  // A template primitive whose template locates none of its identities.
  'scratch/base/primitives/scratch/unlocated/1.0.0.json': JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'primitive',
    name: 'scratch.unlocated',
    definition: {
      version: '1.0.0',
      template: { name: 'scratch-unlocated', version: '1.0.0', id: 'scratch_unlocated' },
    },
  }),
  'scratch/templates/scratch-unlocated/1.0.0.json': JSON.stringify({
    schema: 'tensorspine/2.0',
    model: 'scratch_unlocated',
    version: '1.0.0',
    primitive_libraries: [{ base: '../base/' }],
    quantities: {
      width: {
        type: { kind: 'cardinality' },
        domain: { kind: 'interval', lower: { value: { literal: 1 }, inclusive: true } },
        source: { kind: 'external' },
      },
      eps: {
        type: { kind: 'real' },
        domain: { kind: 'interval', lower: { value: { literal: 0 }, inclusive: false } },
        source: { kind: 'external' },
      },
    },
    constants: {},
    instances: {
      n: {
        primitive: { name: 'norm.rms', version: '1.0.0' },
        arguments: { width: { quantity: 'width' }, eps: { quantity: 'eps' } },
        families: ['norm'],
      },
    },
    compositions: {},
    bindings: {
      values: {},
      parameters: {
        'n.weight': {
          tensor: { name: 'n.weight' },
          members: [{ instance: { kind: 'root', instance: 'n' }, parameter: 'weight' }],
        },
      },
      constants: {},
      states: {},
    },
    interfaces: {
      inputs: {
        hidden: { to: [{ instance: { kind: 'root', instance: 'n' }, port: 'input' }], kind: 'token' },
      },
      outputs: {
        hidden_out: {
          from: { instance: { kind: 'root', instance: 'n' }, port: 'output' },
          generative: false,
        },
      },
    },
  }),
};

describe('the branches the reference base cannot declare', () => {
  const scratch = scratchLibrary(SCRATCH);

  /** A document whose one `scratch.state` is fed and read, with its state ports bound. */
  function stateful(parts: Parameters<typeof document>[0] = {}): Parameters<typeof document>[0] {
    return {
      instances: { s: instance('scratch.state', { width: { quantity: 'd' }, covered: { literal: true } }) },
      inputs: { tokens: { to: [{ instance: root('s'), port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: root('s'), port: 'output' }, generative: false } },
      states: {
        both: { identity: { name: 'both' }, members: [{ instance: root('s'), state: 'both' }] },
        session_only: {
          identity: { name: 'session_only' },
          members: [{ instance: root('s'), state: 'session_only' }],
        },
        uncovered: {
          identity: { name: 'uncovered' },
          members: [{ instance: root('s'), state: 'uncovered' }],
        },
      },
      ...parts,
    };
  }

  it('loads the scratch base beside the repository`s', () => {
    expect(scratch.primitives.has('scratch.state')).toBe(true);
    expect(scratch.primitives.has('norm.rms')).toBe(true);
    expect(scratch.problems).toEqual([]);
  });

  it('refuses members keyed on different axes (V9)', () => {
    // `both` and `session_only` on one identity: the same payload, the same rule, the same
    // stream, and two different key-axis sets — the one V9 refusal all eleven state ports of the
    // reference base agree too well to reach.
    const answer = analysed(
      stateful({
        states: {
          shared: {
            identity: { name: 'shared' },
            members: [
              { instance: root('s'), state: 'both' },
              { instance: root('s'), state: 'session_only' },
            ],
          },
          uncovered: {
            identity: { name: 'uncovered' },
            members: [{ instance: root('s'), state: 'uncovered' }],
          },
        },
      }),
      scratch,
    );
    const printed = formatSemanticProblems([...answer.problems]);
    expect(printed).toContain(
      "[V9] state shared: members keyed on ['instance.session', 'instance.branch'] and " +
        "['instance.session'] cannot share one allocation",
    );
    // The instance key is the identity's indices then the *first* member's key axes (§4.4).
    expect(answer.bindings.instanceKeys.get('shared')).toEqual([
      'instance.session',
      'instance.branch',
    ]);
  });

  it('refuses a present state port no rule matches (V9)', () => {
    const answer = analysed(
      {
        ...stateful(),
        instances: {
          s: instance('scratch.state', { width: { quantity: 'd' }, covered: { literal: false } }),
        },
      },
      scratch,
    );
    const printed = formatSemanticProblems([...answer.problems]);
    expect(printed).toContain(
      '[V9] state uncovered: no rule of scratch.state.uncovered applies to these arguments',
    );
    // With no applicable rule the identity settles nothing: the key axes are still the port's.
    expect(answer.bindings.instanceKeys.get('uncovered')).toEqual([
      'instance.session',
      'instance.branch',
    ]);
  });

  it('refuses a located document whose template locates nothing (V17)', () => {
    const answer = analysed(
      {
        instances: {
          n: norm(),
          t: instance(
            'scratch.unlocated',
            { width: { quantity: 'd' }, eps: { literal: 0.00001 } },
            { weights_location_prefix: ['text.'] },
          ),
        },
        values: {
          into: {
            from: { instance: root('n'), port: 'output' },
            to: { instance: root('t'), port: 'hidden' },
          },
        },
        inputs: { tokens: { to: [{ instance: root('n'), port: 'input' }], kind: 'token' } },
        outputs: {
          out: { from: { instance: root('t'), port: 'hidden_out' }, generative: false },
        },
        parameters: tensor('w', 'n', 'weight', { location: { tensor: ['n.weight'] } }),
      },
      scratch,
    );
    const printed = formatSemanticProblems([...answer.problems]);
    expect(printed).toContain(
      '[V17] scratch.unlocated @t: the document locates its weights, but the template locates ' +
        'none of its identities',
    );
    // The template is analysed on its own terms: it locates nothing and refuses nothing for it
    // (§3.4 — "a template instance without a prefix in a document that locates its weights" is
    // the *caller's* refusal), and its `located` count is what the caller reads.
    expect(printed.filter((line) => line.includes('(in instance'))).toEqual([]);
    const expansion = [...answer.subResults.values()][0];
    expect(expansion?.stats.get('located')).toBe(0n);
    expect(answer.stats.get('located')).toBe(1n);
  });

  it('refuses a multiplicity that resolves to no copies (V7)', () => {
    // Every extent of the reference base is an argument with a lower bound of 1, so a bad count
    // is `UNRESOLVED` there and the tools print the sentinel's address (the parity suite's
    // `elided_sentinel`). Here the argument is unbounded, so the refusal names the count itself.
    const answer = analysed(
      {
        quantities: { d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 0 } } },
        instances: { c: instance('scratch.counted', { width: { quantity: 'd' } }) },
        inputs: { tokens: { to: [{ instance: root('c'), port: 'input' }], kind: 'token' } },
        outputs: { out: { from: { instance: root('c'), port: 'output' }, generative: false } },
        parameters: {
          w: {
            tensor: { name: 'w' },
            members: [{ instance: root('c'), parameter: 'weight' }],
            location: { stack: { axis: 'feature', part: { tensor: ['e.', { coordinate: 'feature' }] } } },
          },
        },
      },
      scratch,
    );
    expect(formatSemanticProblems([...answer.problems])).toContain(
      "[V17] w: stack: axis 'feature' has no coordinates — its extent resolves to 0",
    );
  });

  it('reads no constant binding at all, as the tools read none (finding F8)', () => {
    // `scratch.counted` declares a constant slot and the document binds nothing to it. V7 says
    // "every parameter, constant and state slot present under its `present_when` is bound exactly
    // once", but `analyse` reads `bindings.constants` nowhere and never looks at a constant slot:
    // the port reproduces that, and this test is what states it.
    const answer = analysed(
      {
        instances: { c: instance('scratch.counted', { width: { quantity: 'd' } }) },
        inputs: { tokens: { to: [{ instance: root('c'), port: 'input' }], kind: 'token' } },
        outputs: { out: { from: { instance: root('c'), port: 'output' }, generative: false } },
        parameters: tensor('w', 'c', 'weight'),
        constants: {
          nothing: {
            constant: 'absent',
            members: [{ instance: root('c'), constant: 'table' }],
          },
        },
      },
      scratch,
    );
    // No refusal about the unbound constant slot, and none about the binding naming a constant
    // the document does not declare either.
    expect(formatSemanticProblems([...answer.problems])).toEqual([]);
    expect(answer.stats.get('parameter_slots')).toBe(1n);
  });
});

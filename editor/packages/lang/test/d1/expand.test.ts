import { describe, expect, it } from 'vitest';

import {
  acrossPositions,
  expand,
  expandText,
  items,
  loadLibrary,
  member,
  memorySource,
  parse,
  PyKeyError,
  PyValueError,
  recordDefaults,
  serialize,
  toJsonValue,
  toPython,
  UNRESOLVED,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { repositoryRoot } from '../json/repository.js';
import { nodeSource, overlay } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';

// D1 (feature 1.7), stated directly: the rules of §5.2 the parity layer cannot reach through a
// document of the repository, and the two facts D1 carries beside the graph.
//
// What the parity suite already decides is not repeated here. The corpus and the 73 rejection
// documents settle the identifiers, the canonical listing, the guards of a composition site, the
// record defaults (309 fills), `across_positions` in both its forms, and six of the emitter's
// refusals; thirteen edited cases settle a root instance's guard, a composition over two indices,
// a value cycle, an undecidable `across_positions`, a record argument that is not one, a template
// instance a guard removes, a prefix that does not evaluate, and a template read with no
// assignment. What is left is what needs a *base* the repository does not have:
//
//   * a template primitive citing a second one — the reference base holds exactly one, so nesting
//     below depth 1, the `instances` of an expansion merging into its caller's, `MAX_DEPTH` and
//     the `primitive cycle` refusal have no document;
//   * a template instance a *composition* generates, whose arguments and whose
//     `weights_location_prefix` are evaluated in the site's own indices;
//   * a record field whose `default` carries a `present_when`, a default that does not resolve and
//     one that resolves to `None` — none of the three defaulted fields of the reference base has a
//     guard, and every default it writes is a literal.
//
// The first two are a base built here; the last three are `recordDefaults` and `acrossPositions`
// called directly, which is cheaper than a base and states the rule where it is written.

const schemas = repositorySchemas();
const repository = nodeSource(repositoryRoot);

/** A scratch base beside the repository's, for the template branches nothing in `data/` reaches. */
function scratchLibrary(): Library {
  return loadLibrary(['data/primitive-library', 'scratch/base'], {
    schemas,
    source: overlay(memorySource(SCRATCH), repository),
  });
}

/** The reference base alone, read from the repository. */
function referenceLibrary(): Library {
  return loadLibrary(['data/primitive-library'], { schemas, source: repository });
}

/**
 * A declaration as the loader reads one.
 *
 * Through the core's own parser, not `JSON.parse`: a unit's `{"literal": 1}` is a Python *int* and
 * `{"literal": 1.0}` a float, and the distinction is what `recordDefaults` writes into the node.
 */
function declaration(value: unknown): PyValue {
  return toPython(parse(JSON.stringify(value)));
}

/** A model document's text, with the members the grammar requires and nothing else. */
function documentText(parts: Record<string, unknown>): string {
  return JSON.stringify({
    schema: 'tensorspine/2.0',
    model: 'scratch_model',
    primitive_libraries: [{ base: '../primitive-library/' }],
    quantities: {},
    constants: {},
    instances: {},
    compositions: {},
    bindings: { values: {}, parameters: {}, constants: {}, states: {} },
    interfaces: { inputs: {}, outputs: {} },
    ...parts,
  });
}

/** D1 of a document written here. */
function expanded(parts: Record<string, unknown>, library: Library): PyRecord {
  return member(expandText(documentText(parts), library), 'd1') as PyRecord;
}

/** The node identifiers of an expanded graph, in the order D1 lists them. */
function nodeNames(graph: PyRecord): string[] {
  return items(member(graph, 'nodes') as PyValue).map(([name]) => name);
}

/** One edge as `<from node>.<from port> -> <to node>.<to port>` under its rule. */
function edgeLines(graph: PyRecord): string[] {
  return (member(graph, 'edges') as readonly PyValue[]).map((edge) => {
    const from = member(edge as PyRecord, 'from') as PyRecord;
    const to = member(edge as PyRecord, 'to') as PyRecord;
    return (
      `${member(edge as PyRecord, 'rule') as string}: ` +
      `${member(from, 'node') as string}.${member(from, 'port') as string} -> ` +
      `${member(to, 'node') as string}.${member(to, 'port') as string}`
    );
  });
}

// --- the two facts D1 carries beside the graph ------------------------------

describe('record defaults', () => {
  /** One record argument `opts` whose fields are declared here. */
  const declared = (fields: Record<string, unknown>): PyValue =>
    declaration({ opts: { type: { kind: 'record', fields } } });

  it('fills an absent field in place and leaves a written one alone', () => {
    const args: PyRecord = { opts: { kept: 'mine' } };
    recordDefaults(
      declared({
        kept: { type: { kind: 'enum', values: ['mine', 'theirs'] }, default: { literal: 'theirs' } },
        added: { type: { kind: 'boolean' }, default: { literal: true } },
      }),
      args,
      args,
    );
    expect(args['opts']).toEqual({ kept: 'mine', added: true });
  });

  it('applies a field`s `present_when` against the whole argument map', () => {
    // The path is absolute — `opts.kind` — because "the scope is the instance's whole argument
    // map", which is also what lets a guard read a sibling the same pass has just written.
    const guard = (value: string): unknown => ({
      compare: { operator: 'equal', left: { argument: 'opts.kind' }, right: { literal: value } },
    });
    const fields = {
      kind: { type: { kind: 'enum', values: ['a', 'b'] }, default: { literal: 'a' } },
      onlyA: { type: { kind: 'boolean' }, default: { literal: true }, present_when: guard('a') },
      onlyB: { type: { kind: 'boolean' }, default: { literal: true }, present_when: guard('b') },
    };
    const args: PyRecord = { opts: {} };
    recordDefaults(declared(fields), args, args);
    expect(args['opts']).toEqual({ kind: 'a', onlyA: true });
  });

  it('leaves out a default that does not resolve, and one that is `None`', () => {
    const args: PyRecord = { opts: {} };
    recordDefaults(
      declared({
        unresolved: { type: { kind: 'cardinality' }, default: { argument: 'opts.absent' } },
        settled: { type: { kind: 'cardinality' }, default: { literal: 1 } },
      }),
      args,
      args,
    );
    expect(args['opts']).toEqual({ settled: 1n });
    // `v is not None`: a default reading an argument the document wrote as `null`.
    const nulled: PyRecord = { opts: {}, source: null };
    recordDefaults(
      declared({ copied: { type: { kind: 'cardinality' }, default: { argument: 'source' } } }),
      nulled,
      nulled,
    );
    expect(nulled['opts']).toEqual({});
  });

  it('recurses into a nested record and steps over what is not one', () => {
    const nested = declaration({
        opts: {
          type: {
            kind: 'record',
            fields: {
              inner: {
                type: {
                  kind: 'record',
                  fields: { deep: { type: { kind: 'boolean' }, default: { literal: false } } },
                },
              },
            },
          },
        },
      scalar: { type: { kind: 'record', fields: { any: { default: { literal: 1 } } } } },
    });
    const args: PyRecord = { opts: { inner: {} }, scalar: 3n };
    recordDefaults(nested, args, args);
    expect(args['opts']).toEqual({ inner: { deep: false } });
    expect(args['scalar']).toBe(3n);
  });
});

describe('across_positions', () => {
  const effect = (when: unknown): PyValue => declaration({ effects: { across_positions: { when } } });

  it('is false when the primitive declares none', () => {
    expect(acrossPositions({}, {}, 'x')).toBe(false);
    expect(acrossPositions({ effects: {} }, {}, 'x')).toBe(false);
  });

  it('is the condition on the resolved arguments', () => {
    expect(acrossPositions(effect({ boolean: true }), {}, 'x')).toBe(true);
    const reads = effect({
      compare: { operator: 'greater', left: { argument: 'kernel' }, right: { literal: 1 } },
    });
    expect(acrossPositions(reads, { kernel: 3n }, 'x')).toBe(true);
    expect(acrossPositions(reads, { kernel: 1n }, 'x')).toBe(false);
  });

  it('refuses rather than guessing when an argument it reads does not resolve', () => {
    const reads = effect({
      all: [
        { compare: { operator: 'greater', left: { argument: 'zebra' }, right: { literal: 1 } } },
        { compare: { operator: 'greater', left: { argument: 'alpha' }, right: { literal: 1 } } },
      ],
    });
    // "A condition the arguments leave undecidable refuses the derivation; it never answers
    // `false`" — and the path it names is the first in sorted order, whatever the condition's own.
    expect(() => acrossPositions(reads, { alpha: UNRESOLVED, zebra: UNRESOLVED }, 'conv')).toThrow(
      new PyValueError(
        "conv: across_positions is undecidable — its condition reads argument 'alpha', " +
          'which does not resolve',
      ),
    );
    // An unresolved argument the condition does not read decides nothing, and is not in the map
    // the condition is evaluated over.
    expect(acrossPositions(reads, { alpha: 2n, zebra: 2n, other: UNRESOLVED }, 'conv')).toBe(true);
  });
});

// --- the graph ---------------------------------------------------------------

describe('the expanded graph', () => {
  const library = referenceLibrary();

  /** One `norm.rms` instance under the names its arguments need. */
  const norm = (families: string[] = ['norm']): unknown => ({
    primitive: { name: 'norm.rms', version: '1.0.0' },
    arguments: { width: { quantity: 'd' }, eps: { quantity: 'eps' } },
    families,
  });

  const QUANTITIES = {
    d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
    eps: { type: { kind: 'real' }, source: { kind: 'literal', value: 1e-5 } },
  };

  it('keeps only the external quantities in the head`s assignment', () => {
    const text = documentText({
      quantities: {
        ...QUANTITIES,
        outer: {
          type: { kind: 'cardinality' },
          domain: { kind: 'interval', lower: { value: { literal: 1 }, inclusive: true } },
          source: { kind: 'external' },
        },
      },
      instances: { a: norm() },
      interfaces: {
        inputs: { input: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' } },
        outputs: { out: { from: { instance: { kind: 'root', instance: 'a' }, port: 'output' }, generative: false } },
      },
    });
    const document = expandText(text, library, { assignment: { outer: 4n, unknown: 9n } });
    expect(member(document, 'assignment')).toEqual({ outer: 4n });
    expect(member(document, 'schema')).toBe('tensorspine-derived/2.1');
    expect(member(document, 'model')).toBe('scratch_model');
  });

  it('drops every binding that names a site a guard removed (§5.2 rule 3)', () => {
    const graph = expanded(
      {
        quantities: QUANTITIES,
        instances: {
          a: norm(),
          b: { ...(norm() as object), when: { boolean: false } },
          c: norm(),
        },
        bindings: {
          values: {
            'a.b': {
              from: { instance: { kind: 'root', instance: 'a' }, port: 'output' },
              to: { instance: { kind: 'root', instance: 'b' }, port: 'input' },
            },
            'a.c': {
              from: { instance: { kind: 'root', instance: 'a' }, port: 'output' },
              to: { instance: { kind: 'root', instance: 'c' }, port: 'input' },
            },
          },
          parameters: {},
          constants: {},
          states: {},
        },
        interfaces: {
          inputs: { input: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' } },
          outputs: { out: { from: { instance: { kind: 'root', instance: 'c' }, port: 'output' }, generative: false } },
        },
      },
      library,
    );
    expect(nodeNames(graph)).toEqual(['a', 'c']);
    // No guard is repeated on an edge: the binding is simply not emitted there.
    expect(edgeLines(graph)).toEqual(['a.c: a.output -> c.input']);
    expect(member(graph, 'topological_order')).toEqual(['a', 'c']);
  });

  it('lists nodes by identifier and edges by (source, destination)', () => {
    // Written out of order, and over two indices, so that the listing is the emitter's and not the
    // document's: the identifier carries the indices in *name* order whatever the grid's.
    const graph = expanded(
      {
        quantities: QUANTITIES,
        compositions: {
          stack: {
            indices: {
              layer: { start: { literal: 0 }, stop: { literal: 2 }, step: { literal: 1 } },
              half: { start: { literal: 0 }, stop: { literal: 2 }, step: { literal: 1 } },
            },
            families: ['stack'],
            instances: { z: norm(['tail']), a: norm() },
          },
        },
        interfaces: {
          inputs: {
            input: {
              to: [
                {
                  instance: {
                    kind: 'generated',
                    composition: 'stack',
                    instance: 'a',
                    indices: { layer: { literal: 0 }, half: { literal: 0 } },
                  },
                  port: 'input',
                },
              ],
              kind: 'token',
            },
          },
          outputs: {
            out: {
              from: {
                instance: {
                  kind: 'generated',
                  composition: 'stack',
                  instance: 'z',
                  indices: { layer: { literal: 1 }, half: { literal: 1 } },
                },
                port: 'output',
              },
              generative: false,
            },
          },
        },
      },
      library,
    );
    expect(nodeNames(graph)).toEqual([
      'stack/a[half=0,layer=0]',
      'stack/a[half=0,layer=1]',
      'stack/a[half=1,layer=0]',
      'stack/a[half=1,layer=1]',
      'stack/z[half=0,layer=0]',
      'stack/z[half=0,layer=1]',
      'stack/z[half=1,layer=0]',
      'stack/z[half=1,layer=1]',
    ]);
    // A generated site's families are its own and its composition's, as one sorted set.
    const first = member(member(graph, 'nodes') as PyRecord, 'stack/a[half=0,layer=0]') as PyRecord;
    expect(member(first, 'families')).toEqual(['norm', 'stack']);
    const tail = member(member(graph, 'nodes') as PyRecord, 'stack/z[half=0,layer=0]') as PyRecord;
    expect(member(tail, 'families')).toEqual(['stack', 'tail']);
  });

  it('refuses a value cycle with V6`s own count', () => {
    expect(() =>
      expanded(
        {
          quantities: QUANTITIES,
          instances: { a: norm(), b: norm() },
          bindings: {
            values: {
              there: {
                from: { instance: { kind: 'root', instance: 'a' }, port: 'output' },
                to: { instance: { kind: 'root', instance: 'b' }, port: 'input' },
              },
              back: {
                from: { instance: { kind: 'root', instance: 'b' }, port: 'output' },
                to: { instance: { kind: 'root', instance: 'a' }, port: 'input' },
              },
            },
            parameters: {},
            constants: {},
            states: {},
          },
          interfaces: {
            inputs: { input: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' } },
            outputs: { out: { from: { instance: { kind: 'root', instance: 'b' }, port: 'output' }, generative: false } },
          },
        },
        library,
      ),
    ).toThrow(
      new PyValueError('scratch_model: cyclic graph, 2 node(s) in a cycle — no D1 (V6 rejection)'),
    );
  });

  it('raises Python`s own KeyError for an edge whose destination is not a node', () => {
    // `indegree` is a plain dictionary keyed by the emitted nodes: a binding naming a site outside
    // the composition's ranges is neither refused nor skipped here, it is a `KeyError`.
    expect(() =>
      expanded(
        {
          quantities: QUANTITIES,
          instances: { a: norm() },
          compositions: {
            stack: {
              indices: { layer: { start: { literal: 0 }, stop: { literal: 1 }, step: { literal: 1 } } },
              families: ['stack'],
              instances: { s: norm() },
            },
          },
          bindings: {
            values: {
              far: {
                from: { instance: { kind: 'root', instance: 'a' }, port: 'output' },
                to: {
                  instance: {
                    kind: 'generated',
                    composition: 'stack',
                    instance: 's',
                    indices: { layer: { literal: 7 } },
                  },
                  port: 'input',
                },
              },
            },
            parameters: {},
            constants: {},
            states: {},
          },
          interfaces: {
            inputs: { input: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' } },
            outputs: { out: { from: { instance: { kind: 'root', instance: 'a' }, port: 'output' }, generative: false } },
          },
        },
        library,
      ),
    ).toThrow(new PyKeyError("'stack/s[layer=7]'"));
  });

  it('raises for a primitive the base does not hold, by name alone', () => {
    expect(() =>
      expanded(
        {
          quantities: QUANTITIES,
          instances: {
            a: { primitive: { name: 'norm.rmz', version: '1.0.0' }, arguments: {}, families: ['norm'] },
          },
          interfaces: {
            inputs: { input: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' } },
            outputs: { out: { from: { instance: { kind: 'root', instance: 'a' }, port: 'output' }, generative: false } },
          },
        },
        library,
      ),
    ).toThrow(new PyKeyError("'norm.rmz'"));
  });

  it('leaves the document it was given untouched', () => {
    // `_record_defaults` fills a record **in place**, and the record it fills is the one
    // `static_argument` built — never a subtree of the document. The editor's store holds one tree
    // that every view projects (plan D1), so an expansion that wrote into it would edit the file.
    const text = documentText({
      quantities: QUANTITIES,
      instances: {
        a: {
          primitive: { name: 'attention.dense', version: '1.0.0' },
          arguments: {
            width: { quantity: 'd' },
            heads: { literal: 2 },
            kv_heads: { literal: 1 },
            head_dim: { literal: 4 },
            mask: { literal: 'causal' },
            rope: { record: { theta: { literal: 10000 } } },
          },
          families: ['attention'],
        },
      },
      interfaces: {
        inputs: { input: { to: [{ instance: { kind: 'root', instance: 'a' }, port: 'input' }], kind: 'token' } },
        outputs: { out: { from: { instance: { kind: 'root', instance: 'a' }, port: 'output' }, generative: false } },
      },
    });
    const document = toPython(parse(text));
    const before = serialize(toJsonValue(document));
    const graph = member(expand(document, library), 'd1') as PyRecord;
    // The default landed in the node...
    const node = member(member(graph, 'nodes') as PyRecord, 'a') as PyRecord;
    expect(member(member(node, 'arguments') as PyRecord, 'rope')).toEqual({
      theta: 10000n,
      layout: 'split',
    });
    // ...and nowhere else: the document reads, byte for byte, as it did before the expansion.
    expect(serialize(toJsonValue(document))).toBe(before);
  });
});

// --- template instances ------------------------------------------------------

describe('template instances', () => {
  const library = scratchLibrary();

  it('loads the scratch base beside the repository`s', () => {
    expect(library.problems).toEqual([]);
    expect(library.primitives.has('scratch.outer')).toBe(true);
    expect(library.primitives.has('norm.rms')).toBe(true);
  });

  it('expands a template inside a template, prefixing the nodes of each', () => {
    const graph = expanded(
      {
        quantities: {
          d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
          eps: { type: { kind: 'real' }, source: { kind: 'literal', value: 1e-5 } },
        },
        instances: {
          o: {
            primitive: { name: 'scratch.outer', version: '1.0.0' },
            arguments: { width: { quantity: 'd' }, eps: { quantity: 'eps' } },
            families: ['outer'],
          },
        },
        interfaces: {
          inputs: { input: { to: [{ instance: { kind: 'root', instance: 'o' }, port: 'in' }], kind: 'token' } },
          outputs: { out: { from: { instance: { kind: 'root', instance: 'o' }, port: 'out' }, generative: false } },
        },
      },
      library,
    );
    // "An instance of a template [is identified by] `<instance>/<identifier in the template>`"
    // (§5.2 rule 2), applied at each level.
    expect(nodeNames(graph)).toEqual(['o/i/n']);
    // Every template instance the expansion met, the caller's first and the nested one after it.
    expect(items(member(graph, 'instances') as PyValue).map(([name]) => name)).toEqual(['o', 'o/i']);
    const inner = member(member(graph, 'instances') as PyRecord, 'o/i') as PyRecord;
    expect(member(inner, 'primitive')).toEqual({ name: 'scratch.inner', version: '1.0.0' });
    // The assignment a call site makes is its arguments, evaluated: the caller's `d` reaches the
    // innermost `width` through two levels.
    expect(member(inner, 'arguments')).toEqual({ width: 8n, eps: 1e-5 });
    expect(edgeLines(graph)).toEqual([]);
    // An edge into the instance fans out to the template's own destinations, and the public
    // interface is the innermost port.
    const interfaces = member(graph, 'interfaces') as PyRecord;
    expect(member(member(interfaces, 'inputs') as PyRecord, 'input')).toEqual({
      to: [{ node: 'o/i/n', port: 'input' }],
      kind: 'token',
    });
    expect(member(member(interfaces, 'outputs') as PyRecord, 'out')).toEqual({
      node: 'o/i/n',
      port: 'output',
      generative: false,
    });
  });

  it('evaluates a generated template instance`s arguments and prefix in its own indices', () => {
    const graph = expanded(
      {
        quantities: {
          d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
          eps: { type: { kind: 'real' }, source: { kind: 'literal', value: 1e-5 } },
        },
        compositions: {
          stack: {
            indices: { layer: { start: { literal: 0 }, stop: { literal: 2 }, step: { literal: 1 } } },
            families: ['stack'],
            instances: {
              t: {
                primitive: { name: 'scratch.inner', version: '1.0.0' },
                arguments: {
                  width: { op: 'add', args: [{ quantity: 'd' }, { index: 'layer' }] },
                  eps: { quantity: 'eps' },
                },
                families: ['inner'],
                weights_location_prefix: ['layers.', { index: 'layer' }, '.'],
              },
            },
          },
        },
        interfaces: {
          inputs: {
            input: {
              to: [
                {
                  instance: {
                    kind: 'generated',
                    composition: 'stack',
                    instance: 't',
                    indices: { layer: { literal: 0 } },
                  },
                  port: 'in',
                },
              ],
              kind: 'token',
            },
          },
          outputs: {
            out: {
              from: {
                instance: {
                  kind: 'generated',
                  composition: 'stack',
                  instance: 't',
                  indices: { layer: { literal: 1 } },
                },
                port: 'out',
              },
              generative: false,
            },
          },
        },
      },
      library,
    );
    expect(nodeNames(graph)).toEqual(['stack/t[layer=0]/n', 'stack/t[layer=1]/n']);
    const instances = member(graph, 'instances') as PyRecord;
    expect(items(instances).map(([name]) => name)).toEqual(['stack/t[layer=0]', 'stack/t[layer=1]']);
    const second = member(instances, 'stack/t[layer=1]') as PyRecord;
    expect(member(second, 'arguments')).toEqual({ width: 9n, eps: 1e-5 });
    expect(member(second, 'weights_location_prefix')).toBe('layers.1.');
  });

  it('refuses a template primitive that cites itself', () => {
    expect(() =>
      expanded(
        {
          quantities: {
            d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
            eps: { type: { kind: 'real' }, source: { kind: 'literal', value: 1e-5 } },
          },
          instances: {
            l: {
              primitive: { name: 'scratch.loop', version: '1.0.0' },
              arguments: { width: { quantity: 'd' }, eps: { quantity: 'eps' } },
              families: ['loop'],
            },
          },
          interfaces: {
            inputs: { input: { to: [{ instance: { kind: 'root', instance: 'l' }, port: 'in' }], kind: 'token' } },
            outputs: { out: { from: { instance: { kind: 'root', instance: 'l' }, port: 'out' }, generative: false } },
          },
        },
        library,
      ),
    ).toThrow(new PyValueError('primitive cycle: scratch.loop -> scratch.loop'));
  });

  it('refuses a nesting deeper than the bound of §4.6', () => {
    expect(() =>
      expanded(
        {
          quantities: {
            d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
            eps: { type: { kind: 'real' }, source: { kind: 'literal', value: 1e-5 } },
          },
          instances: {
            deep: {
              primitive: { name: 'scratch.deep0', version: '1.0.0' },
              arguments: { width: { quantity: 'd' }, eps: { quantity: 'eps' } },
              families: ['deep'],
            },
          },
          interfaces: {
            inputs: { input: { to: [{ instance: { kind: 'root', instance: 'deep' }, port: 'in' }], kind: 'token' } },
            outputs: { out: { from: { instance: { kind: 'root', instance: 'deep' }, port: 'out' }, generative: false } },
          },
        },
        library,
      ),
    ).toThrow(new PyValueError('primitive nesting deeper than 8'));
  });
});

// --- the scratch base --------------------------------------------------------

/** A template primitive's unit: the identity, and the template it pins. */
function templateUnit(name: string, template: string, id: string): string {
  return JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'primitive',
    name,
    definition: { version: '1.0.0', template: { name: template, version: '1.0.0', id } },
  });
}

/** A template document: `width` and `eps` external, one instance, one input and one output. */
function templateDocument(
  id: string,
  name: string,
  instance: Record<string, unknown>,
  port: [string, string],
): string {
  const bound = { kind: 'root', instance: name };
  return JSON.stringify({
    schema: 'tensorspine/2.0',
    model: id,
    version: '1.0.0',
    primitive_libraries: [{ base: '../../base/' }],
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
    instances: { [name]: instance },
    compositions: {},
    bindings: { values: {}, parameters: {}, constants: {}, states: {} },
    interfaces: {
      inputs: { in: { to: [{ instance: bound, port: port[0] }], kind: 'token' } },
      outputs: { out: { from: { instance: bound, port: port[1] }, generative: false } },
    },
  });
}

/** `norm.rms` under the template's own external quantities. */
const LEAF = {
  primitive: { name: 'norm.rms', version: '1.0.0' },
  arguments: { width: { quantity: 'width' }, eps: { quantity: 'eps' } },
  families: ['norm'],
};

/** An instance of another template primitive, passing its own quantities down. */
function invokes(name: string): Record<string, unknown> {
  return {
    primitive: { name, version: '1.0.0' },
    arguments: { width: { quantity: 'width' }, eps: { quantity: 'eps' } },
    families: ['nested'],
  };
}

/**
 * The base: a manifest, three template primitives — a leaf, one that invokes it, one that invokes
 * itself — and a chain of nine, which is what it takes to pass `MAX_DEPTH`.
 *
 * The chain is nine because the bound is checked before the recursion: a model's instance is
 * expanded at depth 0, so `scratch.deep8` is the first one met at a depth the bound refuses.
 */
const SCRATCH: Record<string, string> = {
  'scratch/base/primitive-library.json': JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'base',
    name: 'tensorspine.scratch.d1',
    definition: {
      primitive_library: 'tensorspine/scratch-d1',
      title: 'A base for the expansion tests',
      templates: '../templates/',
    },
  }),
  'scratch/base/primitives/scratch/inner/1.0.0.json': templateUnit(
    'scratch.inner',
    'scratch-inner',
    'scratch_inner',
  ),
  'scratch/base/primitives/scratch/outer/1.0.0.json': templateUnit(
    'scratch.outer',
    'scratch-outer',
    'scratch_outer',
  ),
  'scratch/base/primitives/scratch/loop/1.0.0.json': templateUnit(
    'scratch.loop',
    'scratch-loop',
    'scratch_loop',
  ),
  'scratch/templates/scratch-inner/1.0.0.json': templateDocument('scratch_inner', 'n', LEAF, [
    'input',
    'output',
  ]),
  'scratch/templates/scratch-outer/1.0.0.json': templateDocument(
    'scratch_outer',
    'i',
    invokes('scratch.inner'),
    ['in', 'out'],
  ),
  'scratch/templates/scratch-loop/1.0.0.json': templateDocument(
    'scratch_loop',
    'l',
    invokes('scratch.loop'),
    ['in', 'out'],
  ),
};

for (let level = 0; level <= 8; level += 1) {
  SCRATCH[`scratch/base/primitives/scratch/deep${level}/1.0.0.json`] = templateUnit(
    `scratch.deep${level}`,
    `scratch-deep${level}`,
    `scratch_deep${level}`,
  );
  SCRATCH[`scratch/templates/scratch-deep${level}/1.0.0.json`] =
    level === 8
      ? templateDocument(`scratch_deep${level}`, 'n', LEAF, ['input', 'output'])
      : templateDocument(`scratch_deep${level}`, 'd', invokes(`scratch.deep${level + 1}`), [
          'in',
          'out',
        ]);
}

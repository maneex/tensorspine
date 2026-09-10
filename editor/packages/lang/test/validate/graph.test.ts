import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  PyIndexError,
  PyTypeError,
  toPython,
  UNRESOLVED,
  type PyValue,
} from '../../src/expr/index.js';
import { loadLibrary, PrimitiveLibraryError, type Library } from '../../src/library/index.js';
import {
  analyseGraph,
  analyseGraphText,
  compareSiteKeys,
  formatSemanticProblems,
  generatedSite,
  instancePorts,
  keyOf,
  loopEnvs,
  physicalName,
  portShape,
  present,
  reprShape,
  reprTuple,
  rootSite,
  selectSite,
  shapeIdentity,
  shapesAgree,
  valueToken,
  whereOfSite,
  type GraphAnalysis,
  type SemanticProblem,
} from '../../src/validate/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { repositoryRoot } from '../json/repository.js';

// `analyse`'s graph half (feature 1.6b): the sites its guards keep, the references they resolve,
// the value edges, the public interfaces, the indexing domains and the acyclicity.
//
// What the parity suite compares against the tools is in `test/parity/graph.test.ts`, over every
// model document of the repository. What is checked here is what the tools do not answer: the
// pointer each refusal carries beside its words, the identity of a site as a key, and the rules
// no fixture of `tests/rejections` reaches — V6 above all, which the suite carries no case for.
//
// The documents are written here as JSON and read as a document is read (`4096` a whole number,
// `1e-05` a float), against the repository's own reference base: the primitives are real ones,
// so a shape or a domain is the base's and not an invention of the test.

const schemas = repositorySchemas();
const library: Library = loadLibrary(['data/primitive-library'], {
  schemas,
  source: nodeSource(repositoryRoot),
});

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

/** A generated endpoint at the indices given, written as expressions. */
function generated(composition: string, name: string, indices: Record<string, unknown>): object {
  return { kind: 'generated', composition, instance: name, indices };
}

/** A value binding from one port to another. */
function edge(from: object, fromPort: string, to: object, toPort: string, rest: object = {}): object {
  return { from: { instance: from, port: fromPort }, to: { instance: to, port: toPort }, ...rest };
}

/** A whole document: the members the grammar requires, with the parts a case names. */
function document(parts: {
  quantities?: object;
  instances?: object;
  compositions?: object;
  values?: object;
  inputs?: object;
  outputs?: object;
}): PyValue {
  return read(
    JSON.stringify({
      schema: 'tensorspine/2.0',
      model: 'unit_graph',
      primitive_libraries: [{ base: '../primitive-library/' }],
      quantities: parts.quantities ?? {
        d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
      },
      constants: {},
      instances: parts.instances ?? {},
      compositions: parts.compositions ?? {},
      bindings: { values: parts.values ?? {}, parameters: {}, constants: {}, states: {} },
      interfaces: { inputs: parts.inputs ?? {}, outputs: parts.outputs ?? {} },
    }),
  );
}

/** The graph of one document, under the reference base. */
function analyse(parts: Parameters<typeof document>[0]): GraphAnalysis {
  return analyseGraph(document(parts), library);
}

/** The lines a document is refused with, as `analyse` prints them. */
function lines(parts: Parameters<typeof document>[0]): string[] {
  return formatSemanticProblems([...analyse(parts).problems]);
}

/** A normalising instance, `norm.rms` of the reference base: one input, one output. */
function norm(): object {
  return instance('norm.rms', { width: { quantity: 'd' }, eps: { literal: 0.00001 } });
}

/** An adding instance, `residual.add`: two inputs `a` and `b`, one output. */
function add(): object {
  return instance('residual.add', { width: { quantity: 'd' } });
}

/** A document whose one instance is fed by a public input and read by a public output. */
function oneNorm(extra: object = {}): Parameters<typeof document>[0] {
  return {
    instances: { n: { ...norm(), ...extra } },
    inputs: { tokens: { to: [{ instance: root('n'), port: 'input' }], kind: 'token' } },
    outputs: { out: { from: { instance: root('n'), port: 'output' }, generative: false } },
  };
}

/** The problem at a code, for a document that carries exactly one. */
function only(parts: Parameters<typeof document>[0]): SemanticProblem {
  const problems = analyse(parts).problems;
  expect(formatSemanticProblems([...problems])).toHaveLength(1);
  return problems[0] as SemanticProblem;
}

describe('a site is a key, and the key is Python’s tuple', () => {
  it('names a root instance by its name and a generated one by its indices', () => {
    expect(whereOfSite(rootSite('embed'))).toBe('embed');
    expect(
      whereOfSite(
        generatedSite('decoder', 'attn', [
          { name: 'layer', value: 3n },
          { name: 'block', value: 1n },
        ]),
      ),
    ).toBe('decoder/attn[block=1,layer=3]');
  });

  it('sorts the indices by name, however the selector wrote them', () => {
    const one = generatedSite('c', 's', [
      { name: 'b', value: 1n },
      { name: 'a', value: 0n },
    ]);
    const other = generatedSite('c', 's', [
      { name: 'a', value: 0n },
      { name: 'b', value: 1n },
    ]);
    expect(keyOf(one)).toBe(keyOf(other));
    expect(whereOfSite(one)).toBe('c/s[a=0,b=1]');
  });

  it('reads 1, 1.0 and True as one key, as a Python dictionary does', () => {
    const integer = generatedSite('c', 's', [{ name: 'i', value: 1n }]);
    const real = generatedSite('c', 's', [{ name: 'i', value: 1 }]);
    const truth = generatedSite('c', 's', [{ name: 'i', value: true }]);
    expect(keyOf(real)).toBe(keyOf(integer));
    expect(keyOf(truth)).toBe(keyOf(integer));
    // The *name* keeps the value it was written with: `str(1.0)` is `1.0`.
    expect(whereOfSite(real)).toBe('c/s[i=1.0]');
    expect(whereOfSite(integer)).toBe('c/s[i=1]');
  });

  it('keeps a string apart from the number that reads like it', () => {
    expect(valueToken('1')).not.toBe(valueToken(1n));
    expect(valueToken(null)).not.toBe(valueToken('None'));
    expect(valueToken(UNRESOLVED)).not.toBe(valueToken('unresolved'));
    expect(valueToken(-0)).toBe(valueToken(0n));
  });

  it('refuses a value Python cannot hash', () => {
    expect(() => valueToken({ a: 1n })).toThrow(PyTypeError);
    expect(() => valueToken({ a: 1n })).toThrow("unhashable type: 'dict'");
    expect(() => valueToken([1n])).toThrow("unhashable type: 'list'");
  });

  it('orders two keys as Python orders the tuples', () => {
    // `'gen' < 'root'`, then the composition, the site, and the index bindings.
    expect(compareSiteKeys(generatedSite('c', 's', []), rootSite('a'))).toBeLessThan(0);
    expect(compareSiteKeys(rootSite('a'), rootSite('b'))).toBeLessThan(0);
    expect(
      compareSiteKeys(
        generatedSite('c', 's', [{ name: 'i', value: 2n }]),
        generatedSite('c', 's', [{ name: 'i', value: 10n }]),
      ),
    ).toBeLessThan(0);
    expect(
      compareSiteKeys(generatedSite('a', 'z', []), generatedSite('b', 'a', [])),
    ).toBeLessThan(0);
  });
});

describe('the expansion: guards keep or remove a site (§5.2 rule 6)', () => {
  it('emits every site of a grid, and every root instance', () => {
    const answer = analyse({
      compositions: {
        c: {
          indices: { i: { start: { literal: 0 }, stop: { literal: 3 }, step: { literal: 1 } } },
          instances: { n: norm() },
        },
      },
    });
    expect([...answer.sites.values()].map((one) => whereOfSite(one.key))).toEqual([
      'c/n[i=0]',
      'c/n[i=1]',
      'c/n[i=2]',
    ]);
    expect(answer.stats.get('instances')).toBe(3n);
  });

  it('remembers a site its guard removes, and refuses none for it', () => {
    const answer = analyse({
      compositions: {
        c: {
          indices: { i: { start: { literal: 0 }, stop: { literal: 4 }, step: { literal: 1 } } },
          instances: {
            n: {
              ...norm(),
              when: {
                compare: { operator: 'equal', left: { op: 'modulo', args: [{ index: 'i' }, { literal: 2 }] }, right: { literal: 0 } },
              },
            },
          },
        },
      },
    });
    expect([...answer.absent.values()].map((one) => whereOfSite(one))).toEqual([
      'c/n[i=1]',
      'c/n[i=3]',
    ]);
    // A guard that fires false removes the site; it is never a refusal of its own (§5.2 rule 6).
    expect(answer.problems.filter((one) => one.code === 'V10')).toEqual([]);
    expect([...answer.sites.values()].map((one) => whereOfSite(one.key))).toEqual([
      'c/n[i=0]',
      'c/n[i=2]',
    ]);
  });

  it('refuses a guard it cannot decide (V10), never reads it as false', () => {
    const undecidable = { compare: { operator: 'equal', left: { quantity: 'absent' }, right: { literal: 0 } } };
    const problem = only({ instances: { n: { ...norm(), when: undecidable } } });
    expect(problem.code).toBe('V10');
    expect(problem.message).toBe('n: `when` does not resolve');
    expect(problem.path).toBe('/instances/n');
  });

  it('names the composition, the site and the environment of an undecidable guard', () => {
    const undecidable = { compare: { operator: 'equal', left: { quantity: 'absent' }, right: { literal: 0 } } };
    const answer = analyse({
      compositions: {
        c: {
          indices: { i: { start: { literal: 0 }, stop: { literal: 2 }, step: { literal: 1 } } },
          instances: { n: { ...norm(), when: undecidable } },
        },
      },
    });
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V10] c.n{'i': 0}: `when` does not resolve",
      "[V10] c.n{'i': 1}: `when` does not resolve",
    ]);
    expect(answer.problems[0]?.path).toBe('/compositions/c/instances/n');
  });

  it('does not emit a binding whose endpoint a guard removed (§5.2 rule 3)', () => {
    const answer = analyse({
      instances: {
        a: { ...norm(), when: { boolean: false } },
        b: norm(),
      },
      values: { link: edge(root('a'), 'output', root('b'), 'input') },
      inputs: {},
      outputs: { out: { from: { instance: root('b'), port: 'output' }, generative: false } },
    });
    // No V1 for the absent instance; `b` is simply unfed, which is V7's line — and its output,
    // with no domain reaching it, is V5's.
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      '[V7] input port with no producer: norm.rms@b.input',
      '[V5] norm.rms@b.output: domain undetermined',
    ]);
    expect(answer.edges).toEqual([]);
  });
});

describe('references resolve (V1)', () => {
  it('names a primitive the library does not hold', () => {
    const problem = only({ instances: { n: instance('norm.rmz', { width: { quantity: 'd' } }) } });
    expect(problem.code).toBe('V1');
    expect(problem.message).toBe('primitive absent from primitive library: norm.rmz');
    expect(problem.path).toBe('/instances/n/primitive');
  });

  it('names a version the library does not carry', () => {
    const answer = analyse({
      instances: {
        n: {
          primitive: { name: 'norm.rms', version: '9.9.9' },
          arguments: { width: { quantity: 'd' }, eps: { literal: 0.00001 } },
          families: ['norm'],
        },
      },
      inputs: { tokens: { to: [{ instance: root('n'), port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: root('n'), port: 'output' }, generative: false } },
    });
    const problem = answer.problems.find((one) => one.code === 'V1') as SemanticProblem;
    expect(problem.message).toBe('norm.rms: version 9.9.9 != primitive library 1.0.0');
    expect(problem.path).toBe('/instances/n/primitive/version');
  });

  it('names an instance a binding invents, and the port it has not', () => {
    const answer = analyse({
      instances: { n: norm() },
      values: {
        gone: edge(root('n'), 'output', root('missing'), 'input'),
        wrong: edge(root('n'), 'nothing', root('n'), 'input'),
      },
    });
    expect(formatSemanticProblems([...answer.problems])).toContain(
      '[V1] gone{}: to instance does not exist missing',
    );
    expect(formatSemanticProblems([...answer.problems])).toContain(
      "[V1] wrong: norm.rms has no output port 'nothing'",
    );
    expect(answer.problems[0]?.path).toBe('/bindings/values/gone');
  });

  it('names a stream a public input joins without one being declared', () => {
    const answer = analyse({
      ...oneNorm(),
      inputs: {
        tokens: { to: [{ instance: root('n'), port: 'input' }], kind: 'token', stream: 'pixels' },
      },
    });
    const problem = answer.problems.find((one) => one.code === 'V1') as SemanticProblem;
    expect(problem.message).toBe("input tokens: joins unknown stream 'pixels'");
    expect(problem.path).toBe('/interfaces/inputs/tokens');
  });
});

describe('shapes compose (V4) and bindings are total (V7, V13)', () => {
  it('refuses an edge whose extents differ, naming both shapes', () => {
    const answer = analyse({
      quantities: {
        d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
        e: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 16 } },
      },
      instances: {
        a: norm(),
        b: instance('norm.rms', { width: { quantity: 'e' }, eps: { literal: 0.00001 } }),
      },
      values: { link: edge(root('a'), 'output', root('b'), 'input') },
      inputs: { tokens: { to: [{ instance: root('a'), port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: root('b'), port: 'output' }, generative: false } },
    });
    const problem = answer.problems.find((one) => one.code === 'V4') as SemanticProblem;
    expect(problem.message).toBe(
      "link: shapes do not unify norm.rms.output[('model.width', 8)] -> " +
        "norm.rms.input[('model.width', 16)]",
    );
    expect(problem.path).toBe('/bindings/values/link');
  });

  it('refuses an input port fed twice, naming both producers', () => {
    const answer = analyse({
      instances: { a: norm(), b: norm(), c: norm() },
      values: {
        first: edge(root('a'), 'output', root('c'), 'input'),
        second: edge(root('b'), 'output', root('c'), 'input'),
      },
      inputs: {
        one: {
          to: [
            { instance: root('a'), port: 'input' },
            { instance: root('b'), port: 'input' },
          ],
          kind: 'token',
        },
      },
      outputs: { out: { from: { instance: root('c'), port: 'output' }, generative: false } },
    });
    expect(formatSemanticProblems([...answer.problems])).toContain(
      '[V7] input port fed twice: c.input by first and second',
    );
  });

  it('refuses a public input feeding a port a binding already feeds', () => {
    const answer = analyse({
      instances: { a: norm(), b: norm() },
      values: { link: edge(root('a'), 'output', root('b'), 'input') },
      inputs: {
        one: {
          to: [
            { instance: root('a'), port: 'input' },
            { instance: root('b'), port: 'input' },
          ],
          kind: 'token',
        },
      },
      outputs: { out: { from: { instance: root('b'), port: 'output' }, generative: false } },
    });
    expect(formatSemanticProblems([...answer.problems])).toContain(
      '[V7] input one: port b.input also fed by link',
    );
  });

  it('refuses the ports of one public input whose shapes differ (V4)', () => {
    const answer = analyse({
      quantities: {
        d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
        e: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 16 } },
      },
      instances: {
        a: norm(),
        b: instance('norm.rms', { width: { quantity: 'e' }, eps: { literal: 0.00001 } }),
      },
      inputs: {
        one: {
          to: [
            { instance: root('a'), port: 'input' },
            { instance: root('b'), port: 'input' },
          ],
          kind: 'token',
        },
      },
      outputs: {
        first: { from: { instance: root('a'), port: 'output' }, generative: false },
        second: { from: { instance: root('b'), port: 'output' }, generative: false },
      },
    });
    expect(formatSemanticProblems([...answer.problems])).toContain(
      "[V4] input one: feeds a.input[('model.width', 8)] and b.input[('model.width', 16)], " +
        'whose shapes differ',
    );
  });

  it('refuses an unfed input port (V7) and an unread output port (V13), at the site', () => {
    const answer = analyse({ instances: { n: norm() } });
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      '[V7] input port with no producer: norm.rms@n.input',
      '[V13] output consumed by nothing: norm.rms@n.output',
      '[V5] norm.rms@n.output: domain undetermined',
    ]);
    expect(answer.problems.map((one) => one.path)).toEqual([
      '/instances/n',
      '/instances/n',
      '/instances/n',
    ]);
  });

  it('takes a public output as a consumer, and a public input as a producer', () => {
    expect(lines(oneNorm())).toEqual([]);
  });
});

describe('the value graph is acyclic (V6)', () => {
  // `tests/rejections` carries no V6 case at all: the rule is exercised here alone.
  it('counts the instances a cycle holds', () => {
    const answer = analyse({
      instances: { a: add(), b: norm() },
      values: {
        forward: edge(root('a'), 'output', root('b'), 'input'),
        back: edge(root('b'), 'output', root('a'), 'b'),
      },
      inputs: { tokens: { to: [{ instance: root('a'), port: 'a' }], kind: 'token' } },
      outputs: {},
    });
    expect(formatSemanticProblems([...answer.problems])).toContain(
      '[V6] value cycle: 2 instance(s) in a cycle',
    );
    expect(answer.stats.get('dag')).toBe(false);
    expect(answer.order).toEqual([]);
  });

  it('orders the graph and reports a graph with no cycle as one', () => {
    const answer = analyse({
      instances: { a: norm(), b: norm() },
      values: { link: edge(root('a'), 'output', root('b'), 'input') },
      inputs: { tokens: { to: [{ instance: root('a'), port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: root('b'), port: 'output' }, generative: false } },
    });
    expect(answer.stats.get('dag')).toBe(true);
    expect(answer.order.map((one) => whereOfSite(one))).toEqual(['a', 'b']);
  });
});

describe('indexing domains agree (V5, §5.3)', () => {
  it('carries the public input’s domain along every edge', () => {
    const answer = analyse({
      instances: { a: norm(), b: norm() },
      values: { link: edge(root('a'), 'output', root('b'), 'input') },
      inputs: { tokens: { to: [{ instance: root('a'), port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: root('b'), port: 'output' }, generative: false } },
    });
    expect([...answer.domains.values()].map((one) => [whereOfSite(one.site), one.port, ...one.domain])).toEqual([
      ['a', 'input', 'token', 'tokens'],
      ['a', 'output', 'token', 'tokens'],
      ['b', 'input', 'token', 'tokens'],
      ['b', 'output', 'token', 'tokens'],
    ]);
    expect(answer.stats.get('resolved_domains')).toBe(4n);
    expect(answer.ports.outputs.get('out')).toEqual({
      kind: 'token',
      stream: 'tokens',
      shape: [['model.width', 8n]],
    });
  });

  it('refuses an output declared generative that is not of kind token', () => {
    const answer = analyse({
      instances: { n: norm() },
      inputs: { pixels: { to: [{ instance: root('n'), port: 'input' }], kind: 'patch' } },
      outputs: { out: { from: { instance: root('n'), port: 'output' }, generative: true } },
    });
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      '[V5] output out: generative, but of kind patch',
    ]);
    expect(answer.problems[0]?.path).toBe('/interfaces/outputs/out');
  });

  it('refuses an instance whose domain nothing determines', () => {
    const answer = analyse({
      instances: { a: norm(), b: norm() },
      values: { link: edge(root('a'), 'output', root('b'), 'input') },
      outputs: { out: { from: { instance: root('b'), port: 'output' }, generative: false } },
    });
    expect(formatSemanticProblems([...answer.problems])).toContain(
      '[V5] norm.rms@a.output: domain undetermined',
    );
    expect(formatSemanticProblems([...answer.problems])).toContain(
      '[V5] link: the domain of a.output is undetermined',
    );
  });
});

describe('a public input joins a stream at a kind it carries (V19)', () => {
  it('accepts a join at a kind another value on the stream carries', () => {
    // `residual.add` fuses the two: `b` joins the stream `a` delivers, at the same kind (§2.3).
    expect(
      lines({
        instances: { fuse: add() },
        inputs: {
          first: { to: [{ instance: root('fuse'), port: 'a' }], kind: 'token' },
          second: {
            to: [{ instance: root('fuse'), port: 'b' }],
            kind: 'token',
            stream: 'first',
          },
        },
        outputs: { out: { from: { instance: root('fuse'), port: 'output' }, generative: false } },
      }),
    ).toEqual([]);
  });

  it('refuses a join at a kind the stream carries only through the input itself', () => {
    const answer = analyse({
      instances: { fuse: add() },
      inputs: {
        first: { to: [{ instance: root('fuse'), port: 'a' }], kind: 'token' },
        second: { to: [{ instance: root('fuse'), port: 'b' }], kind: 'patch', stream: 'first' },
      },
      outputs: { out: { from: { instance: root('fuse'), port: 'output' }, generative: false } },
    });
    const problem = answer.problems.find((one) => one.code === 'V19') as SemanticProblem;
    expect(problem.message).toBe(
      "input second: joins stream 'first' at kind patch, which that stream carries at no value " +
        "independently of the input (it carries ['token']): the join would count elements the " +
        'stream does not have (§5.3)',
    );
    expect(problem.path).toBe('/interfaces/inputs/second');
  });
});

describe('what the document exposes, and what it cannot be read as', () => {
  it('answers the public interface ports with their evaluated shapes', () => {
    const answer = analyse(oneNorm());
    expect([...answer.ports.inputs]).toEqual([
      ['tokens', { kind: 'token', stream: 'tokens', shape: [['model.width', 8n]] }],
    ]);
    expect([...answer.ports.outputs]).toEqual([
      ['out', { kind: 'token', stream: 'tokens', shape: [['model.width', 8n]] }],
    ]);
  });

  it('reports a duplicate member name as V12 and a scoped-rule refusal as V1', () => {
    const duplicate = analyseGraphText('{"model": "a", "model": "b"}', library);
    // `model.py` writes the rule into its own message and `analyse` prefixes it again, so the
    // line the tools print carries `(V12)` twice (feature 0.3's finding, reproduced).
    expect(formatSemanticProblems([...duplicate.problems])).toEqual([
      "[V12] duplicate member name 'model' (V12)",
    ]);
    expect(duplicate.model).toBeNull();
    const unknownSite = analyseGraph(
      document({
        compositions: {
          c: {
            indices: { i: { start: { literal: 0 }, stop: { literal: 1 }, step: { literal: 1 } } },
            instances: { n: norm() },
            bindings: {
              values: {
                r: { from: { site: 'nowhere', port: 'output' }, to: { site: 'n', port: 'input' } },
              },
            },
          },
        },
      }),
      library,
    );
    expect(formatSemanticProblems([...unknownSite.problems])).toEqual([
      "[V1] composition 'c', binding 'r': no site named 'nowhere'",
    ]);
  });

  it('reads a text and a parsed document alike', () => {
    const parts = oneNorm();
    const text = JSON.stringify({
      schema: 'tensorspine/2.0',
      model: 'unit_graph',
      primitive_libraries: [{ base: '../primitive-library/' }],
      quantities: { d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } } },
      constants: {},
      instances: parts.instances,
      compositions: {},
      bindings: { values: {}, parameters: {}, constants: {}, states: {} },
      interfaces: { inputs: parts.inputs, outputs: parts.outputs },
    });
    const fromText = analyseGraphText(text, library);
    expect(formatSemanticProblems([...fromText.problems])).toEqual([]);
    expect([...fromText.ports.outputs]).toEqual([...analyse(parts).ports.outputs]);
  });

  it('raises where the loader left a template primitive unresolved', () => {
    // `primitive_library.template_path` raises out of `analyse` when the load resolved no
    // document for a template primitive — the one place a caller of a gathered library still
    // meets a `PrimitiveLibraryError`, and the port raises it with the tools' words.
    const unpinned: Library = {
      bases: [],
      byId: new Map([
        [
          'audit.template@1.0.0',
          {
            name: 'audit.template',
            version: '1.0.0',
            definition: read(
              JSON.stringify({
                version: '1.0.0',
                template: { name: 'nowhere', version: '1.0.0', id: 'nowhere' },
              }),
            ),
            file: '<test>',
            base: '<test>',
          },
        ],
      ]),
      primitives: new Map([
        [
          'audit.template',
          read(
            JSON.stringify({
              version: '1.0.0',
              template: { name: 'nowhere', version: '1.0.0', id: 'nowhere' },
            }),
          ),
        ],
      ]),
      axes: new Map(),
      precision: new Map(),
      templates: new Map(),
      problems: [],
    };
    const document = read(
      JSON.stringify({
        schema: 'tensorspine/2.0',
        model: 'unit_graph',
        primitive_libraries: [{ base: './' }],
        quantities: {},
        constants: {},
        instances: {
          t: {
            primitive: { name: 'audit.template', version: '1.0.0' },
            arguments: {},
            families: ['audit'],
          },
        },
        compositions: {},
        bindings: { values: {}, parameters: {}, constants: {}, states: {} },
        interfaces: { inputs: {}, outputs: {} },
      }),
    );
    expect(() => analyseGraph(document, unpinned)).toThrow(PrimitiveLibraryError);
    expect(() => analyseGraph(document, unpinned)).toThrow(
      "template 'nowhere' 1.0.0 was not resolved at load",
    );
  });

  it('raises what the tools raise on a public input feeding nothing', () => {
    // `decl['to'][0]` on an empty list: the grammar requires at least one endpoint (`minItems`),
    // and a document that has none reaches CPython's own `IndexError` in the interface block.
    const empty = document({
      instances: { n: norm() },
      inputs: { tokens: { to: [], kind: 'token' } },
      outputs: { out: { from: { instance: root('n'), port: 'output' }, generative: false } },
    });
    expect(() => analyseGraph(empty, library)).toThrow(PyIndexError);
    expect(() => analyseGraph(empty, library)).toThrow('list index out of range');
  });

  it('raises what the tools raise on an interface written in the old form', () => {
    // `for endpoint in decl['to']` walks a dictionary's *names*, and `endpoint['instance']` on a
    // string is CPython's own refusal — a document the grammar refuses, and the exception it gets.
    const old = document({
      instances: { n: norm() },
      inputs: {
        tokens: { to: { instance: root('n'), port: 'input' }, kind: 'token' },
      },
      outputs: { out: { from: { instance: root('n'), port: 'output' }, generative: false } },
    });
    expect(() => analyseGraph(old, library)).toThrow(PyTypeError);
    expect(() => analyseGraph(old, library)).toThrow("string indices must be integers, not 'str'");
  });
});

describe('the pieces the later stages share', () => {
  it('selects a site at the indices a selector overrides', () => {
    const quantities = new Map<string, PyValue>([['layers', 4n]]);
    const env = new Map<string, PyValue>([['layer', 2n]]);
    const selector = read(
      JSON.stringify(
        generated('decoder', 'ffn_r', { layer: { op: 'subtract', args: [{ index: 'layer' }, { literal: 1 }] } }),
      ),
    );
    expect(whereOfSite(selectSite(selector, quantities, env))).toBe('decoder/ffn_r[layer=1]');
    expect(whereOfSite(selectSite(read(JSON.stringify(root('embed'))), quantities, env))).toBe(
      'embed',
    );
  });

  it('keeps the environments a rule’s guard admits, and refuses one it cannot decide', () => {
    const quantities = new Map<string, PyValue>([['layers', 3n]]);
    const problems: SemanticProblem[] = [];
    const binding = read(
      JSON.stringify({
        for_each: { layer: { start: { literal: 0 }, stop: { quantity: 'layers' }, step: { literal: 1 } } },
        when: { compare: { operator: 'greater_or_equal', left: { index: 'layer' }, right: { literal: 1 } } },
      }),
    );
    const envs = loopEnvs(binding, 'r', quantities, problems);
    expect(envs.map((one) => [...one])).toEqual([
      [['layer', 1n]],
      [['layer', 2n]],
    ]);
    expect(problems).toEqual([]);
    const undecidable = read(
      JSON.stringify({
        for_each: { layer: { start: { literal: 0 }, stop: { literal: 2 }, step: { literal: 1 } } },
        when: { compare: { operator: 'equal', left: { quantity: 'absent' }, right: { literal: 0 } } },
      }),
    );
    expect(loopEnvs(undecidable, 'r', quantities, problems, ['bindings', 'values', 'r'])).toEqual(
      [],
    );
    expect(formatSemanticProblems(problems)).toEqual([
      "[V10] r{'layer': 0}: `when` does not resolve",
      "[V10] r{'layer': 1}: `when` does not resolve",
    ]);
    expect(problems[0]?.path).toBe('/bindings/values/r');
  });

  it('evaluates a shape, its presence and its equality', () => {
    const port = read(
      JSON.stringify({
        shape: {
          axes: [
            { name: 'feature', axis: 'model.width', nature: 'feature', extent: { argument: 'width' } },
          ],
        },
      }),
    );
    expect(portShape(port, { width: 8n })).toEqual([['model.width', 8n]]);
    expect(portShape(read('{}'), {})).toBeNull();
    expect(shapeIdentity(read('{"axes": []}'), {})).toEqual([]);
    // Python's `==` on the extents: `8` and `8.0` are one value.
    expect(shapesAgree(portShape(port, { width: 8n }), portShape(port, { width: 8 }))).toBe(true);
    expect(shapesAgree(portShape(port, { width: 8n }), portShape(port, { width: 9n }))).toBe(false);
    expect(shapesAgree(null, null)).toBe(true);
    expect(shapesAgree(null, [])).toBe(false);
    expect(present(read('{}'), {})).toBe(true);
    expect(
      present(read('{"present_when": {"compare": {"operator": "equal", "left": {"argument": "k"}, "right": {"literal": 1}}}}'), {
        k: 1n,
      }),
    ).toBe(true);
  });

  it('writes a tuple and a shape as Python writes them', () => {
    expect(reprTuple(['model.width', 4096n])).toBe("('model.width', 4096)");
    expect(reprTuple(['only'])).toBe("('only',)");
    expect(reprShape([])).toBe('[]');
    expect(reprShape([['a.b', 2n]])).toBe("[('a.b', 2)]");
  });

  it('evaluates a physical name over literals, indices and coordinates', () => {
    const quantities = new Map<string, PyValue>();
    const value = (expression: PyValue, env: ReadonlyMap<string, PyValue>): PyValue => {
      const name = (expression as { index: string }).index;
      return env.get(name) ?? UNRESOLVED;
    };
    const items = read('["layers.", {"index": "layer"}, ".weight"]');
    expect(
      physicalName(items, new Map([['layer', 3n]]), value, new Map()).name,
    ).toBe('layers.3.weight');
    expect(physicalName(items, new Map(), value, new Map())).toEqual({
      name: null,
      problem: "index 'layer' does not resolve in the physical name",
    });
    const stacked = read('[{"coordinate": "multiplicity"}]');
    expect(physicalName(stacked, new Map(), value, new Map([['multiplicity', 2n]])).name).toBe('2');
    expect(physicalName(stacked, new Map(), value, new Map()).problem).toBe(
      "`coordinate` 'multiplicity' outside a stack over that axis",
    );
    expect(quantities.size).toBe(0);
  });

  it('turns an expansion’s interface into the instance’s ports', () => {
    const ports = instancePorts({
      inputs: new Map([
        ['hidden', { kind: 'token', stream: 'hidden', shape: [['model.width', 3072n]] }],
      ]),
      outputs: new Map([['out', { kind: null, stream: null, shape: null }]]),
    });
    expect(ports).toEqual({
      inputs: {
        hidden: {
          role: 'activation.hidden',
          domain: { kind: 'token', from: { self: true } },
          shape: {
            axes: [
              {
                name: 'width',
                axis: 'model.width',
                nature: 'feature',
                extent: { literal: 3072n },
              },
            ],
          },
        },
      },
      // A kind the expansion left undetermined becomes `inherit`, and no shape is written.
      outputs: { out: { role: 'activation.hidden', domain: { kind: 'inherit', from: { self: true } } } },
    });
  });
});

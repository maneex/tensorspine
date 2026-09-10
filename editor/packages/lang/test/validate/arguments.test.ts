import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  PyKeyError,
  PyTypeError,
  resolveQuantities,
  staticArgument,
  toPython,
  UNRESOLVED,
  type Env,
  type PyRecord,
  type PyValue,
  type Quantities,
} from '../../src/expr/index.js';
import { loadLibrary, primitiveOf } from '../../src/library/index.js';
import { loadModel } from '../../src/model/index.js';
import {
  checkInvariants,
  describeArguments,
  formatSemanticProblems,
  resolveArguments,
  type ArgumentFact,
  type SemanticProblem,
} from '../../src/validate/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource, REFERENCE_BASE, repositoryPath } from '../library/source.js';

// `resolve_arguments`, `_resolve_record`, `_check_argument_domain` and the V8 block of
// `tools/validate.py` (feature 1.6a): V2, V3 and V8 over one instance's arguments, and the facts
// the argument sheet of §4.12 reads.
//
// Declarations and argument maps are written as JSON and read as a document is read, so `32` is
// Python's `int` and `32.0` its `float` — the distinction V3 turns on. What the parity suite
// compares against the tools is in `test/parity/arguments.test.ts`; what is checked here is what
// the tools do not answer: the pointer each problem carries, the attribution of a problem to its
// row, and the facts a sheet renders.

/** A declaration, an argument map, a value: read as a document is read (D12, feature 1.2). */
function value(json: string): PyValue {
  return toPython(parse(json));
}

/** No quantities and no indices: an argument map written in literals alone. */
const LITERALS: Quantities = new Map<string, PyValue>();

/** No index in scope: a site outside a composition. */
const NO_INDEX: Env = new Map<string, PyValue>();

/** `analyse`'s own evaluator, under the quantities and index environment given. */
function evaluator(quantities: Quantities = LITERALS, env: Env = NO_INDEX) {
  return (one: PyValue): PyValue => staticArgument(one, quantities, env);
}

/** Everything the module answers about one instance, from JSON. */
function describe_(definition: string, given: string, quantities?: Quantities, env?: Env) {
  return describeArguments(value(definition), value(given), evaluator(quantities, env));
}

/** The lines a description carries, as `analyse` would print them. */
function linesOf(definition: string, given: string): string[] {
  return formatSemanticProblems([...describe_(definition, given).problems]);
}

/** The fact at one argument path. */
function factAt(facts: readonly ArgumentFact[], path: string): ArgumentFact {
  const found = facts.find((one) => one.path === path);
  expect(found, path).toBeDefined();
  return found as ArgumentFact;
}

const HEADS =
  '{"type": {"kind": "cardinality"}, "required": true, "structural": true, ' +
  '"domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}}}';
const KV_HEADS =
  '{"type": {"kind": "cardinality"}, "required": false, "structural": true, ' +
  '"default": {"argument": "heads"}, "domain": {"kind": "interval", ' +
  '"lower": {"value": {"literal": 1}, "inclusive": true}, ' +
  '"upper": {"value": {"argument": "heads"}, "inclusive": true}}}';

/** `heads` and `kv_heads` as `attention.dense` declares them, the pair §4.6 names. */
const ATTENTION = `{"arguments": {"heads": ${HEADS}, "kv_heads": ${KV_HEADS}}}`;

describe('resolveArguments: defaults (V2)', () => {
  it('applies a default that names another argument, and says where the value came from', () => {
    const answer = describe_(ATTENTION, '{"heads": {"literal": 32}}');
    expect(answer.problems).toEqual([]);
    expect(answer.values['kv_heads']).toBe(32n);
    expect(factAt(answer.facts, 'heads').source).toBe('given');
    const defaulted = factAt(answer.facts, 'kv_heads');
    expect(defaulted.source).toBe('default');
    expect(defaulted.value).toBe(32n);
    expect(defaulted.written).toBeUndefined();
    expect(defaulted.domain).toBe('ok');
  });

  it('keeps the value the document writes over the default', () => {
    const answer = describe_(ATTENTION, '{"heads": {"literal": 32}, "kv_heads": {"literal": 8}}');
    expect(answer.values['kv_heads']).toBe(8n);
    expect(factAt(answer.facts, 'kv_heads').source).toBe('given');
  });

  it('resolves a chain of defaults declared backwards, to a fixpoint', () => {
    const declaration =
      '{"arguments": {' +
      '"c": {"type": {"kind": "cardinality"}, "required": false, "structural": true, ' +
      '"default": {"argument": "b"}}, ' +
      '"b": {"type": {"kind": "cardinality"}, "required": false, "structural": true, ' +
      '"default": {"argument": "a"}}, ' +
      '"a": {"type": {"kind": "cardinality"}, "required": true, "structural": true}}}';
    const answer = describe_(declaration, '{"a": {"literal": 4}}');
    expect(answer.problems).toEqual([]);
    // The given name first, then the defaults in the order they resolved: `b` before `c`.
    expect(Object.keys(answer.values)).toEqual(['a', 'b', 'c']);
    expect(answer.values['c']).toBe(4n);
  });

  it('refuses a default that never resolves, and one that resolves into a cycle', () => {
    expect(
      linesOf(
        '{"arguments": {"scale": {"type": {"kind": "real"}, "required": false, ' +
          '"structural": false, "default": {"argument": "missing"}}}}',
        '{}',
      ),
    ).toEqual(["[V2] default of 'scale' does not resolve"]);
    const cycle =
      '{"arguments": {' +
      '"a": {"type": {"kind": "cardinality"}, "required": false, "structural": true, ' +
      '"default": {"argument": "b"}}, ' +
      '"b": {"type": {"kind": "cardinality"}, "required": false, "structural": true, ' +
      '"default": {"argument": "a"}}}}';
    expect(linesOf(cycle, '{}')).toEqual([
      "[V2] default of 'a' does not resolve",
      "[V2] default of 'b' does not resolve",
    ]);
  });

  it('applies a default whose value is false or zero: what is refused is `None`', () => {
    const answer = describe_(
      '{"arguments": {"cross": {"type": {"kind": "boolean"}, "required": false, ' +
        '"structural": true, "default": {"literal": false}}, ' +
        '"offset": {"type": {"kind": "cardinality"}, "required": false, "structural": false, ' +
        '"default": {"literal": 0}}}}',
      '{}',
    );
    expect(answer.problems).toEqual([]);
    expect(answer.values).toEqual({ cross: false, offset: 0n });
  });
});

describe('resolveArguments: presence (V2) and applicability (V3)', () => {
  const MASKED =
    '{"arguments": {' +
    '"mask": {"type": {"kind": "enum", "values": ["causal", "chunked"]}, "required": true, ' +
    '"structural": true}, ' +
    '"chunk": {"type": {"kind": "cardinality"}, "required": true, "structural": true, ' +
    '"present_when": {"compare": {"operator": "equal", "left": {"argument": "mask"}, ' +
    '"right": {"literal": "chunked"}}}}}}';

  it('refuses a required argument that is missing, and an argument nobody declared', () => {
    // `kv_heads` defaults to `heads`, which is not there to be read: a default that names a
    // refused argument does not resolve, and says so before the missing argument is reported.
    expect(linesOf(ATTENTION, '{"kernel_hint": {"literal": "flash"}}')).toEqual([
      "[V2] unknown argument 'kernel_hint'",
      "[V2] default of 'kv_heads' does not resolve",
      "[V2] required argument missing 'heads'",
    ]);
  });

  it('does not miss a required argument that does not apply', () => {
    const answer = describe_(MASKED, '{"mask": {"literal": "causal"}}');
    expect(answer.problems).toEqual([]);
    const chunk = factAt(answer.facts, 'chunk');
    expect(chunk.applicable).toBe(false);
    expect(chunk.source).toBe('absent');
  });

  it('refuses a value supplied where it does not apply — not ignored (I2)', () => {
    const answer = describe_(MASKED, '{"mask": {"literal": "causal"}, "chunk": {"literal": 128}}');
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V3] argument 'chunk' is present but inapplicable for these arguments",
    ]);
    const chunk = factAt(answer.facts, 'chunk');
    expect(chunk.applicable).toBe(false);
    expect(chunk.source).toBe('given');
    expect(chunk.value).toBe(128n);
    expect(chunk.problems).toHaveLength(1);
  });

  it('misses a required argument that does apply', () => {
    expect(linesOf(MASKED, '{"mask": {"literal": "chunked"}}')).toEqual([
      "[V2] required argument missing 'chunk'",
    ]);
  });
});

describe('resolveArguments: types and domains (V3)', () => {
  it('accepts a real in seconds and refuses one in tokens', () => {
    const physical = (unit: string): string =>
      `{"arguments": {"span": {"type": {"kind": "physical", "unit": "${unit}"}, ` +
      '"required": true, "structural": true}}}';
    expect(linesOf(physical('seconds'), '{"span": {"literal": 0.5}}')).toEqual([]);
    expect(linesOf(physical('tokens'), '{"span": {"literal": 2.5}}')).toEqual([
      "[V3] argument 'span' = 2.5 is not a whole number of tokens (only seconds is real)",
    ]);
    // A whole number written as a real is a whole number of tokens: `float(v) == int(v)`.
    expect(linesOf(physical('tokens'), '{"span": {"literal": 4096.0}}')).toEqual([]);
  });

  it('reads the domain after the type, and only when the type held', () => {
    const answer = describe_(ATTENTION, '{"heads": {"literal": 32}, "kv_heads": {"literal": 0}}');
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V3] argument 'kv_heads' = 0 is below the domain bound 1 (inclusive)",
    ]);
    expect(factAt(answer.facts, 'kv_heads').domain).toBe('refused');
    // Refused once, with its reason: nothing downstream reads it as a value.
    expect(answer.values['kv_heads']).toBe(UNRESOLVED);
    const refused = describe_(ATTENTION, '{"heads": {"literal": 32}, "kv_heads": {"literal": -1}}');
    expect(factAt(refused.facts, 'kv_heads').domain).toBe('unchecked');
  });

  it('evaluates a bound that names another argument in the instance’s own arguments', () => {
    expect(linesOf(ATTENTION, '{"heads": {"literal": 8}, "kv_heads": {"literal": 32}}')).toEqual([
      "[V3] argument 'kv_heads' = 32 is above the domain bound 8 (inclusive)",
    ]);
    // A bound whose argument was refused is skipped, not read as a limit.
    expect(
      linesOf(ATTENTION, '{"heads": {"literal": "many"}, "kv_heads": {"literal": 32}}'),
    ).toEqual(["[V3] argument 'heads' = 'many' is not a cardinality (non-negative integer)"]);
  });

  it('says a value does not resolve when the quantity it names has none', () => {
    const answer = describe_(ATTENTION, '{"heads": {"quantity": "absent"}}');
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      // The sentinel is a value the map carries, so `kv_heads`' default reads it and answers it;
      // `primitive_value` refuses to write it, which is the V2 line.
      "[V2] default of 'kv_heads' does not resolve",
      "[V3] argument 'heads' does not resolve to a value",
    ]);
    expect(factAt(answer.facts, 'heads').value).toBe(UNRESOLVED);
  });

  it('takes an index of the composition as a value', () => {
    const declaration =
      '{"arguments": {"layer": {"type": {"kind": "cardinality"}, "required": true, ' +
      '"structural": true}}}';
    const answer = describeArguments(
      value(declaration),
      value('{"layer": {"index": "layer"}}'),
      evaluator(LITERALS, new Map([['layer', 3n]])),
    );
    expect(answer.problems).toEqual([]);
    expect(answer.values['layer']).toBe(3n);
  });
});

describe('resolveArguments: records', () => {
  const ROPE =
    '{"arguments": {"rope": {"type": {"kind": "record", "fields": {' +
    '"theta": {"type": {"kind": "real"}, "required": true, "structural": false}, ' +
    '"scaling": {"type": {"kind": "record", "fields": {' +
    '"kind": {"type": {"kind": "enum", "values": ["yarn", "linear"]}, "required": true, ' +
    '"structural": true}, ' +
    '"beta_fast": {"type": {"kind": "real"}, "required": true, "structural": false, ' +
    '"present_when": {"compare": {"operator": "equal", ' +
    '"left": {"argument": "rope.scaling.kind"}, "right": {"literal": "yarn"}}}}}}, ' +
    '"required": false, "structural": true}}}, "required": true, "structural": true}}}';

  it('resolves the fields of a record, one row per field, deeper first only where it walked', () => {
    const answer = describe_(
      ROPE,
      '{"rope": {"record": {"theta": {"literal": 500000.0}, "scaling": {"record": ' +
        '{"kind": {"literal": "yarn"}, "beta_fast": {"literal": 32.0}}}}}}',
    );
    expect(answer.problems).toEqual([]);
    expect(answer.facts.map((one) => one.path)).toEqual([
      'rope',
      'rope.theta',
      'rope.scaling',
      'rope.scaling.kind',
      'rope.scaling.beta_fast',
    ]);
    expect(factAt(answer.facts, 'rope.scaling.beta_fast').applicable).toBe(true);
    expect(factAt(answer.facts, 'rope.scaling.beta_fast').value).toBe(32);
  });

  it('reads a field’s `present_when` through the record it is filling in place', () => {
    const answer = describe_(
      ROPE,
      '{"rope": {"record": {"theta": {"literal": 500000.0}, "scaling": {"record": ' +
        '{"kind": {"literal": "linear"}, "beta_fast": {"literal": 32.0}}}}}}',
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V3] argument 'rope.scaling.beta_fast' is present but inapplicable for these arguments",
    ]);
    expect(factAt(answer.facts, 'rope.scaling.beta_fast').applicable).toBe(false);
  });

  it('refuses a scalar where a record is declared, and walks no field of it', () => {
    const answer = describe_(ROPE, '{"rope": {"literal": 0.1}}');
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V3] argument 'rope' = 0.1 is not a record",
    ]);
    expect(answer.facts.map((one) => one.path)).toEqual(['rope']);
    // A record is not replaced by the sentinel: its fields carry their own refusals.
    expect(answer.values['rope']).toBe(0.1);
  });

  it('refuses a field nobody declared, and drops it from the resolved record', () => {
    const answer = describe_(
      ROPE,
      '{"rope": {"record": {"theta": {"literal": 500000.0}, "thetta": {"literal": 1.0}}}}',
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V2] unknown argument 'rope.thetta'",
    ]);
    expect(Object.keys(answer.values['rope'] as PyRecord)).toEqual(['theta']);
  });

  it('puts a field’s refusal on the field’s row, not on the record’s', () => {
    const answer = describe_(
      ROPE,
      '{"rope": {"record": {"theta": {"literal": "big"}, "scaling": {"record": ' +
        '{"kind": {"literal": "yarn"}, "beta_fast": {"literal": 32.0}}}}}}',
    );
    expect(factAt(answer.facts, 'rope').problems).toEqual([]);
    expect(formatSemanticProblems([...factAt(answer.facts, 'rope.theta').problems])).toEqual([
      "[V3] argument 'rope.theta' = 'big' is not a number",
    ]);
  });
});

describe('checkInvariants: V8', () => {
  const MULTIPLE =
    `{"arguments": {"heads": ${HEADS}, "kv_heads": ${KV_HEADS}}, ` +
    '"invariants": [{"holds": {"compare": {"operator": "equal", ' +
    '"left": {"op": "modulo", "args": [{"argument": "heads"}, {"argument": "kv_heads"}]}, ' +
    '"right": {"literal": 0}}}, "description": "heads is a multiple of kv_heads"}]}';

  it('holds, and shows the values it read', () => {
    const answer = describe_(MULTIPLE, '{"heads": {"literal": 32}, "kv_heads": {"literal": 8}}');
    expect(answer.problems).toEqual([]);
    expect(answer.invariants).toEqual([
      {
        description: 'heads is a multiple of kv_heads',
        verdict: 'holds',
        reads: ['heads', 'kv_heads'],
        shown: 'heads = 32, kv_heads = 8',
      },
    ]);
  });

  it('refuses in the tools’ words, with the operands sorted', () => {
    const answer = describe_(MULTIPLE, '{"heads": {"literal": 32}, "kv_heads": {"literal": 3}}');
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V8] 'heads is a multiple of kv_heads' does not hold (heads = 32, kv_heads = 3)",
    ]);
    expect(answer.invariants[0]?.verdict).toBe('fails');
  });

  it('is skipped when it reads an argument V3 refused', () => {
    const answer = describe_(MULTIPLE, '{"heads": {"literal": 32}, "kv_heads": {"literal": 0.5}}');
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V3] argument 'kv_heads' = 0.5 is not a cardinality (non-negative integer)",
    ]);
    expect(answer.invariants[0]?.verdict).toBe('skipped');
    expect(answer.invariants[0]?.shown).toBe('heads = 32');
  });

  it('writes no parenthesis when the condition reads nothing', () => {
    const answer = describe_(
      '{"arguments": {}, "invariants": [{"holds": {"boolean": false}, ' +
        '"description": "nothing ever holds"}]}',
      '{}',
    );
    expect(formatSemanticProblems([...answer.problems])).toEqual([
      "[V8] 'nothing ever holds' does not hold",
    ]);
  });

  it('leaves a path that resolves to nothing out of the values it shows', () => {
    const guarded =
      '{"arguments": {"window": {"type": {"kind": "cardinality"}, "required": false, ' +
      '"structural": true}}, "invariants": [{"holds": {"any": [{"not": {"present": "window"}}, ' +
      '{"compare": {"operator": "greater", "left": {"argument": "window"}, ' +
      '"right": {"literal": 0}}}]}, "description": "a window spans at least one position"}]}';
    expect(describe_(guarded, '{}').invariants).toEqual([
      {
        description: 'a window spans at least one position',
        verdict: 'holds',
        reads: ['window'],
        shown: '',
      },
    ]);
    expect(linesOf(guarded, '{"window": {"literal": 0}}')).toEqual([
      "[V8] 'a window spans at least one position' does not hold (window = 0)",
    ]);
  });

  it('is checked on the arguments the resolution answered, not on the map as written', () => {
    // `checkInvariants` takes the resolved map: the defaults are in it, which is what V8 reads.
    const resolution = resolveArguments(
      value(MULTIPLE),
      value('{"heads": {"literal": 32}}'),
      evaluator(),
    );
    expect(resolution.values['kv_heads']).toBe(32n);
    expect(checkInvariants(value(MULTIPLE), resolution.values).verdicts[0]?.shown).toBe(
      'heads = 32, kv_heads = 32',
    );
  });
});

describe('what a problem points at', () => {
  it('names the member the document writes, and the map when it writes none', () => {
    const answer = describeArguments(
      value(ATTENTION),
      value('{"kv_heads": {"literal": 0}, "kernel_hint": {"literal": 1}}'),
      evaluator(),
      ['instances', 'attn'],
    );
    expect(answer.problems.map((one) => `${one.code} ${one.path}`)).toEqual([
      'V2 /instances/attn/arguments/kernel_hint',
      'V2 /instances/attn/arguments',
      'V3 /instances/attn/arguments/kv_heads',
    ]);
  });

  it('names a field of a record by the path the document writes it at', () => {
    const answer = describeArguments(
      value(
        '{"arguments": {"rope": {"type": {"kind": "record", "fields": {' +
          '"theta": {"type": {"kind": "real"}, "required": true, "structural": false}}}, ' +
          '"required": true, "structural": true}}}',
      ),
      value('{"rope": {"record": {"theta": {"literal": "big"}}}}'),
      evaluator(),
      ['instances', 'attn'],
    );
    expect(answer.problems.map((one) => one.path)).toEqual([
      '/instances/attn/arguments/rope/record/theta',
    ]);
  });
});

describe('names the reading admits', () => {
  it('carries an argument called `__proto__` like any other', () => {
    const declaration =
      '{"arguments": {"__proto__": {"type": {"kind": "cardinality"}, "required": true, ' +
      '"structural": true}, ' +
      '"twice": {"type": {"kind": "cardinality"}, "required": false, "structural": false, ' +
      '"default": {"op": "multiply", "args": [{"argument": "__proto__"}, {"literal": 2}]}}}}';
    const answer = describe_(declaration, '{"__proto__": {"literal": 21}}');
    expect(answer.problems).toEqual([]);
    expect(Object.keys(answer.values)).toEqual(['__proto__', 'twice']);
    expect(answer.values['twice']).toBe(42n);
    expect(Object.getPrototypeOf(answer.values)).toBe(Object.prototype);
  });
});

describe('where the walk raises, the tools raise', () => {
  it('raises on a declaration the grammar would have refused', () => {
    expect(() => describe_('{"invariants": []}', '{}')).toThrow(PyKeyError);
    // `required` is read for an argument the document leaves out, `type` for one it supplies.
    expect(() =>
      describe_('{"arguments": {"heads": {"structural": true, "type": {"kind": "real"}}}}', '{}'),
    ).toThrow(PyKeyError);
    expect(() =>
      describe_(
        '{"arguments": {"heads": {"structural": true, "required": true}}}',
        '{"heads": {"literal": 1}}',
      ),
    ).toThrow(PyKeyError);
  });

  it('raises where a domain compares what Python cannot order', () => {
    expect(() =>
      describe_(
        '{"arguments": {"mask": {"type": {"kind": "enum", "values": ["causal"]}, ' +
          '"required": true, "structural": true, "domain": {"kind": "interval", ' +
          '"lower": {"value": {"literal": 1}, "inclusive": true}}}}}',
        '{"mask": {"literal": "causal"}}',
      ),
    ).toThrow(PyTypeError);
  });
});

describe('the reference base’s own declarations', () => {
  const schemas = repositorySchemas();
  const library = loadLibrary([REFERENCE_BASE], { schemas, source: nodeSource() });
  const model = loadModel(readFileSync(repositoryPath('data', 'models', 'llama3-8b.json'), 'utf8'));
  const quantities = resolveQuantities(model);

  /** The site `llama3-8b` writes at `decoder/attn`, against `attention.dense@1.0.0`. */
  function attention() {
    const composition = (model['compositions'] as PyRecord)['decoder'] as PyRecord;
    const site = (composition['instances'] as PyRecord)['attn'] as PyRecord;
    const definition = primitiveOf(library, {
      name: (site['primitive'] as PyRecord)['name'] as string,
      version: (site['primitive'] as PyRecord)['version'] as string,
    });
    expect(definition, 'attention.dense@1.0.0 is in the reference base').toBeDefined();
    return describeArguments(
      definition as PyValue,
      site['arguments'] as PyValue,
      evaluator(quantities, new Map([['layer', 0n]])),
    );
  }

  it('loads with no refusal', () => {
    expect(library.problems.map((one) => one.message)).toEqual([]);
  });

  it('resolves `llama3-8b`’s attention with `kv_heads` given and 14 arguments in all', () => {
    const answer = attention();
    expect(answer.problems).toEqual([]);
    expect(answer.values['heads']).toBe(32n);
    expect(answer.values['kv_heads']).toBe(8n);
    expect(answer.values['mask']).toBe('causal');
    // Five given, nine defaulted: what the sheet prints as `cross = false (default)`.
    const sources = answer.facts.filter((one) => one.source !== 'absent');
    expect(sources.filter((one) => one.source === 'given').map((one) => one.path)).toEqual([
      'width',
      'heads',
      'head_dim',
      'kv_heads',
      'mask',
      'rope',
      'rope.theta',
      'rope.layout',
    ]);
    expect(sources.filter((one) => one.source === 'default').map((one) => one.path)).toEqual([
      'cross',
      'streaming',
      'kv_source',
      'q_bias',
      'k_bias',
      'v_bias',
      'out_bias',
      'output_gate',
    ]);
  });

  it('says which arguments do not apply to a causal attention', () => {
    const answer = attention();
    expect(answer.facts.filter((one) => !one.applicable).map((one) => one.path)).toEqual([
      'chunk',
    ]);
    // `window` applies under a causal mask and is simply not supplied here.
    expect(factAt(answer.facts, 'window').applicable).toBe(true);
    expect(factAt(answer.facts, 'window').source).toBe('absent');
  });

  it('answers every invariant `attention.dense` declares, with the values it read', () => {
    const answer = attention();
    expect(answer.invariants.map((one) => one.verdict)).toEqual([
      'holds',
      'holds',
      'holds',
      'holds',
    ]);
    expect(answer.invariants[0]).toMatchObject({
      description: 'heads is a multiple of kv_heads',
      shown: 'heads = 32, kv_heads = 8',
    });
    // The yarn bounds read paths that resolve to nothing here: they are left out of the text.
    expect(answer.invariants[1]?.shown).toBe('');
  });

  it('lists every problem of the description on exactly one row', () => {
    const answer = attention();
    const rows: SemanticProblem[] = answer.facts.flatMap((one) => [...one.problems]);
    const invariants = answer.invariants.flatMap((one) =>
      one.problem === undefined ? [] : [one.problem],
    );
    expect([...rows, ...invariants]).toEqual([...answer.problems]);
  });
});

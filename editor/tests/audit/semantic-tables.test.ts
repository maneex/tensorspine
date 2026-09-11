import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { COMPARISONS, OPERATORS, toPython, type PyValue } from '../../packages/lang/src/expr/index.js';
import {
  BYTES,
  d4,
  expandedKeyOf,
  widthOf,
  type ExpandedGraph,
  type ExpandedSite,
} from '../../packages/lang/src/derive/index.js';
import { rootSite } from '../../packages/lang/src/validate/index.js';
import type { PyRecord } from '../../packages/lang/src/expr/index.js';
import { loadSchemas, type Vocabulary } from '../../packages/lang/src/schema/index.js';
import { parse } from '../../packages/lang/src/json/index.js';
import type { Library } from '../../packages/lang/src/library/index.js';
import {
  analyseGraph,
  checkArgumentDomain,
  checkType,
  evaluateLocation,
  formatSemanticProblems,
  storageShape,
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
// The fifth is the transform relations of §5.3, in `packages/lang/src/validate/graph.ts`: the V5
// block reads `merge` by name and lumps `align` and `insert`, so the audit states what §5.3 says
// each relation does to the output's domain and asks the module, relation by relation, over a
// synthetic primitive with two streams. A relation the grammar gained would have no stated
// reading and fails the set equality; one whose reading changed fails the behaviour.
//
// The fourth is the *domain*'s, beside it: `domain['kind'] == 'set'` and everything else read as
// an interval — the tools' own two-branch reading, which has no fall-through at all, so a third
// shape would be silently read as an interval. The audit asks each shape the grammar declares
// which branch decides it, and requires the shapes and the branches to be the same two.
//
// The seventh is the dtype **width** table of `packages/lang/src/derive/figures.ts`: the schema
// declares the sixteen dtype names and says nothing about what one costs, so `BYTES` is a table
// keyed by a vocabulary item and admitted for exactly the reason §1 admits one. Its key set must
// equal the schema's enumeration, and every width must be a positive finite number.
//
// The eighth, ninth and tenth are D4's three state vocabularies of §4.3, in
// `packages/lang/src/derive/d4.ts`: the **evolution**, which the product reads by name twice — the
// visit rule of §7 it takes, and which of the three byte totals it feeds — and the **access
// geometry** and **sharing granularity**, which it reads not at all and carries to the product as
// the rule declares them. All three are stated here and *asked* of `d4`, value by value, over a
// one-state expanded graph: a value the grammar gained would have no stated reading and fails the
// set equality, one whose reading changed fails the behaviour, and a `d4` that began to branch on
// an access geometry or a sharing granularity fails the requirement that changing one of them
// changes exactly one member of the entry.
//
// The sixth is the *location forms* of §3.4, in `packages/lang/src/validate/bindings/locations.ts`:
// `evaluate_location` is four `if`s over the form's own key, ending in "unknown location form", so
// a form the grammar gained would be refused as unknown rather than evaluated. The audit reads the
// forms off the `location` union's discriminating keys and asks the module which branch decides
// each one; a form with no branch, or a branch with no form, fails the set equality.

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
function union(pointer: string): {
  alternatives: readonly { target: string | null; discriminating: readonly string[] }[];
} {
  const found = vocabulary.unionAt(pointer);
  expect(found, `${pointer} is not a union of the loaded schemas`).toBeDefined();
  return found as {
    alternatives: readonly { target: string | null; discriminating: readonly string[] }[];
  };
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

describe('the transform relations of packages/lang/src/validate/graph', () => {
  const RELATION = `${UNIT}#/$defs/domain_transform/properties/relation`;

  // What §5.3 states each relation does to the output's indexing domain, name by name:
  // "`merge` — the output carries the input port's stream at the output's kind"; "`align` — the
  // input port carries another domain and the output stays in the instance's domain"; "`insert`
  // — the input port's elements enter the instance's stream, which the output keeps". The V5
  // block reads `merge` by name and lumps the rest, so a relation the grammar gained would be
  // read as an `align` without anyone saying so. This is the set equality that stops it.
  const STATED: Record<string, 'from_port' | 'own'> = {
    merge: 'from_port',
    align: 'own',
    insert: 'own',
  };

  /** The relations the grammar declares, read from the schema and never typed here. */
  function relations(): string[] {
    const found = vocabulary.enumAt(RELATION);
    expect(found, `${RELATION} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly PyValue[] }).values
      .map((one) => (typeof one === 'string' ? one : JSON.stringify(one)))
      .sort();
  }

  /**
   * A primitive with two inputs and one transformed output.
   *
   * `a` and `b` inherit their domains from the edges into them, so each carries the stream of the
   * public input feeding it; `out` declares a kind, and a transform relates it to `b`. Nothing
   * else is declared: the reading under test is the domain's, and a shape or a slot would only
   * bring another rule into the answer.
   */
  function primitive(relation: string): PyValue {
    const inherit = { domain: { kind: 'inherit', from: { self: true } }, role: 'activation.hidden' };
    return toPython(
      parse(
        JSON.stringify({
          version: '1.0.0',
          arguments: {},
          ports: {
            inputs: { a: inherit, b: inherit },
            outputs: {
              out: { domain: { kind: 'token', from: { self: true } }, role: 'activation.hidden' },
            },
          },
          parameters: {},
          constants: {},
          state_ports: {},
          domain_transforms: [
            { from_port: 'b', to_port: 'out', relation, factor: { literal: 1 } },
          ],
          effects: { reads: [], writes: [] },
          partition_options: [],
        }),
      ),
    );
  }

  /** A primitive library holding that one primitive, as `loadLibrary` would answer it. */
  function libraryWith(definition: PyValue): Library {
    return {
      bases: [],
      byId: new Map([
        [
          'audit.transform@1.0.0',
          { name: 'audit.transform', version: '1.0.0', definition, file: '<audit>', base: '<audit>' },
        ],
      ]),
      primitives: new Map([['audit.transform', definition]]),
      axes: new Map(),
      precision: new Map(),
      templates: new Map(),
      problems: [],
    };
  }

  /**
   * Whose stream the output carries: the transformed port's, or the instance's own.
   *
   * `a` is fed by the public input `first` and `b` by `second`, two streams; `a` is the only
   * untransformed input, so the instance's own domain is `first`'s.
   */
  function readingOf(relation: string): string {
    const endpoint = (port: string) => ({
      instance: { kind: 'root', instance: 'x' },
      port,
    });
    const document = toPython(
      parse(
        JSON.stringify({
          schema: 'tensorspine/2.0',
          model: 'audit_transform',
          primitive_libraries: [{ base: './' }],
          quantities: {},
          constants: {},
          instances: {
            x: {
              primitive: { name: 'audit.transform', version: '1.0.0' },
              arguments: {},
              families: ['audit'],
            },
          },
          compositions: {},
          bindings: { values: {}, parameters: {}, constants: {}, states: {} },
          interfaces: {
            inputs: {
              first: { to: [endpoint('a')], kind: 'token' },
              second: { to: [endpoint('b')], kind: 'token' },
            },
            outputs: { out: { from: endpoint('out'), generative: false } },
          },
        }),
      ),
    );
    const answer = analyseGraph(document, libraryWith(primitive(relation)));
    expect(formatSemanticProblems([...answer.problems]), relation).toEqual([]);
    const stream = answer.ports.outputs.get('out')?.stream;
    expect(stream, relation).toBeTypeOf('string');
    return stream === 'second' ? 'from_port' : 'own';
  }

  it('reads every relation the grammar declares as §5.3 states it', () => {
    // A relation the schema gained, or dropped, fails here rather than being read as an `align`.
    expect(relations()).toEqual(Object.keys(STATED).sort());
    for (const relation of relations()) {
      expect(readingOf(relation), relation).toBe(STATED[relation]);
    }
  });
});

describe('the dtype width table of packages/lang/src/derive', () => {
  const DTYPE = `${MODEL}#/$defs/dtype`;

  /** Every dtype the language declares, in the order the schema writes them. */
  function dtypes(): string[] {
    const found = vocabulary.enumAt(DTYPE);
    expect(found, `${DTYPE} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly unknown[] }).values.map(String);
  }

  it('gives every dtype the language declares a width, and names no other', () => {
    // §1 (d): the core "names a vocabulary item only to attach semantics to it", and a dtype's
    // width is that semantics — the schema says nothing about what a dtype costs. A dtype the
    // language gains without a width, or a width for a name the language has not, fails here.
    expect(Object.keys(BYTES).sort()).toEqual([...dtypes()].sort());
  });

  it('answers a positive width for each of them, and raises for a name it has not', () => {
    for (const dtype of dtypes()) {
      const width = widthOf(dtype);
      expect(Number(width), dtype).toBeGreaterThan(0);
      expect(Number.isFinite(Number(width)), dtype).toBe(true);
      // A whole width keeps Python's integer; a sub-byte one is a real, and that is what makes a
      // `fp4` tensor's `bytes` a float in the derived document (D3).
      expect(typeof width === 'bigint' || Number(width) < 1, dtype).toBe(true);
    }
    expect(() => widthOf('f8e3m4')).toThrowError("'f8e3m4'");
  });
});

describe('the location forms of packages/lang/src/validate/bindings', () => {
  const LOCATION = `${MODEL}#/$defs/location`;

  /** The forms the grammar declares, from the `oneOf`'s own discriminating keys. */
  function forms(): string[] {
    return union(LOCATION).alternatives
      .map((alternative) => {
        expect(alternative.discriminating, 'a location alternative discriminates on one key')
          .toHaveLength(1);
        return alternative.discriminating[0] as string;
      })
      .sort();
  }

  /** A slot of one axis, `feature`, which every form below addresses. */
  const SLOT = toPython(
    parse(
      '{"role": "norm.scale", "sharing": {"kind": "exclusive"}, "shape": {"axes": [' +
        '{"name": "feature", "axis": "model.width", "nature": "feature", ' +
        '"extent": {"literal": 2}}]}}',
    ),
  );

  /** One location of each form, written so that it evaluates rather than refusing. */
  const WRITTEN: Record<string, string> = {
    tensor: '{"tensor": ["a"]}',
    stack: '{"stack": {"axis": "feature", "part": {"tensor": ["a.", {"coordinate": "feature"}]}}}',
    concat: '{"concat": {"axis": "feature", "parts": [{"tensor": ["a"]}, {"tensor": ["b"]}]}}',
    slice: '{"slice": {"tensor": ["a"], "axis": "feature", "offset": {"literal": 0}}}',
  };

  /** Which branch of `evaluate_location` decided a form: the key of its answer, or `unknown`. */
  function branchOf(form: string): string {
    const answer = evaluateLocation(
      toPython(parse(WRITTEN[form] as string)),
      new Map(),
      storageShape(SLOT),
      {},
      (expression) => (expression as { literal?: PyValue }).literal ?? null,
    );
    if (answer.evaluated === null) {
      expect(answer.problems.join(' '), form).toContain('unknown location form');
      return 'unknown';
    }
    expect(answer.problems, form).toEqual([]);
    return Object.keys(answer.evaluated)[0] as string;
  }

  it('evaluates every form the grammar declares by its own branch, and has no fall-through', () => {
    const declared = forms();
    expect(declared.length).toBeGreaterThan(0);
    // A form the schema gained would have no entry here, and one the module lost would answer
    // `unknown`: the set equality catches both directions.
    expect(Object.keys(WRITTEN).sort()).toEqual(declared);
    for (const form of declared) expect(branchOf(form), form).toBe(form);
  });
});

describe('the state vocabularies of packages/lang/src/derive/d4', () => {
  const RULE = `${UNIT}#/$defs/evolution_rule`;

  /** Every value the grammar admits at one member of an evolution rule. */
  function declared(member: string): string[] {
    const pointer = `${RULE}/properties/${member}`;
    const found = vocabulary.enumAt(pointer);
    expect(found, `${pointer} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly unknown[] }).values.map(String).sort();
  }

  /** A precision role the payload below cites, so that the library answers a dtype and a width. */
  const PRECISION = toPython(
    parse('{"admissible": ["bf16"], "default": "bf16", "sensitivity": "quantizable"}'),
  );

  /** A library holding that one role and nothing else: D4 reads the precision table alone. */
  const library: Library = {
    bases: [],
    byId: new Map(),
    primitives: new Map(),
    axes: new Map(),
    precision: new Map([['audit.state', PRECISION]]),
    templates: new Map(),
    problems: [],
  };

  /** `{prefix: '', key: ('root', 'n')}`: the one node of the graph below. */
  const SITE: ExpandedSite = { prefix: '', key: rootSite('n') };

  /**
   * A one-state expanded graph whose rule declares those three values.
   *
   * The payload is two elements of a `bf16` role, so `bytes_per_cached_position` is 4 and a
   * `window`'s span of 2 bounds it at 8: each of the three byte totals is non-zero for exactly the
   * evolution that feeds it, which is what {@link totalOf} reads back.
   */
  function graphWith(evolution: string, access: string, sharing: string): ExpandedGraph {
    const rule =
      `{"when": {"boolean": true}, "evolution": "${evolution}", "access": "${access}", ` +
      `"sharing": "${sharing}", "indexed_by": {"self": true}, "span": {"literal": 2}}`;
    const definition = toPython(
      parse(
        `{"state_ports": {"s": {"present_when": {"boolean": true}, "payload": {"p": {` +
          `"role": "audit.state", "shape": {"axes": [{"name": "a", "axis": "audit.axis", ` +
          `"nature": "feature", "extent": {"literal": 2}}]}}}, "key_axes": ["instance.session"], ` +
          `"operations": {"r": {"effect": "read"}}, "rules": [${rule}]}}}`,
      ),
    );
    return {
      model: toPython(parse('{"interfaces": {"inputs": {}, "outputs": {}}}')) as PyRecord,
      quantities: new Map(),
      resolved: new Map([
        [
          expandedKeyOf(SITE),
          { site: SITE, primitive: 'audit.state', definition, args: {} },
        ],
      ]),
      edges: [],
      domains: new Map(),
      own: new Map(),
      order: [SITE],
      meta: new Map(),
      compositions: [],
      inputsAt: new Map(),
      outputsAt: new Map(),
      tensorInstances: [],
      stateInstances: [
        {
          identity: 's',
          rule: 's',
          members: [{ site: SITE, name: 's' }],
          dtype: null,
          indices: [],
          writer: { site: SITE.key, name: 's' },
        },
      ],
    };
  }

  /** The one entry and the totals D4 answers for such a graph. */
  function inventory(evolution: string, access = 'ring', sharing = 'within_span'): {
    state: PyRecord;
    totals: PyRecord;
  } {
    const answer = d4(graphWith(evolution, access, sharing), library);
    const states = answer['states'] as readonly PyRecord[];
    expect(states, evolution).toHaveLength(1);
    return { state: states[0] as PyRecord, totals: answer['totals'] as PyRecord };
  }

  describe('the evolution table', () => {
    // What D4 does with an evolution, name by name: §7's visit rule — "an `append` or `window`
    // state indexed by its own stream is written once per new element … a `fixed` state is read
    // and written once per element" — and the total of the guide's §6 that it feeds.
    const STATED: Record<string, { visits: string; total: string }> = {
      append: { visits: 'per new element', total: 'append_bytes_per_cached_position' },
      window: { visits: 'per new element', total: 'bounded_bytes' },
      fixed: { visits: 'per element', total: 'fixed_bytes' },
    };

    /** Which of the three byte totals a state of that evolution fed, or `none`. */
    function totalOf(evolution: string): string {
      const { totals } = inventory(evolution);
      const fed = ['append_bytes_per_cached_position', 'bounded_bytes', 'fixed_bytes'].filter(
        (name) => Number(totals[name] as bigint | number) !== 0,
      );
      expect(fed.length, evolution).toBeLessThan(2);
      return fed[0] ?? 'none';
    }

    /** Which visit rule it took: `_visits`' own two wordings for a self-indexed state. */
    function visitsOf(evolution: string): string {
      const visits = inventory(evolution).state['visits'] as PyRecord;
      return visits['write'] === 'once per element' ? 'per element' : 'per new element';
    }

    it('gives every evolution the grammar declares a visit rule and a total, and names no other', () => {
      expect(declared('evolution')).toEqual(Object.keys(STATED).sort());
      for (const evolution of declared('evolution')) {
        expect(visitsOf(evolution), evolution).toBe(STATED[evolution]?.visits);
        expect(totalOf(evolution), evolution).toBe(STATED[evolution]?.total);
      }
    });
  });

  describe('the access and sharing tables', () => {
    // What D4 does with an access geometry (O5.3) and a sharing granularity (§4.3): it carries
    // the rule's own value to the product and reads neither by name — "as consumed properties
    // rather than runtime data-structure names". The reading is proved by asking for both halves:
    // the value arrives at its own member, and *nothing else of the entry moves*. A `d4` that
    // began to branch on one of them — a visit rule chosen by the access, a total chosen by the
    // granularity — fails the second half.
    const ACCESS: Record<string, 'carried'> = {
      logical_position: 'carried',
      ring: 'carried',
      aggregate: 'carried',
      selected: 'carried',
    };
    const SHARING: Record<string, 'carried'> = {
      by_position: 'carried',
      by_source: 'carried',
      within_span: 'carried',
      at_fork_point: 'carried',
    };

    /** Every member of an entry but that one, written down: what must not move. */
    function rest(state: PyRecord, member: string): string {
      const shown = (value: unknown): string =>
        JSON.stringify(value, (_name, held: unknown) =>
          typeof held === 'bigint' ? `${held}n` : held,
        ) ?? 'undefined';
      return Object.entries(state)
        .filter(([name]) => name !== member)
        .map(([name, value]) => `${name}=${shown(value)}`)
        .join(' | ');
    }

    it('carries every access geometry the grammar declares, and reads none of them', () => {
      expect(declared('access')).toEqual(Object.keys(ACCESS).sort());
      const reference = inventory('window', 'ring').state;
      for (const access of declared('access')) {
        const { state } = inventory('window', access);
        expect(state['access'], access).toBe(access);
        expect(rest(state, 'access'), access).toBe(rest(reference, 'access'));
      }
    });

    it('carries every sharing granularity the grammar declares, and reads none of them', () => {
      expect(declared('sharing')).toEqual(Object.keys(SHARING).sort());
      const reference = inventory('window', 'ring', 'within_span').state;
      for (const sharing of declared('sharing')) {
        const { state } = inventory('window', 'ring', sharing);
        expect(state['sharing'], sharing).toBe(sharing);
        expect(rest(state, 'sharing'), sharing).toBe(rest(reference, 'sharing'));
      }
    });
  });
});

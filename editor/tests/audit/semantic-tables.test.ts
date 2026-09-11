import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { COMPARISONS, OPERATORS, toPython, type PyValue } from '../../packages/lang/src/expr/index.js';
import { dtypeOf, SAFETENSORS_DTYPES } from '../../packages/lang/src/artifact/index.js';
import {
  BYTES,
  d2,
  d4,
  d5,
  d6,
  expandAnalysis,
  expandedKeyOf,
  OPERATION_COUNTERS,
  propagate,
  propagationOf,
  PROPAGATIONS,
  STATUSES,
  sumStatus,
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
  type GraphAnalysis,
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
// The fifth is the transform relations of §5.3, read twice by two modules and therefore stated
// twice. `packages/lang/src/validate/graph.ts`'s V5 block reads `merge` by name and lumps `align`
// and `insert`, so the audit states what §5.3 says each relation does to the output's **domain**;
// `packages/lang/src/derive/d2.ts` reads `merge` and `insert` by name and lumps `align`, so it
// states what each does to the output's **count**. Both are asked of the module, relation by
// relation, over one synthetic primitive with two streams. A relation the grammar gained would
// have no stated reading and fails the set equality; one whose reading changed fails the
// behaviour.
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
// The eleventh is the **qualified-value algebra** of §2.2, in
// `packages/lang/src/derive/qualified.ts`: the four statuses O0.5 declares and the row of §2.2's
// propagation table each operator of O0.1's set stands in. The statuses are asked behaviourally —
// `_status` is four membership tests over a set, so a status the grammar gained without a reading
// would fall through to `exact` rather than raise, and a status summed beside an exact value must
// survive as itself. The operators are held to the same walk the first table uses, and the four
// rows are required to stay four: a sum carries a bound through, an inverse flips the second
// operand's, `modulo` (which §2.2 gives no row) keeps only an exact result, and `negate` and
// `absolute` answer nothing at all.
//
// The twelfth is D5's **cost units** in `packages/lang/src/derive/d5.ts`: the four values of
// `cost_entry.per` and the counter of `--validate`'s first derivation each one reads. Both halves
// are asked — the figure a unit reads must be the counter that unit's name is paired with, and a
// correction counted per one unit must leave the other three exact — which is what catches a pair
// written the wrong way round.
//
// The thirteenth, fourteenth and fifteenth are D6's, in `packages/lang/src/derive/d6.ts`: the
// **communications** a partition admits and the **targets** it names, both carried to the product
// and read by no rule of the core — asked the way the eighth and ninth are, by requiring the value
// to arrive at its own member and nothing else of the row to move — and the one operator D6 *does*
// read, `multiply`, which is what makes an axis flattened for O5.10. That last is not a table but
// a single name out of the operator vocabulary, so it is asked of the module operator by operator:
// exactly one must report the loss, and an operator the grammar gained must not.
//
// The sixth is the *location forms* of §3.4, in `packages/lang/src/validate/bindings/locations.ts`:
// `evaluate_location` is four `if`s over the form's own key, ending in "unknown location form", so
// a form the grammar gained would be refused as unknown rather than evaluated. The audit reads the
// forms off the `location` union's discriminating keys and asks the module which branch decides
// each one; a form with no branch, or a branch with no form, fails the set equality.
//
// The sixteenth is the **safetensors dtype table** of `packages/lang/src/artifact/dtypes.ts`, and
// it is the one table of the core whose *keys* are not a schema's vocabulary — plan §1 states the
// exception in as many words: "The safetensors header format is the file format's; its dtype
// vocabulary is a table in the core, mapped onto the schema's `dtype` enum and audited against it
// like every other table." So three things are asked of it. Its **values** must all be dtypes the
// language declares. Every pair `tools/artifact.py`'s own `DTYPES` carries must be here with the
// same value, read out of the Python source, so that a change on either side is a failure rather
// than drift (feature 0.5 put that guard in the browser layer for the spike's copy; here it is the
// core's). And the dtypes of the enumeration that **no** key maps to must be exactly the set named
// below with its reason, so that a dtype the language gains reaches a decision instead of silently
// becoming unreachable from a checkpoint.

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

  /** And what each does to the output's count (§5.3), with the transform's factor at two. */
  const COUNTED: Record<string, PyRecord> = {
    merge: { second: 0.5 },
    align: { first: 1 },
    insert: { first: 1, second: 1 },
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
  function primitive(relation: string, factor: number): PyValue {
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
            { from_port: 'b', to_port: 'out', relation, factor: { literal: factor } },
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
      // The one role the ports declare, so that D2 can ask it for the dtype it defaults to. The
      // name and the width are the test's scaffolding, not a vocabulary the core carries.
      precision: new Map([
        ['activation.hidden', toPython(parse('{"default": "bf16", "sensitivity": "full"}'))],
      ]),
      templates: new Map(),
      problems: [],
    };
  }

  /** The one-instance document: `first` feeds the untransformed `a`, `second` the transformed `b`. */
  function documentOf(): PyValue {
    const endpoint = (port: string) => ({
      instance: { kind: 'root', instance: 'x' },
      port,
    });
    return toPython(
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
  }

  /** That document analysed under a library holding that one primitive, and found valid. */
  function analysisOf(relation: string, factor = 1): GraphAnalysis {
    const answer = analyseGraph(documentOf(), libraryWith(primitive(relation, factor)));
    expect(formatSemanticProblems([...answer.problems]), relation).toEqual([]);
    return answer;
  }

  /**
   * Whose stream the output carries: the transformed port's, or the instance's own.
   *
   * `a` is fed by the public input `first` and `b` by `second`, two streams; `a` is the only
   * untransformed input, so the instance's own domain is `first`'s.
   */
  function readingOf(relation: string): string {
    const stream = analysisOf(relation).ports.outputs.get('out')?.stream;
    expect(stream, relation).toBeTypeOf('string');
    return stream === 'second' ? 'from_port' : 'own';
  }

  /**
   * What the same relation does to the output's **count**, which is D2's own reading of the three
   * names (`packages/lang/src/derive/d2.ts`): a `merge` divides the transformed port's count by
   * the factor, an `insert` adds it to the instance's own, an `align` leaves the instance's own
   * alone. Asked of `d2` over the same document, with a factor of two so that the division shows.
   */
  function countReadingOf(relation: string): PyValue {
    const library = libraryWith(primitive(relation, 2));
    const values = d2(expandAnalysis(analysisOf(relation, 2)), library)[
      'values'
    ] as readonly PyValue[];
    const out = values.find((value) => (value as PyRecord)['value'] === 'x.out');
    expect(out, relation).toBeDefined();
    return (out as PyRecord)['count'] as PyValue;
  }

  it('reads every relation the grammar declares as §5.3 states it', () => {
    // A relation the schema gained, or dropped, fails here rather than being read as an `align`.
    expect(relations()).toEqual(Object.keys(STATED).sort());
    for (const relation of relations()) {
      expect(readingOf(relation), relation).toBe(STATED[relation]);
    }
  });

  it('counts every one of them as §5.3 states the counts', () => {
    // The second reading of the same three names, D2's: "transforms carry element counts — after
    // a `merge` a stream has one element per `factor`; after an `insert` it has the inserted
    // stream's elements in addition". A relation the grammar gained would take no branch in `d2`
    // and be counted as an `align`, which is what this stops.
    expect(relations()).toEqual(Object.keys(COUNTED).sort());
    for (const relation of relations()) {
      expect(countReadingOf(relation), relation).toEqual(COUNTED[relation]);
    }
    // The three readings are distinct, so a port that confused two of them fails above.
    expect(new Set(Object.values(COUNTED).map((count) => JSON.stringify(count))).size).toBe(3);
  });
});

const DTYPE = `${MODEL}#/$defs/dtype`;

/** Every dtype the language declares, in the order the schema writes them. */
function dtypes(): string[] {
  const found = vocabulary.enumAt(DTYPE);
  expect(found, `${DTYPE} is not an enumeration of the loaded schemas`).toBeDefined();
  return (found as { values: readonly unknown[] }).values.map(String);
}

describe('the dtype width table of packages/lang/src/derive', () => {
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

describe('the safetensors dtype table of packages/lang/src/artifact', () => {
  /**
   * The dtypes of the language the safetensors vocabulary has no name for — the stated reading
   * this audit holds the module to, as the transform relations' readings are stated above.
   *
   * `@huggingface/hub` declares the format's twenty-five names (feature 0.5 listed them), and
   * none of them is a four-bit integer; `f8e4m3fn` is a *third* thing beside `F8_E4M3` and
   * `F8_E4M3FNUZ`, and the tools map `F8_E4M3` to `f8e4m3`, so giving `f8e4m3fn` a key would
   * change an answer `tools/artifact.py` gives. A checkpoint therefore cannot carry a tensor of
   * one of these three, and a document that declares one locates nothing that matches — which is
   * a finding for the language, not a mapping for a port to invent.
   */
  const UNSPELLED = ['f8e4m3fn', 'i4', 'u4'];

  /** `tools/artifact.py`'s own `DTYPES`, read out of the Python source. */
  function toolsTable(): Record<string, string> {
    const source = readFileSync(join(repositoryRoot, 'tools', 'artifact.py'), 'utf8');
    const literal = /^DTYPES = (\{[^}]*\})/m.exec(source);
    expect(literal, 'tools/artifact.py no longer opens with a DTYPES dictionary').not.toBeNull();
    return JSON.parse((literal?.[1] ?? '{}').replaceAll("'", '"')) as Record<string, string>;
  }

  it('maps the file format’s names onto dtypes the language declares, and onto nothing else', () => {
    const declared = new Set(dtypes());
    for (const [name, dtype] of Object.entries(SAFETENSORS_DTYPES)) {
      expect(declared.has(dtype), `${name} → ${dtype}`).toBe(true);
    }
    // The mapping is one-to-one: two format names claiming one dtype would make the reader's
    // answer depend on which spelling a checkpoint happened to use.
    expect(new Set(Object.values(SAFETENSORS_DTYPES)).size).toBe(
      Object.keys(SAFETENSORS_DTYPES).length,
    );
  });

  it('carries every pair tools/artifact.py carries, with the same value', () => {
    const tools = toolsTable();
    expect(Object.keys(tools).length).toBeGreaterThan(0);
    for (const [name, dtype] of Object.entries(tools)) {
      expect(dtypeOf(name), name).toEqual({ dtype, known: true });
    }
  });

  it('adds only names whose lower-case form the tools’ fallback already answers', () => {
    // `read_headers` falls back on `name.lower()`. A key this table has and the tools' has not is
    // admissible exactly when the fallback already produced the same string — otherwise the port
    // would be answering something the tools do not, on an input they both can meet.
    const tools = toolsTable();
    const added = Object.keys(SAFETENSORS_DTYPES).filter((name) => !(name in tools));
    expect(added).toEqual(['FP4']);
    for (const name of added) {
      expect(name.toLowerCase(), name).toBe(SAFETENSORS_DTYPES[name]);
    }
  });

  it('leaves exactly the stated dtypes without a spelling', () => {
    const mapped = new Set(Object.values(SAFETENSORS_DTYPES));
    expect(dtypes().filter((dtype) => !mapped.has(dtype)).sort()).toEqual(UNSPELLED);
    // And the stated set is a set of dtypes, not of names nobody declares.
    for (const dtype of UNSPELLED) expect(dtypes(), dtype).toContain(dtype);
  });

  it('passes a name it has not through unchanged, and never lower-cases one', () => {
    // The decision of feature 1.9, asked of the module: a pass-through marked unknown. The
    // vocabulary below is the format's own, as `@huggingface/hub` declares it; every one of these
    // is a name `read_headers` would have answered `name.lower()` for.
    for (const name of [
      'C64',
      'E8M0',
      'F4',
      'F6_E2M3',
      'F6_E3M2',
      'F8_E4M3FNUZ',
      'F8_E5M2FNUZ',
      'F8_E8M0',
      'U16',
      'U32',
      'U64',
      'UE8',
    ]) {
      expect(name in SAFETENSORS_DTYPES, name).toBe(false);
      expect(dtypeOf(name), name).toEqual({ dtype: name, known: false });
    }
    // `known` says exactly whether the table decided the answer, on every input.
    for (const name of [...Object.keys(SAFETENSORS_DTYPES), 'bf16', 'nonsense', '']) {
      expect(dtypeOf(name).known, name).toBe(name in SAFETENSORS_DTYPES);
    }
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

describe('the qualified-value algebra of packages/lang/src/derive/qualified', () => {
  const STATUS = `${MODEL}#/$defs/epistemic_status`;

  /** Every status O0.5 declares, as the schema writes them. */
  function statuses(): string[] {
    const found = vocabulary.enumAt(STATUS);
    expect(found, `${STATUS} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly unknown[] }).values.map(String);
  }

  it('knows exactly the statuses the language declares', () => {
    expect([...STATUSES].sort()).toEqual([...statuses()].sort());
  });

  it('gives each of them a reading the sum row keeps apart from the others', () => {
    // `_status` is four membership tests over a set, so a status the grammar gained without a
    // reading would fall through to `exact` rather than raise. The audit asks behaviourally: a
    // status summed beside an exact value must survive as itself, which a name the tools do not
    // look for cannot do.
    for (const status of statuses()) {
      expect(sumStatus(['exact', status]), status).toBe(status);
    }
    expect(sumStatus(['exact', 'probably'])).toBe('exact');
  });

  it('gives every operator of the algebra a row of §2.2, and names no other', () => {
    // The same walk the operator table above uses: every alternative of the expression union that
    // carries an `op`, on both grammars. §2.2's table is keyed by those names and by nothing else.
    const implemented = Object.keys(PROPAGATIONS).sort();
    for (const pointer of [`${MODEL}#/$defs/scalar_expression`, `${UNIT}#/$defs/expression`]) {
      const declared = operatorsOf(pointer);
      expect(declared.length, pointer).toBeGreaterThan(0);
      expect(implemented, pointer).toEqual(declared);
    }
  });

  it('answers a stated row per operator, and refuses a name it has not', () => {
    const rows = new Set<string>();
    for (const operation of operatorsOf(`${MODEL}#/$defs/scalar_expression`)) {
      rows.add(propagationOf(operation));
    }
    // The four readings are distinct, so a table that collapsed two of them fails here: a sum
    // carries a bound through, an inverse flips the second operand's, an opaque one keeps only an
    // exact result, and a rejection answers nothing at all.
    expect([...rows].sort()).toEqual(['inverse', 'opaque', 'rejection', 'sum']);
    expect(propagate('add', ['exact', 'upper_bound'])).toBe('upper_bound');
    expect(propagate('subtract', ['exact', 'upper_bound'])).toBe('lower_bound');
    expect(propagate('modulo', ['exact', 'upper_bound'])).toBe('estimate');
    expect(() => propagate('negate', ['exact'])).toThrowError('a rejection (§2.2)');
    expect(() => propagationOf('logarithm')).toThrowError('unknown to the qualified-value algebra');
  });
});

describe('the cost units of packages/lang/src/derive/d5', () => {
  const PER = `${UNIT}#/$defs/cost_entry/properties/per`;

  /** Every unit a correction may be counted per (§4.1). */
  function units(): string[] {
    const found = vocabulary.enumAt(PER);
    expect(found, `${PER} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly unknown[] }).values.map(String);
  }

  /**
   * The counters the validator's first derivation keeps, each at a figure of its own.
   *
   * Keyed `ops_per_<unit>` from the **schema's** names, not from the table under audit: a table
   * that paired a unit with another unit's counter would otherwise read back its own mistake, the
   * figures having come from it. `validate.analyse` writes those four names and no others.
   */
  function counters(): Map<string, PyValue> {
    const stats = new Map<string, PyValue>();
    for (const [index, per] of units().entries()) {
      stats.set(`ops_per_${per}`, BigInt(index + 1) * 100n);
    }
    return stats;
  }

  /** A node whose primitive declares one correction counted per that unit, with that status. */
  function graphOf(per: string, status: string): ExpandedGraph {
    const definition = toPython(
      parse(
        `{"parameters": {}, "logical_cost": [{"expression": {"literal": 7}, ` +
          `"status": "${status}", "per": "${per}"}]}`,
      ),
    );
    const site: ExpandedSite = { prefix: '', key: rootSite('n') };
    return {
      model: toPython(parse('{"interfaces": {"inputs": {}, "outputs": {}}}')) as PyRecord,
      quantities: new Map(),
      resolved: new Map([
        [expandedKeyOf(site), { site, primitive: 'p', definition, args: {} }],
      ]),
      edges: [],
      domains: new Map(),
      own: new Map(),
      order: [site],
      meta: new Map(),
      compositions: [],
      inputsAt: new Map(),
      outputsAt: new Map(),
      tensorInstances: [],
      stateInstances: [],
    };
  }

  /** D5 over that node, with empty totals and no split. */
  function costs(per: string, status = 'upper_bound'): PyRecord {
    const totals = toPython(parse('{"totals": {"elements": 0, "bytes": 0}}'));
    const states = toPython(
      parse(
        '{"totals": {"append_bytes_per_cached_position": 0, "bounded_bytes": 0, "fixed_bytes": 0}}',
      ),
    );
    const values = toPython(parse('{"graph_splits": []}'));
    return d5(graphOf(per, status), totals, states, values, counters());
  }

  it('counts a correction per every unit the grammar declares, and per no other', () => {
    expect(OPERATION_COUNTERS.map(([per]) => per)).toEqual(units());
  });

  it('reads one counter per unit, and moves that unit’s status alone', () => {
    // The table's two halves are asked separately: the counter on the right must be the figure
    // `--validate` kept for that unit, and a correction counted per one unit must leave the other
    // three exact — which is what catches a pair written the wrong way round.
    for (const [index, per] of units().entries()) {
      const operations = costs(per)['operations'] as PyRecord;
      expect((operations[per] as PyRecord)['value'], per).toBe(BigInt(index + 1) * 100n);
      for (const other of units()) {
        expect((operations[other] as PyRecord)['status'], `${per}/${other}`).toBe(
          other === per ? 'upper_bound' : 'exact',
        );
      }
    }
  });
});

describe('the decomposition vocabularies of packages/lang/src/derive/d6', () => {
  const COMMUNICATION = `${UNIT}#/$defs/communication`;
  const TARGET = `${UNIT}#/$defs/partition_target`;

  /** Every logical communication a partition may admit (§7, O7.1). */
  function communications(): string[] {
    const found = vocabulary.enumAt(COMMUNICATION);
    expect(found, `${COMMUNICATION} is not an enumeration of the loaded schemas`).toBeDefined();
    return (found as { values: readonly unknown[] }).values.map(String);
  }

  /** Every form a partition's target takes, by the key that discriminates it. */
  function targets(): string[] {
    return union(TARGET)
      .alternatives.flatMap((alternative) => [...alternative.discriminating])
      .sort();
  }

  /** `{prefix: '', key: ('root', 'n')}`: the one node of the graphs below. */
  const SITE: ExpandedSite = { prefix: '', key: rootSite('n') };

  /** A one-node expanded graph whose primitive declares that definition. */
  function graphOf(definition: string): ExpandedGraph {
    return {
      model: toPython(parse('{"interfaces": {"inputs": {}, "outputs": {}}}')) as PyRecord,
      quantities: new Map(),
      resolved: new Map([
        [
          expandedKeyOf(SITE),
          { site: SITE, primitive: 'audit.partition', definition: toPython(parse(definition)), args: {} },
        ],
      ]),
      edges: [],
      domains: new Map(),
      own: new Map(),
      order: [SITE],
      // `_structural_graph_splits` reads `meta[key]` for every resolved node, unguarded.
      meta: new Map([[expandedKeyOf(SITE), { families: new Set<string>(), composition: null }]]),
      compositions: [],
      inputsAt: new Map(),
      outputsAt: new Map(),
      tensorInstances: [],
      stateInstances: [],
    };
  }

  /** The one partition row D6 answers for a primitive declaring that target and communication. */
  function partition(target: string, communication: string): PyRecord {
    const definition =
      `{"parameters": {}, "partition_options": [{"target": ${target}, ` +
      `"communication": ${communication}}]}`;
    const answer = d6(graphOf(definition), splits, states, []);
    const rows = answer['partition_options'] as readonly PyRecord[];
    expect(rows, `${target} / ${communication}`).toHaveLength(1);
    return rows[0] as PyRecord;
  }

  /** How many information-loss rows a slot whose one axis is written that way produces. */
  function losses(axis: Record<string, unknown>): number {
    const definition = JSON.stringify({
      parameters: { w: { shape: { axes: [{ name: 'f', axis: 'audit.axis', nature: 'feature', ...axis }] } } },
    });
    return (d6(graphOf(definition), splits, states, [])['information_loss'] as readonly PyValue[])
      .length;
  }

  /** Every member of a row but that one, written down: what must not move. */
  function rest(row: PyRecord, member: string): string {
    const shown = (value: unknown): string =>
      JSON.stringify(value, (_name, held: unknown) => (typeof held === 'bigint' ? `${held}n` : held)) ??
      'undefined';
    return Object.entries(row)
      .filter(([name]) => name !== member)
      .map(([name, value]) => `${name}=${shown(value)}`)
      .join(' | ');
  }

  const splits = toPython(parse('{"graph_splits": []}'));
  const states = toPython(parse('{"states": []}'));

  it('carries every communication the grammar declares, and reads none of them', () => {
    // What D6 does with a communication: it wraps a single one in a list — "always a list", so a
    // consumer reads one shape whether the primitive admits one pattern or several — and reads it
    // by name not at all. Proved by asking for both halves: the value arrives at its own member,
    // and *nothing else of the row moves*. A `d6` that began to branch on a communication — a
    // granularity chosen by the pattern, a target rewritten for `none` — fails the second half.
    const declared = communications();
    expect(declared.length).toBeGreaterThan(0);
    const reference = partition('{"any_axis": true}', `"${declared[0] as string}"`);
    for (const communication of declared) {
      const row = partition('{"any_axis": true}', `"${communication}"`);
      expect(row['communication'], communication).toEqual([communication]);
      expect(rest(row, 'communication'), communication).toBe(rest(reference, 'communication'));
    }
    // A declared list is carried as written, in its own order, and is not re-wrapped.
    const several = partition('{"any_axis": true}', JSON.stringify(declared));
    expect(several['communication']).toEqual(declared);
  });

  it('carries every partition target the grammar declares, and reads none of them', () => {
    const shapes: Record<string, string> = {
      argument_axis: '{"argument_axis": "audit.axis"}',
      instance_key_axis: '{"instance_key_axis": "instance.session"}',
      payload_axis: '{"payload_axis": {"state": "s", "component": "c", "axis": "audit.axis"}}',
      any_axis: '{"any_axis": true}',
      none: '{"none": true}',
    };
    expect(Object.keys(shapes).sort()).toEqual(targets());
    const reference = partition(shapes['any_axis'] as string, '"none"');
    for (const [name, written] of Object.entries(shapes)) {
      const row = partition(written, '"none"');
      expect(row['target'], name).toEqual(toPython(parse(written)));
      expect(rest(row, 'target'), name).toBe(rest(reference, 'target'));
    }
  });

  it('flattens on `multiply` alone, of every operator the grammar declares', () => {
    // The one place D6 names a member of the operator vocabulary: "a flattened shape declares its
    // decomposition" (O5.10), and a flattened axis is one whose extent is the *product* of the
    // axes it stands for. The reading is asked of the module, operator by operator, over one slot
    // whose axis carries an extent written with that operator: exactly `multiply` must report the
    // loss, and every other operator — an operator the grammar gained among them — must not.
    const declared = operatorsOf(`${UNIT}#/$defs/expression`);
    expect(declared).toContain('multiply');
    const product = { op: 'multiply', args: [{ literal: 2 }, { literal: 3 }] };
    for (const operator of declared) {
      const extent = { op: operator, args: [{ literal: 2 }, { literal: 3 }] };
      expect(losses({ extent }), operator).toBe(operator === 'multiply' ? 1 : 0);
    }
    // An extent that is no operation at all is no flattening either.
    expect(losses({ extent: { literal: 4096 } })).toBe(0);
    expect(losses({ extent: { argument: 'width' } })).toBe(0);
    // And a product that declares its factors is a decomposition, not a loss.
    expect(
      losses({ extent: product, factors: [{ axis: 'audit.a', extent: { literal: 6 } }] }),
    ).toBe(0);
  });
});

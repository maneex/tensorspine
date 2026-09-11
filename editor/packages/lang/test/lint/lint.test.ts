import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { toJsonValue, toPython, type PyValue } from '../../src/expr/index.js';
import { parse, serialize } from '../../src/json/index.js';
import {
  loadLibrary,
  memorySource,
  type Library,
  type LibrarySource,
} from '../../src/library/index.js';
import {
  baseName,
  calledPrimitives,
  distinctFindings,
  formatLintFinding,
  lint,
  lintReport,
  modelAdvisories,
  uncalledPrimitives,
  unreferencedVocabulary,
  type LintDocument,
  type LintFinding,
} from '../../src/lint/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { repositoryRoot } from '../json/repository.js';

// Lint (feature 1.10): the port of `tools/lint.py`.
//
// The parity suite (`test/parity/lint.test.ts`) compares whole `--lint` invocations with the
// tools', set by set. What is checked here is what the tools do not answer, and what neither the
// corpus nor an edit of a repository file can reach:
//
//   - the **pointer** each finding carries beside its words (plan §3): a curation question is
//     about a *file*, and the Problems panel wants a place to click;
//   - the reading `_called_primitives` takes when a document is off the grammar — a primitive
//     name that is not a string, and one Python cannot hash at all;
//   - the short circuit that keeps a template from being walked twice, which no repository
//     document can show (the one template of the repository is instantiated by one document);
//   - the rules **separately**, so that `lint`'s order is a statement rather than an accident;
//   - the one place the port's answer is *not* the tools', measured rather than reasoned, so that
//     the gap is visible and cannot go stale.

const schemas = repositorySchemas();
const repository = nodeSource(repositoryRoot);
const REFERENCE = ['data/primitive-library'];

/** The reference base alone, gathered once. */
const reference: Library = loadLibrary(REFERENCE, { schemas, source: repository });

/** The text of a repository file. */
function read(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}

/** A document of the repository as `--lint` is given one. */
function document(path: string): LintDocument {
  return { path, text: read(path) };
}

/** One document with a few values changed, serialised as `--lint` would read it from a file. */
function edited(path: string, edit: (value: Record<string, PyValue>) => void): LintDocument {
  const value = toPython(parse(read(path))) as Record<string, PyValue>;
  edit(value);
  return { path, text: serialize(toJsonValue(value)) };
}

/**
 * A base of the tests' own beside the reference base.
 *
 * Feature 1.6c's shape: the memory holds the scratch base's files, everything else comes from the
 * repository, so a unit written here resolves against the real axes and precision roles.
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
  return loadLibrary([...REFERENCE, 'scratch/base'], { schemas, source: combined });
}

/**
 * The reference base with its one template pin instrumented: how often the document is read.
 *
 * The tools open the template's file on every walk; the port reads the document the load already
 * pinned, so "how often it is read" is what states that the walk happens at all.
 */
function countingLibrary(): [Library, () => number] {
  let reads = 0;
  const counting: Library = {
    ...reference,
    templates: new Map(
      [...reference.templates].map(([key, pin]) => [
        key,
        {
          ...pin,
          get document(): PyValue {
            reads += 1;
            return pin.document;
          },
        },
      ]),
    ),
  };
  return [counting, () => reads];
}

/** One unit of the scratch base: a repository unit with a few values changed. */
function copiedUnit(source: string, edit: (unit: Record<string, unknown>) => void): string {
  const unit = JSON.parse(read(source)) as Record<string, unknown>;
  edit(unit);
  return JSON.stringify(unit);
}

describe('the base name a document-scoped finding is written under', () => {
  it('is the text after the last slash, as os.path.basename gives it', () => {
    expect(baseName('data/models/llama3-8b.json')).toBe('llama3-8b.json');
    expect(baseName('llama3-8b.json')).toBe('llama3-8b.json');
    // `os.path.basename('a/b/')` is `''`, and a finding about such a path would print nothing;
    // no caller produces one, and the reading is stated rather than guarded.
    expect(baseName('data/models/')).toBe('');
  });
});

/** A document that instantiates the one template primitive of the repository and nothing else. */
const TEMPLATE_INSTANCE = toPython(
  parse(
    '{"instances": {"text": {"primitive": ' +
      '{"name": "decoder.causal_yarn", "version": "1.0.0"}}}, "compositions": {}}',
  ),
);

describe('the primitives a document calls', () => {
  it('follows a template instance into the template it pins', () => {
    const called = calledPrimitives(
      toPython(parse(read('data/models/shieldstral-3b-composite.json'))),
      reference,
    );
    // The composite instantiates the one template primitive of the repository, and the
    // primitives the *template* calls are called through it — which is why the corpus lints
    // clean although no document names `attention.dense` beside the composite's template.
    expect(called.has('s"decoder.causal_yarn"')).toBe(true);
    expect(called.has('s"attention.dense"')).toBe(true);
  });

  it('reads a template document once, however many instances name it', () => {
    // Two instances of one template primitive: the name enters `seen` before the walk, so the
    // template's document is read once. No document of the repository has two.
    const composite = toPython(
      parse(read('data/models/shieldstral-3b-composite.json')),
    ) as Record<string, PyValue>;
    const instances = composite['instances'] as Record<string, PyValue>;
    const doubled = {
      ...composite,
      instances: { ...instances, text_again: instances['text'] as PyValue },
    } as unknown as PyValue;
    const [counting, reads] = countingLibrary();
    calledPrimitives(doubled, counting);
    expect(reads()).toBe(1);
  });

  it('does not walk a primitive the set it was given already carries', () => {
    // `seen.add(name)` runs *before* the recursion, and `seen` is shared down it. That is what
    // stops a template whose own document instantiates it from recurring for ever, and it is
    // what makes the closure the same however the names are ordered — the tools iterate a
    // Python set there, so a walk that depended on the order could not be reproduced at all.
    //
    // The document is two members long because that is what makes the recursion observable at
    // all: measured, `shieldstral-3b-composite` — the one document of the repository that
    // instantiates a template — calls every primitive the template calls itself, so following
    // the template adds nothing to its called set. Here the template's own four are the whole
    // difference. The document is off the grammar, and deliberately: this rule runs before
    // anything checks it.
    const [counting, reads] = countingLibrary();
    const whole = calledPrimitives(TEMPLATE_INSTANCE, counting);
    expect(reads()).toBe(1);
    expect([...whole].sort()).toEqual([
      's"attention.dense"',
      's"decoder.causal_yarn"',
      's"ffn.gated"',
      's"norm.rms"',
      's"residual.add"',
    ]);

    const [again, none] = countingLibrary();
    const seen = new Set(['s"decoder.causal_yarn"']);
    const withoutTemplate = calledPrimitives(TEMPLATE_INSTANCE, again, seen);
    expect(none()).toBe(0);
    expect(withoutTemplate).toBe(seen);
    expect([...withoutTemplate]).toEqual(['s"decoder.causal_yarn"']);
  });

  it('looks up a name that is not a string in no dictionary, and hashes one that cannot be', () => {
    // `--lint` runs this rule before anything checks the grammar, so a document off it reaches
    // here. A Python dictionary of string keys answers `None` for a non-string key; a key it
    // cannot hash raises, and the message is CPython's.
    const numeric = edited('data/models/llama3-8b.json', (value) => {
      const instances = value['instances'] as Record<string, PyValue>;
      const embed = instances['embed'] as Record<string, PyValue>;
      instances['embed'] = { ...embed, primitive: { name: 7n, version: '1.0.0' } };
    });
    const called = calledPrimitives(toPython(parse(numeric.text)), reference);
    expect(called.has('#7')).toBe(true);

    const unhashable = edited('data/models/llama3-8b.json', (value) => {
      const instances = value['instances'] as Record<string, PyValue>;
      const embed = instances['embed'] as Record<string, PyValue>;
      instances['embed'] = { ...embed, primitive: { name: {}, version: '1.0.0' } };
    });
    expect(() => calledPrimitives(toPython(parse(unhashable.text)), reference)).toThrow(
      "unhashable type: 'dict'",
    );
  });

  it('raises where the tools raise: the members it indexes are not checked first', () => {
    const without = edited('data/models/llama3-8b.json', (value) => {
      delete value['compositions'];
    });
    // The grammar requires `compositions`, and `model_advisories` would have reported the
    // document as off the schema — but `uncalled_primitives` runs first and indexes it.
    expect(() => uncalledPrimitives([without], reference)).toThrow("'compositions'");
  });
});

describe('the findings a curation question carries', () => {
  it('names the unit file of an uncalled primitive', () => {
    const findings = uncalledPrimitives([document('data/models/llama3-8b.json')], reference);
    const found = findings.find((one) => one.message.includes("'norm.layer'"));
    expect(found?.rule).toBe('uncalled');
    expect(found?.scope).toBe('primitive_library');
    expect(found?.file).toBe('data/primitive-library/primitives/norm/layer/1.0.0.json');
  });

  it('names the unit file of an axis and of a precision role no primitive cites', () => {
    const library = scratchLibrary({
      'scratch/base/axes/scratch/unused.json': copiedUnit(
        'data/primitive-library/axes/model/width.json',
        (unit) => {
          unit['name'] = 'scratch.unused';
        },
      ),
      'scratch/base/precision/scratch/unused.json': copiedUnit(
        'data/primitive-library/precision/norm/scale.json',
        (unit) => {
          unit['name'] = 'scratch.unused';
        },
      ),
    });
    expect(library.problems).toEqual([]);
    const findings = unreferencedVocabulary(library);
    expect(findings.map((one) => one.message)).toEqual([
      "axis 'scratch.unused' is cited by no primitive",
      "precision role 'scratch.unused' is cited by no primitive",
    ]);
    expect(findings.map((one) => one.file)).toEqual([
      'scratch/base/axes/scratch/unused.json',
      'scratch/base/precision/scratch/unused.json',
    ]);
    // The axis comes first because the tools walk the axes before the roles, and the two
    // messages differ although the names are the same: the two loops are not one.
    expect(findings.every((one) => one.rule === 'vocabulary')).toBe(true);
  });

  it('skips a storage axis, and reads the vocabulary against the highest version alone', () => {
    // A *citation* is a string equal to the name — `_strings` yields whole strings, never the
    // member names — so what makes `scratch.only` cited by version 1.0.0 is the shape's own
    // `axis`, and the same name inside a sentence would count for nothing.
    const definition = (axis: string, version: string): string =>
      copiedUnit('data/primitive-library/primitives/norm/rms/1.0.0.json', (unit) => {
        unit['name'] = 'scratch.demo';
        const inner = unit['definition'] as Record<string, unknown>;
        inner['version'] = version;
        const parameters = inner['parameters'] as Record<string, Record<string, never>>;
        const shape = (parameters['weight'] as unknown as { shape: { axes: { axis: string }[] } })
          .shape;
        (shape.axes[0] as { axis: string }).axis = axis;
      });
    const library = scratchLibrary({
      // A `storage` axis nothing cites at all: "cited by the derivation (D3's storage axis,
      // §3.4), never by a primitive shape", so it is not a finding.
      'scratch/base/axes/scratch/stored.json': copiedUnit(
        'data/primitive-library/axes/storage/multiplicity.json',
        (unit) => {
          unit['name'] = 'scratch.stored';
        },
      ),
      // An axis version 1.0.0 shapes a parameter on and version 1.1.0 does not.
      'scratch/base/axes/scratch/only.json': copiedUnit(
        'data/primitive-library/axes/model/width.json',
        (unit) => {
          unit['name'] = 'scratch.only';
        },
      ),
      'scratch/base/primitives/scratch/demo/1.0.0.json': definition('scratch.only', '1.0.0'),
      'scratch/base/primitives/scratch/demo/1.1.0.json': definition('model.width', '1.1.0'),
    });
    expect(library.problems).toEqual([]);
    expect(unreferencedVocabulary(library).map((one) => one.message)).toEqual([
      "axis 'scratch.only' is cited by no primitive",
    ]);
    // And the version an uncalled line names is the definition's own, not `1.0.0` — every
    // primitive of the reference base is at `1.0.0`, so nothing in the repository shows it.
    const uncalled = uncalledPrimitives([document('data/models/llama3-8b.json')], library);
    expect(uncalled.map((one) => one.message)).toContain(
      "primitive 'scratch.demo' 1.1.0 is in the primitive library, " +
        'called by none of the 1 model(s) linted',
    );
  });

  it('points an advisory and an off-schema report at the document they are about', () => {
    const off = edited('data/models/llama3-8b.json', (value) => {
      const quantities = value['quantities'] as Record<string, PyValue>;
      const width = { ...(quantities['d'] as Record<string, PyValue>) };
      delete width['type'];
      quantities['d'] = width;
    });
    const findings = modelAdvisories([off], { schemas, library: reference });
    expect(findings).toHaveLength(1);
    expect(findings[0]?.rule).toBe('off-schema');
    expect(findings[0]?.scope).toBe('llama3-8b.json');
    expect(findings[0]?.file).toBe('data/models/llama3-8b.json');
    expect(findings[0]?.message).toContain('off the schema, not analysed (--validate refuses it)');
  });

  it('passes over a document that still needs an assignment', () => {
    // `--lint` takes no `--assign`, so a template is never analysed: `missing_assignment` is
    // non-empty and the document is skipped in silence.
    const template = document('data/models/decoder-causal-yarn/1.0.0.json');
    expect(modelAdvisories([template], { schemas, library: reference })).toEqual([]);
  });
});

describe('the order the curation questions are listed in', () => {
  /** A definition small enough to stand in a base of two primitives and nothing else. */
  const DEFINITION = {
    version: '1.0.0',
    arguments: {},
    ports: { inputs: {}, outputs: {} },
    parameters: {},
    constants: {},
    state_ports: {},
    effects: { reads: [], writes: [] },
    partition_options: [{ target: { none: true }, communication: 'none' }],
  };

  it('is Python sorted: by code point, where UTF-16 would disagree', () => {
    // `sorted(cat['primitives'].items())` orders by code point, and JavaScript's own `<` orders
    // by UTF-16 code unit — the two disagree exactly when a surrogate pair meets a code point
    // above U+E000. Every name the *grammar* admits is ASCII (`qualified_name` is
    // `^[A-Za-z_][A-Za-z0-9_-]*(\.…)*$`), so no document and no exploded base can tell the two
    // apart; a **monolithic** base can, because its primitive names are map keys the loader
    // reads without a schema. Measured against `tools/lint.py` on the same base: `a\uFFFD`
    // first, `a\u{10000}` second.
    const base = JSON.stringify({
      schema: 'tensorspine-primitive-library-unit/2.0',
      primitives: { 'a\u{10000}': DEFINITION, 'a\uFFFD': DEFINITION },
    });
    const library = loadLibrary(['mono.json'], {
      schemas,
      source: memorySource({ 'mono.json': base }),
    });
    expect(library.problems).toEqual([]);
    expect(uncalledPrimitives([], library).map((one) => one.message)).toEqual([
      "primitive 'a\uFFFD' 1.0.0 is in the primitive library, " +
        'called by none of the 0 model(s) linted',
      "primitive 'a\u{10000}' 1.0.0 is in the primitive library, " +
        'called by none of the 0 model(s) linted',
    ]);
  });
});

describe('the report', () => {
  const finding = (scope: string, message: string, rule: LintFinding['rule']): LintFinding => ({
    rule,
    scope,
    message,
    file: `${scope}.file`,
  });

  it('keeps one finding per (scope, message), whatever else differs', () => {
    const findings = [
      finding('a.json', 'said once', 'advisory'),
      { ...finding('a.json', 'said once', 'off-schema'), file: 'elsewhere' },
      finding('b.json', 'said once', 'advisory'),
    ];
    // The tools deduplicate on the pair the line is printed from and on nothing else, so the
    // second row goes although it is a different rule about a different file.
    expect(distinctFindings(findings).map((one) => one.rule)).toEqual(['advisory', 'advisory']);
    expect(lintReport(findings)).toEqual([
      '  W  a.json: said once',
      '  W  b.json: said once',
      '  2 advisory finding(s) — nothing blocking',
    ]);
  });

  it('says nothing to report when there is none', () => {
    expect(lintReport([])).toEqual(['  nothing to report']);
  });

  it('writes a finding as the tools write one', () => {
    expect(formatLintFinding(finding('primitive_library', 'something', 'uncalled'))).toBe(
      '  W  primitive_library: something',
    );
  });

  it('runs the three rules in the tools order', () => {
    const findings = lint([document('data/models/llama3-8b.json')], {
      schemas,
      library: reference,
    });
    const rules = [...new Set(findings.map((one) => one.rule))];
    // `llama3-8b` alone leaves primitives uncalled and cites every axis and role, so the middle
    // rule contributes nothing; what the order states is that the uncalled lines come first.
    expect(rules).toEqual(['uncalled']);
    expect(findings.every((one) => one.scope === 'primitive_library')).toBe(true);
  });
});

describe('where the port does not answer what the tools answer', () => {
  it('reports an unsupported revision in the schema stage words, not read_json own', () => {
    // `validate.structural` runs three stages: the JSON layer (V12), then
    // `primitive_library.read_json` — which refuses the legacy field layout and an unsupported
    // `schema` revision — and only then the grammar. Feature 1.1 ported the first and the third;
    // the middle one is not in `structuralText`, and closing the hole needs a rendering
    // `StructuralProblem` has not (the tools print that line *bare*, where the duplicate's is
    // `[V12]`-prefixed) and a path `structuralText` is not given. So the two disagree here, on a
    // document no corpus or rejection file of the repository carries.
    //
    // Measured on this box with `tools/tensorspine --lint`, not reasoned:
    //
    //   W  legacy-revision.json: off the schema, not analysed (--validate refuses it):
    //   /tmp/…/legacy-revision.json: unsupported revision 'tensorspine/1.0'; convert supported
    //   legacy input with python3 tools/migrate.py INPUT -o OUTPUT (V12)
    //
    // The verdict is the same — the document is off the grammar and is not analysed — and the
    // line that says why is the schema's rather than the loader's.
    const legacy = edited('data/models/llama3-8b.json', (value) => {
      value['schema'] = 'tensorspine/1.0';
    });
    const findings = modelAdvisories([legacy], { schemas, library: reference });
    expect(findings.map((one) => one.message)).toEqual([
      "off the schema, not analysed (--validate refuses it): schema: 'tensorspine/2.0' was expected",
    ]);
  });
});

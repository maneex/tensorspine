import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  followAnchor,
  formatProblems,
  loadSchemas,
  parse,
  parseAnchor,
  type SchemaObject,
  type SchemaRegistry,
} from '../../packages/lang/src/index.js';
import { referenceTags, SchemaShapes } from '../../packages/store/src/index.js';
import { auditPresentation } from '../../packages/ui/src/presentation/audit.js';
import { presentation } from '../../packages/ui/src/presentation/load.js';
import { registry, repositoryRoot } from '../../packages/ui/test/presentation/source.js';
import { editorRoot, readEditorFile } from './tree.js';

// Catching rule (a) of the implementation plan's §1, in CI:
//
//   > A startup audit resolves every key of `presentation.json` against the loaded schemas; a key
//   > that resolves nowhere is an error in the log, and every enum value and `oneOf` member of the
//   > loaded schemas that no binding names is listed as "rendered generically".
//
// The startup half runs in the application (`startPresentation`) and is exercised by
// `packages/ui/test/presentation/`. This is the half that fails the build, and it asks for more
// than the startup one does: at startup an unresolved key is an error **in the log**, because a
// user with a workspace of its own schemas must still get an editor; here it is a refusal,
// because the schemas the build vendors are the repository's own and a binding that misses them
// is a mistake nobody should be able to commit.
//
// It audits against `schemas/` rather than against `apps/web/public/vendor/schemas/`. The vendor
// is a byte-for-byte copy of that directory — `vendor.test.ts` runs the script and proves it on
// every run — and it is gitignored, so a suite that read it would be green or absent depending on
// whether somebody had built lately. Auditing the source is auditing what is shipped.

const PRESENTATION = 'packages/ui/src/presentation.json';
const SCHEMA = 'schemas/tensorspine-editor-presentation.schema.json';
const NOTE = 'schemas/TENSORSPINE-EDITOR-PRESENTATION.md';
const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const DERIVED = 'https://tensorspine.dev/schema/2.1/derived.json';

const file = presentation();
const audit = auditPresentation(registry(), file);

describe('every binding names a place of the schemas the build vendors', () => {
  it('resolves, all of them, with nothing to report', () => {
    expect(audit.problems.map((one) => `${one.code} ${one.anchor}: ${one.message}`)).toEqual([]);
    expect(audit.resolved).toBe(audit.bindings);
    expect(audit.bindings).toBeGreaterThan(40);
  });

  it('lists what renders generically, which is most of the language and no error', () => {
    expect(audit.generic.length).toBeGreaterThan(audit.bindings);
  });

  it('names three of the schemas the vendor carries, and neither of the other two', () => {
    // The vendor copies `schemas/` whole (feature 0.2), and the registry indexes whatever `$id`s
    // it finds. The bindings reach the three the editor renders: the model grammar, the unit
    // vocabulary, the derived products. Not `documentation.json` — its fields are rendered by the
    // generic walker like any other, and a `description` is help text rather than presentation —
    // and not `fixture.json`, which is the witness format and no part of this plan.
    const named = new Set(file.anchors.map((anchor) => parseAnchor(anchor)?.schema));
    expect([...named].sort()).toEqual(
      [MODEL, 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json', DERIVED].sort(),
    );
    for (const id of named) expect(registry().byId(id ?? ''), id).toBeDefined();
  });
});

describe('the one data file of the interface', () => {
  it('is where plan §1 puts it, and the no-hard-coding audit exempts that path alone', () => {
    expect(existsSync(join(editorRoot, PRESENTATION))).toBe(true);
    expect(readEditorFile('tests/audit/no-hard-coding.test.ts')).toContain(PRESENTATION);
  });

  it('holds vocabulary of the schemas, which is what the exemption is for', () => {
    // The point of the exemption is that the vocabulary lives *here*: a file that named none
    // would mean the symbols had gone somewhere else, which is the thing (b) forbids.
    const text = readEditorFile(PRESENTATION);
    for (const value of ['floor_divide', 'greater_or_equal', 'not_equal']) {
      expect(text, value).toContain(`"${value}"`);
    }
  });

  it('is JSON the core accepts: no duplicate member name, one binding per anchor', () => {
    // `parse` is the lexeme-preserving parser of feature 0.3, which refuses a duplicate member
    // name (V12). A second binding for one anchor would otherwise be dropped by `JSON.parse` in
    // silence, and half the file would be a binding nobody applies.
    expect(() => parse(readEditorFile(PRESENTATION))).not.toThrow();
    expect(new Set(file.anchors).size).toBe(file.anchors.length);
  });

  it('ends with a newline and is written as the repository writes JSON', () => {
    const text = readEditorFile(PRESENTATION);
    expect(text.endsWith('}\n')).toBe(true);
    expect(text).not.toMatch(/\t/);
  });
});

describe('the format has a schema, and the file is on it', () => {
  const schemaRegistry: SchemaRegistry = loadSchemas(
    [{ path: SCHEMA, text: readEditorFile(SCHEMA) }],
    { origin: 'editor/schemas' },
  );

  it('ships with its companion note, as every schema of this project does', () => {
    expect(existsSync(join(editorRoot, SCHEMA))).toBe(true);
    expect(existsSync(join(editorRoot, NOTE))).toBe(true);
    expect(readEditorFile(NOTE)).toContain(PRESENTATION);
  });

  it('conforms to it', () => {
    const tree = parse(readEditorFile(PRESENTATION));
    expect(formatProblems(schemaRegistry.structural(tree, 'presentation'))).toEqual([]);
    expect(schemaRegistry.conforms(tree, 'presentation')).toBe(true);
  });

  it('refuses a binding member the loader would refuse, so the two say the same thing', () => {
    const off = parse('{"a#": {"widgit": "expression"}}');
    expect(schemaRegistry.conforms(off, 'presentation')).toBe(false);
  });

  it('refuses a value the loader lets through, which is where the two differ on purpose', () => {
    // The loader admits an unknown `widget` — "unknown constructs get the generic widget" — and
    // the schema does not. That is the division of labour: a typo cannot reach a release, and a
    // rendering the running interface has never heard of still renders.
    const unknown = parse('{"a#": {"widget": "sankey"}}');
    expect(schemaRegistry.conforms(unknown, 'presentation')).toBe(false);
  });
});

describe('every figure of the derived products is bound', () => {
  /** The derived schema itself, or a failure naming it rather than a silent empty walk. */
  function derivedDocument(): SchemaObject {
    const found = registry().byId(DERIVED);
    if (found === undefined) throw new Error(`${DERIVED} is not in the registry`);
    return found.document;
  }

  /** Every property of the derived schema whose name says bytes, by its anchor. */
  function byteFigures(): string[] {
    const found: string[] = [];
    const walk = (node: unknown, pointer: string, inProperties: boolean): void => {
      if (Array.isArray(node)) {
        node.forEach((one, index) => walk(one, `${pointer}/${String(index)}`, false));
        return;
      }
      if (node === null || typeof node !== 'object') return;
      for (const [keyword, one] of Object.entries(node as Record<string, unknown>)) {
        const step = `${pointer}/${keyword.replace(/~/g, '~0').replace(/\//g, '~1')}`;
        if (inProperties && keyword.includes('bytes')) found.push(`${DERIVED}#${step}`);
        walk(one, step, keyword === 'properties');
      }
    };
    walk(derivedDocument(), '', false);
    return found;
  }

  /** The places a figure's binding may be written at: itself, what it refers to, that map's values. */
  function candidates(anchor: string): string[] {
    const followed = followAnchor(registry(), anchor);
    if (followed === null) return [anchor];
    return [anchor, followed.anchor, `${followed.anchor}/additionalProperties`];
  }

  const figures = byteFigures();

  it('finds the byte figures in the schema rather than listing them here', () => {
    expect(figures.length).toBeGreaterThanOrEqual(20);
  });

  it('binds every one of them, so no component converts a byte count of its own', () => {
    // The component inventory's §7: "No component adds, converts or rounds a byte count; `bytes`
    // is a presentation binding over the product's own number." A byte figure added to a derived
    // product fails here until it is bound, which is the only way that rule survives a schema
    // change.
    const unbound = figures.filter(
      (anchor) => !candidates(anchor).some((one) => file.at(one)?.format === 'bytes'),
    );
    expect(unbound).toEqual([]);
  });

  it('binds nothing as a size that the schema does not call one', () => {
    const sizes = file.anchors.filter((anchor) => file.at(anchor)?.format === 'bytes');
    const reached = new Set(figures.flatMap((anchor) => candidates(anchor)));
    expect(sizes.filter((anchor) => !reached.has(anchor))).toEqual([]);
  });
});

// Every map the model schema keys by a *name* — the store finds them by reading the schemas, not
// from a list — and what a name of it refers to. A map with no referent binding is not an
// oversight if the reason is written down; a map with one that nothing explains is.
interface NameKeyedMap {
  /** The member the map is written under, as the store's reference index reports it. */
  readonly tag: string;
  /** The anchors of the declarations it holds, where it holds declarations. */
  readonly declares: readonly string[];
  /** Why those, or why none. */
  readonly why: string;
}

const KEYED: readonly NameKeyedMap[] = [
  {
    tag: 'quantities',
    declares: [`${MODEL}#/properties/quantities`],
    why: 'the document’s quantities, named by `{"quantity": …}` anywhere in it',
  },
  {
    tag: 'constants',
    declares: [`${MODEL}#/properties/constants`],
    why: 'the document’s constants at the root, named by `{"constant": …}`. The other `constants` — `bindings.constants` — is keyed by rule names, which nothing refers to',
  },
  {
    tag: 'instances',
    declares: [
      `${MODEL}#/properties/instances`,
      `${MODEL}#/$defs/composition_definition/properties/instances`,
    ],
    why: 'two declarations under one tag, and the reason the anchor is the place: a root instance is named by a selector with no `composition` beside it, a site by one that names its composition',
  },
  {
    tag: 'compositions',
    declares: [`${MODEL}#/properties/compositions`],
    why: 'the compositions, named by the `composition` of a generated selector',
  },
  {
    tag: 'indices',
    declares: [`${MODEL}#/$defs/composition_definition/properties/indices`],
    why: 'a composition’s own index ranges. The other `indices` maps are *assignments* — a generated selector’s, a scoped endpoint’s override — and are references to these',
  },
  {
    tag: 'inputs',
    declares: [`${MODEL}#/$defs/interfaces/properties/inputs`],
    why: 'the public inputs, named as the `stream` another input joins',
  },
  {
    tag: 'outputs',
    declares: [`${MODEL}#/$defs/interfaces/properties/outputs`],
    why: 'the public outputs; nothing names one, and the empty `refers` says so',
  },
  {
    tag: 'for_each',
    declares: [],
    why: 'a binding rule’s own index ranges. They *are* declarations — a top-level rule may introduce an index that exists nowhere else — but their scope is one rule rather than a composition, and no corpus document writes one by hand (every one of the 706 is `model.normalise` hoisting a scoped rule). Feature 2.13, which edits binding rules, is where that scope is decided',
  },
  {
    tag: 'arguments',
    declares: [],
    why: 'the argument names are declared by the *primitive*, in its unit, not by the document; the declaration is `argument_declaration`, and its referent binding belongs with the primitive editor (feature 3.3)',
  },
  {
    tag: 'record',
    declares: [],
    why: 'the fields of a record argument, declared by the unit’s `record_type`, for the same reason',
  },
  {
    tag: 'values',
    declares: [],
    why: 'keyed by rule names (`qualified_name`), which nothing in a document refers to: a rule name is a label, which is why `value_binding` carries `label: "$key"` and no referent',
  },
  {
    tag: 'parameters',
    declares: [],
    why: 'keyed by rule names, as `values` is',
  },
  {
    tag: 'states',
    declares: [],
    why: 'keyed by rule names, as `values` is',
  },
];

describe('what the document names, and where it is declared', () => {
  const tags = referenceTags(new SchemaShapes(registry()), 'model');

  it('covers every map the schemas key by a name, and no other', () => {
    // Set equality, as the core's semantic tables are held to (§1 d): a grammar that grew a
    // name-keyed map fails here until somebody says what its names refer to.
    expect(KEYED.map((one) => one.tag).sort()).toEqual([...tags.keys.keys()].sort());
  });

  it('names an anchor that is bound and declares something, wherever it names one', () => {
    for (const row of KEYED) {
      for (const anchor of row.declares) {
        expect(file.at(anchor)?.declares, anchor).toBeDefined();
      }
    }
  });

  it('accounts for every declaration the file carries, and for every one it does not', () => {
    const declared = file.anchors.filter((anchor) => file.at(anchor)?.declares !== undefined);
    expect([...declared].sort()).toEqual([...KEYED.flatMap((one) => one.declares)].sort());
    expect(KEYED.filter((one) => one.declares.length === 0).map((one) => one.tag)).toEqual([
      'for_each',
      'arguments',
      'record',
      'values',
      'parameters',
      'states',
    ]);
  });

  it('gives a reason on every row, whether it declares or not', () => {
    for (const row of KEYED) expect(row.why.length, row.tag).toBeGreaterThan(20);
  });

  it('carries no reference rule without a declaration to hang it on', () => {
    for (const anchor of file.anchors) {
      const binding = file.at(anchor);
      if (binding?.refers === undefined) continue;
      expect(binding.declares, anchor).toBeDefined();
    }
  });
});

describe('the unit side of the store, stated rather than assumed', () => {
  it('answers no reference tag at all, which is feature 3.2’s to close', () => {
    // Measured, not guessed: the unit schema reaches its `definition` through the root's
    // `allOf: [{if, then}, …]`, and `SchemaShapes.root` builds its shape from the root node alone
    // — `allOf` is flattened by the `$ref`-following path (`follow`/`gather`), which `root` does
    // not take. So `definition` reduces to `{"type": "object"}` and no `propertyNames` of the unit
    // schema is reachable. The presentation bindings for the unit side are keyed by *anchors* and
    // resolve against the registry, so this feature is unaffected; the unit store (feature 3.2)
    // and the walker (2.3) are not, and this is the line that says so.
    const shapes = new SchemaShapes(registry());
    const unit = referenceTags(shapes, 'primitive-library-unit');
    expect([...unit.keys.keys()]).toEqual([]);
    expect([...unit.values.keys()]).toEqual([]);
    const definition = shapes.member(shapes.root('primitive-library-unit'), 'definition');
    expect(definition.all.map((place) => place.pointer)).toEqual(['/properties/definition']);
  });
});

describe('the schemas this audit reads are the repository’s own', () => {
  it('reads the directory the build vendors, not a copy of it', () => {
    const document = registry().byId(MODEL);
    expect(document).toBeDefined();
    const onDisk = readFileSync(join(repositoryRoot, 'schemas', 'tensorspine.schema.json'), 'utf8');
    expect((JSON.parse(onDisk) as SchemaObject)['$id']).toBe(MODEL);
  });
});

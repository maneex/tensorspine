import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import ts from 'typescript';
import { describe, expect, it } from 'vitest';

import { loadSchemas, type Vocabulary } from '../../packages/lang/src/schema/index.js';
import { editorRoot, readEditorFile } from './tree.js';

// The catching rule (b) of the implementation plan's §1: "A CI test extracts every enum literal
// of the four schemas and every operator name and fails the build if one appears as a string
// literal in GUI source outside `presentation.json`, the core's semantic tables and the tests."
//
// The rule it catches is the governing one: *an item of information that can be inferred from a
// schema is never hard-coded in the GUI*. A select's options are the schema's `enum`; a tagged
// union's tags are the `required` keys of its `oneOf` members; a symbol for an operator is a
// binding in `presentation.json`. A component that types `'floor_divide'` has stopped reading and
// started remembering, and the schemas are then two sources of truth instead of one.
//
// **What it scans, and why not more.** The scope is the *interface*'s sources: every package of
// the workspace but `packages/lang`, and only the `src/` subtree of each. Three reasons, each
// measured rather than assumed:
//
//   * `src/` and not the package: `apps/web/public/vendor/` holds the vendored corpus and the
//     reference base (feature 0.2), which carry every enum literal of the language by
//     construction — they are *data the editor reads*, and a walk of `apps` would report all of
//     them. Feature 0.2 recorded the hazard for this feature; scoping to `src/` is the answer,
//     and it also puts the tests out of scope, which is the third exemption (b) names.
//   * `packages/lang` is out because §0.2 gives it a different rule: it "may name a vocabulary
//     item only in a semantic table whose key set a test proves equal to the schema's enum" —
//     which is catching rule (d), `semantic-tables.test.ts`, seventeen tables today. Measured with
//     the scanner below: **44 of the core's 83 source files** carry a vocabulary literal, 257
//     occurrences over 58 of the 101 values, because the core *is* the reading of the grammar
//     (`if ('self' in rule.indexed_by)`, `BYTES['bf16']`, the location forms). A grep there would
//     need an exemption list longer than the rule, and an exemption list that long is the
//     staleness hazard feature 1.3 warned about. (d) is what holds the core honest.
//   * the four schemas and not the five files `schemas/` holds: the fifth is
//     `tensorspine-fixture.schema.json`, the unit-fixture format the generators write (feature
//     0.2). It is no part of what the editor renders. The set of schema files is asserted below,
//     so a sixth forces a decision instead of slipping past.
//
// **What counts as a literal.** The source is parsed by TypeScript's own parser, not grepped:
// a comment quoting `floor_divide` is documentation, an identifier named `window` is a global,
// and `describe('the dtype table')` is prose — only a string literal token is a literal. Module
// specifiers are not one: `import … from './value.js'` names a file, and a file name is not a
// rendering of the vocabulary.
//
// The comparison is the **whole literal**, not a substring of it, and that is a decision rather
// than a convenience. Thirty-two of the hundred and one values are ordinary English — `add`,
// `value`, `read`, `write`, `none`, `min`, `max`, `first`, `second`, `unit`, `element`, `other`,
// `base`, `record`, `real`, `window` — so a substring rule would report every sentence it ever
// writes, and a check nobody can keep green is a check that gets deleted. The literal has to *be*
// the vocabulary item, which is the shape every real occurrence takes: `kind === 'layer'`,
// `options: ['layer', 'family']`, `{ add: '+' }`. A template literal with no substitution is one
// literal and is compared whole; the spans of one with substitutions are sentence fragments and
// are not — so `` `${p}bf16` `` escapes the rule. Stated, not hidden: it is an evasion nobody
// writes by accident, and closing it costs every message the interface prints.
//
// **What is exempt.** `presentation.json` — the one data file §1 admits, "keyed by JSON pointers
// into the schemas" — and nothing else. The exemption is not a name the scanner skips: the scan
// reads *scripts*, and the rule that keeps it honest is the stronger one §1 states directly —
// **the interface carries one data file, and it is that one**. So a binding table written as a
// TypeScript module is reported like any other literal, and a second data file fails whether it
// carries vocabulary or not.
//
// Feature 2.2 wrote that file, so the exemption is live and the tests below state both halves of
// it: the interface carries exactly one data file and it is that path, and the file **is** where
// the vocabulary went — `tests/audit/presentation.test.ts` requires it to carry operator names
// and resolves every one of its keys against the schemas. An exemption with nothing behind it
// would mean the symbols had gone somewhere else, which is what (b) is for. (Feature 1.3's
// lesson, from the no-Python audit: an exemption ships with a test that names what uses it, or it
// goes stale.)

const repositoryRoot = resolve(editorRoot, '..');

/** The four schemas the plan names, by the `$id` each file declares. */
const FOUR = {
  'tensorspine.schema.json': 'https://tensorspine.dev/schema/2.0/model.json',
  'tensorspine-primitive-library-unit.schema.json':
    'https://tensorspine.dev/schema/2.0/primitive-library-unit.json',
  'tensorspine-documentation.schema.json': 'https://tensorspine.dev/schema/2.0/documentation.json',
  'tensorspine-derived.schema.json': 'https://tensorspine.dev/schema/2.1/derived.json',
} as const;

/**
 * The fifth file of `schemas/`, and the reason it is not part of this rule.
 *
 * `tensorspine-fixture.schema.json` is the unit-fixture format — what a witness records for the
 * reference generator (`docs/TENSORSPINE-FIXTURE.md`). The editor neither writes nor renders one:
 * a primitive's witness is out of this plan's scope (§0, D15). Its own vocabulary is therefore not
 * the GUI's to avoid, and naming it here is what keeps the exclusion visible.
 */
const FIFTH = 'tensorspine-fixture.schema.json';

/** The one data file the interface may carry (§1, "lives in **one** data file"). */
const PRESENTATION = 'packages/ui/src/presentation.json';

/** The repository's own schemas, as the registry loads them at startup. */
function vocabularyOfRepository(): Vocabulary {
  const directory = join(repositoryRoot, 'schemas');
  const files = readdirSync(directory)
    .filter((name) => name.endsWith('.json'))
    .map((name) => ({ path: `schemas/${name}`, text: readFileSync(join(directory, name), 'utf8') }));
  return loadSchemas(files, { origin: 'schemas' }).vocabulary();
}

const vocabulary = vocabularyOfRepository();

/** Every string an enumeration of the four schemas declares — the operator names among them. */
function vocabularyOfTheFour(): ReadonlySet<string> {
  const words = new Set<string>();
  const wanted = new Set<string>(Object.values(FOUR));
  for (const one of vocabulary.enums) {
    if (!wanted.has(one.schema)) continue;
    for (const value of one.values) if (typeof value === 'string') words.add(value);
  }
  return words;
}

const words = vocabularyOfTheFour();

/** The workspace's package globs, as `pnpm-workspace.yaml` declares them. */
function workspaceGlobs(): string[] {
  const found: string[] = [];
  for (const line of readEditorFile('pnpm-workspace.yaml').split(/\r?\n/)) {
    const match = /^\s*-\s*'([^']+)'\s*$/.exec(line) ?? /^\s*-\s*"([^"]+)"\s*$/.exec(line);
    if (match?.[1] !== undefined) found.push(match[1]);
  }
  return found;
}

/**
 * The packages this rule covers: every workspace package but the language core.
 *
 * Derived from the workspace's own globs rather than listed, so that `apps/desktop` (§5.1) and
 * `services/api` join the audit by existing — the one way an exemption cannot go stale is to have
 * none.
 */
function guiPackages(): string[] {
  const found: string[] = [];
  for (const glob of workspaceGlobs()) {
    const [directory, star] = glob.split('/');
    if (directory === undefined || star !== '*') {
      throw new Error(`pnpm-workspace.yaml declares a glob this audit cannot read: ${glob}`);
    }
    const root = join(editorRoot, directory);
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const path = `${directory}/${entry.name}`;
      if (path !== 'packages/lang') found.push(path);
    }
  }
  return found.sort();
}

/** Every file under a package's `src/`, as paths relative to `editor/`. */
function sourcesOf(pkg: string): string[] {
  const root = join(editorRoot, pkg, 'src');
  if (!existsSync(root)) return [];
  const found: string[] = [];
  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) walk(path);
      else found.push(`${pkg}/src/${path.slice(root.length + 1).split('\\').join('/')}`);
    }
  };
  walk(root);
  return found.sort();
}

const SCRIPT = /\.(ts|tsx|js|jsx|mjs|cjs)$/;

/** One place a vocabulary item was typed instead of read. */
export interface Offence {
  readonly path: string;
  readonly line: number;
  readonly value: string;
}

/**
 * Every string literal of a TypeScript source whose text is a vocabulary item.
 *
 * The parser decides what a literal is, which is the whole point: a comment, an identifier and a
 * property name written bare are not literals, and a template literal's spans are.
 */
export function offencesIn(path: string, text: string, admitted: ReadonlySet<string>): Offence[] {
  const source = ts.createSourceFile(path, text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const found: Offence[] = [];
  const isSpecifier = (node: ts.Node): boolean => {
    const parent: ts.Node | undefined = node.parent;
    if (parent === undefined) return false;
    return (
      ((ts.isImportDeclaration(parent) || ts.isExportDeclaration(parent)) &&
        parent.moduleSpecifier === node) ||
      (ts.isCallExpression(parent) && parent.expression.kind === ts.SyntaxKind.ImportKeyword) ||
      ts.isImportTypeNode(parent)
    );
  };
  const walk = (node: ts.Node): void => {
    const literal =
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      node.kind === ts.SyntaxKind.TemplateHead ||
      node.kind === ts.SyntaxKind.TemplateMiddle ||
      node.kind === ts.SyntaxKind.TemplateTail;
    if (literal && !isSpecifier(node)) {
      const value = (node as ts.LiteralLikeNode).text;
      if (admitted.has(value)) {
        const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
        found.push({ path, line: line + 1, value });
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  return found;
}

const packages = guiPackages();
const sources = packages.flatMap((pkg) => sourcesOf(pkg));
const scripts = sources.filter((path) => SCRIPT.test(path));
const offences = scripts.flatMap((path) => offencesIn(path, readEditorFile(path), words));

describe('the vocabulary this rule is about', () => {
  it('is every enumeration of the four schemas the plan names', () => {
    const files = readdirSync(join(repositoryRoot, 'schemas'))
      .filter((name) => name.endsWith('.json'))
      .sort();
    expect(files).toEqual([...Object.keys(FOUR), FIFTH].sort());

    for (const [file, id] of Object.entries(FOUR)) {
      const declared = JSON.parse(readFileSync(join(repositoryRoot, 'schemas', file), 'utf8')) as {
        $id?: string;
      };
      expect(declared.$id, file).toBe(id);
    }
  });

  it('holds the operators, the dtypes and the closed vocabularies of §3.4 and §4.3', () => {
    // Not a count — a schema may gain a value — but the anchors §1 names by hand, so an
    // extraction that quietly found nothing cannot pass for an interface that names nothing.
    expect(words.size).toBeGreaterThan(90);
    for (const anchor of [
      'floor_divide',
      'absolute',
      'greater_or_equal',
      'bf16',
      'all_reduce',
      'logical_position',
      'at_fork_point',
      'quantizable',
      'upper_bound',
      'cached_position',
    ]) {
      expect(words, anchor).toContain(anchor);
    }
  });

  it('leaves the fifth schema of the directory out, and says which', () => {
    const fixture = vocabulary.enums.filter(
      (one) => one.schema === 'https://tensorspine.dev/schema/2.0/fixture.json',
    );
    expect(fixture.length).toBeGreaterThan(0);
    const only = new Set<string>();
    for (const one of fixture) {
      for (const value of one.values) if (typeof value === 'string') only.add(value);
    }
    // `integration` is the fixture format's alone; every other value it writes is a dtype the
    // model schema declares too, so the exclusion loses nothing the GUI could hard-code.
    expect([...only].filter((value) => !words.has(value)).sort()).toEqual(['integration', 'unit']);
  });
});

describe('the sources this rule covers', () => {
  it('is every package of the workspace but the language core', () => {
    expect(packages).toContain('packages/ui');
    expect(packages).toContain('packages/store');
    expect(packages).toContain('apps/web');
    expect(packages).not.toContain('packages/lang');
  });

  it('finds a `src/` in each of them, so none is silently unscanned', () => {
    const without = packages.filter((pkg) => sourcesOf(pkg).length === 0);
    expect(without).toEqual([]);
    expect(scripts.length).toBeGreaterThan(0);
  });

  it('scans no vendored file: the corpus carries the whole vocabulary by construction', () => {
    expect(sources.filter((path) => path.includes('/vendor/'))).toEqual([]);
    expect(sources.filter((path) => /\/(test|e2e)\//.test(path))).toEqual([]);
  });

  it('leaves the core to rule (d), on a premise that is measured and not assumed', () => {
    // Why `packages/lang` is out is a number, not an opinion: the core names the vocabulary in
    // file after file because the core *is* the reading of the grammar, so a grep there would be
    // an exemption list longer than the rule. If that ever stopped being true — if the core came
    // to name a vocabulary item in only a handful of tables — the scoping decision above would
    // want taking again rather than inheriting, and this is what says so.
    const core = sourcesOf('packages/lang').filter((path) => SCRIPT.test(path));
    const carrying = core.filter((path) => offencesIn(path, readEditorFile(path), words).length > 0);
    expect(core.length).toBeGreaterThan(50);
    expect(carrying.length).toBeGreaterThan(core.length / 4);
  });
});

describe('the interface names no vocabulary of the schemas', () => {
  it('types no enum value and no operator name as a string literal', () => {
    expect(offences.map((one) => `${one.path}:${String(one.line)}: ${one.value}`)).toEqual([]);
  });

  it('carries one data file, and it is `presentation.json`', () => {
    const data = sources.filter((path) => path.endsWith('.json'));
    expect(data).toEqual([PRESENTATION]);
  });

  it('leaves that file to the presentation audit, which resolves every key of it', () => {
    // The scan reads scripts, so the data file is out of its reach by construction — and that is
    // exactly why something else has to answer for it. `tests/audit/presentation.test.ts` is
    // that something: every key a place of the loaded schemas, every byte figure bound, every
    // name-keyed map of the grammar accounted for.
    expect(scripts).not.toContain(PRESENTATION);
    expect(existsSync(join(editorRoot, 'tests/audit/presentation.test.ts'))).toBe(true);
  });
});

/**
 * Every identity the repository's own reference base carries, read off its file tree.
 *
 * `<base>/primitives/<a>/<b>/<version>.json` **is** the identity: §8.2 makes the path and what is
 * written inside agree, and the loader refuses a disagreement. So the set is read from the
 * directory rather than loaded, which keeps this audit a scan over files and not a second library
 * load — and a primitive added to the base joins the rule by existing.
 */
function identitiesOfReferenceBase(): ReadonlySet<string> {
  const root = join(repositoryRoot, 'data', 'primitive-library', 'primitives');
  const found = new Set<string>();
  const walk = (current: string, name: readonly string[]): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        walk(join(current, entry.name), [...name, entry.name]);
        continue;
      }
      if (!entry.name.endsWith('.json')) continue;
      const identity = name.join('.');
      found.add(identity);
      found.add(`${identity}@${entry.name.slice(0, -'.json'.length)}`);
    }
  };
  walk(root, []);
  return found;
}

const identities = identitiesOfReferenceBase();
const named = scripts.flatMap((path) => offencesIn(path, readEditorFile(path), identities));

describe('the interface names no primitive of the library either', () => {
  // Feature 2.21's own half of catching rule (b), and the reason it is a rule of its own: an
  // identity is **data of a base**, not vocabulary of a schema, so the extraction above finds it
  // and the enum scan never could. The list a chooser offers is `library.byId` — the core's
  // projection of what was gathered — and a component that typed `attention.dense` would be
  // offering a primitive whether or not any base provides one, which is the same mistake one
  // level along from a hard-coded enum.
  //
  // The tests may type one, and do: they are what says what the catalog answers.

  it('finds the identities in the base rather than listing them here', () => {
    expect(identities.size).toBeGreaterThan(60);
    // Both forms, since both are what a component would be tempted to write: the name a document
    // pins and the `name@version` the library keys by.
    expect([...identities].filter((one) => one.includes('@')).length * 2).toBe(identities.size);
  });

  it('types no identity of the reference base as a string literal', () => {
    expect(named.map((one) => `${one.path}:${String(one.line)}: ${one.value}`)).toEqual([]);
  });

  it('would report one, which is what says the scan reaches them', () => {
    const one = [...identities].find((name) => name.includes('@')) ?? '';
    expect(
      offencesIn('packages/ui/src/probe.ts', `export const p = '${one}';`, identities).map(
        (found) => found.value,
      ),
    ).toEqual([one]);
  });
});

describe('the scan itself', () => {
  const bite = (text: string): string[] =>
    offencesIn('packages/ui/src/probe.ts', text, words).map((one) => one.value);

  it('reports a vocabulary item typed as a string', () => {
    expect(bite(`export const symbol = 'floor_divide';`)).toEqual(['floor_divide']);
    expect(bite('export const dtype = `bf16`;')).toEqual(['bf16']);
    expect(bite('export const kind = `${String(1)}` + `all_reduce`;')).toEqual(['all_reduce']);
    expect(bite(`export type Relation = 'merge' | 'align' | 'insert';`)).toEqual([
      'merge',
      'align',
      'insert',
    ]);
    expect(bite(`export const map = { ['modulo']: 1 };`)).toEqual(['modulo']);
  });

  it('reports nothing for prose, an identifier or a file name', () => {
    expect(bite('// the evaluator says what floor_divide does\nexport const a = 1;')).toEqual([]);
    expect(bite('export const m = `nothing to add for ${String(1)}`;')).toEqual([]);
    expect(bite('/** bf16 and f32 are dtypes. */\nexport const a = 1;')).toEqual([]);
    expect(bite('export const merge = 1; export const x = { insert: 2 };')).toEqual([]);
    expect(bite(`import { a } from './value.js';\nexport const b = a;`)).toEqual([]);
    expect(bite(`export { a } from './merge.js';`)).toEqual([]);
    expect(bite(`export const a = import('./add.js');`)).toEqual([]);
  });

  it('reads every script of the interface, and each one exactly once', () => {
    expect(new Set(scripts).size).toBe(scripts.length);
    for (const path of scripts) expect(existsSync(join(editorRoot, path)), path).toBe(true);
  });
});

import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import vitestConfig from '../../vitest.config.js';
import { editorRoot, filesUnder, readEditorFile } from './tree.js';

// The job of feature 1.12: "the CI job assembles §0.5's oracle, runs every parity suite, the
// semantic-table audits and the no-hard-coding grep (plan §1 b, d), and fails on any divergence
// with the message that moved."
//
// The job itself is the thing this feature builds, so the job itself is what this file tests. Its
// steps are two — `pnpm oracle`, then `pnpm check` — and everything that decides whether those
// two steps mean anything is checked here:
//
//   * the workflow runs the oracle **before** the suites, with the one interpreter the oracle
//     needs, in the workspace the suites live in;
//   * every check the plan's §6 table names is either run by a layer of `pnpm check` today, or
//     carried by a named feature that will run it — transcribed, so a check cannot quietly stop
//     being anybody's;
//   * every parity suite is included by the parity project and reads the oracle, so a suite that
//     invented its own expectation would be visible;
//   * no test of any layer is disabled: `.only` and an unconditional `.skip` are refused
//     outright, and the conditions a test may skip under are a closed, declared set;
//   * the semantic-table audit names, in each of its sections, a module of the core that exists;
//   * the expectations that are **not** the oracle's are declared, one row each, with the feature
//     that will give them one.
//
// What this file does not do is compare anything with the tools: that is the parity layer's, and
// the oracle's integrity is held in `packages/lang/test/parity/job.test.ts`, where the oracle is.

const repositoryRoot = resolve(editorRoot, '..');
const workflow = join(repositoryRoot, '.github', 'workflows', 'editor.yml');

/** A path relative to `editor/` that must exist. */
function editorPath(path: string): string {
  return join(editorRoot, path);
}

describe('the CI job', () => {
  const text = existsSync(workflow) ? readFileSync(workflow, 'utf8') : '';

  it('is the workflow the plan’s §9 Q2 decided on: the tools run in CI as the oracle', () => {
    expect(existsSync(workflow), '.github/workflows/editor.yml').toBe(true);
    expect(text).toMatch(/^\s*editor:\s*$/m);
    expect(text).toMatch(/working-directory:\s*editor/);
  });

  it('assembles the oracle before it runs the suites', () => {
    // The step, not the prose: the workflow's own header names `pnpm check` in a comment.
    const step = (command: string): number => {
      const found = new RegExp(`run:\\s*${command}\\s*$`, 'm').exec(text);
      return found === null ? -1 : found.index;
    };
    const oracle = step('pnpm oracle');
    const check = step('pnpm check');
    expect(oracle, 'the job runs `pnpm oracle` as a step').toBeGreaterThan(-1);
    expect(check, 'the job runs `pnpm check` as a step').toBeGreaterThan(-1);
    expect(oracle).toBeLessThan(check);
  });

  it('installs what the oracle needs, and nothing of Python reaches the editor', () => {
    // `tools/` imports `jsonschema` and nothing else; the standing rule of §0.2 is that Python
    // runs here and in the build scripts, never at runtime.
    expect(text).toMatch(/setup-python/);
    expect(text).toMatch(/jsonschema/);
    expect(text).toMatch(/playwright install .*chromium/);
  });

  it('runs on every push and every pull request', () => {
    expect(text).toMatch(/^\s*push:\s*$/m);
    expect(text).toMatch(/^\s*pull_request:\s*$/m);
  });
});

// The checks table of the plan's §6, transcribed. A row either names the files that run it today
// or the feature that will; the two together are the whole table, so a check cannot be lost
// between phases by nobody claiming it.
interface Check {
  readonly check: string;
  readonly proves: string;
  /** Files that run it today, relative to `editor/`. */
  readonly runs?: readonly string[];
  /** The feature whose block builds it, when no layer runs it yet. */
  readonly owner?: string;
}

const CHECKS: readonly Check[] = [
  {
    check: 'Parity job',
    proves:
      "the core's verdicts, wording and products equal the tools' on the corpus, the reference " +
      'base, the rejection suite (model and library cases) and the signature suite (D2)',
    runs: ['tests/oracle/generate.py', 'packages/lang/test/parity/job.test.ts'],
  },
  {
    check: 'Semantic-table audit (§1 d)',
    proves:
      'every operator, condition form, evolution, access, sharing, communication, transform ' +
      'relation and dtype the schemas list has exactly one implementation in the core',
    runs: ['tests/audit/semantic-tables.test.ts'],
  },
  {
    check: 'Schema snapshot test',
    proves: 'every `$def` of the four schemas renders; a schema change is a reviewed diff',
    runs: ['packages/ui/test/snapshots/forms.test.ts', 'packages/ui/test/forms/walk.test.ts'],
  },
  {
    check: 'No-hard-coding audit (§1 b)',
    proves: 'no schema vocabulary typed into GUI source',
    runs: ['tests/audit/no-hard-coding.test.ts'],
  },
  {
    check: 'Presentation audit at startup',
    proves: 'every binding resolves; unbound constructs listed',
    runs: [
      'tests/audit/presentation.test.ts',
      'packages/ui/test/presentation/audit.test.ts',
      'packages/ui/test/snapshots/presentation.test.ts',
    ],
  },
  {
    check: 'Corpus open test',
    proves: 'every `data/models/*.json` and the template: zero problems, six products, a diagram',
    owner: '2.6 — open, save, tabs, drafts (the reading), 3.1 — the library activity (the units)',
  },
  {
    check: 'Corpus and library round-trip',
    proves: 'unedited save is byte-identical (D12); edited-and-reverted is deep-equal',
    runs: ['packages/lang/test/json/roundtrip.test.ts'],
  },
  {
    check: 'Gesture on-schema test',
    proves: 'every canvas and form gesture keeps every corpus document and unit on the grammar',
    owner: '2.10 — the instance sheet and the argument sheet',
  },
  {
    check: 'Expression round-trip',
    proves: 'text ↔ JSON identity over every expression of the corpus and the reference base',
    owner: '2.11 — expression editors',
  },
  {
    check: 'Build-from-scratch e2e',
    proves: '`llama3-8b` from an empty document and `norm.rms` from an empty unit, in the GUI',
    owner: '2.19 — phase-2 exit, and 3.3 — primitive editor forms',
  },
  {
    check: 'Platform leak build',
    proves: 'the app builds against the stub `Platform`; the round-trip runs in headless Chromium',
    owner: '2.4 — `Platform`: browser workspace, settings, drafts, stub',
  },
  {
    check: 'Timings',
    proves: 'Appendix A refreshed by phase 1 and on every release',
    runs: ['packages/lang/test/api/timings.test.ts'],
  },
];

describe('the checks of the plan’s §6', () => {
  it('names every row as run by a layer or owed by a feature, never both and never neither', () => {
    for (const row of CHECKS) {
      const stated = (row.runs === undefined ? 0 : 1) + (row.owner === undefined ? 0 : 1);
      expect(stated, row.check).toBe(1);
    }
    expect(CHECKS).toHaveLength(12);
  });

  it('finds every file a row says runs it', () => {
    for (const row of CHECKS) {
      for (const path of row.runs ?? []) {
        expect(existsSync(editorPath(path)), `${row.check}: ${path}`).toBe(true);
      }
    }
  });

  it('runs seven of them today; the other five name the feature that will', () => {
    const run = CHECKS.filter((row) => row.runs !== undefined).map((row) => row.check);
    expect(run).toEqual([
      'Parity job',
      'Semantic-table audit (§1 d)',
      'Schema snapshot test',
      'No-hard-coding audit (§1 b)',
      'Presentation audit at startup',
      'Corpus and library round-trip',
      'Timings',
    ]);
    expect(CHECKS.filter((row) => row.owner !== undefined)).toHaveLength(5);
  });
});

const parityRoot = 'packages/lang/test/parity';

function paritySuites(): string[] {
  return filesUnder(parityRoot).filter((path) => path.endsWith('.test.ts'));
}

describe('every parity suite is run, and reads the oracle', () => {
  const suites = paritySuites();

  it('is included by the parity project of the Vitest configuration', () => {
    const projects = (vitestConfig as { test?: { projects?: unknown[] } }).test?.projects ?? [];
    const parity = projects
      .map((project) => (project as { test?: { name?: string; root?: string; include?: string[] } }).test)
      .find((project) => project?.name === 'parity');
    expect(parity?.root).toBe('./packages/lang');
    expect(parity?.include).toEqual(['test/parity/**/*.test.ts']);
    expect(suites.length).toBeGreaterThan(15);
  });

  it('reads what the oracle wrote, directly or through a shared fixture', () => {
    const shared = readEditorFile(`${parityRoot}/derived.ts`);
    expect(shared).toContain("from './oracle.js'");
    const without = suites.filter((path) => {
      const text = readEditorFile(path);
      return !/from '\.\/(oracle|derived)\.js'/.test(text);
    });
    expect(without).toEqual([]);
  });
});

/**
 * The conditions under which a test of this workspace may not run.
 *
 * Every one is a fact about the machine, never about the code: the oracle has not been generated
 * (a developer before `pnpm oracle`; in CI the parity layer asserts that it has), a checkpoint is
 * not on this disk (`TENSORSPINE_MODEL_ARTIFACTS`, `TENSORSPINE_CHECKPOINT`), the browser engine
 * has no OPFS, or the Hub is not reachable. A skip that is not one of these is a test being
 * turned off, which is what the standing rule forbids — and the list being closed is what makes
 * a new way of not running a test a decision somebody takes rather than a line somebody writes.
 */
const SKIP_CONDITIONS = [
  '!generated',
  '!oracleGenerated',
  '!oracleGenerated()',
  'inCI',
  'llamaCheckpoint === null',
  'shieldstral === null',
  '!(await hubReachable())',
  'result.skipped !== undefined',
  'shard === null',
] as const;

function suiteFiles(): string[] {
  return [
    ...filesUnder('packages').filter((path) => /\/test\//.test(path) && path.endsWith('.ts')),
    ...filesUnder('tests/audit').filter((path) => path.endsWith('.ts')),
    ...filesUnder('apps/web/e2e').filter((path) => path.endsWith('.ts')),
  ];
}

describe('no test of any layer is disabled', () => {
  const files = suiteFiles();

  it('reads every suite of every layer', () => {
    expect(files.length).toBeGreaterThan(60);
    for (const path of files) expect(existsSync(editorPath(path)), path).toBe(true);
  });

  it('carries no `.only`, no `.todo` and no unconditional `.skip`', () => {
    const offences: string[] = [];
    for (const path of files) {
      const text = readEditorFile(path);
      for (const match of text.matchAll(/\b(it|test|describe|suite)\.(only|todo)\s*\(/g)) {
        offences.push(`${path}: ${match[1] ?? ''}.${match[2] ?? ''}`);
      }
      // `.skip(` is Playwright's conditional skip when its first argument is an expression, and a
      // test turned off when it is the test's own name.
      for (const match of text.matchAll(/\b(it|test|describe|suite)\.skip\s*\(\s*(.)/g)) {
        if (match[2] === "'" || match[2] === '"' || match[2] === '`') {
          offences.push(`${path}: ${match[1] ?? ''}.skip('…')`);
        }
      }
      for (const match of text.matchAll(/\b(xit|xtest|xdescribe)\s*\(/g)) {
        offences.push(`${path}: ${match[1] ?? ''}(`);
      }
    }
    expect(offences).toEqual([]);
  });

  it('skips only on a fact about the machine, from a closed set of conditions', () => {
    const found = new Set<string>();
    for (const path of files) {
      const text = readEditorFile(path);
      for (const match of text.matchAll(/\.(?:skipIf|runIf)\(([^)]*(?:\([^)]*\))?[^)]*)\)\(/g)) {
        found.add((match[1] ?? '').trim());
      }
      for (const match of text.matchAll(/\b(?:it|test|describe|suite)\.skip\(\s*([^,]+),/g)) {
        found.add((match[1] ?? '').trim());
      }
    }
    expect(found.size).toBeGreaterThan(0);
    expect([...found].sort()).toEqual([...SKIP_CONDITIONS].sort());
  });
});

describe('the semantic-table audits', () => {
  const path = 'tests/audit/semantic-tables.test.ts';
  const text = readEditorFile(path);
  const titles = [...text.matchAll(/^describe\('([^']+)'/gm)].map((match) => match[1] ?? '');

  it('is one file of the audit layer, with a section per table', () => {
    expect(existsSync(editorPath(path))).toBe(true);
    expect(titles.length).toBeGreaterThanOrEqual(14);
  });

  it('names, in each section, a module of the core that exists', () => {
    const named = titles.flatMap((title) => {
      const match = /packages\/lang\/src\/[A-Za-z0-9/_-]+/.exec(title);
      return match === null ? [] : [match[0]];
    });
    expect(named.length).toBeGreaterThanOrEqual(12);
    for (const module of named) {
      const there = existsSync(editorPath(module)) || existsSync(editorPath(`${module}.ts`));
      expect(there, module).toBe(true);
    }
  });
});

/**
 * The expectations of this workspace that are **not** the oracle's.
 *
 * Every parity suite compares the core with what `tools/` answered on material the oracle
 * generated from the repository's own files. These do not: each was read off the tools **once**,
 * by hand, for a document, a unit or a schema no file of the repository carries — so nothing
 * regenerates them and a change in the tools would not move them. They are listed rather than
 * left to be noticed, because that is the difference between a recorded fact and an assumption.
 *
 * Feature 1.13 (finding F8) emptied most of it: the documents behind four of the seven rows are
 * files of `editor/tests/fixtures/` now, the oracle runs `--validate`, `--d1`, `--derive` and
 * `artifact.check` over them as it does over the corpus, and `test/parity/fixtures.test.ts`
 * compares. What may not leave is an expectation whose *input* is not a document — a schema built
 * per keyword, a label `--view` prints, a D3 written by hand — because no fixture can produce one;
 * each such row says so, and says what would own it.
 */
interface Recorded {
  readonly path: string;
  readonly subject: string;
  readonly from: string;
  readonly owner: string;
}

const NOT_THE_ORACLE: readonly Recorded[] = [
  {
    path: 'packages/lang/test/schema/messages.json',
    subject: "one case per keyword of draft 2020-12, both readings of a refusal's wording",
    from: 'tools/schema.py — `check`, `deepest` and `format_error`, on a schema per keyword',
    owner: 'feature 1.1; the corpus reaches five or six of the twenty-odd forms',
  },
  {
    path: 'packages/lang/test/schema/messages.test.ts',
    subject: 'the suite that holds the core to those lines',
    from: 'the fixture beside it',
    owner: 'feature 1.1',
  },
  {
    path: 'packages/lang/test/derive/view-labels.json',
    subject: 'the value-type labels `--view` prints on three models',
    from: 'tools/tensorspine --view, recorded once while `view.py` is in the repository (F6)',
    owner: 'feature 1.8c; the labels move into the editor when `view.py` goes',
  },
  {
    path: 'packages/lang/test/artifact/check.test.ts',
    subject: "V17's lines on `tests/run_artifact.py`'s cases, over hand-built D3s",
    from: 'artifact.check run on the same input',
    owner:
      'nobody: a D3 written by hand under the identity `t` is not a document, so no fixture ' +
      'produces it. Every *shape* it pins is the oracle’s — `out/artifact/index.json`’s `forms` ' +
      'carry the stack, concat, slice and multiplicity branches and `fixtures` now carry a ' +
      'concat and a stack at a dimension that is not 0, both from a document (feature 1.13) — ' +
      'and what is left here is the wording over those literal inputs. Giving it one means ' +
      'adding cases to `_artifact_forms`, which is feature 1.9’s step and not a fixture',
  },
];

/** The prose an author writes when a figure was read off the tools once. */
const TAKEN_BY_HAND =
  /taken from `tools\/|recorded by hand|having been taken from|hand-recorded|'s own on that document/;

/** A file's text with its comment markers removed and its wrapped prose joined. */
function prose(path: string): string {
  return readEditorFile(path)
    .replace(/^\s*(\/\/|\*)\s?/gm, ' ')
    .replace(/\s+/g, ' ');
}

describe('the expectations that are not the oracle’s', () => {
  it('names a file that exists, its subject, where it came from and who will own it', () => {
    for (const row of NOT_THE_ORACLE) {
      expect(existsSync(editorPath(row.path)), row.path).toBe(true);
      expect(row.subject.length, row.path).toBeGreaterThan(20);
      expect(row.from.length, row.path).toBeGreaterThan(10);
      expect(row.owner.length, row.path).toBeGreaterThan(8);
    }
    // Feature 1.13 (F8) gave four of the seven an oracle: the one-instance caller and the scratch
    // base became documents of `editor/tests/fixtures/`, which the oracle runs the tools over, so
    // `derive/source.ts`, `derive/expand.test.ts` and `derive/d2.test.ts` left this register with
    // them. What is left of the fourth is stated in its own row.
    expect(NOT_THE_ORACLE).toHaveLength(4);
    expect(NOT_THE_ORACLE.filter((row) => row.owner.startsWith('nobody'))).toHaveLength(1);
  });

  it('accounts for every fixture the workspace commits beside a suite', () => {
    // The oracle's own output is gitignored and regenerated; a `.json` committed under a test
    // directory is, by construction, an expectation nothing regenerates.
    const committed = filesUnder('packages')
      .filter((path) => /\/test\//.test(path) && path.endsWith('.json'))
      .filter((path) => !path.includes('__snapshots__'));
    const declared = new Set(NOT_THE_ORACLE.map((row) => row.path));
    expect(committed.filter((path) => !declared.has(path))).toEqual([]);
    expect(committed.length).toBeGreaterThan(0);
  });

  it('accounts for every suite whose own prose says a figure was read off the tools', () => {
    const declared = new Set(NOT_THE_ORACLE.map((row) => row.path));
    const saying = filesUnder('packages')
      .filter((path) => /\/test\//.test(path) && path.endsWith('.ts'))
      .filter((path) => TAKEN_BY_HAND.test(prose(path)));
    // Both directions, since feature 1.13 emptied most of the register: nothing says it without a
    // row, and every suite the register still names says it — so a suite that stopped being
    // hand-recorded and stayed listed is as visible as one that started and did not.
    expect(saying.filter((path) => !declared.has(path))).toEqual([]);
    expect([...saying].sort()).toEqual(
      NOT_THE_ORACLE.filter((row) => row.path.endsWith('.ts'))
        .map((row) => row.path)
        .sort(),
    );
  });

  it('names no suite of the parity layer, which is held to the oracle by construction', () => {
    // A parity suite compares the core with what the oracle recorded; one carrying a figure read
    // off the tools by hand would be a parity suite in name only. Every row below is therefore in
    // the unit layer, and this is what keeps it so.
    const inParity = NOT_THE_ORACLE.filter((row) => row.path.startsWith(`${parityRoot}/`));
    expect(inParity).toEqual([]);
    for (const row of NOT_THE_ORACLE) expect(row.path.startsWith('packages/lang/test/')).toBe(true);
  });
});

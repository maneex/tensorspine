import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

import { describe, expect, it } from 'vitest';

import { comparePythonStrings } from '../../src/index.js';
import {
  editorRoot,
  oracleGenerated,
  oracleOut,
  readOracleManifest,
  repositoryRoot,
} from './oracle.js';

// The oracle's integrity (feature 1.12): what has to be true before any suite of this layer means
// anything.
//
// A parity suite compares the core with what `tools/` answered. It reads that answer out of
// `editor/tests/oracle/out/`, which is gitignored, regenerated, and — on a developer machine —
// as old as the last `pnpm oracle`. Three things can make the comparison a lie, and each has a
// test here:
//
//   * **The oracle is stale.** The tools changed and the recorded answers did not, so the suites
//     hold the core to a version of the language nobody runs any more. The manifest carries a
//     digest of `tools/` and of `schemas/` (plan §9 Q2: "a recorded hash of `tools/` and
//     `schemas/` checked by CI against staleness"); this recomputes both and requires them to
//     match the working tree. It is the one check that makes "the tools run in CI as the oracle"
//     true of a developer's run as well.
//   * **A recorded answer is read by nobody.** The generator writes eighteen products; a product
//     no suite opens is an expectation that does not exist. Every one is required to be named by
//     a suite of this layer, or to be declared below with the reason and the feature that will
//     read it.
//   * **The tools are not reproducible.** `validate.analyse` reads a Python set at
//     `mine = next(iter(agree))` and interpolates `repr(expr.UNRESOLVED)` — an address — into one
//     line, so two runs on one document can differ (features 1.6b and 1.6c measured both). The
//     generator answers by dropping what the choice decides and recording that it did; this
//     requires the places where it did to be **exactly** the two documents those features named,
//     over every product the generator writes. A third one is a build failure here rather than a
//     flake in a suite three features away.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/**
 * `generate.py`'s `digest`, in TypeScript.
 *
 * ```python
 * for base, _, names in sorted(os.walk(directory)):
 *     if '__pycache__' in base: continue
 *     for name in sorted(names):
 *         if not name.endswith(suffixes): continue
 *         path = os.path.join(base, name)
 *         sha.update(os.path.relpath(path, ROOT).encode())
 *         sha.update(open(path, 'rb').read())
 * ```
 *
 * The two sorts are Python's, which orders by code point; every path under `tools/` and
 * `schemas/` is ASCII, so `comparePythonStrings` and a plain comparison agree there — it is used
 * all the same, because a file named in anything but ASCII would otherwise make this reading
 * quietly differ from the generator's.
 */
function digest(directory: string, suffixes: readonly string[]): string {
  const directories: string[] = [];
  const walk = (current: string): void => {
    directories.push(current);
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.isDirectory()) walk(join(current, entry.name));
    }
  };
  walk(directory);
  directories.sort(comparePythonStrings);

  const sha = createHash('sha256');
  for (const base of directories) {
    if (base.includes('__pycache__')) continue;
    const names = readdirSync(base, { withFileTypes: true })
      .filter((entry) => !entry.isDirectory())
      .map((entry) => entry.name)
      .sort(comparePythonStrings);
    for (const name of names) {
      if (!suffixes.some((suffix) => name.endsWith(suffix))) continue;
      const path = join(base, name);
      sha.update(Buffer.from(relative(repositoryRoot, path), 'utf8'));
      sha.update(readFileSync(path));
    }
  }
  return sha.digest('hex');
}

/** The suites of this layer, as text: what names a product decides whether it is read. */
function paritySources(): { path: string; text: string }[] {
  const directory = join(editorRoot, 'packages', 'lang', 'test', 'parity');
  return readdirSync(directory)
    .filter((name) => name.endsWith('.ts'))
    .sort()
    .map((name) => ({ path: name, text: readFileSync(join(directory, name), 'utf8') }));
}

/**
 * Which suite of this layer reads each product the generator writes.
 *
 * A product no suite opens is an expectation that does not exist, so the eighteen are listed
 * against their readers rather than left to be noticed. A row with no reader carries the reason it
 * has none and who will give it one — the only such row today is the generated argument schemas,
 * which are a **vendored artifact the editor consumes and never regenerates** (plan §7 F5): the
 * copy that matters is the one in the static build, and `tests/audit/vendor.test.ts` regenerates
 * and checks that on every `pnpm check`.
 *
 * The list is held from both sides: a product the generator gains has no row and fails, a row
 * naming a product it stopped writing fails, and a row naming a suite that does not exist fails.
 */
interface Reader {
  readonly product: string;
  readonly suites: readonly string[];
  readonly why?: string;
}

const READERS: readonly Reader[] = [
  { product: 'arguments', suites: ['arguments.test.ts'] },
  { product: 'artifact', suites: ['artifact.test.ts'] },
  { product: 'bindings', suites: ['bindings.test.ts'] },
  { product: 'd1', suites: ['d1.test.ts'] },
  { product: 'derive', suites: ['derived.ts'] },
  { product: 'expansion', suites: ['d1.test.ts', 'signatures.test.ts'] },
  { product: 'expressions', suites: ['expressions.test.ts'] },
  { product: 'fixtures', suites: ['fixtures.test.ts'] },
  { product: 'graph', suites: ['graph.test.ts'] },
  { product: 'library', suites: ['library.test.ts'] },
  { product: 'lint', suites: ['lint.test.ts'] },
  { product: 'model', suites: ['model.test.ts'] },
  {
    product: 'primitive-schema',
    suites: [],
    why:
      'a generated artifact the editor consumes and never regenerates (plan §7 F5). The vendored '
      + 'copy is the one that matters, and `tests/audit/vendor.test.ts` runs the vendor script '
      + 'itself, so the artifacts are regenerated from the tools and checked on every `pnpm '
      + "check`. The oracle keeps its own because §0.5 asks for it; feature 2.10's argument sheet "
      + 'is what will read a generated schema as an expectation.',
  },
  { product: 'quantities', suites: ['quantities.test.ts'] },
  { product: 'rejections', suites: ['graph.test.ts', 'bindings.test.ts', 'library.test.ts'] },
  { product: 'signatures', suites: ['signatures.test.ts'] },
  { product: 'structural', suites: ['structural.test.ts'] },
  { product: 'validate', suites: ['validate-output.test.ts'] },
];

describe('the oracle the parity layer reads', () => {
  it.runIf(inCI)('has been generated before the suites, as the CI job runs it', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('was generated from the tools this working tree carries', () => {
    const manifest = readOracleManifest();
    expect(digest(join(repositoryRoot, 'tools'), ['.py', 'tensorspine']), 'tools/').toBe(
      manifest.tools_sha256,
    );
    expect(digest(join(repositoryRoot, 'schemas'), ['.json']), 'schemas/').toBe(
      manifest.schemas_sha256,
    );
  });

  it.skipIf(!generated)('says which interpreter and which jsonschema answered', () => {
    const manifest = readOracleManifest();
    expect(manifest.generated_by).toBe('editor/tests/oracle/generate.py');
    expect(manifest.python).toMatch(/^3\.\d+\.\d+$/);
    expect(manifest.jsonschema).toMatch(/^\d+\.\d+\.\d+$/);
    // Recorded so that a flake can be told from a defect by reading a line (see `hash_seed`).
    expect(manifest).toHaveProperty('python_hash_seed');
    expect(manifest.python_hash_seed === null || typeof manifest.python_hash_seed === 'string').toBe(
      true,
    );
  });

  it.skipIf(!generated)('is read whole: every product it wrote is opened by a named suite', () => {
    const sources = new Map(paritySources().map((source) => [source.path, source.text]));
    const products = readdirSync(oracleOut, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
    expect(products.length).toBeGreaterThan(10);
    expect(READERS.map((row) => row.product).sort()).toEqual(products);

    for (const row of READERS) {
      for (const suite of row.suites) {
        expect(sources.has(suite), `${row.product}: ${suite}`).toBe(true);
        // Anti-typo, not proof: the suite that reads a product names it somewhere.
        expect(
          new RegExp(`\\b${row.product.replace('-', '.')}\\b`).test(sources.get(suite) ?? ''),
          `${suite} does not name ${row.product}`,
        ).toBe(true);
      }
      if (row.suites.length === 0) {
        expect((row.why ?? '').length, row.product).toBeGreaterThan(40);
      } else {
        expect(row.why, row.product).toBeUndefined();
      }
    }
    expect(READERS.filter((row) => row.suites.length === 0)).toHaveLength(1);
  });
});

/**
 * The two documents where the tools answer differently from one run to the next.
 *
 * `tests/rejections/models/v5-fusion-without-join.json` — an instance whose inputs disagree, so
 * `mine = next(iter(agree))` takes a hash-seeded one of them and the whole error list follows
 * (feature 1.6b found the choice, feature 1.6c found that it moves the *count*: 131 lines under
 * `PYTHONHASHSEED=0`, 105 under 3). `tests/rejections/models/v3-streams-below-two.json` — one
 * line interpolates `repr(expr.UNRESOLVED)`, whose address changes between runs (feature 1.6c).
 */
const ARBITRARY = 'tests/rejections/models/v5-fusion-without-join.json';
const ELIDED = 'tests/rejections/models/v3-streams-below-two.json';

describe('where the tools are not reproducible, and only there', () => {
  it.skipIf(!generated)('flags a hash-seeded reading in two products, and nowhere else', () => {
    const flagged = new Map<string, number>();
    for (const entry of readdirSync(oracleOut, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const index = join(oracleOut, entry.name, 'index.json');
      if (!statSync(index, { throwIfNoEntry: false })?.isFile()) continue;
      // The text, not the reading: the two largest indexes are 14 and 17 MB, and what is asked
      // of them is whether a flag occurs at all.
      const text = readFileSync(index, 'utf8');
      const count = (flag: string): number => text.split(`"${flag}": true`).length - 1;
      const total = count('arbitrary_own') + count('elided_sentinel');
      if (total > 0) flagged.set(entry.name, total);
    }
    // `graph` records the coin flip once (feature 1.6b's half of `analyse`); `bindings` records
    // it and the elided address, one each (feature 1.6c's whole function).
    expect([...flagged].sort()).toEqual([
      ['bindings', 2],
      ['graph', 1],
    ]);
  });

  it.skipIf(!generated)('names the two documents in the suites that own them', () => {
    const sources = new Map(paritySources().map((source) => [source.path, source.text]));
    expect(sources.get('graph.test.ts')).toContain(ARBITRARY);
    expect(sources.get('bindings.test.ts')).toContain(ARBITRARY);
    expect(sources.get('bindings.test.ts')).toContain(ELIDED);
  });

  it.skipIf(!generated)('drops what the choice decides rather than recording one reading', () => {
    // The fixture's own statement of the rule: the flagged case carries no `errors`, so a parity
    // comparison of them cannot pass under one seed and fail under another.
    const bindings = JSON.parse(
      readFileSync(join(oracleOut, 'bindings', 'index.json'), 'utf8'),
    ) as { documents: { path: string; errors?: unknown; error?: unknown; arbitrary_own?: boolean }[] };
    const arbitrary = bindings.documents.find((one) => one.path === ARBITRARY);
    expect(arbitrary?.arbitrary_own).toBe(true);
    expect(arbitrary?.errors).toBeUndefined();
    // Every other document carries what the tools answered — the lines, or the exception they
    // raised instead (`structural-interface-old-form.json`, feature 1.6a) — so the one dropped
    // reading is the only one a suite cannot compare.
    const settled = bindings.documents.filter((one) => one.arbitrary_own !== true);
    expect(settled.length).toBe(bindings.documents.length - 1);
    for (const one of settled) expect(one.errors ?? one.error, one.path).toBeDefined();
  });
});

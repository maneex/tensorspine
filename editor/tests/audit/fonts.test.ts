import { createRequire } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

import ts from 'typescript';
import { describe, expect, it } from 'vitest';

import { editorRoot, filesUnder, readEditorFile } from './tree.js';

// Feature 2.18 — the two families, served from the application's own origin.
//
// Feature 2.5 asked for them the way the documentation site asks (`docs/style/doc.html`'s one
// `<link>` to `fonts.googleapis.com`) and left the files here: "vendoring the files is 2.18's".
// A static page deployed beside that site should need nothing outside itself — its schemas, its
// corpus, its reference base and its generated artifacts already travel with it (D11, §1, F5) —
// and the type was the last thing that left the origin.
//
// So what this audit holds is the *sameness*: the faces the editor vendors are the faces the site
// asks for, minus the serif the component inventory forbids here ("No serif anywhere — that is
// what separates this from the documentation site"), and which family is the serif is read from
// the site's own stylesheet rather than decided. Both ends are read, so a weight added to the
// site's request or a family renamed in the packages fails here rather than shipping as a page
// drawn in the fallback stack.

const repositoryRoot = resolve(editorRoot, '..');
const require = createRequire(join(editorRoot, 'packages', 'ui', 'package.json'));

/** The site's own template, which carries its one request for type. */
const SITE_TEMPLATE = 'docs/style/doc.html';

/** The site's stylesheet, which says which of the families it asks for is the serif. */
const SITE_STYLE = 'docs/style/primitive-library.css';

/** Where the editor declares its faces. */
const FONTS = 'packages/ui/src/shell/fonts.css';

function repositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}

/**
 * A text with its comments taken out — feature 1.12's rule, one layer down.
 *
 * A comment that names a font host is documentation (this very file's `fonts.css` explains what
 * the site does and why the editor stopped doing it); a `url()` or a `<link>` is a request. The
 * three comment forms of the three languages scanned here: CSS and TypeScript block comments,
 * line comments, and HTML comments.
 */
export function withoutComments(text: string): string {
  return text
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|\s)\/\/[^\n]*/g, '$1');
}

/** One family and the weights a request asks for. */
type Faces = Map<string, number[]>;

/**
 * The families and weights `docs/style/doc.html` asks Google Fonts for.
 *
 * The CSS2 API states a family as `Family+Name:<axes>@<tuples>`, one `family=` parameter each;
 * the weight is the `wght` coordinate, which is the last number of every tuple. A family asked
 * for without axes is the regular weight alone.
 */
export function requestedBy(html: string): Faces {
  const link = /https:\/\/fonts\.googleapis\.com\/css2\?([^"']+)/.exec(html);
  if (link === null) throw new Error(`${SITE_TEMPLATE}: no Google Fonts request to read`);
  const asked: Faces = new Map();
  for (const parameter of (link[1] ?? '').split('&')) {
    if (!parameter.startsWith('family=')) continue;
    const value = decodeURIComponent(parameter.slice('family='.length)).replace(/\+/g, ' ');
    const [name = '', axes] = value.split(':');
    if (axes === undefined) {
      asked.set(name, [400]);
      continue;
    }
    const tuples = (axes.split('@')[1] ?? '').split(';');
    asked.set(
      name,
      tuples.map((tuple) => Number(tuple.split(',').at(-1))).sort((a, b) => a - b),
    );
  }
  return asked;
}

/** The family the site sets its headings in — the one the editor does not carry. */
export function serifOf(css: string): string {
  const declared = /--serif:\s*"([^"]+)"/.exec(css);
  if (declared === null) throw new Error(`${SITE_STYLE}: no --serif to read the serif family from`);
  return declared[1] ?? '';
}

/** The `@fontsource` package a family's files are in, by the package's own naming. */
function packageOf(family: string): string {
  return `@fontsource/${family.toLowerCase().replace(/\s+/g, '-')}`;
}

/** What `fonts.css` imports: the package and the weight of each stylesheet. */
export function importedBy(css: string): Faces {
  const found: Faces = new Map();
  for (const match of css.matchAll(/@import\s+'(@fontsource\/[a-z0-9-]+)\/(\d+)\.css'/g)) {
    const name = match[1] ?? '';
    const weights = found.get(name) ?? [];
    weights.push(Number(match[2]));
    found.set(name, weights.sort((a, b) => a - b));
  }
  return found;
}

const asked = requestedBy(repositoryFile(SITE_TEMPLATE));
const serif = serifOf(repositoryFile(SITE_STYLE));
const imported = importedBy(readEditorFile(FONTS));

describe('the faces the editor vendors', () => {
  it('are the site’s own request, minus the serif the inventory forbids', () => {
    const wanted = new Map(
      [...asked].filter(([family]) => family !== serif).map(([family, weights]) => [packageOf(family), weights]),
    );
    expect(wanted.size).toBeGreaterThan(0);
    expect([...imported].sort()).toEqual([...wanted].sort());
    // And the serif is asked for by the site and by nobody here.
    expect(asked.has(serif)).toBe(true);
    expect([...imported.keys()]).not.toContain(packageOf(serif));
  });

  it('are the families the tokens name, in the order they name them', () => {
    // A vendored face nothing asks for is dead weight; a stack whose first family is not vendored
    // is a page drawn in the fallback. `tokens.css` is the design's and is held byte for byte, so
    // this reads it rather than changing it.
    const tokens = readEditorFile('packages/ui/src/shell/tokens.css');
    for (const [token, family] of [
      ['--sans', 'IBM Plex Sans'],
      ['--mono', 'IBM Plex Mono'],
    ] as const) {
      const stack = new RegExp(`${token}:\\s*"([^"]+)"`).exec(tokens)?.[1];
      expect(stack, `${token} names ${family} first`).toBe(family);
      expect([...imported.keys()]).toContain(packageOf(family));
    }
  });
});

describe('every stylesheet it imports', () => {
  const sheets = [...imported].flatMap(([name, weights]) =>
    weights.map((weight) => ({ specifier: `${name}/${String(weight)}.css`, name, weight })),
  );

  it('is a file of the workspace’s own lockfile, not a request to anyone', () => {
    expect(sheets.length).toBeGreaterThan(0);
    for (const sheet of sheets) {
      expect(() => require.resolve(sheet.specifier), sheet.specifier).not.toThrow();
    }
  });

  it('declares the family and the weight its name claims', () => {
    for (const sheet of sheets) {
      const css = readFileSync(require.resolve(sheet.specifier), 'utf8');
      const families = new Set([...css.matchAll(/font-family:\s*'([^']+)'/g)].map((one) => one[1]));
      const weights = new Set([...css.matchAll(/font-weight:\s*(\d+)/g)].map((one) => Number(one[1])));
      expect([...families], sheet.specifier).toEqual([
        [...asked.keys()].find((family) => packageOf(family) === sheet.name),
      ]);
      expect([...weights], sheet.specifier).toEqual([sheet.weight]);
      // Upright only: the design writes no italic, and an italic face would be a file nobody
      // draws with.
      expect([...new Set([...css.matchAll(/font-style:\s*(\w+)/g)].map((one) => one[1]))]).toEqual(['normal']);
    }
  });

  it('keeps a `unicode-range` per subset, which is what makes several of them one family', () => {
    // The package's per-subset stylesheets (`latin-400.css`) carry none, so two of them imported
    // together would be two faces of the same family and weight and the later would simply win.
    // The per-weight stylesheet is the Google request's own shape: one face per subset, each with
    // the range it answers for.
    for (const sheet of sheets) {
      const css = readFileSync(require.resolve(sheet.specifier), 'utf8');
      const faces = [...css.matchAll(/@font-face/g)].length;
      const ranges = [...css.matchAll(/unicode-range:/g)].length;
      expect(faces, sheet.specifier).toBeGreaterThan(1);
      expect(ranges, sheet.specifier).toBe(faces);
    }
  });

  it('carries every subset its family publishes, which is what the site’s request delivers', () => {
    // The per-weight stylesheet is the whole family at that weight: one face per subset, each
    // with its range, which is exactly what the Google stylesheet answers with. So a reader gets
    // the latin subset for a page of English and the greek one only if something writes Greek —
    // the same bytes over the wire as before, from this origin instead of a third party.
    for (const sheet of sheets) {
      const css = readFileSync(require.resolve(sheet.specifier), 'utf8');
      const metadata = JSON.parse(
        readFileSync(join(dirname(require.resolve(`${sheet.name}/package.json`)), 'metadata.json'), 'utf8'),
      ) as { subsets: string[] };
      expect(metadata.subsets.length, sheet.specifier).toBeGreaterThan(0);
      for (const subset of metadata.subsets) {
        expect(css, `${sheet.specifier} — ${subset}`).toContain(
          `-${subset}-${String(sheet.weight)}-normal.woff2`,
        );
      }
    }
  });

  it('covers the Greek the corpus writes, in the family that has it', () => {
    // `data/models/` and `data/primitive-library/` are English with two Greek letters in their
    // prose (`θ`, `Φ`). IBM Plex Sans publishes a `greek` subset and the editor carries it; IBM
    // Plex Mono publishes none at all, so a Greek letter inside an identifier or a shape falls
    // back to the machine's own face — which is what the site's request does too, and what the
    // token stacks are for.
    const sans = sheets.filter((sheet) => sheet.name === packageOf('IBM Plex Sans'));
    expect(sans.length).toBeGreaterThan(0);
    for (const sheet of sans) {
      expect(readFileSync(require.resolve(sheet.specifier), 'utf8'), sheet.specifier).toContain(
        `-greek-${String(sheet.weight)}-normal.woff2`,
      );
    }
    const mono = JSON.parse(
      readFileSync(
        join(dirname(require.resolve(`${packageOf('IBM Plex Mono')}/package.json`)), 'metadata.json'),
        'utf8',
      ),
    ) as { subsets: string[] };
    expect(mono.subsets).not.toContain('greek');
  });

  it('names files that are there, in both formats it offers', () => {
    for (const sheet of sheets) {
      const at = dirname(require.resolve(sheet.specifier));
      const urls = [...readFileSync(require.resolve(sheet.specifier), 'utf8').matchAll(/url\((\.\/[^)]+)\)/g)].map(
        (one) => one[1] ?? '',
      );
      expect(urls.length, sheet.specifier).toBeGreaterThan(0);
      for (const url of urls) expect(existsSync(join(at, url)), `${sheet.specifier} → ${url}`).toBe(true);
    }
  });
});

describe('the page that draws with them', () => {
  it('asks no font host for anything, in any source or any page of the application', () => {
    const hosts = ['fonts.googleapis.com', 'fonts.gstatic.com'];
    const offences: string[] = [];
    const scanned = [
      ...filesUnder('apps/web/src'),
      ...filesUnder('apps/web/e2e/page'),
      'apps/web/index.html',
      'apps/web/stub.html',
      ...filesUnder('packages/ui/src'),
    ];
    for (const path of scanned) {
      const text = withoutComments(readEditorFile(path));
      for (const host of hosts) if (text.includes(host)) offences.push(`${path}: ${host}`);
    }
    expect(offences).toEqual([]);
    // And the check is not vacuous: the site's own template still carries the request the editor
    // replaced, and it is the one this audit reads its list of faces from.
    expect(withoutComments(repositoryFile(SITE_TEMPLATE))).toContain(hosts[0]);
  });

  it('loads the faces from the stylesheet the shell ships, and from one place', () => {
    // `fonts.css` is imported by `shell.css`, which is `@tensorspine/ui/style.css` — so every
    // deployment that draws the shell draws it in these faces, and a second declaration of the
    // same face somewhere else would be a second answer to the same question.
    expect(readEditorFile('packages/ui/src/shell/shell.css')).toContain("@import './fonts.css';");
    const declaring = filesUnder('packages/ui/src').filter((path) => {
      if (!path.endsWith('.css')) return false;
      const css = withoutComments(readEditorFile(path));
      return css.includes('@font-face') || (path !== FONTS && css.includes('@fontsource'));
    });
    expect(declaring).toEqual([]);
  });

  it('is not a component’s business: no source imports a font', () => {
    // Parsed, not grepped (feature 1.12's rule): a comment naming a package is documentation.
    const offences: string[] = [];
    for (const path of [...filesUnder('packages/ui/src'), ...filesUnder('apps/web/src')]) {
      if (!/\.tsx?$/.test(path)) continue;
      const source = ts.createSourceFile(path, readEditorFile(path), ts.ScriptTarget.ESNext, true, ts.ScriptKind.TSX);
      const walk = (node: ts.Node): void => {
        if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
          if (node.moduleSpecifier.text.startsWith('@fontsource')) offences.push(`${path}: ${node.moduleSpecifier.text}`);
        }
        ts.forEachChild(node, walk);
      };
      walk(source);
    }
    expect(offences).toEqual([]);
  });
});

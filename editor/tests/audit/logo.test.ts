import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { MONOGRAM } from '../../packages/ui/src/shell/logo.js';
import { monogramOf, moduleFor, MODULE, NAV } from '../../scripts/logo.js';
import { HELP_PAGES } from '../../packages/ui/src/shell/store.js';
import { editorRoot, readEditorFile } from './tree.js';

// Two things the shell takes from the repository rather than inventing, and the rule for each.
//
//   * **the logo** — §4.21: "The logo is the monogram of `docs/tensorspine.svg` arranged as
//     `docs/style/nav.html` does — placed and scaled, never redrawn". So the vendored copy has to
//     *be* that arrangement, and this is what says so: the same extraction the vendoring script
//     makes, compared with what it wrote. The project's standing note on the logo is the same
//     rule in one line — use it as it stands, placement and scale only.
//   * **the Help menu's links** — §4.4 names six documents; the documentation site builds its
//     pages from `docs/*.md` with `tools/site.sh`'s own slug rule and lists the generated ones in
//     `nav.html`. A link that named a page the site does not build would be a dead link nobody
//     would notice until they clicked it.

const repositoryRoot = resolve(editorRoot, '..');

/** The text of a file named relative to the repository root. */
function repositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}

describe('the wordmark’s monogram', () => {
  it('is the one element `docs/style/nav.html` arranges, character for character', () => {
    expect(MONOGRAM).toBe(monogramOf(repositoryFile(NAV)));
  });

  it('is what the vendoring script would write today', () => {
    // The module is generated and committed; this is what stops the two from parting company —
    // a hand edit of either fails here rather than surviving to a release.
    expect(readEditorFile(MODULE)).toBe(moduleFor(monogramOf(repositoryFile(NAV))));
  });

  it('is the drawing of `docs/tensorspine.svg`, cropped to the mark and not redrawn', () => {
    const source = repositoryFile('docs/tensorspine.svg');
    // Every path, circle and gradient stop of the monogram comes from the file: the `d`
    // attributes are the test, because a redrawing is exactly a change to those.
    const paths = [...MONOGRAM.matchAll(/ d="([^"]+)"/g)].map((match) => match[1] ?? '');
    expect(paths.length).toBeGreaterThan(10);
    for (const d of paths) expect(source, d.slice(0, 24)).toContain(d);
    // The viewBox is the crop, and it is not the file's own page.
    expect(MONOGRAM).toContain('viewBox="90.5 135.2 29.1 61.6"');
    expect(source).toContain('viewBox="0 0 210 297"');
  });

  it('keeps both spellings of a gradient reference, which is the trap', () => {
    // `nav.html` says why: "a gradient that inherits its stops through the namespaced form alone
    // disappears wherever that form is stripped". Both are present, so either reader works.
    const namespaced = [...MONOGRAM.matchAll(/xlink:href="#/g)].length;
    const plain = [...MONOGRAM.matchAll(/ href="#/g)].length;
    expect(namespaced).toBeGreaterThan(20);
    expect(plain).toBe(namespaced);
  });

  it('is hidden from assistive technology: the bar names the lockup, not the drawing', () => {
    expect(MONOGRAM).toContain('aria-hidden="true"');
  });
});

describe('the Help menu’s links', () => {
  /** The pages `tools/site.sh` builds under `spec/`, by its own rule: the file name, lower-cased. */
  function specPages(): Set<string> {
    const skipped = new Set(['PRIMITIVE-LIBRARY-REFERENCE', 'PLAN-DOCUMENTATION', 'TENSORSPINE_MODEL_TSPL']);
    const pages = new Set<string>();
    for (const name of readdirSync(join(repositoryRoot, 'docs'))) {
      if (!name.endsWith('.md')) continue;
      const stem = name.slice(0, -3);
      if (skipped.has(stem)) continue;
      pages.add(`spec/${stem.toLowerCase()}.html`);
    }
    return pages;
  }

  it('reads `tools/site.sh`’s own exclusions, so this test cannot drift from the build', () => {
    const site = repositoryFile('tools/site.sh');
    expect(site).toContain('PRIMITIVE-LIBRARY-REFERENCE|PLAN-DOCUMENTATION|TENSORSPINE_MODEL_TSPL');
    expect(site).toContain('$out/spec/$slug.html');
  });

  it('names six documents, and the site builds every one of them', () => {
    const pages = specPages();
    const nav = repositoryFile('docs/style/nav.html');
    expect(Object.keys(HELP_PAGES)).toHaveLength(6);
    for (const [command, page] of Object.entries(HELP_PAGES)) {
      if (page.startsWith('spec/')) {
        expect(pages, `${command} → ${page}`).toContain(page);
      } else {
        // A generated page: the site's own navigation is what lists those.
        expect(nav, `${command} → ${page}`).toContain(`%ROOT%${page}`);
      }
    }
  });

  it('resolves against the directory the editor is served beside, never against a host', () => {
    for (const page of Object.values(HELP_PAGES)) {
      expect(page.startsWith('http'), page).toBe(false);
      expect(page.startsWith('/'), page).toBe(false);
    }
  });

  it('leaves the source of each link where the repository keeps it', () => {
    for (const source of [
      'docs/SPECIFICATION.md',
      'docs/TENSORSPINE-MODEL_JSON.md',
      'docs/TENSORSPINE-PRIMITIVE-LIBRARY-UNIT.md',
      'docs/GLOSSARY.md',
      'docs/DERIVED-PRODUCTS.md',
    ]) {
      expect(existsSync(join(repositoryRoot, source)), source).toBe(true);
    }
  });
});

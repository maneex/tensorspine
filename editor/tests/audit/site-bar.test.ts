import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { moduleFor, MODULE, NAV, navOf } from '../../scripts/site-nav.ts';
import { EDITOR_ENTRY, EDITOR_GROUP } from '../../packages/ui/src/shell/site.js';
import { SITE_HOME, SITE_NAV } from '../../packages/ui/src/shell/site-nav.js';
import { editorRoot, readEditorFile } from './tree.js';

// Feature 2.18 — the documentation site's bar above the editor (component inventory §2, S18).
//
//   > The editor is one page of the site, under the site's own bar, exactly as the model views
//   > are today — so a reader who opens a model view and wants to change it is already in the
//   > right place. (S18)
//
// Two things travel from the site into the editor and neither may be written by hand: the
// **entries**, which are `docs/style/nav.html`'s, and the **palette**, which is
// `docs/style/primitive-library.css`'s. This audit reads both sources and holds the copies to
// them — the same rule `tests/audit/logo.test.ts` holds the monogram to, for the same reason: the
// editor cannot fetch the site's own files (it is a page that must run from a folder of its own,
// and from a workspace with no site at all), so it carries copies, and a copy nobody checks
// drifts.

const repositoryRoot = resolve(editorRoot, '..');

/** The site's own stylesheet: where its palette is declared. */
const SITE_STYLE = 'docs/style/primitive-library.css';

/** Where the bar's rules are. */
const SITE_BAR_CSS = 'packages/ui/src/shell/site-bar.css';

function repositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}

const nav = navOf(repositoryFile(NAV));

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

/** The `:root` custom properties of a stylesheet. */
function rootTokens(css: string): Record<string, string> {
  const start = css.indexOf(':root {');
  const end = css.indexOf('}', start);
  const found: Record<string, string> = {};
  for (const match of css.slice(start, end).matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)) {
    found[match[1] ?? ''] = (match[2] ?? '').trim();
  }
  return found;
}

describe('the site’s entries, vendored', () => {
  it('are what `docs/style/nav.html` declares, entry for entry', () => {
    expect(SITE_HOME).toBe(nav.home);
    expect(SITE_NAV).toEqual(nav.groups);
    // Three groups and thirteen entries when this was written; what is asserted is that there are
    // several of each, so the extraction cannot silently answer an empty bar.
    expect(SITE_NAV.length).toBeGreaterThan(1);
    expect(SITE_NAV.flatMap((group) => group.entries).length).toBeGreaterThan(5);
  });

  it('are what the vendoring script would write today', () => {
    expect(readEditorFile(MODULE)).toBe(moduleFor(nav));
  });

  it('refuse rather than guess when the source stops looking like itself', () => {
    expect(() => navOf('<nav></nav>')).toThrow('no <a class="wordmark">');
    expect(() => navOf('<a class="wordmark" href="/somewhere">x</a>')).toThrow('not below %ROOT%');
    expect(() =>
      navOf('<a class="wordmark" href="%ROOT%index.html">x</a><div class="nav-main"></div>'),
    ).toThrow('holds no entry');
  });

  it('name pages the site builds, and never a host', () => {
    const pages = specPages();
    const generated = repositoryFile(NAV);
    for (const entry of SITE_NAV.flatMap((group) => group.entries)) {
      expect(entry.href.startsWith('http'), entry.href).toBe(false);
      expect(entry.href.startsWith('/'), entry.href).toBe(false);
      if (entry.href.startsWith('spec/')) expect(pages, entry.label).toContain(entry.href);
      // A generated page or the front page: the site's own navigation is what lists those, and
      // `tools/site.sh` is what writes them.
      else expect(generated, entry.label).toContain(`%ROOT%${entry.href}`);
    }
    expect(SITE_HOME).toBe('index.html');
  });

  it('leave the site’s own `data-nav` identities behind', () => {
    // They are how `tools/site.sh` marks the page being read, and the editor is never one of the
    // site's pages — and one of them, `specification`, is also an enum value of the documentation
    // schema, which catching rule (b) reports in interface source. Rightly: a whole-literal scan
    // has no way to tell the site's vocabulary from the language's. So the identities are what the
    // extraction *matches on* and are not what it carries.
    expect(readEditorFile(MODULE)).not.toContain('data-nav');
    expect(repositoryFile(NAV)).toContain('data-nav="specification"');
    for (const entry of SITE_NAV.flatMap((group) => group.entries)) {
      expect(Object.keys(entry).sort()).toEqual(['href', 'label']);
    }
  });

  it('carry no entry for the editor, which is why the bar supplies one', () => {
    // The site does not link to the editor yet, and the editor may not edit `docs/` (the standing
    // rule): `SiteBar` marks the page being read itself. The day a documentation change set adds
    // one, this fails and the duplicate goes.
    expect(repositoryFile(NAV)).not.toContain('data-nav="editor"');
    expect(SITE_NAV.map((group) => group.name)).toContain(EDITOR_GROUP);
    expect(EDITOR_ENTRY).toBe('Editor');
  });
});

describe('the bar’s palette', () => {
  const site = rootTokens(repositoryFile(SITE_STYLE));
  const css = readEditorFile(SITE_BAR_CSS);
  const declared = [...css.matchAll(/(--site-[a-z0-9-]+)\s*:\s*([^;]+);/g)].map((match) => ({
    name: match[1] ?? '',
    value: (match[2] ?? '').trim(),
  }));

  it('is the site’s own, token for token', () => {
    expect(declared.length).toBeGreaterThan(4);
    for (const token of declared) {
      const theirs = token.name.replace('--site-', '--');
      expect(site[theirs], `${token.name} is ${SITE_STYLE}'s ${theirs}`).toBe(token.value);
    }
  });

  it('is used, all of it, and nothing else is', () => {
    // Every `--site-*` declared is written somewhere, and no colour of the *editor's* palette is:
    // the bar belongs to the site, which has one palette and no theme, so a bar that darkened
    // with the editor would be the editor pretending the site follows it.
    for (const token of declared) {
      const used = [...css.matchAll(new RegExp(`var\\(${token.name}\\)`, 'g'))].length;
      expect(used, token.name).toBeGreaterThan(0);
    }
    const editorTokens = [...css.matchAll(/var\((--[a-z0-9-]+)\)/g)]
      .map((match) => match[1] ?? '')
      .filter((name) => !name.startsWith('--site-'));
    // `--sans` is the one exception and it is deliberate: the bar is set in the same face as the
    // site's, which is the face the build vendors (`fonts.css`).
    expect([...new Set(editorTokens)]).toEqual(['--sans']);
  });

  it('states the two measures of the lockup the site states', () => {
    // §4.21 and inventory §1: "placed and scaled only, on the two measures the site uses (gap
    // 0.21 H, name 0.44 H)". The site's own narrow-screen lockup is H = 44 px, and the design's
    // `.docbar` is drawn at exactly that.
    const height = /\.docbar \.wordmark svg\.mono \{[^}]*height:\s*([\d.]+)px/.exec(css)?.[1];
    const gap = /\.docbar \.wordmark \{[^}]*gap:\s*([\d.]+)px/.exec(css)?.[1];
    const name = /\.docbar \.wordmark \.name \{[^}]*font-size:\s*([\d.]+)px/.exec(css)?.[1];
    expect(height).toBeDefined();
    expect(Number(gap) / Number(height)).toBeCloseTo(0.21, 2);
    expect(Number(name) / Number(height)).toBeCloseTo(0.44, 2);
    // And the site draws the same two measures at its own sizes.
    const theirs = repositoryFile(SITE_STYLE);
    expect(theirs).toContain('.sitebar .mono { display: block; height: 56px; width: auto; }');
    expect(theirs).toContain('.sitebar .wordmark { flex-shrink: 0; display: flex; align-items: center; gap: 11.9px; }');
  });
});

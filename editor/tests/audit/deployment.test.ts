import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  BASE_VARIABLE,
  EDITOR_DIRECTORY,
  baseFor,
  normaliseBase,
  publishedBase,
  siteDocuments,
  siteRootOf,
  siteUrlsIn,
} from '../../apps/web/deploy.ts';
import { documentationBase } from '../../packages/ui/src/shell/store.js';
import { editorRoot, readEditorFile } from './tree.js';

// Feature 2.18 — where the static application is published, and by what.
//
//   > **D11.** The first target is a static build — HTML, JS, the vendored schemas and reference
//   > base — deployed on GitHub Pages **beside the documentation site** (under the site's bar, as
//   > the model views are), with no server at all.
//
// Two things have to agree for "beside" to be true, and a third has to carry it there:
//
//   * the **base path** the build names its own chunks under, which is the site's path with the
//     editor's directory below it;
//   * `documentationBase()`, which the Help menu's links and the site's bar are resolved
//     against — one directory up from the page, which is the site root only if the editor sits
//     exactly one directory below it;
//   * the **Pages workflow**, which is the second file outside `editor/` this plan allows: one
//     job, one artifact, because GitHub Pages serves one site per repository and the editor
//     travels inside the site's own upload.
//
// The site's path is never typed: it is read from the repository's documents, which name the
// published site wherever they link into it. A site that moves moves the editor with it, and the
// day the two disagree this is what says so.

const repositoryRoot = resolve(editorRoot, '..');

/** The Pages workflow: the one file outside `editor/` this feature writes. */
const PAGES_WORKFLOW = '.github/workflows/pages.yml';

/** The editor's own CI workflow, feature 0.1's — the other file the plan allows. */
const EDITOR_WORKFLOW = '.github/workflows/editor.yml';

function repositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}

const workflow = repositoryFile(PAGES_WORKFLOW);

describe('the published site the editor sits beside', () => {
  it('is named by the repository’s own documents, and they agree', () => {
    const texts = siteDocuments().map((path) => readFileSync(path, 'utf8'));
    const named = texts.flatMap((text) => siteUrlsIn(text));
    // Seventeen links into the site across the README and the guides when this was written; what
    // matters is that there are several and that every one of them says the same site.
    expect(named.length).toBeGreaterThan(5);
    expect(new Set(named).size).toBe(1);
    expect(siteRootOf(texts)).toMatch(/^\/[A-Za-z0-9._-]+\/$/);
  });

  it('refuses rather than guesses where the documents say nothing, or say two things', () => {
    expect(() => siteRootOf(['no link here'])).toThrow(BASE_VARIABLE);
    expect(() =>
      siteRootOf(['https://a.github.io/one/', 'https://a.github.io/two/']),
    ).toThrow('2 different sites');
  });

  it('reads a site URL wherever a document writes one', () => {
    expect(siteUrlsIn('see [the status page](https://maneex.github.io/tensorspine/status/)')).toEqual([
      'https://maneex.github.io/tensorspine/',
    ]);
    expect(siteUrlsIn('https://example.com/tensorspine/')).toEqual([]);
  });
});

describe('the base path the build is made under', () => {
  it('is the site’s path with the editor’s directory below it', () => {
    const texts = siteDocuments().map((path) => readFileSync(path, 'utf8'));
    expect(publishedBase()).toBe(`${siteRootOf(texts)}${EDITOR_DIRECTORY}/`);
  });

  it('leaves the documentation exactly one directory up, where the Help menu looks for it', () => {
    // `documentationBase()` answers the directory above the page; the editor's own directory is
    // what makes that the site root. The two are decided in different packages and this is the
    // one place they are compared.
    const site = siteRootOf(siteDocuments().map((path) => readFileSync(path, 'utf8')));
    expect(documentationBase(`https://example.test${publishedBase()}`)).toBe(
      `https://example.test${site}`,
    );
  });

  it('is what a deployment says when it says something', () => {
    expect(baseFor({ [BASE_VARIABLE]: '/somewhere/else/' })).toBe('/somewhere/else/');
    // `actions/configure-pages` answers `/tensorspine` for a project site and `/` for a user
    // site; the workflow appends `/editor/` to either, and both have to come out as a base a
    // browser reads.
    expect(baseFor({ [BASE_VARIABLE]: `/tensorspine/${EDITOR_DIRECTORY}/` })).toBe(
      `/tensorspine/${EDITOR_DIRECTORY}/`,
    );
    expect(baseFor({ [BASE_VARIABLE]: `//${EDITOR_DIRECTORY}/` })).toBe(`/${EDITOR_DIRECTORY}/`);
  });

  it('is the repository’s where a deployment says nothing, an empty value included', () => {
    expect(baseFor({})).toBe(publishedBase());
    expect(baseFor({ [BASE_VARIABLE]: '' })).toBe(publishedBase());
    expect(baseFor({ [BASE_VARIABLE]: '   ' })).toBe(publishedBase());
  });

  it('is written as a browser reads one: leading and trailing separator, no doubles', () => {
    expect(normaliseBase('/')).toBe('/');
    expect(normaliseBase('')).toBe('/');
    expect(normaliseBase('editor')).toBe('/editor/');
    expect(normaliseBase('/a//b')).toBe('/a/b/');
  });

  it('is the base the Vite build actually uses', async () => {
    // Read from the configuration itself rather than restated: `vite.config.ts` is what the build
    // runs on, and a base written twice is a base that can differ.
    const config = (await import('../../apps/web/vite.config.ts')).default;
    const resolved = typeof config === 'function' ? config({ mode: 'production', command: 'build' }) : config;
    expect(resolved.base).toBe(baseFor());
  });
});

describe('the Pages workflow', () => {
  it('is one of the two files outside `editor/` this plan allows, and both are there', () => {
    expect(workflow.length).toBeGreaterThan(0);
    expect(repositoryFile(EDITOR_WORKFLOW)).toContain('pnpm check');
  });

  it('deploys on `main`, and on nothing else', () => {
    expect(workflow).toMatch(/on:\n\s+push:\n\s+branches: \[main\]/);
    expect(workflow).toContain('workflow_dispatch:');
    expect(workflow).toContain('uses: actions/deploy-pages@');
  });

  it('asks Pages for the base path before the editor is built, and hands it over', () => {
    const configure = workflow.indexOf('actions/configure-pages@');
    const build = workflow.indexOf(BASE_VARIABLE);
    expect(configure).toBeGreaterThan(0);
    expect(build).toBeGreaterThan(configure);
    expect(workflow).toContain(`${BASE_VARIABLE}: \${{ steps.pages.outputs.base_path }}/${EDITOR_DIRECTORY}/`);
    expect(workflow).toContain('id: pages');
  });

  it('builds the editor with the workspace’s own command, dependencies frozen', () => {
    expect(workflow).toContain('pnpm install --frozen-lockfile');
    expect(workflow).toContain('pnpm build');
    expect(workflow).toContain('working-directory: editor');
    // `pnpm build` is the vendoring and the Vite build, which is what makes the artifact carry
    // the schemas, the corpus and the reference base (D11, §1).
    const manifest = JSON.parse(readEditorFile('package.json')) as { scripts: Record<string, string> };
    expect(manifest.scripts['build']).toContain('pnpm run vendor');
  });

  it('puts the editor inside the site’s own artifact, at the directory the base names', () => {
    // One artifact, because GitHub Pages serves one site per repository: a second workflow
    // uploading a second one would replace the site rather than sit beside it.
    expect(workflow).toContain(`mkdir -p _site/${EDITOR_DIRECTORY}`);
    expect(workflow).toContain(`cp -r editor/apps/web/dist/. _site/${EDITOR_DIRECTORY}/`);
    expect([...workflow.matchAll(/upload-pages-artifact@/g)]).toHaveLength(1);
    expect(workflow).toMatch(/path: _site\n/);
    // And the site is still built by the tools, before the editor is put beside it.
    expect(workflow.indexOf('tools/site.sh _site')).toBeGreaterThan(0);
    expect(workflow.indexOf('tools/site.sh _site')).toBeLessThan(
      workflow.indexOf(`cp -r editor/apps/web/dist/.`),
    );
  });
});

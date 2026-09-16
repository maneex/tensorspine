import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

import { navOf, NAV } from '../../../scripts/site-nav.ts';
import { appBase, appUrl, origin, platformUrl } from '../playwright.config.ts';

/**
 * Feature 2.18 — the static build, as GitHub Pages serves it.
 *
 * D11 puts the editor on Pages **beside the documentation site**, and three of the four things
 * that follow can only be asked of a browser looking at the built artifact:
 *
 *   - **the base path is real.** The build names its own chunks, its workers and its stylesheet
 *     under `<site>/editor/`, the preview serves them only there, and a page that ignored the
 *     base would 404 rather than pass. Every other suite of this layer runs against that same
 *     build, which is what "the CI e2e passes on the build" means.
 *   - **nothing leaves the origin.** The schemas, the corpus, the reference base, the generated
 *     artifacts and — this feature's own — the two type families are all served from the page's
 *     own directory. A static page that computes everything in the browser has no reason to tell
 *     a third party who opened it.
 *   - **the documentation site's bar** is above the application, with the site's entries and the
 *     site's lockup, resolving one directory up (S18, component inventory §2).
 *   - **the Examples workspace opens every corpus document with zero problems and its six
 *     products** — the block's own test, and the half of §0's done-criterion that a browser can
 *     answer.
 *
 * The corpus *round trip* (open, save, byte for byte) is feature 2.6's suite, which runs against
 * this same build through `BrowserWorkspace`; it is not repeated here.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');

/** The corpus, by the path the Examples workspace holds it at (its root is the vendor's `data`). */
function corpusPaths(): string[] {
  const found: string[] = [];
  const walk = (at: string, prefix: string): void => {
    for (const entry of readdirSync(join(repository, at), { withFileTypes: true })) {
      if (entry.isDirectory()) walk(`${at}/${entry.name}`, `${prefix}${entry.name}/`);
      else if (entry.name.endsWith('.json')) found.push(`models/${prefix}${entry.name}`);
    }
  };
  walk('data/models', '');
  return found.sort();
}

/**
 * Whether a corpus document is a template — §4.3's own test: "a document with external
 * quantities, carrying `version`".
 *
 * Read from the file rather than from a path, so the day the corpus gains a second template this
 * suite counts it as one.
 */
function isTemplate(path: string): boolean {
  const document = JSON.parse(readFileSync(join(repository, 'data', path), 'utf8')) as {
    version?: unknown;
  };
  return document.version !== undefined;
}

async function open(page: Page, url = appUrl): Promise<void> {
  await page.goto(url);
  await expect(page.locator('#root')).toHaveAttribute('data-state', 'ready');
  await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
}

/** Run a command of §4.4 by its menu entry, which is the way a person reaches it. */
async function command(page: Page, menu: string, id: string): Promise<void> {
  await page.locator(`nav.menu > div > button:text-is("${menu}")`).click();
  await page.locator(`.menu-pop button[data-command="${id}"]`).click();
}

/** Open the Examples workspace the way the empty state offers it. */
async function openExamples(page: Page): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
}

test.describe('the base path the build is published under', () => {
  test('is where the page, its chunks and its vendored material are, and nowhere else', async ({
    page,
  }) => {
    // The page as it is served: the module and the stylesheet are named under the base.
    const html = await (await page.request.get(appUrl)).text();
    const named = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((match) => match[1] ?? '');
    expect(named.length).toBeGreaterThan(1);
    for (const one of named) expect(one.startsWith(appBase), one).toBe(true);

    // And the base is load-bearing: the same asset at the origin root is not there.
    const asset = named.find((one) => one.endsWith('.js')) ?? '';
    expect(asset).not.toBe('');
    expect((await page.request.get(`${origin}${asset}`)).status()).toBe(200);
    expect((await page.request.get(`${origin}${asset.slice(appBase.length - 1)}`)).status()).toBe(404);

    // The vendored material is resolved against the page, not against the origin (D11, §1).
    expect((await page.request.get(`${appUrl}vendor/vendor.json`)).status()).toBe(200);
    expect((await page.request.get(`${origin}/vendor/vendor.json`)).status()).toBe(404);
  });

  test('is where the preview server puts the application, root included', async ({ page }) => {
    // A statement about `vite preview` and `vite dev`, which answer the origin root with a
    // redirect to the base — not about Pages, where the site root is the documentation site's own
    // front page and the editor is the directory beside it. It is here because it is what makes
    // every other suite of this layer, which opens `/`, a suite about the published build.
    const response = await page.goto('/');
    expect(response?.url()).toBe(appUrl);
    await expect(page.locator('#root')).toHaveAttribute('data-state', 'ready');
  });

  test('is the whole application: the worker, the core and the vendored library all under it', async ({
    page,
  }) => {
    const asked: string[] = [];
    page.on('request', (request) => {
      asked.push(request.url());
    });
    await open(page);
    await openExamples(page);
    await command(page, 'File', 'file.open-model');
    await page.locator('.dlg [data-document="models/llama3-8b.json"]').click();
    await expect(page.locator('.gcanvas[data-canvas="models/llama3-8b.json"]')).toBeVisible();
    await expect(page.locator('.bar [data-pill="validation"]')).toHaveText('no problems', {
      timeout: 60_000,
    });

    // Everything the page asked anyone for, and every one of them is this build's own.
    const outside = [...new Set(asked)].filter((url) => !url.startsWith(appUrl));
    expect(outside).toEqual([]);
    // Not a vacuous claim: the worker's chunk, the vendor and a font are all among them.
    const all = asked.join('\n');
    expect(all).toMatch(/\/assets\/worker-[^/]+\.js/);
    expect(all).toContain('vendor/vendor.json');
    expect(all).toMatch(/ibm-plex-(sans|mono)-latin-\d+-normal-[^/]+\.woff2/);
  });

  test('draws in the faces the build carries, not in the fallback stack', async ({ page }) => {
    await open(page);
    const loaded = await page.evaluate(async () => {
      await document.fonts.ready;
      return [...document.fonts].map((face) => `${face.family} ${face.weight} ${face.status}`);
    });
    // The five faces the site asks for, minus the serif (component inventory §1).
    expect(loaded.filter((one) => one.startsWith('IBM Plex Sans')).length).toBeGreaterThan(0);
    expect(loaded.filter((one) => one.startsWith('IBM Plex Mono')).length).toBeGreaterThan(0);
    expect(loaded.filter((one) => one.startsWith('Newsreader'))).toEqual([]);
    expect(
      await page.evaluate(() => document.fonts.check('13px "IBM Plex Sans"')),
    ).toBe(true);
  });
});

test.describe('the documentation site’s bar (S18)', () => {
  test('carries the site’s own entries, in the site’s own order', async ({ page }) => {
    await open(page);
    const bar = page.locator('.docbar');
    await expect(bar).toBeVisible();

    const nav = navOf(readFileSync(join(repository, NAV), 'utf8'));
    const declared = nav.groups.flatMap((group) => group.entries);
    // Read out of the DOM rather than by visibility: a folded group's links are there and are
    // the same links, which is the claim — that the bar carries every entry the site declares.
    // Every href resolves one directory above the page, which is where the site is (D11) — the
    // same directory `documentationBase()` answers for the Help menu, and no host anywhere.
    const site = new URL('../', appUrl).href;
    const shown = await bar.locator('.doclinks a').evaluateAll((all) =>
      all.map((one) => `${(one as HTMLAnchorElement).href} ${one.textContent ?? ''}`),
    );
    expect(shown).toEqual(declared.map((entry) => `${site}${entry.href} ${entry.label}`));
  });

  test('shows what the site shows and folds what the site folds', async ({ page }) => {
    await open(page);
    const nav = navOf(readFileSync(join(repository, NAV), 'utf8'));
    const shown = nav.groups.filter((group) => group.label === null);
    const folded = nav.groups.filter((group) => group.label !== null);
    expect(shown.length).toBeGreaterThan(0);
    expect(folded.length).toBeGreaterThan(0);

    // The site's destinations, in the bar; its guides and its generated documents, behind the
    // site's own two summaries. Fourteen links in a row do not fit the width S18 draws.
    for (const entry of shown.flatMap((group) => group.entries)) {
      await expect(page.locator(`.docbar .doclinks a:text-is("${entry.label}")`)).toBeVisible();
    }
    await expect(page.locator('.docbar .docmenu > summary')).toHaveText(
      folded.map((group) => group.label ?? ''),
    );
    const first = folded[0];
    const inside = page.locator(`.docbar .doclinks a:text-is("${first?.entries[0]?.label ?? ''}")`);
    await expect(inside).toBeHidden();
    await page.locator('.docbar .docmenu > summary').first().click();
    await expect(inside).toBeVisible();
    // And the whole bar still fits: nothing of it is cut off to the right.
    const width = await page.locator('.docbar .doclinks').evaluate((one) => ({
      scroll: one.scrollWidth,
      client: one.clientWidth,
    }));
    expect(width.scroll).toBeLessThanOrEqual(width.client);
  });

  test('marks the editor as the page being read, the site having no entry for it', async ({
    page,
  }) => {
    await open(page);
    await expect(page.locator('.docbar .doclinks .cur')).toHaveText('Editor');
    await expect(page.locator('.docbar .doclinks .cur')).toHaveAttribute('aria-current', 'page');
    // A mark, not a link: there is nowhere for it to go.
    await expect(page.locator('.docbar .doclinks .cur')).not.toHaveAttribute('href', /./);
  });

  test('carries the one lockup on the page, and the application’s bar carries none', async ({
    page,
  }) => {
    await open(page);
    // The site's lockup: the drawing of `docs/tensorspine.svg`, placed and scaled, with the name
    // beside it — and the front page behind both.
    const lockup = page.locator('.docbar .wordmark');
    await expect(lockup).toHaveAttribute('href', new URL('../index.html', appUrl).href);
    await expect(lockup.locator('svg.mono')).toBeVisible();
    await expect(lockup).toHaveText('TensorSpine');
    expect(await lockup.locator('svg.mono').evaluate((one) => Math.round(one.getBoundingClientRect().height))).toBe(44);
    // One to a page: two identical lockups stacked read as two applications.
    await expect(page.locator('.wordmark')).toHaveCount(1);
    await expect(page.locator('header.bar .wordmark')).toHaveCount(0);
    await expect(page.locator('header.bar .sitekind')).toHaveCount(0);
    // The menus are still every one of §4.4's.
    await expect(page.locator('header.bar nav.menu > div > button')).toHaveCount(7);
  });

  test('takes the site’s palette and not the editor’s theme', async ({ page }) => {
    await open(page);
    const ground = async (): Promise<string> =>
      page.locator('.docbar').evaluate((one) => getComputedStyle(one).backgroundColor);
    const light = await ground();
    await page.locator('nav.menu > div > button:text-is("View")').click();
    await page.locator('[data-command="view.theme-dark"]').click();
    await expect(page.locator('.app')).toHaveAttribute('data-scheme', 'dark');
    // The site has one palette; a bar that darkened with the editor would be the editor
    // pretending the site follows it.
    expect(await ground()).toBe(light);
    expect(light).toBe('rgb(247, 245, 240)');
  });

  test('has no axe violation of its own, in either theme', async ({ page }) => {
    for (const theme of ['dark', 'light'] as const) {
      await open(page);
      await page.locator('nav.menu > div > button:text-is("View")').click();
      await page.locator(`[data-command="view.theme-${theme}"]`).click();
      await expect(page.locator('.app')).toHaveAttribute('data-scheme', theme);
      // The wordmark is a logotype: WCAG 1.4.3 exempts brand text from the contrast requirement,
      // and repainting it would be redrawing the logo (§4.21). Every other rule runs over it.
      const audit = async (): Promise<string[]> => {
        const result = await new AxeBuilder({ page }).include('.docbar').exclude('.wordmark').analyze();
        const lockup = await new AxeBuilder({ page })
          .include('.docbar .wordmark')
          .disableRules(['color-contrast'])
          .analyze();
        return [...result.violations, ...lockup.violations].flatMap((one) =>
          one.nodes.map((node) => `${theme}: ${one.id} ${node.target.join(' ')}`),
        );
      };
      expect(await audit()).toEqual([]);
      // And with a group open, which is a state the bar only reaches on a gesture.
      await page.locator('.docbar .docmenu > summary').first().click();
      await expect(page.locator('.docbar .docmenu[open] .docmenu-panel')).toBeVisible();
      expect(await audit()).toEqual([]);
    }
  });
});

test.describe('the Examples workspace on the published build', () => {
  test('opens every corpus document with zero problems and its six products', async ({ page }) => {
    test.setTimeout(300_000);
    await open(page);
    await openExamples(page);

    const paths = corpusPaths();
    expect(paths.length).toBeGreaterThanOrEqual(15);
    const models = paths.filter((path) => !isTemplate(path));
    const templates = paths.filter((path) => isTemplate(path));
    // The corpus's fourteen documents, and the one template, which §0.5 opens "under its
    // documented assignment" — the editor has no command that supplies one yet (§4.6 is feature
    // 3.6's), so it is opened and shown to say what it needs.
    expect(models.length).toBe(14);
    expect(templates.length).toBe(1);

    await command(page, 'View', 'view.derived');
    for (const path of models) {
      await command(page, 'File', 'file.open-model');
      await page.locator(`.dlg [data-document="${path}"]`).click();
      await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
      // A read-only workspace says what it is, never silently (§4.3, inventory §5, S18).
      await expect(page.locator('.banner.info'), path).toContainText(
        'These are the examples the build carries.',
      );
      await expect(page.locator('.bar [data-pill="validation"]'), path).toHaveText('no problems', {
        timeout: 120_000,
      });
      await expect(page.locator('.bar [data-pill="derivation"]'), path).toHaveText('derived · fresh', {
        timeout: 120_000,
      });
      // The six products the specification names, from the core's own derived document.
      await expect(page.locator('.derived .der-rail .row .n'), path).toHaveText([
        'D1',
        'D2',
        'D3',
        'D4',
        'D5',
        'D6',
      ]);
    }

    for (const path of templates) {
      await command(page, 'File', 'file.open-model');
      await page.locator(`.dlg [data-document="${path}"]`).click();
      await expect(page.locator('nav.tabs .tab-slot.on .bdg'), path).toHaveText('template', {
        timeout: 120_000,
      });
      // The grammar and the library still answer — "a template with no assignment is skipped, not
      // failed" — and the derivation is what has nothing to work from. S17 draws `assignment
      // needed` in the status bar; the bar says `not derived`, which is what feature 2.6 built,
      // and the command that would supply one (`Set Preview Assignment…`, §4.6) is feature 3.6's.
      await expect(page.locator('.bar [data-pill="validation"]'), path).toHaveText('no problems', {
        timeout: 120_000,
      });
      await expect(page.locator('.bar [data-pill="derivation"]'), path).toHaveText('not derived');
    }
    await expect(page.locator('nav.tabs .tab')).toHaveCount(paths.length);
    // Not one of them was touched by being read.
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(0);
  });
});

test.describe('what the first document costs to fetch', () => {
  test('is one request per set rather than one per file, for the same bytes', async ({ page }) => {
    await page.goto(platformUrl);
    await expect(page.locator('#out')).toHaveAttribute('data-state', 'ready');
    const outcome = (await page.evaluate(async () => {
      const probes = window.platformProbe as Record<string, () => Promise<unknown>>;
      return probes['transport']?.();
    })) as { ok: boolean; detail: Record<string, unknown> };
    expect(outcome.ok, JSON.stringify(outcome.detail)).toBe(true);

    const detail = outcome.detail;
    // The schemas, the corpus and the reference base: what the first document of a session reads
    // whole (feature 2.6's `gatherSchemas` and `gatherBases`).
    expect(detail['files']).toBeGreaterThan(140);
    expect(detail['sameTexts']).toBe(true);
    expect(detail['sameCount']).toBe(true);
    // Three requests instead of a hundred and fifty, and the same text out of each.
    expect((detail['bundles'] as unknown[]).length).toBeGreaterThanOrEqual(3);
    expect(detail['bundledMs']).toBeLessThan(Number(detail['directMs']));
    // A file no bundle covers is still read from the file itself, which is what a build with no
    // bundles does for everything. The vendored library reference is one file and gets none.
    expect(detail['uncovered']).toBe('generated/primitive-library.md');
    expect(detail['uncoveredBytes']).toBeGreaterThan(1000);
    console.log(
      `transport: ${String(detail['files'])} files — ${Number(detail['directMs']).toFixed(1)} ms one at a time, ` +
        `${Number(detail['bundledMs']).toFixed(1)} ms in ${String((detail['bundles'] as unknown[]).length)} bundles`,
    );
  });
});

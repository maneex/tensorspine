import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { chromium, expect, test, type Locator, type Page } from '@playwright/test';

/**
 * Feature 2.8 — the Problems panel, in a browser.
 *
 * The five claims the feature's block names, each of which needs a real page and a real core to be
 * worth anything:
 *
 *   - **`llama3-8b` shows zero problems** — the panel is empty because the core said nothing, and
 *     the bar and the tab agree with it;
 *   - **a primitive named `norm.rmz` shows the V1 row with the tools' wording**, and it shows it
 *     *within the debounce*: the two halves are asked separately, because no feature before 2.10
 *     gives the interface a way to type a primitive name — the sheet's primitive select is 2.10's
 *     and the editable JSON source is 2.17's — so the wording is asked of a document that carries
 *     it and the debounce of the edits this feature's predecessors do provide;
 *   - **clicking navigates** — including to a place the core's own pointer does not name, which
 *     is feature 2.7's finding answered: a refusal about a composition-scoped binding names the
 *     *written* place now, and the tree opens down to it;
 *   - **the banner appears when a stage refused**, with the rows below it marked as the core's
 *     continuation (plan §3);
 *   - **a code links to its anchor in the specification**, from the map `pnpm anchors` generates.
 *
 * Everything that is a question about *how the rows are arranged* rather than about a browser is
 * asked in `packages/ui/test/problems/`, over the whole corpus.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const MODEL = 'models/llama3-8b.json';
const modelText = readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8');

/** Open the application and wait for it to have decided what to open (D11's own startup). */
async function open(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.locator('#root')).toHaveAttribute('data-state', 'ready');
  await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
}

/** Run a command of §4.4 by its menu entry, which is the way a person reaches it. */
async function command(page: Page, menu: string, id: string): Promise<void> {
  await page.locator(`nav.menu > div > button:text-is("${menu}")`).click();
  await page.locator(`.menu-pop button[data-command="${id}"]`).click();
}

/** Open the Examples workspace and one document of it, the way the empty state offers them. */
async function openExample(page: Page, path = MODEL): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.doc-json[data-path="${path}"]`)).toBeVisible();
}

/** The bottom panel's Problems tab, and its body. */
function panel(page: Page): Locator {
  return page.locator('.panel .panel-body');
}

/** The rows the panel is showing, as `severity | source | code | message | location`. */
async function rows(page: Page): Promise<string[]> {
  return panel(page)
    .locator('.prow')
    .evaluateAll((found) =>
      found.map((one) => {
        const part = (selector: string): string =>
          (one.querySelector(selector)?.textContent ?? '').trim();
        return [
          one.getAttribute('data-severity') ?? '',
          part('.src'),
          part('.code'),
          part('.msg'),
          part('.loc'),
        ].join(' | ');
      }),
    );
}

/** Wait until the core has answered for the document on screen. */
async function checked(page: Page): Promise<void> {
  await expect(page.locator('.bar [data-pill="validation"]')).not.toHaveText('checking…', {
    timeout: 60_000,
  });
  await expect(page.locator('.bar [data-pill="validation"]')).not.toHaveText('not checked', {
    timeout: 60_000,
  });
}

/**
 * A Chromium with a profile on disk, so that the Origin Private File System and the directory
 * handles survive — feature 2.6's own measure, taken for the same reason.
 */
async function inPersistentProfile(base: string, body: (page: Page) => Promise<void>): Promise<void> {
  const context = await chromium.launchPersistentContext('', { baseURL: base });
  try {
    const page = await context.newPage();
    await page.goto(`${base}/`);
    await body(page);
  } finally {
    await context.close();
  }
}

/**
 * Copy the vendored `data/` into an Origin Private File System folder, rewriting the documents a
 * case wants changed, and make the picker answer it.
 *
 * The native picker cannot be driven by an automated browser, so what is replaced is the *picker
 * call*; everything below it is the application's own code (feature 2.4's measure, and 2.6's).
 * The rewrites are how a case gets a document the editor has no gesture for yet: the wording of a
 * refusal is a fact about the core, and a document that carries it is the honest way to ask for it.
 */
async function seedPickedFolder(
  page: Page,
  edits: Readonly<Record<string, [string, string]>> = {},
): Promise<number> {
  return page.evaluate(async (rewrites: Record<string, [string, string]>) => {
    const manifest = (await (await fetch('vendor/vendor.json')).json()) as {
      files: { path: string }[];
      examples: { root: string };
    };
    const root = await navigator.storage.getDirectory();
    const picked = await root.getDirectoryHandle('picked', { create: true });
    const prefix = `${manifest.examples.root}/`;
    const wanted = manifest.files.filter((one) => one.path.startsWith(prefix));
    await Promise.all(
      wanted.map(async (one) => {
        const at = one.path.slice(prefix.length);
        let text = await (await fetch(`vendor/${one.path}`)).text();
        const rewrite = rewrites[at];
        if (rewrite !== undefined) text = text.replace(rewrite[0], rewrite[1]);
        const segments = at.split('/');
        let directory = picked;
        for (const segment of segments.slice(0, -1)) {
          directory = await directory.getDirectoryHandle(segment, { create: true });
        }
        const file = await directory.getFileHandle(segments.at(-1) as string, { create: true });
        const writable = await file.createWritable();
        await writable.write(text);
        await writable.close();
      }),
    );
    Object.defineProperty(window, 'showDirectoryPicker', {
      configurable: true,
      value: () => Promise.resolve(picked),
    });
    return wanted.length;
  }, edits as Record<string, [string, string]>);
}

/** Open the seeded folder and one document of it. */
async function openPicked(page: Page, path = MODEL): Promise<void> {
  await page.locator('.nothing .btn:text-is("Open Folder…")').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('picked');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.doc-json[data-path="${path}"]`)).toBeVisible();
}

test.describe('a corpus document', () => {
  test('shows zero problems, and the bar, the tab and the panel all say so', async ({ page }) => {
    await open(page);
    await openExample(page);
    await checked(page);

    await expect(page.locator('.bar [data-pill="validation"]')).toHaveText('no problems');
    await expect(page.locator('footer.status [data-validation]')).toHaveText('no problems');
    await expect(panel(page).locator('.prow')).toHaveCount(0);
    await expect(panel(page).locator('.banner')).toHaveCount(0);
    await expect(panel(page).locator('.empty-line')).toHaveText('No problems.');
    // §4.17's count: nothing to count, so the tab carries no chip at all.
    await expect(page.locator('.panel .panel-tabs .ptab:text-is("Problems") .n')).toHaveCount(0);
  });

  test('reveals the panel when the count in the bar is chosen (§4.17)', async ({ page }) => {
    await open(page);
    await openExample(page);
    await checked(page);
    await page.locator('.panel .panel-tabs .ptab:text-is("Log")').click();
    await expect(page.locator('.panel .panel-tabs .ptab[aria-selected="true"]')).toHaveText('Log');
    await page.locator('.bar [data-pill="validation"]').click();
    await expect(page.locator('.panel .panel-tabs .ptab[aria-selected="true"]')).toContainText(
      'Problems',
    );
  });
});

test.describe('a refusal of the core', () => {
  test('shows the V1 row with the tools’ wording, and links its code to the specification', async ({
    baseURL,
  }) => {
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      // `norm.rms` at its first occurrence is `final_n`'s pin; nothing else moves.
      const copied = await seedPickedFolder(page, {
        [MODEL]: ['"norm.rms"', '"norm.rmz"'],
      });
      expect(copied).toBeGreaterThan(140);
      await openPicked(page);
      await checked(page);

      const row = panel(page).locator('.prow').first();
      await expect(row).toHaveAttribute('data-source', 'semantic');
      await expect(row).toHaveAttribute('data-code', 'V1');
      await expect(row.locator('.msg')).toHaveText(
        '[V1] primitive absent from primitive library: norm.rmz',
      );
      await expect(row.locator('.loc')).toHaveText('instances/final_n/primitive');
      // §4.17: "a problem's code links to its anchor in the specification (generated map, §1)".
      await expect(row.locator('.code a')).toHaveAttribute(
        'href',
        /spec\/specification\.html#6--static-semantics$/,
      );
      await expect(row.locator('.code a')).toHaveAttribute('title', '§6 — Static semantics');

      // The bar and the tab count the same rows the panel holds.
      await expect(page.locator('.bar [data-pill="validation"]')).toContainText('problems');
      await expect(page.locator('.panel .panel-tabs .ptab:text-is("Problems") .n')).toHaveClass(/bad/);
    });
  });

  test('appears within the debounce, without a save and without a command', async ({ page }) => {
    await open(page);
    await openExample(page);
    await checked(page);

    // Deleting the instance the public input feeds leaves the endpoint behind — the grammar
    // requires it (feature 2.1) — and the core refuses the document. §5.4 debounces the semantic
    // stage at 300 ms; nothing else has to happen for the row to arrive.
    await page.locator('.tree [data-row="/instances/embed"]').click();
    await page.keyboard.press('Delete');
    await page.locator('.dlg [data-confirm="remove"]').click();

    const row = panel(page).locator('.prow[data-code="V1"]').first();
    await expect(row).toBeVisible({ timeout: 5_000 });
    await expect(row.locator('.msg')).toHaveText('[V1] input tokens: instance does not exist');
    await expect(row.locator('.loc')).toHaveText('interfaces/inputs/tokens');
  });
});

test.describe('clicking a row', () => {
  test('selects the place it names, and the sheet opens on it', async ({ page }) => {
    await open(page);
    await openExample(page);
    await checked(page);
    await page.locator('.tree [data-row="/instances/embed"]').click();
    await page.keyboard.press('Delete');
    await page.locator('.dlg [data-confirm="remove"]').click();
    await expect(panel(page).locator('.prow[data-code="V1"]').first()).toBeVisible({ timeout: 5_000 });

    await panel(page).locator('.prow[data-code="V1"]').first().locator('.msg').click();
    await expect(page.locator('.insp [data-place]')).toHaveText('/interfaces/inputs/tokens');
    await expect(page.locator('.tree [data-row="/interfaces/inputs/tokens"]')).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  test('opens the tree down to a place a closed group was hiding', async ({ baseURL }) => {
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      // A scoped parameter rule naming a slot `ffn.gated` has not. The refusal is about a place
      // §5.2 rule 7 hoists out of the composition — feature 2.7 found the red dot landing on the
      // root's Bindings for it — and the row names the place the author wrote.
      await seedPickedFolder(page, { [MODEL]: ['"parameter": "gate"', '"parameter": "gatez"'] });
      await openPicked(page);
      await checked(page);

      const place = '/compositions/decoder/bindings/parameters/ffn.gate';
      const row = panel(page).locator(`.prow[data-place="${place}"]`).first();
      await expect(row).toBeVisible();
      await expect(row.locator('.msg')).toContainText(
        "[V7] parameter decoder.ffn.gate: ffn.gated has no parameter 'gatez'",
      );
      // Thirty-two iterations, one line: the fold says how many the core answered.
      await expect(row.locator('.msg .times')).toHaveText('× 32');
      // The composition's own bindings are closed by default (§4.5), so the row is not on screen.
      await expect(page.locator(`.tree [data-row="${place}"]`)).toHaveCount(0);
      await row.locator('.msg').click();
      await expect(page.locator(`.tree [data-row="${place}"]`)).toHaveAttribute(
        'aria-selected',
        'true',
      );
      await expect(page.locator('.insp [data-place]')).toHaveText(place);
    });
  });
});

test.describe('a stage that refused', () => {
  test('shows the banner, and marks the rows below it as the core’s continuation', async ({
    baseURL,
  }) => {
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      await seedPickedFolder(page, {
        [MODEL]: ['"base": "../primitive-library/"', '"base": "../nowhere/"'],
      });
      await openPicked(page);
      await checked(page);

      const banner = panel(page).locator('.banner');
      await expect(banner).toHaveAttribute('data-banner', 'after-refusal');
      await expect(banner).toHaveClass(/stop/);
      await expect(banner).toContainText('The primitive library refused.');
      // The refusal itself is the loader's, in the tools' own words, and the rows under it are
      // what the core could still decide (plan §3).
      await expect(panel(page).locator('.prow[data-source="library"]').first()).toBeVisible();
      await expect(panel(page).locator('.prow[data-source="semantic"]').first()).toBeVisible();
    });
  });
});

test.describe('the editor’s own notices', () => {
  test('says a quantity is read nowhere, and its fix removes it with the confirmation', async ({
    page,
  }) => {
    await open(page);
    // Measured, and a finding about the corpus: `deepseek-v4-pro` declares `kv_heads` and names
    // it nowhere. Neither `--validate` nor `--lint` has a rule that says so; §4.17 gives the row
    // to the editor, at `notice`.
    await openExample(page, 'models/deepseek-v4-pro.json');
    await checked(page);

    const row = panel(page).locator('.prow[data-source="editor"]');
    await expect(row).toHaveCount(1);
    await expect(row).toHaveAttribute('data-severity', 'notice');
    await expect(row.locator('.msg')).toContainText('quantity kv_heads is declared and read nowhere');
    // The bar counts refusals and advisories: a notice is neither, so it still says none.
    await expect(page.locator('.bar [data-pill="validation"]')).toHaveText('no problems');

    // "Every fix is shown before it is applied" (§4.17): the confirmation of §4.4 comes up first.
    await row.locator('.fix[data-fix="remove-declaration"]').click();
    const dialog = page.locator('.dlg');
    await expect(dialog.locator('.dlg-head h2')).toHaveText('Remove quantity kv_heads?');
    await dialog.locator('[data-confirm="remove"]').click();

    await expect(page.locator('.doc-json')).not.toContainText('"kv_heads"');
    await expect(panel(page).locator('.prow[data-source="editor"]')).toHaveCount(0, {
      timeout: 10_000,
    });
  });
});

test.describe('Model ▸ Lint', () => {
  test('lints the workspace’s own model documents, and the corpus lints clean', async ({ page }) => {
    await open(page);
    await openExample(page);
    await checked(page);

    await command(page, 'Model', 'model.lint');
    // The set is the workspace's models: `--lint`'s answer is a function of it (feature 1.10),
    // one document alone would report thirty-two of the base's primitives as called by nobody,
    // and the corpus of fourteen lints clean. The Examples workspace also holds the reference
    // base's 131 units, which `--lint` would crash on — so a clean run is also the filter working.
    await page.locator('.panel .panel-tabs .ptab:text-is("Log")').click();
    await expect(page.locator('.panel .panel-body .logline').last()).toContainText('candidate file(s)');
    await page.locator('.panel .panel-tabs .ptab:text-is("Problems")').click();
    await expect(panel(page).locator('.prow[data-source="lint"]')).toHaveCount(0, {
      timeout: 30_000,
    });
    await expect(panel(page).locator('.empty-line')).toHaveText('No problems.');
  });
});

test.describe('the filter and the grouping', () => {
  test('keep what the reader asks for and say how many they held back', async ({ baseURL }) => {
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      await seedPickedFolder(page, { [MODEL]: ['"parameter": "gate"', '"parameter": "gatez"'] });
      await openPicked(page);
      await checked(page);

      const before = (await rows(page)).length;
      expect(before).toBeGreaterThan(1);

      // The text filter, matched against the message, the code, the place and the file.
      await page.locator('.panel .panel-right input[data-control="filter"]').fill('gatez');
      await expect(panel(page).locator('[data-hidden-rows]')).toBeVisible();
      expect((await rows(page)).length).toBeLessThan(before);
      await page.locator('.panel .panel-right input[data-control="filter"]').fill('');

      // The severity segment: turning errors off leaves a panel with nothing in it, said plainly.
      await page.locator('.panel .panel-right .seg button[data-severity="error"]').click();
      await expect(panel(page).locator('.prow')).toHaveCount(0);
      await page.locator('.panel .panel-right .seg button[data-severity="error"]').click();
      expect((await rows(page)).length).toBe(before);

      // The grouping: by node the rows are under the places the document declares; flat, one list.
      await expect(panel(page).locator('.pgrp').first()).toBeVisible();
      await page.locator('.panel .panel-right select[data-control="grouping"]').selectOption('flat');
      await expect(panel(page).locator('.pgrp')).toHaveCount(0);
    });
  });
});

test.describe('the panel is accessible', () => {
  for (const theme of ['light', 'dark'] as const) {
    test(`has no axe violation in the ${theme} theme, with rows and a banner`, async ({ baseURL }) => {
      await inPersistentProfile(baseURL ?? '', async (page) => {
        await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
        await seedPickedFolder(page, { [MODEL]: ['"parameter": "gate"', '"parameter": "gatez"'] });
        await openPicked(page);
        await checked(page);
        await command(page, 'View', `view.theme-${theme}`);
        await expect(page.locator('.prow').first()).toBeVisible();

        const result = await new AxeBuilder({ page })
          .include('.panel')
          .disableRules(['color-contrast'])
          .analyze();
        // The contrast rule is run on its own below, over the whole page, as feature 2.5's own
        // pass does: what is excluded there is the wordmark alone.
        expect(result.violations).toEqual([]);

        const contrast = await new AxeBuilder({ page })
          .include('.panel')
          .withRules(['color-contrast'])
          .analyze();
        expect(contrast.violations).toEqual([]);
      });
    });
  }
});

test.describe('the document itself', () => {
  test('is untouched by everything the panel does', async ({ page }) => {
    await open(page);
    await openExample(page);
    await checked(page);
    await page.locator('.panel .panel-right input[data-control="filter"]').fill('anything');
    await page.locator('.panel .panel-right select[data-control="grouping"]').selectOption('unit');
    expect(await page.locator('.doc-json').innerText()).toBe(modelText);
  });
});

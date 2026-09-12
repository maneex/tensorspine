import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { chromium, expect, test, type Page } from '@playwright/test';

import { sourceText } from './source-text.js';

/**
 * Feature 2.6 — open and save, in a browser.
 *
 * The four claims the feature's block names, each of which needs a real engine to be worth
 * anything:
 *
 *   - **Open Examples → every corpus document opens.** The problems and the canvas are 2.8's and
 *     2.9's; what is asked here is that the tab opens and the JSON loads, for all fifteen.
 *   - **An unedited Save through a folder the browser writes is byte-identical.** The picker is
 *     not drivable, so the *picker call* is what is replaced — by a handle to an Origin Private
 *     File System directory, which answers the same interface — and everything below it is the
 *     application's own code. The folder is a copy of the repository's `data/`, which is what
 *     §4.3 says a workspace is.
 *   - **A draft restored after reload.** The autosave is on blur as well as on its timer (§4.3),
 *     and a document the workspace never held — a New Model — is restored from its draft, because
 *     losing a user's typing to a reload is the one thing an autosave must never do.
 *   - **Snapshot mode Save downloads.** The Examples workspace is read-only, so Save hands the
 *     document's own bytes to the browser and the folder is left alone (S18).
 *
 * Everything that is a question about *what the editor does* rather than about a browser is asked
 * in `packages/ui/test/documents/store.test.ts`, where it can be asked exhaustively.
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

/** Open the Examples workspace the way the empty state offers it. */
async function openExamples(page: Page): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
}

/** Open one document of the workspace through File ▸ Open Model…. */
async function openDocument(page: Page, path: string): Promise<void> {
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
}

test.describe('the Examples workspace', () => {
  test('opens, says what it is, and opens every document of the corpus', async ({ page }) => {
    await open(page);
    // S17's empty state until something is opened, which is where the examples are offered.
    await expect(page.locator('.nothing h1')).toHaveText('No workspace is open.');
    await openExamples(page);

    // The banner of a workspace that cannot be written back (inventory §5, S18).
    await expect(page.locator('.banner.info b')).toHaveCount(0);
    await expect(page.locator('nav.tabs')).toHaveCount(0);

    const paths = corpusPaths();
    expect(paths.length).toBeGreaterThanOrEqual(15);
    for (const path of paths) {
      await openDocument(page, path);
      // The JSON that loaded is the file's own, which is the bytes a Save would write (D12).
      const shown = await sourceText(page, path);
      expect(shown.length, path).toBeGreaterThan(100);
    }
    await expect(page.locator('nav.tabs .tab')).toHaveCount(paths.length);
    // Every tab is named after its document, and none of them is dirty.
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(0);
    await expect(page.locator('.banner.info')).toBeVisible();
  });

  test('answers the bar’s two pills and the status bar’s eight fields from the core', async ({
    page,
  }) => {
    await open(page);
    await openExamples(page);
    await openDocument(page, MODEL);

    // The model id and the revision tag, read off the document at the two members the *schema*
    // names — the one it requires as free text, the one it fixes.
    await expect(page.locator('footer.status [data-document]')).toHaveText('llama3-8b');
    await expect(page.locator('footer.status [data-tag]')).toHaveText('tensorspine/2.0');

    // The core's validation and derivation state, as the pills and the bar (inventory §2).
    await expect(page.locator('.bar [data-pill="validation"]')).toHaveText('no problems', {
      timeout: 60_000,
    });
    await expect(page.locator('.bar [data-pill="derivation"]')).toHaveText('derived · fresh', {
      timeout: 60_000,
    });

    // D3 total bytes · D5 operations per element · D4 append bytes per cached position · D2 peak
    // live bytes per element — the four figures §4.2 names, at the places `presentation.json`
    // marks in the derived document, rendered as `view.py` renders them.
    const figures = page.locator('footer.status .fig.d b');
    await expect(figures).toHaveText(['14.96 GiB', '15.01 Gop', '128 KiB', '509 KiB']);
    // The exact number is the tooltip's: no component rounds a figure away (inventory §7).
    await expect(page.locator('footer.status .fig.d').first()).toHaveAttribute(
      'title',
      '16 060 522 496',
    );
  });

  test('Save hands the document’s own bytes over, the folder being read-only (S18)', async ({
    page,
  }) => {
    await open(page);
    await openExamples(page);
    await openDocument(page, MODEL);

    // The button says what it will do rather than pretending to write.
    const save = page.locator('.doc-foot .btn.pri');
    await expect(save).toHaveText('Save — download llama3-8b.json');
    const [download] = await Promise.all([page.waitForEvent('download'), save.click()]);
    expect(download.suggestedFilename()).toBe('llama3-8b.json');
    expect(readFileSync(await download.path(), 'utf8')).toBe(modelText);
  });
});

/**
 * A page in a **persistent profile**, which is what a `FileSystemDirectoryHandle` in IndexedDB
 * needs: in Playwright's default (ephemeral) context, Chromium does not refuse to read one back —
 * it kills the renderer. Feature 0.6 measured it, 2.4 recorded it, and 2.6 inherits it.
 */
async function inPersistentProfile(
  base: string,
  body: (page: Page) => Promise<void>,
): Promise<void> {
  const context = await chromium.launchPersistentContext('', {});
  try {
    const page = await context.newPage();
    await page.goto(`${base}/`);
    await body(page);
  } finally {
    await context.close();
  }
}

/**
 * Copy the vendored `data/` into an Origin Private File System folder, and make the picker answer
 * it.
 *
 * The native picker cannot be driven by an automated browser, so what is replaced is the *picker
 * call* — everything below it, the handle store, the permission dance, the reads and the writes,
 * is the application's own code (feature 2.4 took the same measure for the platform's own suite).
 */
async function seedPickedFolder(page: Page): Promise<number> {
  return page.evaluate(async () => {
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
        const text = await (await fetch(`vendor/${one.path}`)).text();
        const segments = one.path.slice(prefix.length).split('/');
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
  });
}

/** What the picked folder holds at a path, read back out of the Origin Private File System. */
async function readPicked(page: Page, path: string): Promise<string> {
  return page.evaluate(async (at: string) => {
    const root = await navigator.storage.getDirectory();
    let directory = await root.getDirectoryHandle('picked');
    const segments = at.split('/');
    for (const segment of segments.slice(0, -1)) {
      directory = await directory.getDirectoryHandle(segment);
    }
    const file = await directory.getFileHandle(segments.at(-1) as string);
    return (await file.getFile()).text();
  }, path);
}

test.describe('a folder the browser writes', () => {
  test('saves an unedited corpus document back byte for byte (D12)', async ({ baseURL }) => {
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      const copied = await seedPickedFolder(page);
      expect(copied).toBeGreaterThan(140);

      await page.locator('.nothing .btn:text-is("Open Folder…")').click();
      await expect(page.locator('footer.status [data-workspace]')).toHaveText('picked');
      await openDocument(page, MODEL);
      // A folder the browser writes says nothing: there is nothing to warn about (S18).
      await expect(page.locator('.banner')).toHaveCount(0);

      const save = page.locator('.doc-foot .btn.pri');
      await expect(save).toHaveText(`Save ${MODEL}`);
      await save.click();
      await expect(page.locator('.toast')).toContainText('saved');
      expect(await readPicked(page, MODEL)).toBe(modelText);
    });
  });

  test('round-trips the whole corpus through it, byte for byte — the feature’s own done-when', async ({
    baseURL,
  }) => {
    test.setTimeout(180_000);
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      await seedPickedFolder(page);
      await page.locator('.nothing .btn:text-is("Open Folder…")').click();
      await expect(page.locator('footer.status [data-workspace]')).toHaveText('picked');

      // Every document of the corpus, opened and saved through the workspace the browser writes:
      // the parser keeps the number lexemes and the member order, the serializer writes Python's
      // own floats, and the folder holds afterwards exactly what it held before (D12).
      for (const path of corpusPaths()) {
        await openDocument(page, path);
        await page.locator(`.doc-foot .btn.pri[data-save$="${path}"]`).click();
        await expect(page.locator('.toast')).toContainText(path);
        expect(await readPicked(page, path), path).toBe(
          readFileSync(join(repository, 'data', path), 'utf8'),
        );
      }
      await expect(page.locator('nav.tabs .tab')).toHaveCount(corpusPaths().length);
      // And not one of them is dirty: a document saved unedited is a document the file has.
      await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(0);
    });
  });

  test('reopens it after a reload with no picker and no gesture at all (D11)', async ({
    baseURL,
  }) => {
    await inPersistentProfile(baseURL ?? '', async (page) => {
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      await seedPickedFolder(page);
      await page.locator('.nothing .btn:text-is("Open Folder…")').click();
      await expect(page.locator('footer.status [data-workspace]')).toHaveText('picked');
      await openDocument(page, MODEL);

      // The handle is kept in IndexedDB, so the workspace comes back with the grant the browser
      // still holds — and the tabs with it (§4.3's "tabs restore on relaunch").
      await page.reload();
      await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
      await expect(page.locator('footer.status [data-workspace]')).toHaveText('picked');
      await expect(page.locator(`.gcanvas[data-canvas="${MODEL}"]`)).toBeVisible();
      await expect(page.locator('.nothing')).toHaveCount(0);
    });
  });
});

test.describe('a folder dropped on the window (§4.3, D11)', () => {
  test('is read before the transfer is disabled, and says what it is', async ({ page }) => {
    // The other half of the rule feature 2.4 stated on the interface: a `DataTransfer` is
    // disabled the moment the synchronous part of the handler returns, so the platform is built
    // before the window can be dropped on and the handler calls `openDrop` with nothing awaited
    // before it. A handler one `await` too late opens an empty snapshot.
    await open(page);
    await page.evaluate((text) => {
      const transfer = new DataTransfer();
      transfer.items.add(new File([text], 'llama3-8b.json', { type: 'application/json' }));
      window.dispatchEvent(
        new DragEvent('drop', { dataTransfer: transfer, bubbles: true, cancelable: true }),
      );
    }, modelText);
    await expect(page.locator('footer.status [data-workspace]')).toHaveText('dropped folder');
    // A snapshot states itself in a banner, never silently (S18, inventory §5).
    await openDocument(page, 'llama3-8b.json');
    await expect(page.locator('.banner.warn b')).toHaveText('This folder is a read-only snapshot.');
    await expect(page.locator('.doc-foot .btn.pri')).toHaveText('Save — download llama3-8b.json');
  });
});

test.describe('Download Workspace as Zip (§4.4)', () => {
  test('hands over an archive of the folder, with the editor’s unsaved work in it', async ({
    page,
  }) => {
    await open(page);
    await openExamples(page);
    await command(page, 'File', 'file.new-model');
    await expect(page.locator('.gcanvas[data-canvas="untitled.json"]')).toBeVisible();
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      command(page, 'File', 'file.download-zip'),
    ]);
    expect(download.suggestedFilename()).toBe('Examples.zip');
    const archive = readFileSync(await download.path());
    // A ZIP, by its own signature, and large enough to be the corpus and the reference base.
    expect(archive.subarray(0, 4).toString('latin1')).toBe('PK\u0003\u0004');
    expect(archive.length).toBeGreaterThan(500_000);
    // The document the workspace has never held is in it, which is what the command is for.
    expect(archive.toString('latin1')).toContain('untitled.json');
  });
});

test.describe('the accessibility of what this feature draws (§4.21)', () => {
  /**
   * The page, checked by axe — feature 2.5's own audit, with its one exemption.
   *
   * The wordmark is excluded from the colour-contrast rule alone, and WCAG 1.4.3 is why: "Text
   * that is part of a logo or brand name has no contrast requirement". Nothing else is excluded,
   * and what is added here is what this feature draws: an open document, the banner a read-only
   * workspace states itself with, the toast, and the dialogs of §4.3.
   */
  async function audit(page: Page): Promise<string[]> {
    const result = await new AxeBuilder({ page }).exclude('.wordmark').analyze();
    const wordmark = await new AxeBuilder({ page })
      .include('.wordmark')
      .disableRules(['color-contrast'])
      .analyze();
    return [...result.violations, ...wordmark.violations].flatMap((one) =>
      one.nodes.map((node) => `${one.id} ${node.target.join(' ')}`),
    );
  }

  for (const theme of ['dark', 'light'] as const) {
    test(`finds none on an open document and a dialog in the ${theme} theme`, async ({ page }) => {
      await open(page);
      await page.locator('nav.menu > div > button:text-is("View")').click();
      await page.locator(`[data-command="view.theme-${theme}"]`).click();
      await expect(page.locator('.app')).toHaveAttribute('data-scheme', theme);

      await openExamples(page);
      await openDocument(page, MODEL);
      // The banner, the document, the tab strip, the pills and the status bar's eight fields.
      await expect(page.locator('.banner.info')).toBeVisible();
      await expect(page.locator('.bar [data-pill="validation"]')).toBeVisible();
      expect(await audit(page)).toEqual([]);

      // The toast a Save leaves behind, and the dialog of Open Recent ▸.
      await Promise.all([page.waitForEvent('download'), page.locator('.doc-foot .btn.pri').click()]);
      await expect(page.locator('.toast')).toBeVisible();
      expect(await audit(page)).toEqual([]);

      await command(page, 'File', 'file.open-recent');
      await expect(page.locator('.dlg')).toBeVisible();
      expect(await audit(page)).toEqual([]);
    });
  }
});

test.describe('the autosave and its restore (§4.3)', () => {
  test('brings a document the workspace never held back after a reload', async ({ page }) => {
    await open(page);
    await openExamples(page);

    // A document made from nothing is dirty from birth and has no file behind it — which is what
    // makes it the case an autosave exists for.
    await command(page, 'File', 'file.new-model');
    await expect(page.locator('.gcanvas[data-canvas="untitled.json"]')).toBeVisible();
    await command(page, 'File', 'file.new-template');
    await expect(page.locator('.gcanvas[data-canvas="untitled-2.json"]')).toBeVisible();
    const template = await sourceText(page, 'untitled-2.json');
    expect(template).toContain('"version": "1.0.0"');
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(2);

    // §4.3: "every 30 s **and on blur**". The blur is what a suite can ask for.
    await page.evaluate(() => {
      window.dispatchEvent(new Event('blur'));
    });
    await page.waitForTimeout(250);

    await page.reload();
    await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
    // The workspace holds neither file — it is the vendored corpus, read-only — so the only place
    // either tab can have come from is its draft.
    await expect(page.locator('nav.tabs .tab')).toHaveCount(2);
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(2);
    await page.locator('nav.tabs .tab[data-tab$="untitled-2.json"]').click();
    expect(await sourceText(page, 'untitled-2.json')).toBe(template);
  });
});

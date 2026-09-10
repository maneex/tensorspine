import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium, expect, test, type Page } from '@playwright/test';

import { CASES, PAGE_CASES, type EngineReport } from '../../../spikes/static/report.ts';
import { staticBase, staticOrigin, staticUrl } from '../playwright.config';

/**
 * Feature 0.6 — the static build and the browser workspace.
 *
 * The spike's page (`editor/spikes/static/page/`) is built **under a base path** and served
 * under it by the third web server of `playwright.config.ts`. What is held to account here is
 * what plan §2 D11 promises:
 *
 *   - a static build that works from under `/editor/`, beside the documentation site;
 *   - the File System Access path: a folder listed, read, written **in place**, watched by
 *     polling, with the relative resolution `primitive_libraries[].base` needs;
 *   - the handle kept in IndexedDB, so the workspace reopens after a reload with one prompt;
 *   - the read-only snapshot path: a folder upload, a banner that says so, and a Save that
 *     downloads the document's own bytes while the folder on disk is left alone.
 *
 * The writable path runs against an Origin Private File System directory handle — the same
 * `FileSystemDirectoryHandle` API a picked folder gives — because no automated browser can
 * drive the native picker. Which engines *have* that picker is measured by
 * `pnpm spike:static` and recorded in `editor/spikes/static/engines.json`; the last test here
 * holds that record to the shape the note quotes.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const spike = join(repository, 'editor', 'spikes', 'static');

/** A corpus document and a base manifest, read at the source: the bytes the editor must keep. */
const MODEL = 'data/models/llama3-8b.json';
const MANIFEST = 'data/primitive-library/primitive-library.json';
const modelText = readFileSync(join(repository, MODEL), 'utf8');
const manifestText = readFileSync(join(repository, MANIFEST), 'utf8');

/**
 * A workspace on disk, laid out like `data/`: a model, and the base its `primitive_libraries`
 * resolves to beside it. This is what the folder upload is pointed at.
 */
const fixture = mkdtempSync(join(tmpdir(), 'tensorspine-spike-0.6-'));
const workspaceDirectory = join(fixture, 'workspace');
mkdirSync(join(workspaceDirectory, 'models'), { recursive: true });
mkdirSync(join(workspaceDirectory, 'primitive-library'), { recursive: true });
writeFileSync(join(workspaceDirectory, 'models', 'llama3-8b.json'), modelText, 'utf8');
writeFileSync(join(workspaceDirectory, 'primitive-library', 'primitive-library.json'), manifestText, 'utf8');

test.afterAll(() => {
  rmSync(fixture, { recursive: true, force: true });
});

/** The window the spike's page installs its entry points on. */
type SpikeWindow = Window & {
  spike: {
    readonly run: (names: readonly string[], engine?: string, attended?: boolean) => Promise<EngineReport>;
    readonly capabilities: () => Record<string, boolean>;
  };
};

/**
 * A page in a **persistent profile**, which is what a `FileSystemDirectoryHandle` in IndexedDB
 * needs: in Playwright's default (ephemeral) context, Chromium 153 does not refuse to read one
 * back — it kills the renderer. `editor/spikes/static/NOTE.md` records the measurement; the
 * consequence for every later feature that touches recent workspaces is this helper.
 */
async function inPersistentProfile(body: (page: Page) => Promise<void>): Promise<void> {
  const profile = mkdtempSync(join(tmpdir(), 'tensorspine-spike-0.6-profile-'));
  const context = await chromium.launchPersistentContext(profile, {});
  try {
    await body(await context.newPage());
  } finally {
    await context.close();
    rmSync(profile, { recursive: true, force: true });
  }
}

/** Open the page and wait for the shell to have wired itself up. */
async function openPage(page: Page): Promise<void> {
  await page.goto(staticUrl);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
}

/** Run the named cases in the page, as the cross-engine runner does, and answer the report. */
async function runCases(page: Page, names: readonly string[]): Promise<EngineReport> {
  await openPage(page);
  return page.evaluate((asked) => (window as unknown as SpikeWindow).spike.run(asked, 'playwright'), names as string[]);
}

/** The one case of a report, with its failure as the message when it did not pass. */
function only(report: EngineReport, name: string): EngineReport['cases'][number] {
  const found = report.cases.find((one) => one.name === name);
  expect(found, `the page ran no case named ${name}`).toBeDefined();
  const one = found as EngineReport['cases'][number];
  expect(one.error ?? 'none', `the case ${name} refused`).toBe('none');
  expect(one.skipped ?? 'not skipped', `the case ${name} was skipped`).toBe('not skipped');
  expect(one.ok).toBe(true);
  return one;
}

test('the static build is served under its base path, and the origin root is not the application', async ({ page }) => {
  const response = await page.goto(staticUrl);
  expect(response?.status()).toBe(200);

  // A build that named its assets from the root would 404 here rather than pass by accident.
  const root = await page.request.get(`${staticOrigin}/`);
  expect(root.status()).toBe(404);

  const html = readFileSync(join(spike, 'dist', 'index.html'), 'utf8');
  expect(html).toContain(`src="${staticBase}assets/`);
  expect(html).not.toMatch(/src="\/assets\//);

  const detail = only(await runCases(page, [CASES.basePath]), CASES.basePath).detail ?? {};
  expect(detail['base']).toBe(staticBase);
  expect(detail['underRoot']).toBe(false);
  expect(String(detail['module'])).toContain(`${staticBase}assets/`);
});

test('a folder the browser can write is listed, read, written in place and resolved', async ({ page }) => {
  const result = only(await runCases(page, [CASES.directoryRoundTrip]), CASES.directoryRoundTrip);
  const detail = result.detail ?? {};
  expect(detail['files']).toEqual(['models/spike.json', 'primitive-library/primitive-library.json']);
  // The refusal is the optimistic concurrency of §5.2, in the words the workspace gave it.
  expect(String(detail['conflict'])).toContain('changed since it was read');
});

test('the polling watch reports a change made behind the workspace, and stops when unsubscribed', async ({ page }) => {
  const result = only(await runCases(page, [CASES.directoryWatch]), CASES.directoryWatch);
  const detail = result.detail ?? {};
  expect(detail['events']).toBe(3);
  for (const key of ['msToChanged', 'msToAdded', 'msToRemoved']) {
    expect(typeof detail[key], `${key} was not measured`).toBe('number');
  }
});

test('a revision is a modification time and a size, and the page says what it can tell apart', async ({ page }) => {
  const detail = only(await runCases(page, [CASES.revisionResolution]), CASES.revisionResolution).detail ?? {};
  // A measurement, not a contract: `editor/spikes/static/NOTE.md` says what follows from it.
  expect(typeof detail['distinguishableImmediately']).toBe('boolean');
  expect(detail['distinguishableAfter20ms']).toBe(true);
});

test('a directory handle survives IndexedDB and comes back as the same folder', async () => {
  await inPersistentProfile(async (page) => {
    const detail = only(await runCases(page, [CASES.handlePersistence]), CASES.handlePersistence).detail ?? {};
    expect(detail['name']).toBe('persisted');
    expect(detail['granted']).toBe(true);
  });
});

test('the workspace opens through the picker, saves in place, and reopens after a reload with one prompt', async () => {
  await inPersistentProfile(async (page) => {
    // The native picker is not drivable, so the *picker call* is what is replaced — by a handle
    // to a folder of the Origin Private File System, which answers the same interface. Everything
    // below the picker is the application's own code. The permission model is stubbed too, and
    // counted: a reopen must ask at most once, and never open a picker again.
    // The grant is kept in `sessionStorage`, so it survives a reload the way a real one survives
    // it: the browser remembers that the user said yes, and does not ask again in that tab.
    await page.addInitScript(() => {
      const KEY = 'spike-permission';
      const read = (): { state: string; queried: number; requested: number } =>
        JSON.parse(sessionStorage.getItem(KEY) ?? '{"state":"prompt","queried":0,"requested":0}') as {
          state: string;
          queried: number;
          requested: number;
        };
      const write = (value: { state: string; queried: number; requested: number }): void => {
        sessionStorage.setItem(KEY, JSON.stringify(value));
      };
      Object.defineProperty(FileSystemHandle.prototype, 'queryPermission', {
        configurable: true,
        value: () => {
          const value = read();
          value.queried += 1;
          write(value);
          return Promise.resolve(value.state);
        },
      });
      Object.defineProperty(FileSystemHandle.prototype, 'requestPermission', {
        configurable: true,
        value: () => {
          const value = read();
          value.requested += 1;
          value.state = 'granted';
          write(value);
          return Promise.resolve('granted');
        },
      });
      Object.defineProperty(window, 'showDirectoryPicker', {
        configurable: true,
        value: async () => (await navigator.storage.getDirectory()).getDirectoryHandle('picked', { create: true }),
      });
    });

    await openPage(page);

    // Seed the folder the stubbed picker will hand over — a workspace laid out like `data/`.
    await page.evaluate(
      async ([model, manifest]) => {
        const root = await navigator.storage.getDirectory();
        const picked = await root.getDirectoryHandle('picked', { create: true });
        const put = async (folder: string, name: string, text: string): Promise<void> => {
          const directory = await picked.getDirectoryHandle(folder, { create: true });
          const writable = await (await directory.getFileHandle(name, { create: true })).createWritable();
          await writable.write(text);
          await writable.close();
        };
        await put('models', 'llama3-8b.json', model ?? '');
        await put('primitive-library', 'primitive-library.json', manifest ?? '');
      },
      [modelText, manifestText],
    );

    await page.getByTestId('open-folder').click();
    await expect(page.locator('body')).toHaveAttribute('data-workspace', 'directory');
    await expect(page.locator('body')).toHaveAttribute('data-writable', 'true');
    await expect(page.getByTestId('status')).toContainText('picked · read-write');

    const files = page.getByTestId('file');
    await expect(files).toHaveText(['models/llama3-8b.json', 'primitive-library/primitive-library.json']);

    await files.first().click();
    // The corpus document reaches the editor byte for byte, through the handle alone.
    await expect(page.getByTestId('editor')).toHaveValue(modelText);

    const edited = modelText.replace('"model": "llama3-8b"', '"model": "llama3-8b-edited"');
    await page.getByTestId('editor').fill(edited);
    await page.getByTestId('save').click();
    await expect(page.getByTestId('log')).toContainText('Saved models/llama3-8b.json in place');

    // Written in place: the folder itself now holds the edit, read back through a fresh handle.
    const onDisk = await page.evaluate(async () => {
      const picked = await (await navigator.storage.getDirectory()).getDirectoryHandle('picked');
      const models = await picked.getDirectoryHandle('models');
      return (await (await models.getFileHandle('llama3-8b.json')).getFile()).text();
    });
    expect(onDisk).toBe(edited);

    // The reload: the handle is in IndexedDB, so the workspace is offered by name, not re-picked.
    await page.reload();
    await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
    await expect(page.locator('body')).toHaveAttribute('data-state', 'no-workspace');
    const recent = page.getByTestId('recent');
    await expect(recent).toHaveText(['picked']);

    await recent.click();
    await expect(page.locator('body')).toHaveAttribute('data-workspace', 'directory');
    await expect(page.getByTestId('log')).toContainText('permission granted (asked once)');

    // And it is the same folder, with the edit still in it.
    await page.getByTestId('file').first().click();
    await expect(page.getByTestId('editor')).toHaveValue(edited);

    // One prompt, and one only: a second reload finds the grant still held and reopens the
    // workspace with no gesture at all — which is what D11 promises the user.
    await page.reload();
    await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
    await expect(page.locator('body')).toHaveAttribute('data-workspace', 'directory');
    await expect(page.getByTestId('log')).toContainText('without a prompt');
    const permission = await page.evaluate(() => JSON.parse(sessionStorage.getItem('spike-permission') ?? '{}') as { requested?: number });
    expect(permission.requested, 'the workspace asked for permission more than once').toBe(1);
  });
});

test('a folder upload is a read-only snapshot, and Save downloads the document without touching the folder', async ({
  page,
}) => {
  await openPage(page);
  await page.getByTestId('upload').setInputFiles(workspaceDirectory);

  await expect(page.locator('body')).toHaveAttribute('data-workspace', 'snapshot');
  await expect(page.locator('body')).toHaveAttribute('data-writable', 'false');
  // Stated in a banner, never silent (S18).
  await expect(page.getByTestId('banner')).toBeVisible();
  await expect(page.getByTestId('banner')).toContainText('read-only snapshot');
  await expect(page.getByTestId('banner')).toContainText('downloads the file for you to put back');
  await expect(page.getByTestId('status')).toContainText('workspace · read-only snapshot');

  // The folder's own name is stripped: the paths are the workspace's, as on the writable side.
  await expect(page.getByTestId('file')).toHaveText(['models/llama3-8b.json', 'primitive-library/primitive-library.json']);

  await page.getByTestId('file').first().click();
  await expect(page.getByTestId('editor')).toHaveValue(modelText);
  // The button says what it will actually do, rather than pretending to write.
  await expect(page.getByTestId('save')).toHaveText('Save — download llama3-8b.json');

  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('save').click()]);
  expect(download.suggestedFilename()).toBe('llama3-8b.json');
  expect(readFileSync(await download.path(), 'utf8')).toBe(modelText);
  await expect(page.getByTestId('log')).toContainText('the folder itself is untouched');

  // And the folder really is untouched: the snapshot is a copy the page was handed.
  expect(readFileSync(join(workspaceDirectory, 'models', 'llama3-8b.json'), 'utf8')).toBe(modelText);
});

test('the snapshot path is exercised on the uploaded folder, and refuses to create a directory', async ({ page }) => {
  await openPage(page);
  await page.getByTestId('upload').setInputFiles(workspaceDirectory);

  const report = await page.evaluate(
    (asked) => (window as unknown as SpikeWindow).spike.run(asked, 'playwright'),
    [CASES.snapshotUpload, CASES.snapshotSave] as string[],
  );

  const upload = only(report, CASES.snapshotUpload).detail ?? {};
  // The files came from the input, not from the page's own stand-in.
  expect(upload['source']).toBe('input');
  expect(upload['folder']).toBe('workspace');
  expect(upload['files']).toBe(2);
  expect(upload['paths']).toEqual(['models/llama3-8b.json', 'primitive-library/primitive-library.json']);
  expect(upload['bytes']).toBe(modelText.length);
  expect(upload['resolved']).toBe('primitive-library');
  expect(String(upload['refusal'])).toContain('read-only snapshot');

  const save = only(report, CASES.snapshotSave).detail ?? {};
  expect(save['source']).toBe('input');
  expect(save['name']).toBe('llama3-8b.json');
  expect(save['bytes']).toBe(Buffer.byteLength(modelText, 'utf8'));
});

test('the note records which browsers have the writable picker', async ({ page }) => {
  // "Done when … the note records which browsers have the writable picker." The record is a
  // measurement, not a sentence: `pnpm spike:static` writes it, and this holds it to its shape.
  const path = join(spike, 'engines.json');
  expect(existsSync(path), 'the cross-engine measurement has not been run').toBe(true);
  const measured = JSON.parse(readFileSync(path, 'utf8')) as {
    cases: string[];
    bases: Record<string, { base: string; script: string }>;
    engines: { engine: string; available: boolean; report?: EngineReport }[];
  };

  expect(measured.cases).toEqual(PAGE_CASES);
  expect(measured.engines.map((one) => one.engine)).toEqual(['chromium', 'firefox', 'webkitgtk']);
  for (const one of measured.engines) {
    expect(one.available, `${one.engine} was not measured`).toBe(true);
    expect(typeof one.report?.capabilities['showDirectoryPicker'], `${one.engine} has no picker row`).toBe('boolean');
  }
  // The base path is a build parameter, and the note's table says what each build named.
  expect(measured.bases['editor']?.script).toContain(`${staticBase}assets/`);

  // What this box's Chromium answers today must be what the record says it answered.
  await openPage(page);
  const here = await page.evaluate(() => (window as unknown as SpikeWindow).spike.capabilities());
  const chromium = measured.engines.find((one) => one.engine === 'chromium')?.report?.capabilities ?? {};
  expect(here['showDirectoryPicker']).toBe(chromium['showDirectoryPicker']);
  expect(here['opfs']).toBe(chromium['opfs']);

  const note = readFileSync(join(spike, 'NOTE.md'), 'utf8');
  for (const engine of ['Chromium', 'Firefox', 'WebKit']) expect(note).toContain(engine);
});

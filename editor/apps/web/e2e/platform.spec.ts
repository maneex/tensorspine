import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium, expect, test, type Page } from '@playwright/test';

import { platformUrl, stubUrl } from '../playwright.config';

/**
 * Feature 2.4 — `Platform` in a browser.
 *
 * What is held to account here is what plan §5.2 promises and D11 deploys:
 *
 *   - the File System Access path: a folder listed, read, **written in place**, refused on a stale
 *     revision, watched by polling, with the relative resolution `primitive_libraries[].base`
 *     needs;
 *   - the read-only snapshot: a folder upload and a drop, where Save downloads the document's own
 *     bytes and the folder is left alone;
 *   - the **Examples** workspace over the material `pnpm vendor` copied out of the repository;
 *   - settings in `localStorage`, drafts and remembered handles in IndexedDB — each of them
 *     surviving a reload, and each of them surviving a browser that refuses to store anything;
 *   - the **platform leak build**: the application built against the stub `Platform` carries no
 *     platform implementation at all (plan §6).
 *
 * The writable path runs against an Origin Private File System directory handle — the same
 * `FileSystemDirectoryHandle` interface a picked folder gives — because no automated browser can
 * drive the native picker. Which engines *have* that picker is feature 0.6's measurement, recorded
 * in `editor/spikes/static/engines.json`.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const dist = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'dist');

/** A corpus document, read at the source: the bytes the editor must carry unchanged. */
const MODEL = 'data/models/llama3-8b.json';
const modelText = readFileSync(join(repository, MODEL), 'utf8');

/**
 * A folder on disk laid out like `data/`, for the folder upload to be pointed at.
 *
 * Made in `beforeAll`, because this suite runs `fullyParallel`: a worker is reused across test
 * groups and `afterAll` runs once per group, so a fixture made at the module's top level — which
 * runs once per worker — is gone for the file's next test in that worker.
 */
const fixture = mkdtempSync(join(tmpdir(), 'tensorspine-2.4-'));
const workspaceDirectory = join(fixture, 'workspace');

test.beforeAll(() => {
  mkdirSync(join(workspaceDirectory, 'models'), { recursive: true });
  mkdirSync(join(workspaceDirectory, 'primitive-library'), { recursive: true });
  writeFileSync(join(workspaceDirectory, 'models', 'llama3-8b.json'), modelText, 'utf8');
  writeFileSync(
    join(workspaceDirectory, 'primitive-library', 'primitive-library.json'),
    readFileSync(join(repository, 'data/primitive-library/primitive-library.json'), 'utf8'),
    'utf8',
  );
});

test.afterAll(() => {
  rmSync(fixture, { recursive: true, force: true });
});

/** What a probe of the page answers. */
interface Outcome {
  readonly ok: boolean;
  readonly detail: Record<string, unknown>;
  readonly ms: number;
}

declare global {
  interface Window {
    platformProbe: unknown;
  }
}

/** Open the driver page and wait for it to have wired itself up. */
async function openPage(page: Page): Promise<void> {
  await page.goto(platformUrl);
  await expect(page.locator('#out')).toHaveAttribute('data-state', 'ready');
}

/** Run one probe and require it to have answered. */
async function probe(page: Page, name: string, argument?: string): Promise<Record<string, unknown>> {
  const outcome = await page.evaluate(
    ([call, one]) => {
      const probes = window.platformProbe as Record<string, (value?: string) => Promise<Outcome>>;
      return probes[call]?.(one) as Promise<Outcome>;
    },
    [name, argument] as const,
  );
  expect(outcome, `the page ran no probe named ${name}`).toBeDefined();
  expect(outcome.detail['error'] ?? 'none', `the probe ${name} refused`).toBe('none');
  expect(outcome.ok).toBe(true);
  return outcome.detail;
}

/**
 * A page in a **persistent profile**, which is what a `FileSystemDirectoryHandle` in IndexedDB
 * needs: in Playwright's default (ephemeral) context, Chromium does not refuse to read one back —
 * it kills the renderer. Feature 0.6 measured it; this is the consequence, and 2.6 inherits it.
 */
async function inPersistentProfile(body: (page: Page) => Promise<void>): Promise<void> {
  const profile = mkdtempSync(join(tmpdir(), 'tensorspine-2.4-profile-'));
  const context = await chromium.launchPersistentContext(profile, {});
  try {
    await body(await context.newPage());
  } finally {
    await context.close();
    rmSync(profile, { recursive: true, force: true });
  }
}

test('a folder the browser can write is listed, read, written in place and resolved', async ({ page }) => {
  await openPage(page);
  const detail = await probe(page, 'directory');

  expect(detail['root']).toEqual({
    kind: 'directory',
    id: 'directory:platform-directory',
    name: 'platform-directory',
    writable: true,
  });
  expect(detail['files']).toEqual(['models/llama3-8b.json', 'primitive-library/primitive-library.json']);
  expect(detail['entries']).toEqual(['directory models', 'directory primitive-library']);

  // The corpus document reaches the editor byte for byte, through the handle alone, and goes back.
  expect(detail['roundTrips']).toBe(true);
  expect(detail['writtenInPlace']).toBe(true);
  expect(detail['revisionMoved']).toBe(true);

  // The optimistic concurrency of §5.2, and a refusal that changed nothing.
  expect(String(detail['conflict'])).toContain('conflict: models/llama3-8b.json changed since it was read');
  expect(detail['unchangedAfterConflict']).toBe(true);

  // `write` never makes the directories above the file; `mkdir` does (feature 0.6's finding 5).
  expect(String(detail['typo'])).toBe('not-found: no directory modles in the workspace');
  expect(detail['made']).toEqual(['bases/mine/primitives/norm.rms.json']);
  expect(String(detail['twice'])).toContain('conflict:');
  expect(String(detail['notFound'])).toBe('not-found: no file models/nothing.json in the workspace');

  // `resolve` of `../primitive-library/` from a model path — what a document's bases mean.
  expect(detail['resolved']).toBe('primitive-library');
  expect(detail['resolvedTemplates']).toBe('models');
});

test('the polling watch reports a change made behind the workspace, and stops when unsubscribed', async ({
  page,
}) => {
  await openPage(page);
  const detail = await probe(page, 'watch');
  expect(detail['events']).toEqual([
    'changed models/llama3-8b.json',
    'added models/added.json',
    'removed models/added.json',
  ]);
  const times = detail['times'] as Record<string, number>;
  for (const kind of ['changed', 'added', 'removed']) {
    expect(typeof times[kind], `${kind} was not timed`).toBe('number');
  }
  expect(detail['quietAfterUnsubscribe']).toBe(true);
});

test('a folder upload is a read-only snapshot, and Save hands the document to the user', async ({ page }) => {
  await openPage(page);
  await page.locator('#upload').setInputFiles(workspaceDirectory);
  const detail = await probe(page, 'upload');

  expect(detail['source']).toBe('input');
  // The chosen folder's own name is stripped: the paths are the writable side's.
  expect(detail['root']).toEqual({
    kind: 'snapshot',
    id: 'upload:workspace',
    name: 'workspace',
    writable: false,
  });
  expect(detail['files']).toEqual(['models/llama3-8b.json', 'primitive-library/primitive-library.json']);
  expect(detail['bytes']).toBe(modelText.length);
  expect(detail['resolved']).toBe('primitive-library');
});

test('a snapshot Save downloads the document’s own bytes and moves nothing', async ({ page }) => {
  await openPage(page);
  const detail = await probe(page, 'snapshot');

  expect(detail['roundTrips']).toBe(true);
  // §5.2: "`write` produces a download and reports the revision as the snapshot's".
  expect(detail['revisionUnmoved']).toBe(true);
  expect(detail['untouched']).toBe(true);
  expect(detail['current']).toBe(true);
  const download = detail['download'] as { name: string; bytes: number };
  expect(download.name).toBe('llama3-8b.json');
  expect(download.bytes).toBe(Buffer.byteLength(modelText, 'utf8'));
  // The one part of §4.3 this path cannot do at all, refused in words that name the way out.
  expect(String(detail['mkdir'])).toContain('read-only: workspace is read-only');
  expect(String(detail['mkdir'])).toContain('writable directory picker');
});

test('a real folder upload is delivered to the browser as a download', async ({ page }) => {
  // The probe above builds the download without handing it over, so that an unattended run does
  // not litter the machine. That the browser really takes it is asked once, here.
  await openPage(page);
  await page.locator('#upload').setInputFiles(workspaceDirectory);
  const [download] = await Promise.all([page.waitForEvent('download'), probe(page, 'download')]);
  expect(download.suggestedFilename()).toBe('llama3-8b.json');
  expect(readFileSync(await download.path(), 'utf8')).toBe(modelText);
  // And the folder on disk is untouched: the snapshot is a copy the page was handed.
  expect(readFileSync(join(workspaceDirectory, 'models', 'llama3-8b.json'), 'utf8')).toBe(modelText);
});

test('a drop is read before the transfer is disabled, so the dropped files are the workspace', async ({
  page,
}) => {
  // A `DataTransfer` is disabled the moment the synchronous part of the `drop` handler returns.
  // `openDrop` awaits `getAsFileSystemHandle()` — the writable path, which is what Chromium can
  // give for a *directory* — so anything it must read from the transfer has to be read before
  // that await. A single file is the case: no directory handle comes back, and what is left is
  // what was read first (the review repair `910539b`).
  await openPage(page);
  await page.evaluate((text) => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([text], 'llama3-8b.json', { type: 'application/json' }));
    window.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, bubbles: true, cancelable: true }));
  }, modelText);
  await expect(page.locator('body')).toHaveAttribute('data-drop', 'snapshot');

  const detail = await probe(page, 'drop');
  expect(detail['files']).toEqual(['llama3-8b.json']);
  expect((detail['opened'] as { writable: boolean }).writable).toBe(false);
});

test('the Examples workspace opens the corpus and the reference base the build vendored', async ({ page }) => {
  await openPage(page);
  const detail = await probe(page, 'examples');

  expect(detail['root']).toEqual({ kind: 'examples', id: 'examples', name: 'Examples', writable: false });
  // Every model of the repository's corpus, by the name it has there.
  const models = readdirSync(join(repository, 'data/models')).sort((a, b) => a.localeCompare(b));
  expect(detail['models']).toEqual(models);
  expect(detail['bytes']).toBe(modelText.length);
  // The layout of `data/` is kept, so a document resolves its bases exactly as it does there.
  expect(detail['resolved']).toBe('primitive-library');
  expect(detail['manifest']).toEqual({
    root: 'data',
    models: 'data/models',
    primitive_library: 'data/primitive-library',
  });
  expect(detail['baseFiles']).toBe(
    readdirSync(join(repository, 'data/primitive-library'), { recursive: true }).filter((name) =>
      String(name).endsWith('.json'),
    ).length,
  );
  // The revision is the manifest's digest, which cannot move while the page is open.
  expect(detail['stableRevision']).toBe(true);
  expect(String(detail['revision'])).toMatch(/^[0-9a-f]{64}$/);
  // Read-only, with Save As to copy a document out (§4.3).
  expect((detail['download'] as { name: string }).name).toBe('llama3-8b.json');
  expect(detail['schemas']).toBe(
    readdirSync(join(repository, 'schemas')).filter((name) => name.endsWith('.json')).length,
  );
});

test('the reference base reads the same whichever way it is gathered', async ({ page }) => {
  // What feature 2.6 hands `loadLibrary` is a map of path to text. The workspace's own walk and
  // the vendor's bulk read must answer the same one — the listing of one against the manifest's
  // prefix arithmetic of the other.
  await openPage(page);
  const detail = await probe(page, 'base');
  expect(detail['files']).toBe(
    readdirSync(join(repository, 'data/primitive-library'), { recursive: true }).filter((name) =>
      String(name).endsWith('.json'),
    ).length,
  );
  expect(detail['sameKeys']).toBe(true);
  expect(detail['sameTexts']).toBe(true);
});

test('settings live in localStorage and come back after a reload', async ({ page }) => {
  await openPage(page);
  const first = await probe(page, 'settings');
  expect(first['persistent']).toBe(true);
  expect(first['theme']).toBe('light');
  expect(first['changed']).toEqual(['theme', 'panel.size', 'unlocked.bases', 'theme', 'panel.size']);
  expect(first['keys']).toEqual(['theme', 'unlocked.bases']);
  expect(first['raw']).toBe('"light"');

  await openPage(page);
  const second = await probe(page, 'settings', 'again');
  expect(second['theme']).toBe('light');
  expect(second['bases']).toEqual(['data/primitive-library']);
  expect(second['keys']).toEqual(['theme', 'unlocked.bases']);
});

test('settings survive a browser that refuses to store anything at all', async ({ page }) => {
  // `localStorage` is not merely empty in a private window or where site data is blocked: reading
  // the property itself throws. The store must keep working, for this session, and say that it
  // will not be remembered.
  await page.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get: () => {
        throw new DOMException('site data is blocked', 'SecurityError');
      },
    });
  });
  await openPage(page);
  const detail = await probe(page, 'settings');
  expect(detail['persistent']).toBe(false);
  expect(detail['theme']).toBe('light');
  expect(detail['bases']).toEqual(['data/primitive-library']);
  expect(detail['raw']).toBeNull();
});

test('drafts live in IndexedDB and come back after a reload', async ({ page }) => {
  await openPage(page);
  const first = await probe(page, 'drafts');
  expect(first['persistent']).toBe(true);
  expect(first['text']).toBe('{"model": "llama3-8b-edited"}\n');
  expect(first['all']).toEqual(['w2 models/other.json 2', 'w1 models/llama3-8b.json 30']);
  expect(first['mine']).toBe(1);

  await openPage(page);
  const second = await probe(page, 'drafts', 'clear');
  expect(second['text']).toBe('{"model": "llama3-8b-edited"}\n');
  expect(second['revision']).toBe('r1');
  expect(second['cleared']).toBe(0);
});

test('drafts survive a browser that refuses IndexedDB', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'indexedDB', {
      configurable: true,
      get: () => {
        throw new DOMException('site data is blocked', 'SecurityError');
      },
    });
  });
  await openPage(page);
  const detail = await probe(page, 'drafts');
  expect(detail['persistent']).toBe(false);
  expect(detail['text']).toBe('{"model": "llama3-8b-edited"}\n');
});

test('a directory handle survives IndexedDB and comes back as the same folder', async () => {
  await inPersistentProfile(async (page) => {
    await openPage(page);
    const detail = await probe(page, 'recents');
    expect(detail['persistent']).toBe(true);
    expect(detail['entries']).toBe(1);
    expect(detail['sameId']).toBe(true);
    expect(detail['name']).toBe('platform-recents');
    expect(detail['sameEntry']).toBe(true);
    // Chromium answers a grant for an OPFS handle; Firefox has no permission model at all and
    // answers `unsupported`, which the code reads as usable (feature 0.6).
    expect(['granted', 'unsupported']).toContain(detail['permission']);
  });
});

test('Open Folder opens a folder in place, remembers it, and reopens it by name', async () => {
  await inPersistentProfile(async (page) => {
    // The native picker is not drivable, so the *picker call* is what is replaced — by a handle to
    // a folder of the Origin Private File System, which answers the same interface. Everything
    // below the picker is the application's own code.
    await page.addInitScript(() => {
      Object.defineProperty(window, 'showDirectoryPicker', {
        configurable: true,
        value: async () => (await navigator.storage.getDirectory()).getDirectoryHandle('picked', { create: true }),
      });
    });
    await openPage(page);
    const detail = await probe(page, 'openFolder');

    expect((detail['opened'] as { kind: string; name: string }).kind).toBe('directory');
    expect((detail['opened'] as { name: string }).name).toBe('picked');
    expect((detail['opened'] as { writable: boolean }).writable).toBe(true);
    expect(detail['files']).toEqual(['models/llama3-8b.json']);

    // Remembered by name, reopened by id, and the same folder under the same identity.
    expect(detail['remembers']).toBe(true);
    expect(detail['recent']).toEqual(['picked']);
    expect(detail['sameIdentity']).toBe(true);
    expect(detail['readBack']).toBe(modelText.length);
    // …and the last folder reopened with no gesture at all, the grant still being held (D11).
    expect(detail['grantedName']).toBe('picked');
    expect(detail['changes']).toEqual(['directory', 'directory', 'directory']);
    expect(detail['forgotten']).toBe(0);
  });
});

test('the platform is assembled from every interface of §5.2', async ({ page }) => {
  await openPage(page);
  const detail = await probe(page, 'platform');

  expect(detail['describes']).toBe('browser');
  // Before anything is opened the editor holds a workspace all the same, and it saves.
  expect(detail['empty']).toEqual({ kind: 'empty', id: 'unopened', name: 'no workspace', writable: false });
  expect(detail['savedWithNothingOpen']).toBe('absent');
  expect((detail['workspace'] as { kind: string }).kind).toBe('examples');
  expect(detail['writablePicker']).toBe(true);
  // Feature 4.1 fills the checkpoints; this feature declares the interface and ships none.
  expect(detail['checkpoints']).toBe(0);
  // `NoAuth`: no session, and a sign-in refused in the deployment's own words (Q8).
  expect(detail['session']).toBeNull();
  expect(String(detail['signIn'])).toBe('unsupported: this deployment has no accounts: everything the editor computes runs in your browser');
  expect(detail['settings']).toBe(true);
  expect(detail['drafts']).toBe(true);
  expect(detail['remembers']).toBe(true);
  expect((detail['download'] as { name: string }).name).toBe('note.txt');
});

test('the application builds against the stub platform, and that page carries no platform at all', async ({
  page,
}) => {
  // Plan §6's "Platform leak build", D11: "CI builds the web app against a stub platform from
  // phase 2 on, so a platform leak is a build failure". The build is the one this suite runs
  // against; what is asked here is that the renderer really reached for nothing else, which is a
  // question about the **emitted chunks** and not about the source.
  await page.goto(stubUrl);
  const root = page.locator('#root');
  await expect(root).toHaveAttribute('data-state', 'ready');
  await expect(root).toHaveAttribute('data-platform', 'stub');
  await expect(root).toHaveAttribute('data-workspace', 'memory');
  await expect(root).toContainText('@tensorspine/lang');

  const html = readFileSync(join(dist, 'stub.html'), 'utf8');
  const chunks = [...html.matchAll(/(?:src|href)="\/(assets\/[^"]+\.js)"/g)].map((match) => match[1] ?? '');
  expect(chunks.length).toBeGreaterThan(1);
  const source = chunks.map((chunk) => readFileSync(join(dist, chunk), 'utf8')).join('\n');
  for (const api of [
    'showDirectoryPicker',
    'getDirectoryHandle',
    'createWritable',
    'webkitGetAsEntry',
    'localStorage',
    'indexedDB',
    'createObjectURL',
  ]) {
    expect(source, `the stub build carries ${api}`).not.toContain(api);
  }

  // And the application's own page does carry it, so the check above is not vacuous.
  const application = readFileSync(join(dist, 'index.html'), 'utf8');
  const carried = [...application.matchAll(/(?:src|href)="\/(assets\/[^"]+\.js)"/g)]
    .map((match) => readFileSync(join(dist, match[1] ?? ''), 'utf8'))
    .join('\n');
  expect(carried).toContain('showDirectoryPicker');
});

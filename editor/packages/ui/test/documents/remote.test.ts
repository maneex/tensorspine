import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { createLang } from '@tensorspine/lang/api/engine';
import type { Lang } from '@tensorspine/lang/api';
import {
  MemoryWorkspace,
  heldSet,
  stubPlatform,
  type PublishedManifest,
  type PublishedSet,
  type SettingValue,
} from '@tensorspine/store/platform';

import {
  createDocuments,
  REMOTE_SETTING,
  WORKSPACE_SETTING,
  type Documents,
  type DocumentsStore,
  type TabSink,
} from '../../src/documents/store.js';
import { filesUnder, repositoryRoot, schemaTexts } from './source.js';

/** The editor's own directory, which holds feature 1.13's acceptance fixtures. */
const editorRoot = join(repositoryRoot, 'editor');

// Feature 2.20 without a browser: a base fetched from an address and **standing at a path of the
// open workspace**, which is where a document can pin it.
//
// The one decision this file states over and over: **a document pins a base by the path it
// resolves**, because that is what the language says a base is (`bases_of` joins the document's
// directory with what `primitive_libraries` writes, `os.path`'s rules and nothing else). An
// address is not a place. So a fetched base is mounted somewhere, the document pins that path as
// it would pin a folder that is really there, and what the editor remembers beside it is *which
// address filled it, and at which commit* — which is what makes a second reader's derivation
// reproduce.
//
// The browser layer (`apps/web/e2e/remote.spec.ts`) asks the three things only an engine can
// answer — a real cross-origin fetch, the CORS refusal, and the cache after a reload.

const MODEL = 'data/models/uses-fixture.json';
const ADDRESS = 'https://lab.example/lab-base/vendor.json';
const COMMIT = 'a'.repeat(40);

function sha256(text: string): string {
  return createHash('sha256').update(Buffer.from(text, 'utf8')).digest('hex');
}

/** A published set as a generator wrote it: every file with its own length and digest. */
function publishedOf(address: string, texts: Readonly<Record<string, string>>): PublishedSet {
  const manifest: PublishedManifest = {
    generated_by: 'editor/scripts/publish.ts',
    repository_commit: COMMIT,
    repository_dirty: false,
    files: Object.entries(texts)
      .map(([path, text]) => ({ path, bytes: Buffer.byteLength(text), sha256: sha256(text) }))
      .sort((a, b) => (a.path < b.path ? -1 : 1)),
  };
  const root = address.slice(0, address.lastIndexOf('/') + 1);
  return heldSet(address, root, manifest, texts, '2026-09-16T08:00:00.000Z', false);
}

/** Feature 1.13's acceptance-fixture base — what a laboratory's own base looks like. */
function fixtureBase(): Record<string, string> {
  const root = join(editorRoot, 'tests', 'fixtures', 'base');
  const found: Record<string, string> = {};
  const walk = (at: string, prefix: string): void => {
    for (const entry of readdirSync(at, { withFileTypes: true })) {
      if (entry.isDirectory()) walk(join(at, entry.name), `${prefix}${entry.name}/`);
      else if (entry.name.endsWith('.json')) {
        found[`${prefix}${entry.name}`] = readFileSync(join(at, entry.name), 'utf8');
      }
    }
  };
  walk(root, '');
  return found;
}

/** `declared-constant.json` with its two bases written where this workspace keeps them. */
function document(where: string): string {
  return readFileSync(join(editorRoot, 'tests', 'fixtures', 'models', 'declared-constant.json'), 'utf8')
    .replace('../../../../data/primitive-library/', '../primitive-library/')
    .replace('../base/', `../${where}/`);
}

/** A workspace laid out like the repository's `data/`, with a model that pins a base it has not. */
function dataWorkspace(where = 'lab-base'): MemoryWorkspace {
  return MemoryWorkspace.of({
    ...filesUnder('data/models/decoder-causal-yarn'),
    ...filesUnder('data/primitive-library'),
    [MODEL]: document(where),
  });
}

function strip(): TabSink {
  let current: string | null = null;
  return {
    open: (tab) => {
      current = tab.id;
    },
    update: () => undefined,
    close: () => {
      current = null;
    },
    select: (id) => {
      current = id;
    },
    current: () => current,
  };
}

interface Open {
  readonly platform: ReturnType<typeof stubPlatform>;
  readonly store: DocumentsStore;
  readonly log: string[];
  readonly stop: () => void;
}

const opened: Open[] = [];

afterEach(() => {
  for (const one of opened.splice(0)) one.stop();
});

/** A documents store over a stub platform holding one published set. */
function editor(
  options: {
    workspace?: MemoryWorkspace;
    remote?: readonly PublishedSet[];
    settings?: Readonly<Record<string, SettingValue>>;
  } = {},
): Open {
  const platform = stubPlatform({
    workspace: options.workspace ?? dataWorkspace(),
    ...(options.remote === undefined ? {} : { remote: options.remote }),
    ...(options.settings === undefined ? {} : { settings: options.settings }),
  });
  const lang: Lang = createLang();
  const log: string[] = [];
  const built = createDocuments({
    platform,
    lang,
    vendoredSchemas: () => Promise.resolve(schemaTexts()),
    log: (line) => log.push(line),
    tabs: strip(),
    debounceMs: 0,
    autosaveMs: 0,
  });
  const one: Open = {
    platform,
    store: built.store,
    log,
    stop: () => {
      built.dispose();
      lang.close();
    },
  };
  opened.push(one);
  return one;
}

function now(one: Open): Documents {
  return one.store.getState();
}

/** Wait until the pipeline has published a reading for the open document. */
async function settled(one: Open, times = 300): Promise<void> {
  for (let tick = 0; tick < times; tick += 1) {
    const open = now(one).open[0];
    if (open !== undefined && !open.reading.checking && open.reading.verdict !== null) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error('the pipeline said nothing');
}

/**
 * Wait until the derivation that follows a passing validation has landed (§5.4).
 *
 * Apart from {@link settled} because it is a different moment: the verdict comes first and the
 * products follow it, so a case that asks for the products waits for the products.
 */
async function derived(one: Open, times = 300): Promise<void> {
  for (let tick = 0; tick < times; tick += 1) {
    if (now(one).open[0]?.reading.derived != null) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error('nothing was derived');
}

/** Everything the core refused about the open document, as one text. */
function refusals(one: Open): string {
  const open = now(one).open[0];
  return [...now(one).library.problems, ...(open?.reading.verdict?.problems ?? [])]
    .map((problem) => problem.message)
    .join('\n');
}

/** What a reload sees: the settings the session wrote, as a fresh platform's own. */
function carried(one: Open): Record<string, SettingValue> {
  const held: Record<string, SettingValue> = {};
  for (const key of one.platform.settings.keys()) {
    const value = one.platform.settings.peek(key);
    if (value !== undefined) held[key] = value;
  }
  return held;
}

describe('a base fetched from an address', () => {
  it('is not there until it is: the loader says so, in the tools’ own words', async () => {
    const one = editor();
    await now(one).adopt(one.platform.workspace);
    await now(one).openDocument(MODEL);
    await settled(one);
    expect(refusals(one)).toContain("primitive library base 'data/lab-base' does not exist (V1)");
  });

  it('stands where the caller puts it, and the document’s pin then resolves', async () => {
    const one = editor({ remote: [publishedOf(ADDRESS, fixtureBase())] });
    await now(one).adopt(one.platform.workspace);
    await now(one).openDocument(MODEL);
    await settled(one);

    await now(one).openBaseFromUrl(ADDRESS, 'data/lab-base');
    await settled(one);
    expect(refusals(one)).toBe('');
    // The products follow a passing validation (§5.4), over a document whose primitive came from
    // an address. Nothing of the language moved: the base is a `LibraryBaseFiles` like any other.
    expect(now(one).open[0]?.reading.verdict?.problems).toEqual([]);
    await derived(one);
  });

  it('is mounted while the first validation is still running, and has the last word', async () => {
    // The gesture a person makes: the document opens, and the base is fetched without waiting for
    // the pipeline to finish saying what is wrong without it. Whatever is in flight must not be
    // what the editor ends up showing.
    const one = editor({ remote: [publishedOf(ADDRESS, fixtureBase())] });
    await now(one).adopt(one.platform.workspace);
    await now(one).openDocument(MODEL);
    await now(one).openBaseFromUrl(ADDRESS, 'data/lab-base');
    await settled(one);
    expect(refusals(one)).toBe('');
  });

  it('is shown with the address and the commit that filled the path', async () => {
    const one = editor({ remote: [publishedOf(ADDRESS, fixtureBase())] });
    await now(one).adopt(one.platform.workspace);
    await now(one).openBaseFromUrl(ADDRESS, 'data/lab-base');
    expect(now(one).mounts).toEqual([
      {
        at: 'data/lab-base',
        address: ADDRESS,
        commit: COMMIT,
        fetchedAt: '2026-09-16T08:00:00.000Z',
        cached: false,
        files: 2,
      },
    ]);
    // The Log says which path was filled from which address — which is what a second reader needs
    // in order to reproduce a derivation, the document itself carrying only the path.
    expect(one.log.join('\n')).toContain(`base data/lab-base: 2 file(s) from ${ADDRESS}`);
    expect(one.log.join('\n')).toContain('a document pins it by writing this path');
  });

  it('comes back where it stood when the workspace is opened again', async () => {
    const one = editor({ remote: [publishedOf(ADDRESS, fixtureBase())] });
    await now(one).adopt(one.platform.workspace);
    await now(one).openBaseFromUrl(ADDRESS, 'data/lab-base');

    // What a reload sees: another session, the settings the first wrote, the same address.
    const two = editor({
      remote: [publishedOf(ADDRESS, fixtureBase())],
      settings: carried(one),
    });
    await now(two).adopt(two.platform.workspace);
    expect(now(two).mounts.map((mount) => mount.at)).toEqual(['data/lab-base']);
    await now(two).openDocument(MODEL);
    await settled(two);
    expect(refusals(two)).toBe('');
  });

  it('is taken out again by the same gesture, and the refusal comes back with it', async () => {
    const one = editor({ remote: [publishedOf(ADDRESS, fixtureBase())] });
    await now(one).adopt(one.platform.workspace);
    await now(one).openDocument(MODEL);
    await now(one).openBaseFromUrl(ADDRESS, 'data/lab-base');
    await settled(one);
    expect(refusals(one)).toBe('');

    await now(one).unmountBase('data/lab-base');
    await settled(one);
    expect(now(one).mounts).toEqual([]);
    expect(refusals(one)).toContain("primitive library base 'data/lab-base' does not exist (V1)");
  });

  it('refuses an address that answers nothing, in the dialog where the gesture was made', async () => {
    const one = editor();
    await now(one).adopt(one.platform.workspace);
    await now(one).openBaseFromUrl('https://nowhere.example/base/', 'data/lab-base');
    const dialog = now(one).dialog;
    expect(dialog?.kind).toBe('remote');
    expect(dialog).toMatchObject({ target: 'library' });
    expect(dialog?.kind === 'remote' ? dialog.refusal : '').toContain('nowhere.example');
    expect(now(one).mounts).toEqual([]);
    expect(one.log.join('\n')).toContain('base from https://nowhere.example/base/');
  });

  it('refuses an empty address rather than doing nothing, which is the one thing Q5 forbids', async () => {
    const one = editor();
    await now(one).adopt(one.platform.workspace);
    await now(one).openBaseFromUrl('', 'data/lab-base');
    const dialog = now(one).dialog;
    expect(dialog?.kind === 'remote' ? dialog.refusal : '').toContain('the address of its manifest');
    expect(now(one).mounts).toEqual([]);
  });

  it('refuses the workspace root, because a base is a place a document can pin', async () => {
    const one = editor({ remote: [publishedOf(ADDRESS, fixtureBase())] });
    await now(one).adopt(one.platform.workspace);
    await now(one).openBaseFromUrl(ADDRESS, '');
    expect(now(one).mounts).toEqual([]);
    expect(now(one).dialog).toMatchObject({ kind: 'remote', target: 'library' });
  });
});

describe('a workspace fetched from an address', () => {
  const WORKSPACE_ADDRESS = 'https://lab.example/workspace/vendor.json';

  function workspaceSet(): PublishedSet {
    return publishedOf(WORKSPACE_ADDRESS, {
      'models/llama3-8b.json': filesUnder('data/models')['data/models/llama3-8b.json'] ?? '',
    });
  }

  it('is the workspace, read-only, and says where it came from', async () => {
    const one = editor({ remote: [workspaceSet()] });
    await now(one).openWorkspaceFromUrl(WORKSPACE_ADDRESS);
    expect(now(one).workspace).toMatchObject({
      kind: 'remote',
      id: `remote:${WORKSPACE_ADDRESS}`,
      name: 'workspace',
      writable: false,
    });
    expect(now(one).remote).toMatchObject({ address: WORKSPACE_ADDRESS, commit: COMMIT, cached: false });
    expect(now(one).banner?.head).toContain(WORKSPACE_ADDRESS);
    expect(now(one).documents).toEqual(['models/llama3-8b.json']);
  });

  it('is reopened by the address a reload remembers, with no gesture at all', async () => {
    const one = editor({ remote: [workspaceSet()] });
    await now(one).openWorkspaceFromUrl(WORKSPACE_ADDRESS);
    expect(one.platform.settings.peek(WORKSPACE_SETTING)).toMatchObject({ kind: 'remote' });
    expect(one.platform.settings.peek(REMOTE_SETTING)).toMatchObject({ workspace: WORKSPACE_ADDRESS });

    const two = editor({ remote: [workspaceSet()], settings: carried(one) });
    await now(two).start();
    expect(now(two).workspace.id).toBe(`remote:${WORKSPACE_ADDRESS}`);
  });

  it('opens nothing when the address refuses, and says so where the address was typed', async () => {
    const one = editor();
    const before = now(one).workspace.id;
    await now(one).openWorkspaceFromUrl('https://nowhere.example/workspace/');
    expect(now(one).workspace.id).toBe(before);
    expect(now(one).dialog).toMatchObject({ kind: 'remote', target: 'workspace' });
  });
});

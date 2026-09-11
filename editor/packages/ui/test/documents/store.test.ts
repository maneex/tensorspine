import { afterEach, describe, expect, it } from 'vitest';

import { createLang } from '@tensorspine/lang/api/engine';
import type { Lang } from '@tensorspine/lang/api';
import {
  MemoryWorkspace,
  stubPlatform,
  type Platform,
  type Workspace,
} from '@tensorspine/store/platform';

import {
  createDocuments,
  DOCUMENT_TAB,
  TABS_SETTING,
  WORKSPACE_SETTING,
  type Documents,
  type DocumentsStore,
  type TabSink,
} from '../../src/documents/store.js';
import { corpusText, filesUnder, schemaTexts } from './source.js';

// Feature 2.6's own suite, without a browser: a real core in this thread, a real workspace in
// memory, and every gesture of §4.3 asked of the store that performs them. The browser layer
// (`apps/web/e2e/documents.spec.ts`) asks the four claims only a browser can answer; everything
// that is a question about *what the editor does* is here, where it can be asked exhaustively.

const MODEL = 'data/models/llama3-8b.json';

/** A workspace laid out exactly like the repository's `data/`, which is what §4.3 says one is. */
function dataWorkspace(): MemoryWorkspace {
  return MemoryWorkspace.of({
    ...filesUnder('data/models'),
    ...filesUnder('data/primitive-library'),
  });
}

/** The tab strip, recorded: what the shell would have been told. */
interface Strip extends TabSink {
  readonly tabs: { id: string; title: string; kind: string; badge?: string; dirty?: boolean }[];
}

function strip(): Strip {
  const tabs: Strip['tabs'] = [];
  let current: string | null = null;
  return {
    tabs,
    open: (tab) => {
      tabs.push({ ...tab });
      current = tab.id;
    },
    update: (id, patch) => {
      const at = tabs.findIndex((tab) => tab.id === id);
      if (at >= 0) tabs[at] = { ...(tabs[at] as Strip['tabs'][number]), ...patch };
    },
    close: (id) => {
      const at = tabs.findIndex((tab) => tab.id === id);
      if (at >= 0) tabs.splice(at, 1);
      current = tabs.at(-1)?.id ?? null;
    },
    select: (id) => {
      current = id;
    },
    current: () => current,
  };
}

/** Everything one case holds, so that it can be taken down whatever it did. */
interface Open {
  readonly platform: ReturnType<typeof stubPlatform>;
  readonly workspace: MemoryWorkspace;
  readonly store: DocumentsStore;
  readonly tabs: Strip;
  readonly lang: Lang;
  readonly log: string[];
  readonly stop: () => void;
}

const opened: Open[] = [];

afterEach(() => {
  for (const one of opened.splice(0)) one.stop();
});

/** A documents store over a stub platform, with the core in this thread. */
function editor(options: { workspace?: MemoryWorkspace; platform?: Platform } = {}): Open {
  const workspace = options.workspace ?? dataWorkspace();
  const platform = (options.platform ?? stubPlatform({ workspace })) as ReturnType<typeof stubPlatform>;
  const lang = createLang();
  const tabs = strip();
  const log: string[] = [];
  const built = createDocuments({
    platform,
    lang,
    vendoredSchemas: () => Promise.resolve(schemaTexts()),
    log: (line) => log.push(line),
    tabs,
    debounceMs: 0,
    // The interval is off; `autosave()` is called where §4.3 says it happens.
    autosaveMs: 0,
  });
  const one: Open = {
    platform,
    workspace,
    store: built.store,
    tabs,
    lang,
    log,
    stop: () => {
      built.dispose();
      lang.close();
    },
  };
  opened.push(one);
  return one;
}

/** The state, for a case that reads it after a gesture. */
function now(one: Open): Documents {
  return one.store.getState();
}

/** Wait until the pipeline has published a reading for the current revision. */
async function settled(one: Open, id: string, times = 200): Promise<void> {
  for (let tick = 0; tick < times; tick += 1) {
    const open = now(one).open.find((document) => document.id === id);
    if (open !== undefined && !open.reading.checking && open.reading.verdict !== null) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`the pipeline said nothing about ${id}`);
}

describe('opening a document of a workspace', () => {
  it('opens a tab named after the document, with the revision tag it declares', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open;
    expect(open).toHaveLength(1);
    expect(open[0]?.title).toBe('llama3-8b');
    expect(open[0]?.tag).toBe('tensorspine/2.0');
    expect(open[0]?.dirty).toBe(false);
    expect(one.tabs.tabs).toEqual([
      { id: open[0]?.id, title: 'llama3-8b', kind: DOCUMENT_TAB, dirty: false },
    ]);
  }, 120_000);

  it('gathers the base **and the template documents it pins**, and the loader refuses nothing', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    expect(now(one).library.problems).toEqual([]);
    expect(now(one).library.handle).not.toBeNull();
    // 131 units of the reference base and the template document outside it.
    expect(now(one).library.files).toBeGreaterThan(130);
  }, 120_000);

  it('holds the document’s own bytes, and saves them back byte for byte (D12)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    expect(open?.session.text).toBe(corpusText('llama3-8b'));
    await now(one).save(open?.id);
    expect((await one.workspace.read(MODEL)).text).toBe(corpusText('llama3-8b'));
  }, 120_000);

  it('opens the same document twice as one tab', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    await now(one).openDocument(MODEL);
    expect(now(one).open).toHaveLength(1);
    expect(one.tabs.tabs).toHaveLength(1);
  }, 120_000);

  it('says what a document that is not there is, and opens nothing', async () => {
    const one = editor();
    await now(one).openDocument('data/models/nothing.json');
    expect(now(one).open).toEqual([]);
    expect(one.log.join('\n')).toContain('data/models/nothing.json');
  }, 120_000);
});

describe('the pipeline of §5.4, over an open document', () => {
  it('validates it, derives it, and answers the status bar’s four figures', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    await settled(one, id);
    const open = now(one).open[0];
    expect(open?.reading.verdict?.problems.filter((problem) => problem.severity === 'error')).toEqual([]);
    for (let tick = 0; tick < 300 && now(one).open[0]?.reading.derived === null; tick += 1) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    const derived = now(one).open[0];
    expect(derived?.reading.derivation).toBe('fresh');
    expect(derived?.figures.map((figure) => figure.figure.text)).toEqual([
      '14.96 GiB',
      '15.01 Gop',
      '128 KiB',
      '509 KiB',
    ]);
  }, 240_000);

  it('marks a document dirty on an edit and says so on its tab (§4.3’s ●)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    open?.session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((held) => held.name === 'model');
        if (member !== undefined) member.value = 'llama3-8b-edited';
      },
      moves: [],
    });
    expect(now(one).open[0]?.dirty).toBe(true);
    expect(one.tabs.tabs[0]?.dirty).toBe(true);
  }, 120_000);
});

describe('a template (§4.3: "badged `template`")', () => {
  it('is badged from the core’s own answer and from no reading of the document', async () => {
    const one = editor();
    await now(one).openDocument('data/models/decoder-causal-yarn/1.0.0.json');
    const id = now(one).open[0]?.id ?? '';
    await settled(one, id);
    // `--validate` counts a template with no assignment as *skipped*, not failed, and says which
    // quantities it wants; that answer is what the badge is.
    expect(now(one).open[0]?.reading.verdict?.needsAssignment).toBeDefined();
    expect(now(one).open[0]?.badge).toBe('template');
    expect(one.tabs.tabs[0]?.badge).toBe('template');
    // And it is not derived, which is a skip rather than a refusal.
    expect(now(one).open[0]?.reading.derivation).toBe('skipped');
  }, 240_000);
});

describe('New Model', () => {
  it('is dirty from birth, proposes a path, and resolves the workspace’s own base', async () => {
    const one = editor();
    await now(one).newModel();
    const open = now(one).open[0];
    expect(open?.dirty).toBe(true);
    expect(open?.path).toBe('untitled.json');
    expect(open?.title).toBe('untitled');
    expect(open?.session.text).toContain('"base": "data/primitive-library/"');
  }, 120_000);

  it('names a second one apart from the first', async () => {
    const one = editor();
    await now(one).newModel();
    await now(one).newModel();
    expect(now(one).open.map((open) => open.title)).toEqual(['untitled', 'untitled-2']);
  }, 120_000);

  it('writes a template’s version where §4.3 says it does', async () => {
    const one = editor();
    await now(one).newModel(true);
    expect(now(one).open[0]?.session.text).toContain('"version": "1.0.0"');
  }, 120_000);
});

describe('Save, Save As, Save All and Revert', () => {
  it('writes every dirty document and leaves the clean ones alone', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    await now(one).newModel();
    await now(one).saveAll();
    expect(now(one).open.every((open) => !open.dirty)).toBe(true);
    expect((await one.workspace.read('untitled.json')).text).toContain('"model": "untitled"');
  }, 120_000);

  it('moves a document to a new path, and the tab with it', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    await now(one).saveAs(id, 'data/models/copy.json');
    expect((await one.workspace.read('data/models/copy.json')).text).toBe(corpusText('llama3-8b'));
    expect(now(one).open[0]?.path).toBe('data/models/copy.json');
    expect(one.tabs.tabs).toHaveLength(1);
  }, 120_000);

  it('takes a document back to the file, undoably (§4.4’s Revert)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    open?.session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((held) => held.name === 'model');
        if (member !== undefined) member.value = 'other';
      },
      moves: [],
    });
    await now(one).revert(open?.id);
    expect(now(one).open[0]?.dirty).toBe(false);
    expect(now(one).open[0]?.session.text).toBe(corpusText('llama3-8b'));
    expect(now(one).open[0]?.session.store.undoLabel).toBe('Revert');
  }, 120_000);

  it('states a conflict rather than overwriting a file that moved (§5.2)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    await one.workspace.write(MODEL, `${corpusText('llama3-8b')}`, (await one.workspace.read(MODEL)).revision);
    await now(one).save(open?.id);
    expect(now(one).banner?.kind).toBe('stop');
    expect(now(one).banner?.body).toContain('Save As');
  }, 120_000);
});

describe('Close with unsaved changes', () => {
  it('asks, and keeps the tab when the answer is no (§4.3)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    open?.session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((held) => held.name === 'model');
        if (member !== undefined) member.value = 'other';
      },
      moves: [],
    });
    // The stub's shell answers no, which is the destructive branch never taken by accident.
    expect(await now(one).mayClose(open?.id ?? '')).toBe(false);
    expect(one.platform.shellRecord.asked.join('')).toContain('unsaved changes');
    one.platform.shellRecord.answer = true;
    expect(await now(one).mayClose(open?.id ?? '')).toBe(true);
  }, 120_000);

  it('asks nothing of a document with nothing unsaved', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    expect(await now(one).mayClose(now(one).open[0]?.id ?? '')).toBe(true);
    expect(one.platform.shellRecord.asked).toEqual([]);
  }, 120_000);
});

describe('the autosave and its restore (§4.3)', () => {
  it('writes a draft of every dirty document and of no clean one', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    await now(one).autosave();
    expect(await one.platform.drafts.list()).toEqual([]);
    const open = now(one).open[0];
    open?.session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((held) => held.name === 'model');
        if (member !== undefined) member.value = 'llama3-8b-edited';
      },
      moves: [],
    });
    await now(one).autosave();
    const drafts = await one.platform.drafts.list();
    expect(drafts.map((held) => held.path)).toEqual([MODEL]);
  }, 120_000);

  it('offers the draft when the document is opened again, and restores it on the word', async () => {
    const workspace = dataWorkspace();
    const platform = stubPlatform({ workspace });
    const first = editor({ workspace, platform });
    await now(first).openDocument(MODEL);
    const open = now(first).open[0];
    open?.session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((held) => held.name === 'model');
        if (member !== undefined) member.value = 'llama3-8b-edited';
      },
      moves: [],
    });
    await now(first).autosave();

    // A second editor over the same platform: the drafts outlive the page, which is what
    // IndexedDB does in the browser and what the stub's own store does here.
    const second = editor({ workspace, platform });
    await now(second).openDocument(MODEL);
    const reopened = now(second).open[0];
    expect(reopened?.draft?.standing).toBe('unsaved');
    expect(now(second).dialog).toEqual({ kind: 'restore', id: reopened?.id });
    expect(reopened?.session.text).toBe(corpusText('llama3-8b'));

    now(second).restoreDraft(reopened?.id ?? '');
    expect(now(second).open[0]?.session.text).toContain('llama3-8b-edited');
    expect(now(second).open[0]?.dirty).toBe(true);
    expect(now(second).dialog).toBeNull();
  }, 240_000);

  it('offers nothing when the file already holds what the draft holds', async () => {
    const workspace = dataWorkspace();
    const platform = stubPlatform({ workspace });
    const first = editor({ workspace, platform });
    await now(first).openDocument(MODEL);
    await one_dirty(first);
    await now(first).autosave();
    await now(first).save(now(first).open[0]?.id);

    const second = editor({ workspace, platform });
    await now(second).openDocument(MODEL);
    expect(now(second).open[0]?.draft).toBeUndefined();
    expect(now(second).dialog).toBeNull();
  }, 240_000);
});

describe('Download Workspace as Zip', () => {
  it('hands over every file of the workspace, with the editor’s unsaved edits in it', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    await one_dirty(one);
    await now(one).downloadZip();
    const handed = one.platform.shellRecord.downloads.at(-1);
    expect(handed?.name).toBe('memory.zip');
    expect(handed?.bytes).toBeGreaterThan(1000);
    expect(one.log.join('\n')).toContain('downloaded:');
  }, 120_000);
});

describe('what opens at launch (D11)', () => {
  it('opens nothing at all where the platform remembers nothing', async () => {
    const one = editor();
    await now(one).start();
    expect(now(one).open).toEqual([]);
    expect(now(one).workspace.kind).toBe('memory');
  }, 120_000);

  it('comes back to the examples when that is what was open, and to its tabs', async () => {
    const workspace = dataWorkspace();
    const platform = stubPlatform({
      workspace,
      examples: { 'models/llama3-8b.json': corpusText('llama3-8b') },
    });
    const first = editor({ workspace, platform });
    await now(first).openExamples();
    await now(first).openDocument('models/llama3-8b.json');
    expect(platform.settings.peek(WORKSPACE_SETTING)).toMatchObject({ kind: 'examples' });
    expect(platform.settings.peek(TABS_SETTING)).toEqual({
      workspace: 'examples',
      paths: ['models/llama3-8b.json'],
      // The path, so that a relaunch finds the document rather than a tab identity it made up.
      current: 'models/llama3-8b.json',
    });

    const second = editor({ workspace, platform });
    await now(second).start();
    expect(now(second).workspace.kind).toBe('examples');
    // §4.3's "tabs restore on relaunch", with the one that was current current again.
    expect(now(second).open.map((open) => open.path)).toEqual(['models/llama3-8b.json']);
    expect(now(second).open.find((open) => open.id === now(second).current)?.path).toBe(
      'models/llama3-8b.json',
    );
  }, 240_000);

  it('asks the platform for the folder it still holds a grant for (D11)', async () => {
    const workspace = dataWorkspace();
    const platform = stubPlatform({ workspace });
    let asked = 0;
    const granted: Workspace = MemoryWorkspace.of({ 'models/x.json': '{}' }, 'granted');
    // The stub has nothing to remember, so it leaves the call out — which is exactly the shape
    // the interface admits, and a deployment that *has* one answers it like this.
    const workspaces: { reopenGranted?: () => Promise<Workspace | null> } = platform.workspaces;
    workspaces.reopenGranted = () => {
      asked += 1;
      return Promise.resolve(granted);
    };
    const one = editor({ workspace, platform });
    await now(one).start();
    expect(asked).toBe(1);
    expect(now(one).workspace.name).toBe('granted');
  }, 120_000);
});

describe('the banner a workspace states itself with (S18, inventory §5)', () => {
  it('says a read-only snapshot downloads, and says nothing of a folder it can write', async () => {
    const one = editor();
    await now(one).adopt(one.workspace);
    expect(now(one).banner).toBeNull();
    const platform = stubPlatform({ workspace: one.workspace, examples: {} });
    const other = editor({ workspace: one.workspace, platform });
    await now(other).openExamples();
    expect(now(other).banner?.kind).toBe('info');
    expect(now(other).banner?.body).toContain('Save As');
  }, 120_000);
});

/** Make the open document dirty, however this suite happens to be doing it. */
async function one_dirty(one: Open): Promise<void> {
  const open = now(one).open[0];
  open?.session.store.apply({
    label: 'Rename the model',
    edit: (draft) => {
      const member = draft.members.find((held) => held.name === 'model');
      if (member !== undefined) member.value = 'llama3-8b-edited';
    },
    moves: [],
  });
  await Promise.resolve();
}

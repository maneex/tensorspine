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
  documentTab,
  DOCUMENT_TAB,
  EXPANDED_SUFFIX,
  EXPANDED_TAB,
  NO_EMITTED_VIEW,
  SOURCE_SUFFIX,
  SOURCE_TAB,
  sourceReading,
  sourceStanding,
  TABS_SETTING,
  WORKSPACE_SETTING,
  type Documents,
  type DocumentsStore,
  type OpenDocument,
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

describe('the expanded graph’s tab and its filters (§4.9, feature 2.16)', () => {
  it('opens a tab of its own, which goes when the document goes', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    now(one).openExpanded(id);
    expect(one.tabs.tabs.map((tab) => tab.kind)).toEqual([DOCUMENT_TAB, EXPANDED_TAB]);
    expect(one.tabs.tabs[1]?.title).toBe('llama3-8b › expanded graph');
    expect(one.tabs.tabs[1]?.id).toBe(`${id}${EXPANDED_SUFFIX}`);
    // Asking twice asks the strip twice, and one tab is the *shell*'s answer (`openTab` keeps the
    // one it has) — as it is for a drill-in. The browser layer is where that is asserted.
    // A command with no argument is about the document behind the tab.
    expect(documentTab(`${id}${EXPANDED_SUFFIX}`)).toBe(id);
    // And a view of a document is not a document: it closes without asking.
    expect(await now(one).mayClose(`${id}${EXPANDED_SUFFIX}`)).toBe(true);
    // Letting the document go takes its views with it (the shell is what closes the document's
    // own tab, which is why closing it is what called this).
    now(one).closeDocument(id);
    expect(one.tabs.tabs.map((tab) => tab.kind)).toEqual([DOCUMENT_TAB]);
  }, 120_000);

  it('records the two filters §5.5 puts in the sidecar, and keeps the other three in session', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    const held = (): OpenDocument | undefined => now(one).open.find((open) => open.id === id);
    expect(held()?.emitted).toEqual(NO_EMITTED_VIEW);

    now(one).setEmittedView({ families: ['decoder'] }, id);
    now(one).setEmittedView({ indices: { layer: { from: 0, to: 3 } } }, id);
    expect(held()?.session.layout.layout.expanded_view).toEqual({
      filters: { families: ['decoder'], indices: { layer: { from: 0, to: 3 } } },
    });

    // The other three are session state: the sidecar's schema declares no member for them, so
    // nothing is written and the file does not move.
    const revision = held()?.session.layout.revision ?? 0;
    now(one).setEmittedView({ search: 'attn', primitives: ['norm.rms'], node: 'embed' }, id);
    expect(held()?.emitted.search).toBe('attn');
    expect(held()?.emitted.primitives).toEqual(['norm.rms']);
    expect(held()?.emitted.node).toBe('embed');
    expect(held()?.session.layout.revision).toBe(revision);
  }, 120_000);

  it('seeds the filters from the sidecar a document was opened with', async () => {
    const workspace = dataWorkspace();
    await workspace.write(
      'data/models/llama3-8b.layout.json',
      `${JSON.stringify(
        {
          schema: 'tensorspine-editor-layout/1',
          positions: {},
          collapsed: [],
          expanded_view: { filters: { families: ['decoder'], indices: { layer: { to: 5 } } } },
        },
        null,
        2,
      )}\n`,
    );
    const one = editor({ workspace });
    await now(one).openDocument(MODEL);
    expect(now(one).open[0]?.emitted.families).toEqual(['decoder']);
    expect(now(one).open[0]?.emitted.indices).toEqual({ layer: { to: 5 } });
    expect(now(one).open[0]?.emitted.search).toBe('');
  }, 120_000);

  it('selects the place a D1 node identifier names, and its index with it', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    expect(now(one).selectNode('decoder/attn[layer=3]', id)).toBe(true);
    const open = now(one).open.find((each) => each.id === id);
    expect(open?.selection).toEqual(['compositions', 'decoder', 'instances', 'attn']);
    // §4.18's "the folded node **and the index**", which is §4.8's scrubber.
    expect(open?.scrub['/compositions/decoder']).toBe('layer=3');
    // A root instance has no index and no composition to scrub.
    expect(now(one).selectNode('embed', id)).toBe(true);
    expect(now(one).open.find((each) => each.id === id)?.selection).toEqual(['instances', 'embed']);
    // And an identifier no box stands for says so rather than selecting something else.
    expect(now(one).selectNode('nothing/at[all=0]', id)).toBe(false);
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

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Feature 2.17 — the JSON source view's half of the store: §4.10's two-way sync, §3's off-schema
// source and its confirmed save, and the flush every path that reads a document goes through.
// What a browser has to answer is in `apps/web/e2e/source.spec.ts`; these are the questions about
// *what the editor does*, asked where they can be asked exhaustively.
// ─────────────────────────────────────────────────────────────────────────────────────────────

describe('an edit typed into the JSON source (§4.10)', () => {
  it('is taken into the tree as one named command, and one Undo gives the bytes back', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const before = open?.session.text ?? '';
    now(one).sourceEdit(before.replace('"literal": 32', '"literal": 16'), id);

    const after = now(one).open[0];
    expect(after?.session.text).toContain('"literal": 16');
    expect(after?.session.store.undoLabel).toBe('Edit the JSON source');
    now(one).undo(id);
    expect(now(one).open[0]?.session.text).toBe(before);
  }, 120_000);

  it('leaves a number the edit did not touch written exactly as the file writes it', async () => {
    // D12's own claim, through the source view: `1e-05` is not `0.00001`, and an edit somewhere
    // else in the document may not change it. The whole file is compared, not the one lexeme.
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const before = open?.session.text ?? '';
    expect(before).toContain('"value": 1e-05');
    now(one).sourceEdit(before.replace('"literal": 32', '"literal": 16'), open?.id ?? '');
    const after = now(one).open[0]?.session.text ?? '';
    expect(after).toContain('"value": 1e-05');
    expect(after).toBe(before.replace('"literal": 32', '"literal": 16'));
  }, 120_000);

  it('takes a text that leaves the grammar, and says the source is off it (D5, Q5)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    now(one).sourceEdit((open?.session.text ?? '').replace('"quantities": {', '"kernels": {},\n  "quantities": {'), id);
    const after = now(one).open.find((document) => document.id === id);
    // The tree took it — no gesture is refused for a semantic reason — and the grammar says so.
    expect(after?.session.text).toContain('"kernels": {}');
    expect(sourceReading(after as OpenDocument)?.pending).toBe(false);
    expect(sourceStanding(after as OpenDocument)).toBe('off-grammar');
    expect((after?.reading.structural ?? []).map((row) => row.message).join(' ')).toContain(
      "'kernels' was unexpected",
    );
  }, 120_000);

  it('does not take a text that is not JSON, and states the core’s own refusal', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const before = open?.session.text ?? '';
    now(one).sourceEdit(`${before.slice(0, 40)}`, id);
    const after = now(one).open.find((document) => document.id === id);
    expect(after?.session.text).toBe(before);
    const source = sourceReading(after as OpenDocument);
    expect(source?.pending).toBe(true);
    expect(source?.refusedAt).toBeGreaterThan(0);
    expect(source?.problems[0]?.code).toBe('V12');
    // CPython's own words and CPython's own position, which is what `--validate` would print.
    expect(source?.problems[0]?.message).toMatch(/line \d+ column \d+ \(char \d+\)/);
    expect(sourceStanding(after as OpenDocument)).toBe('pending');
  }, 120_000);

  it('does not take a text that is JSON and not a document', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const before = open?.session.text ?? '';
    now(one).sourceEdit('[1, 2, 3]\n', id);
    const after = now(one).open.find((document) => document.id === id);
    expect(after?.session.text).toBe(before);
    expect(sourceReading(after as OpenDocument)?.pending).toBe(true);
    expect(sourceReading(after as OpenDocument)?.problems.length).toBeGreaterThan(0);
  }, 120_000);

  it('commands nothing for a text that denotes the document it already holds', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const at = open?.session.store.revision ?? 0;
    // The very bytes…
    now(one).sourceEdit(open?.session.text ?? '', id);
    expect(now(one).open[0]?.session.store.revision).toBe(at);
    // …and a text that *denotes* them: spaced differently, and typed back after a refusal. The
    // serializer is the writer of record (D12) and would write the same bytes either way, so the
    // command log is left alone and only the pane's own text is recorded.
    now(one).sourceEdit('{ "nope"', id);
    expect(sourceReading(now(one).open[0] as OpenDocument)?.pending).toBe(true);
    const spaced = (open?.session.text ?? '').replace('{\n  "schema"', '{\n\n  "schema"');
    now(one).sourceEdit(spaced, id);
    expect(now(one).open[0]?.session.store.revision).toBe(at);
    expect(now(one).open[0]?.dirty).toBe(false);
    const source = sourceReading(now(one).open[0] as OpenDocument);
    expect(source?.pending).toBe(false);
    expect(source?.text).toBe(spaced);
  }, 120_000);

  it('goes stale the moment the document moves for another reason', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    now(one).sourceEdit((open?.session.text ?? '').replace('"literal": 32', '"literal": 16'), id);
    expect(sourceReading(now(one).open[0] as OpenDocument)).not.toBeNull();
    now(one).undo(id);
    // The pane's reading is of a document that no longer exists: it is a projection again.
    expect(sourceReading(now(one).open[0] as OpenDocument)).toBeNull();
    expect(sourceStanding(now(one).open[0] as OpenDocument)).toBeNull();
  }, 120_000);
});

describe('saving what the source view wrote (§3, §4.10)', () => {
  it('asks before it writes a document the source left off the grammar, and takes no for an answer', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const path = open?.path ?? '';
    const file = (await one.workspace.read(path)).text;
    now(one).sourceEdit((open?.session.text ?? '').replace('"quantities": {', '"kernels": {},\n  "quantities": {'), id);

    one.platform.shellRecord.answer = false;
    await now(one).save(id);
    expect(one.platform.shellRecord.asked.at(-1)).toContain('off the grammar');
    expect((await one.workspace.read(path)).text).toBe(file);
    // The core's refusal is in the log, as §3 asks.
    expect(one.log.join('\n')).toContain("'kernels' was unexpected");

    one.platform.shellRecord.answer = true;
    await now(one).save(id);
    expect((await one.workspace.read(path)).text).toContain('"kernels": {}');
  }, 120_000);

  it('asks before it writes a document whose source it could not read', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    now(one).sourceEdit('{ "model":', id);
    one.platform.shellRecord.answer = false;
    await now(one).save(id);
    expect(one.platform.shellRecord.asked.at(-1)).toContain('not a document');
  }, 120_000);

  it('asks nothing of a New Model, which is off the grammar by construction', async () => {
    // Feature 2.6: the skeleton is the required members, empty, and carries three structural
    // problems until the author adds a site and an interface. §3's confirmation is about a source
    // *edit*, not about the editor's own starting point.
    const one = editor();
    await now(one).newModel();
    one.platform.shellRecord.asked.length = 0;
    await now(one).saveAll();
    expect(one.platform.shellRecord.asked).toEqual([]);
    expect((await one.workspace.read('untitled.json')).text).toContain('"model": "untitled"');
  }, 120_000);

  it('takes what a source view is holding before it reads the document', async () => {
    // A `Ctrl+S` one keystroke after an edit writes what the reader is looking at: the pane
    // registers its flush and every path that reads the bytes goes through it first.
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const typed = (open?.session.text ?? '').replace('"literal": 32', '"literal": 16');
    now(one).holdSource(id, () => {
      now(one).sourceEdit(typed, id);
    });
    await now(one).save(id);
    expect((await one.workspace.read(open?.path ?? '')).text).toContain('"literal": 16');
  }, 120_000);

  it('takes it when the Save is asked of the *source tab*, whose id carries a suffix', async () => {
    // `Ctrl+S` typed inside the source view hands the current tab in, and that tab is the view's
    // — feature 2.16's own warning about `documentTab`, met one feature along. A flush keyed on
    // the tab rather than on the document would write what the document held a moment ago.
    const one = editor();
    await now(one).openDocument(MODEL);
    const open = now(one).open[0];
    const id = open?.id ?? '';
    const typed = (open?.session.text ?? '').replace('"literal": 32', '"literal": 16');
    now(one).holdSource(id, () => {
      now(one).sourceEdit(typed, id);
    });
    await now(one).save(`${id}${SOURCE_SUFFIX}`);
    expect((await one.workspace.read(open?.path ?? '')).text).toContain('"literal": 16');
  }, 120_000);
});

describe('Show in JSON (§4.10)', () => {
  it('opens the source tab and asks it for the place that is selected', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    now(one).selectPlace(['instances', 'final_n'], id);
    now(one).showSource(id);
    expect(one.tabs.tabs.map((tab) => tab.kind)).toContain(SOURCE_TAB);
    expect(now(one).open[0]?.reveal?.pointer).toBe('/instances/final_n');
    // Asking again for the same place asks again: a reader who scrolled away means it.
    const seq = now(one).open[0]?.reveal?.seq ?? 0;
    now(one).showSource(id);
    expect(now(one).open[0]?.reveal?.seq).toBe(seq + 1);
  }, 120_000);

  it('is what a Problems row asks for too — §4.17’s third navigation', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id ?? '';
    now(one).revealPlace(['quantities', 'd'], id);
    expect(now(one).open[0]?.reveal?.pointer).toBe('/quantities/d');
    expect(now(one).open[0]?.selection).toEqual(['quantities', 'd']);
  }, 120_000);
});

import { describe, expect, it } from 'vitest';

import {
  DocumentSession,
  draftStanding,
  hasOverrides,
  readSidecar,
  sidecarOf,
} from '../../src/documents/session.js';
import { newDocument } from '../../src/documents/skeleton.js';
import { emptyLayout, writeLayout } from '../../src/layout.js';
import { ABSENT, MemoryWorkspace, PlatformError } from '../../src/platform/index.js';
import { corpusText, shapes } from '../source.js';

// One open document (§4.3): what dirty means, what a save writes, what a revert takes back, and
// what a draft is worth beside the file it was taken from.

const MODEL = 'models/llama3-8b.json';
const text = corpusText('llama3-8b');

/** A workspace laid out like `data/`, holding one corpus document. */
function workspace(): MemoryWorkspace {
  return MemoryWorkspace.of({ [MODEL]: text });
}

/** The session a workspace's own document opens as. */
async function opened(place: MemoryWorkspace): Promise<DocumentSession> {
  const read = await place.read(MODEL);
  return DocumentSession.open(read.text, shapes(), {
    workspace: place.root().id,
    path: MODEL,
    revision: read.revision,
    layout: await readSidecar(place, MODEL),
  });
}

describe('a document opened from a workspace', () => {
  it('is not dirty, and holds the bytes the file holds (D12)', async () => {
    const session = await opened(workspace());
    expect(session.dirty).toBe(false);
    expect(session.text).toBe(text);
    expect(session.name).toBe('llama3-8b.json');
  });

  it('saves back byte for byte, and the workspace holds what it held', async () => {
    const place = workspace();
    const session = await opened(place);
    const saved = await session.save(place);
    expect((await place.read(MODEL)).text).toBe(text);
    expect(saved.revision).toBe((await place.read(MODEL)).revision);
    expect(saved.sidecar).toBeUndefined();
    expect(session.dirty).toBe(false);
  });

  it('is dirty after an edit and clean again after the save that follows it', async () => {
    const place = workspace();
    const session = await opened(place);
    session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const at = draft.members.findIndex((member) => member.name === 'model');
        const member = draft.members[at];
        if (member !== undefined) member.value = 'llama3-8b-edited';
      },
      moves: [],
    });
    expect(session.dirty).toBe(true);
    await session.save(place);
    expect(session.dirty).toBe(false);
    expect((await place.read(MODEL)).text).toContain('llama3-8b-edited');
  });

  it('is clean again when an undo puts it back where the file is', async () => {
    const session = await opened(workspace());
    session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((one) => one.name === 'model');
        if (member !== undefined) member.value = 'other';
      },
      moves: [],
    });
    expect(session.dirty).toBe(true);
    session.store.undo();
    // The revision counts states and an undo makes a new one, so a text comparison is what a
    // dirty dot would have to be — and it is: the tree is the file's again.
    expect(session.text).toBe(text);
  });

  it('refuses a save whose file moved under it, and changes nothing (§5.2)', async () => {
    const place = workspace();
    const session = await opened(place);
    await place.write(MODEL, `${text}`, (await place.read(MODEL)).revision);
    const refused = await session.save(place).catch((error: unknown) => error);
    expect(refused).toBeInstanceOf(PlatformError);
    expect((refused as PlatformError).reason).toBe('conflict');
  });
});

describe('the layout sidecar', () => {
  it('is named beside the document, as §5.5 writes it', () => {
    expect(sidecarOf(MODEL)).toBe('models/llama3-8b.layout.json');
    expect(sidecarOf('models/llama3-8b')).toBe('models/llama3-8b.layout.json');
  });

  it('is written only when the layout has overrides (§4.3)', async () => {
    const place = workspace();
    const session = await opened(place);
    expect(hasOverrides(emptyLayout())).toBe(false);
    await session.save(place);
    await expect(place.read(sidecarOf(MODEL))).rejects.toThrow(PlatformError);

    session.layout.move(['instances', 'embed'], { x: 10, y: 20 });
    expect(session.layoutDirty).toBe(true);
    const saved = await session.save(place);
    expect(saved.sidecar).toBe(sidecarOf(MODEL));
    expect((await place.read(sidecarOf(MODEL))).text).toBe(writeLayout(session.layout.layout));
    expect(session.dirty).toBe(false);
  });

  it('is read back where it is there and shrugged off where it is not', async () => {
    const place = MemoryWorkspace.of({
      [MODEL]: text,
      [sidecarOf(MODEL)]: writeLayout({
        ...emptyLayout(),
        positions: { 'instances/embed': { x: 1, y: 2 } },
      }),
    });
    expect((await readSidecar(place, MODEL)).positions).toEqual({ 'instances/embed': { x: 1, y: 2 } });
    expect((await readSidecar(workspace(), MODEL)).positions).toEqual({});
  });
});

describe('Revert', () => {
  it('takes the document back to the file, and is itself undoable (D13)', async () => {
    const place = workspace();
    const session = await opened(place);
    session.store.apply({
      label: 'Rename the model',
      edit: (draft) => {
        const member = draft.members.find((one) => one.name === 'model');
        if (member !== undefined) member.value = 'other';
      },
      moves: [],
    });
    const applied = session.revert(text, (await place.read(MODEL)).revision);
    expect(applied.label).toBe('Revert');
    expect(session.text).toBe(text);
    expect(session.dirty).toBe(false);
    expect(session.store.undoLabel).toBe('Revert');
    session.store.undo();
    expect(session.text).toContain('"model": "other"');
  });
});

describe('Restore', () => {
  it('puts a draft back and leaves the document dirty, which is what a draft is', async () => {
    const place = workspace();
    const session = await opened(place);
    const draft = '{\n  "model": "from the draft"\n}\n';
    const applied = session.restore(draft);
    expect(applied.label).toBe('Restore the draft');
    // Unsaved work: the file has not got it, and the next Save is what puts it there.
    expect(session.dirty).toBe(true);
    expect(session.text).toBe(draft);
    // And undoable, like every other command (D13).
    session.store.undo();
    expect(session.text).toBe(text);
  });
});

describe('a document made from nothing', () => {
  it('is dirty from birth, since the workspace has never had it (§4.3)', () => {
    const session = new DocumentSession(newDocument(shapes(), { name: 'untitled' }), shapes(), {
      workspace: 'memory:x',
      path: 'untitled.json',
      revision: ABSENT,
      saved: false,
    });
    expect(session.dirty).toBe(true);
    expect(session.revision).toBe(ABSENT);
  });

  it('is written where nothing was, and a second write of the same path is a conflict', async () => {
    const place = MemoryWorkspace.of({});
    const session = new DocumentSession(newDocument(shapes(), { name: 'untitled' }), shapes(), {
      workspace: place.root().id,
      path: 'untitled.json',
      revision: ABSENT,
      saved: false,
    });
    await session.save(place);
    expect(session.dirty).toBe(false);
    const second = new DocumentSession(newDocument(shapes(), { name: 'untitled' }), shapes(), {
      workspace: place.root().id,
      path: 'untitled.json',
      revision: ABSENT,
      saved: false,
    });
    await expect(second.save(place)).rejects.toThrow(PlatformError);
  });
});

describe('a draft beside the file it was taken from', () => {
  const draft = {
    workspace: 'w',
    path: MODEL,
    text: 'edited',
    revision: 'r1',
    savedAt: '2026-09-11T12:00:00.000Z',
  };

  it('is nothing to offer when the file already holds it', () => {
    expect(draftStanding({ ...draft, text }, text, 'r1')).toBeNull();
  });

  it('is unsaved work when it was taken against the revision the file is still at', () => {
    expect(draftStanding(draft, text, 'r1')).toBe('unsaved');
  });

  it('is behind the file when the file has moved since', () => {
    expect(draftStanding(draft, text, 'r2')).toBe('behind');
  });

  it('is what a session offers of itself, text and revision together', async () => {
    const session = await opened(workspace());
    const taken = session.draftOf(new Date('2026-09-11T12:00:00.000Z'));
    expect(taken.path).toBe(MODEL);
    expect(taken.text).toBe(text);
    expect(taken.savedAt).toBe('2026-09-11T12:00:00.000Z');
    expect(draftStanding(taken, text, session.revision)).toBeNull();
  });
});

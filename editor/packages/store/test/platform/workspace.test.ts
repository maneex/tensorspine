import { describe, expect, it } from 'vitest';

import {
  ABSENT,
  MemoryWorkspace,
  PlatformError,
  ReadOnlyWorkspace,
  listTree,
  readTree,
  snapshotOf,
  textsOf,
  within,
  type Delivered,
  type UploadedFile,
  type WatchEvent,
} from '../../src/platform/index.js';

// The two behaviours every `Workspace` of §5.2 has, asked of the implementations that name no
// platform. The browser's are asked the same questions in `apps/web/e2e/platform.spec.ts`, over a
// real directory handle and a real folder upload; what is here is what a suite can ask without a
// browser, and what the stub of D11 is held to.

/** A file as a folder upload hands it over — the structural minimum a browser `File` satisfies. */
function uploaded(path: string, text: string, lastModified = 1_700_000_000_000): UploadedFile {
  return {
    name: path.split('/').at(-1) ?? path,
    webkitRelativePath: path,
    size: new TextEncoder().encode(text).length,
    lastModified,
    text: () => Promise.resolve(text),
  };
}

/** The refusal a call answered with, as a caller reads it. */
async function refusal(call: Promise<unknown>): Promise<PlatformError> {
  try {
    await call;
  } catch (error) {
    expect(error).toBeInstanceOf(PlatformError);
    return error as PlatformError;
  }
  throw new Error('the call was not refused');
}

describe('a workspace the editor can write', () => {
  const open = (): MemoryWorkspace =>
    MemoryWorkspace.of({
      'models/llama3-8b.json': '{"model": "llama3-8b"}\n',
      'primitive-library/primitive-library.json': '{"schema": "x"}\n',
    });

  it('says what it is', () => {
    expect(open().root()).toEqual({
      kind: 'memory',
      id: 'memory:memory',
      name: 'memory',
      writable: true,
    });
  });

  it('lists a directory, files and directories together, sorted by name', async () => {
    const workspace = open();
    expect(await workspace.list('')).toEqual([
      { name: 'models', path: 'models', kind: 'directory' },
      { name: 'primitive-library', path: 'primitive-library', kind: 'directory' },
    ]);
    expect(await workspace.list('models')).toEqual([
      { name: 'llama3-8b.json', path: 'models/llama3-8b.json', kind: 'file' },
    ]);
    expect((await refusal(workspace.list('nowhere'))).reason).toBe('not-found');
  });

  it('reads a file with the revision it is at, and refuses one that is not there', async () => {
    const workspace = open();
    const read = await workspace.read('models/llama3-8b.json');
    expect(read.text).toBe('{"model": "llama3-8b"}\n');
    expect(read.revision).not.toBe(ABSENT);
    expect((await refusal(workspace.read('models/nothing.json'))).reason).toBe('not-found');
  });

  it('writes in place and moves the revision', async () => {
    const workspace = open();
    const before = await workspace.read('models/llama3-8b.json');
    const written = await workspace.write('models/llama3-8b.json', 'edited', before.revision);
    expect(written.revision).not.toBe(before.revision);
    expect((await workspace.read('models/llama3-8b.json')).text).toBe('edited');
  });

  it('refuses a write whose expectation the file no longer holds, and changes nothing', async () => {
    // The optimistic concurrency of §5.2: what makes an external change visible instead of
    // silently overwritten.
    const workspace = open();
    const stale = (await workspace.read('models/llama3-8b.json')).revision;
    await workspace.write('models/llama3-8b.json', 'somebody else', stale);
    const error = await refusal(workspace.write('models/llama3-8b.json', 'mine', stale));
    expect(error.reason).toBe('conflict');
    expect(error.message).toContain('changed since it was read');
    expect(error.revisions?.expected).toBe(stale);
    expect((await workspace.read('models/llama3-8b.json')).text).toBe('somebody else');
  });

  it('creates a file only where the caller expected none', async () => {
    const workspace = open();
    await workspace.write('models/new.json', '{}', ABSENT);
    expect((await refusal(workspace.write('models/new.json', '{}', ABSENT))).reason).toBe('conflict');
  });

  it('never creates the directories above a file: `mkdir` is what makes a folder', async () => {
    // Feature 0.6's finding 5. The File System Access API makes the wrong behaviour a
    // one-character difference, so both implementations are held to the right one.
    const workspace = open();
    expect((await refusal(workspace.write('typo/x.json', '{}'))).reason).toBe('not-found');
    await workspace.mkdir('bases/mine/primitives');
    await workspace.write('bases/mine/primitives/x.json', '{}');
    expect(await workspace.list('bases')).toEqual([
      { name: 'mine', path: 'bases/mine', kind: 'directory' },
    ]);
  });

  it('reports a change made behind it, and stops when unsubscribed', async () => {
    const workspace = open();
    const seen: WatchEvent[] = [];
    const stop = workspace.watch('models', (event) => seen.push(event));
    await workspace.write('models/llama3-8b.json', 'edited');
    await workspace.write('primitive-library/primitive-library.json', 'not watched');
    await workspace.write('models/added.json', '{}');
    await workspace.remove('models/added.json');
    stop();
    await workspace.write('models/llama3-8b.json', 'after the unsubscribe');
    expect(seen.map((event) => `${event.kind} ${event.path}`)).toEqual([
      'changed models/llama3-8b.json',
      'added models/added.json',
      'removed models/added.json',
    ]);
  });

  it('resolves a relative reference against the file it is written in', () => {
    expect(open().resolve('models/llama3-8b.json', '../primitive-library/')).toBe('primitive-library');
  });
});

describe('a workspace the editor cannot write', () => {
  const delivered: Delivered[] = [];
  const open = (): ReadOnlyWorkspace => {
    delivered.length = 0;
    return new ReadOnlyWorkspace(
      snapshotOf([
        uploaded('workspace/models/llama3-8b.json', '{"model": "llama3-8b"}\n'),
        uploaded('workspace/primitive-library/primitive-library.json', '{"schema": "x"}\n'),
      ]),
      (name, text) => {
        delivered.push({ name, path: name, bytes: text.length });
        return Promise.resolve();
      },
    );
  };

  it('strips the chosen folder’s own name, so its paths are the writable side’s', async () => {
    const workspace = open();
    expect(workspace.root()).toEqual({
      kind: 'snapshot',
      id: 'upload:workspace',
      name: 'workspace',
      writable: false,
    });
    expect(await listTree(workspace, '')).toEqual([
      'models/llama3-8b.json',
      'primitive-library/primitive-library.json',
    ]);
    expect(workspace.resolve('models/llama3-8b.json', '../primitive-library/')).toBe(
      'primitive-library',
    );
  });

  it('hands Save to the user as a download and leaves the revision where it was', async () => {
    const workspace = open();
    const read = await workspace.read('models/llama3-8b.json');
    const written = await workspace.write('models/llama3-8b.json', 'edited', read.revision);
    expect(written.revision).toBe(read.revision);
    expect(delivered).toEqual([{ name: 'llama3-8b.json', path: 'llama3-8b.json', bytes: 6 }]);
    expect(workspace.lastDelivered?.path).toBe('models/llama3-8b.json');
    // Nothing moved: the copy the page holds still answers what it was given.
    expect((await workspace.read('models/llama3-8b.json')).text).toBe('{"model": "llama3-8b"}\n');
  });

  it('saves a file it never held, which is one the user will put somewhere', async () => {
    const workspace = open();
    expect((await workspace.write('models/new.json', '{}', ABSENT)).revision).toBe(ABSENT);
    expect(delivered.map((one) => one.name)).toEqual(['new.json']);
  });

  it('refuses to create a folder, in words that name the way out', async () => {
    // §4.3's "New Base creates the folder and manifest" has no answer here: a browser that cannot
    // write a folder cannot, and one downloaded file is not a base (feature 0.6's finding 6).
    const error = await refusal(open().mkdir('bases/mine'));
    expect(error.reason).toBe('read-only');
    expect(error.message).toContain('writable directory picker');
  });

  it('registers a watcher and never calls it, because nothing can change under a copy', async () => {
    const workspace = open();
    const seen: WatchEvent[] = [];
    const stop = workspace.watch('', (event) => seen.push(event));
    await workspace.write('models/llama3-8b.json', 'edited');
    stop();
    expect(seen).toEqual([]);
  });

  it('names itself for the drop when the files share no folder', async () => {
    const workspace = new ReadOnlyWorkspace(snapshotOf([uploaded('llama3-8b.json', '{}')]), () =>
      Promise.resolve(),
    );
    expect(workspace.root().name).toBe('dropped folder');
    expect(await listTree(workspace, '')).toEqual(['llama3-8b.json']);
  });

  it('reads its files lazily: a listing opens none of them', async () => {
    let reads = 0;
    const counting: UploadedFile = {
      ...uploaded('workspace/models/llama3-8b.json', '{}'),
      text: () => {
        reads += 1;
        return Promise.resolve('{}');
      },
    };
    const workspace = new ReadOnlyWorkspace(snapshotOf([counting]), () => Promise.resolve());
    await workspace.list('models');
    expect(reads).toBe(0);
    await workspace.read('models/llama3-8b.json');
    expect(reads).toBe(1);
  });
});

describe('a subtree of a workspace, as the core is handed one', () => {
  const workspace = MemoryWorkspace.of({
    'data/models/llama3-8b.json': '{"model": "llama3-8b"}\n',
    'data/primitive-library/primitive-library.json': '{"schema": "x"}\n',
    'data/primitive-library/primitives/norm.rms/1.0.0.json': '{"name": "norm.rms"}\n',
    'data/primitive-library/notes.md': 'not a unit\n',
    'data/primitive-library/.hidden/x.json': 'skipped\n',
  });

  it('answers every file under it by its path in the workspace', async () => {
    const files = await readTree(workspace, 'data/primitive-library', { suffix: '.json' });
    expect(Object.keys(files).sort()).toEqual([
      'data/primitive-library/primitive-library.json',
      'data/primitive-library/primitives/norm.rms/1.0.0.json',
    ]);
    expect(files['data/primitive-library/primitives/norm.rms/1.0.0.json']).toBe(
      '{"name": "norm.rms"}\n',
    );
  });

  it('skips a name beginning with a dot, as the loader’s own glob does', async () => {
    const files = await readTree(workspace, 'data', { suffix: '.json' });
    expect(Object.keys(files).some((path) => path.includes('.hidden'))).toBe(false);
  });

  it('stops where it is told to', async () => {
    expect(await listTree(workspace, 'data', { suffix: '.json', depth: 1 })).toEqual([
      'data/models/llama3-8b.json',
      'data/primitive-library/primitive-library.json',
    ]);
  });

  it('refuses a path that is not under the directory it was given', () => {
    expect(within('data/models/llama3-8b.json', 'data')).toBe('data/models/llama3-8b.json');
    expect(() => within('elsewhere/x.json', 'data')).toThrow(PlatformError);
  });
});

describe('a set of texts as a read-only workspace', () => {
  it('is what the stub’s Examples workspace is made of', async () => {
    const workspace = new ReadOnlyWorkspace(
      textsOf('examples', 'examples', 'Examples', { 'models/llama3-8b.json': '{}\n' }),
      () => Promise.resolve(),
    );
    expect(workspace.root().kind).toBe('examples');
    expect(workspace.root().writable).toBe(false);
    const read = await workspace.read('models/llama3-8b.json');
    // A set built once cannot change, so two reads answer one revision and a save against it is
    // never in conflict with itself.
    expect((await workspace.read('models/llama3-8b.json')).revision).toBe(read.revision);
    expect((await workspace.write('models/llama3-8b.json', 'x', read.revision)).revision).toBe(
      read.revision,
    );
  });
});

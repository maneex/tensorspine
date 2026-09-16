import { createHash } from 'node:crypto';

import { describe, expect, it } from 'vitest';

import {
  PlatformError,
  ReadOnlyWorkspace,
  addressOf,
  ageOf,
  fetchPublishedSet,
  heldSet,
  nameOfRoot,
  publishedFileSet,
  publishedId,
  publishedName,
  publishedWorkspace,
  PUBLISHED_MANIFEST,
  readPublishedManifest,
  textsUnder,
  type PublishedSet,
  type Retrieve,
} from '../../src/platform/index.js';

// A **published file set** read from an address — feature 2.20's reading half.
//
// Plain HTTP has no directory listing, so a set of files cannot be discovered and has to be
// declared: an address is the address of a *manifest*, and what this file holds to account is what
// that manifest is worth. Three claims, each of which the feature's block asks for by name:
//
//   * **Integrity.** Every file is checked against the sha256 the manifest gives it, whichever way
//     it arrived — on its own or out of a bundle — and one that disagrees refuses the whole set,
//     **naming the path**.
//   * **Trust.** What comes back is data: nothing here parses a document and nothing here decides
//     what a unit means. That is the core's, through the unit schema and the loader (§9 Q6).
//   * **The transport.** A manifest may declare the set a second time as one bundle, and reading
//     through it is one request instead of one per file — feature 2.18's own measurement, one
//     origin further away. What a bundle does not carry is fetched on its own; a bundle the host
//     does not serve at all is a *declared file that is missing*, which is a set that was not
//     served whole. The digests are always the files' own.
//
// Nothing here fetches anything: `Retrieve` and `Digest` are handed in, which is the same line
// feature 1.9 drew for a checkpoint's header — the bytes are the caller's, the format is ours.

function sha256(text: string): string {
  return createHash('sha256').update(Buffer.from(text, 'utf8')).digest('hex');
}

/** A published set as a generator writes one: the files, their digests, and one bundle. */
function published(
  files: Readonly<Record<string, string>>,
  options: { bundle?: boolean; commit?: string | null; omit?: readonly string[] } = {},
): { served: Record<string, string>; manifest: string } {
  const records = Object.entries(files)
    .map(([path, text]) => ({ path, bytes: Buffer.byteLength(text), sha256: sha256(text) }))
    .sort((a, b) => (a.path < b.path ? -1 : 1));
  const served: Record<string, string> = { ...files };
  const bundles: { name: string; path: string; covers: string; files: number; bytes: number }[] = [];
  if (options.bundle !== false && records.length > 1) {
    const carried = Object.fromEntries(
      Object.entries(files).filter(([path]) => !(options.omit ?? []).includes(path)),
    );
    const text = `${JSON.stringify(carried)}\n`;
    served['bundles/all.json'] = text;
    bundles.push({
      name: 'all',
      path: 'bundles/all.json',
      covers: '',
      files: records.length,
      bytes: Buffer.byteLength(text),
    });
    records.push({
      path: 'bundles/all.json',
      bytes: Buffer.byteLength(text),
      sha256: sha256(text),
    });
    records.sort((a, b) => (a.path < b.path ? -1 : 1));
  }
  const manifest = `${JSON.stringify(
    {
      generated_by: 'editor/scripts/publish.ts',
      repository_commit: options.commit === undefined ? 'a'.repeat(40) : options.commit,
      repository_dirty: false,
      ...(bundles.length === 0 ? {} : { bundles }),
      files: records,
    },
    null,
    2,
  )}\n`;
  served[PUBLISHED_MANIFEST] = manifest;
  return { served, manifest };
}

/** A host that answers the paths it was given, counting what it was asked for. */
function host(
  served: Readonly<Record<string, string>>,
  root = 'https://lab.example/base/',
): { retrieve: Retrieve; asked: string[] } {
  const asked: string[] = [];
  const retrieve: Retrieve = (url) => {
    asked.push(url);
    if (!url.startsWith(root)) return Promise.reject(new Error(`${url} is not on this host`));
    const text = served[url.slice(root.length)];
    if (text === undefined) {
      return Promise.reject(new PlatformError(`${url} answered 404 Not Found`, 'not-found'));
    }
    return Promise.resolve(text);
  };
  return { retrieve, asked };
}

const digest = (text: string): Promise<string> => Promise.resolve(sha256(text));

const UNITS: Record<string, string> = {
  'primitive-library.json': '{"schema": "x", "kind": "y"}\n',
  'primitives/a/1.0.0.json': '{"a": 1}\n',
  'primitives/b/1.0.0.json': '{"b": 2}\n',
};

describe('the address of a set', () => {
  it('is the manifest’s own, whether the manifest or its directory was named', () => {
    const manifest = 'https://lab.example/base/vendor.json';
    const root = 'https://lab.example/base/';
    expect(addressOf(manifest)).toEqual({ manifest, root });
    expect(addressOf(root)).toEqual({ manifest, root });
    // A directory written without its separator is a directory: every server writes it either way.
    expect(addressOf('https://lab.example/base')).toEqual({ manifest, root });
    expect(addressOf('  https://lab.example/base/  ')).toEqual({ manifest, root });
  });

  it('drops a query and a fragment, which address a page and not a set', () => {
    expect(addressOf('https://lab.example/base/?v=2#top').root).toBe('https://lab.example/base/');
  });

  it('refuses what is not an address, and says what one is', () => {
    expect(() => addressOf('')).toThrow(PlatformError);
    expect(() => addressOf('lab.example/base')).toThrow(/is not an address/);
    expect(() => addressOf('file:///home/me/base/')).toThrow(/http or https/);
  });

  it('names a set after the last segment of its root, and after the host where it has none', () => {
    expect(nameOfRoot('https://lab.example/base/')).toBe('base');
    expect(nameOfRoot('https://lab.example/')).toBe('lab.example');
  });
});

describe('a manifest', () => {
  it('is read as the generator wrote it — the files, the digests and the commit', () => {
    const { manifest } = published(UNITS);
    const read = readPublishedManifest(manifest, 'https://lab.example/base/vendor.json');
    expect(read.files.map((one) => one.path)).toEqual([
      'bundles/all.json',
      'primitive-library.json',
      'primitives/a/1.0.0.json',
      'primitives/b/1.0.0.json',
    ]);
    expect(read.repository_commit).toBe('a'.repeat(40));
    expect(read.bundles?.[0]?.covers).toBe('');
  });

  it('answers a null commit where the generator ran outside a checkout', () => {
    const { manifest } = published(UNITS, { commit: null });
    expect(readPublishedManifest(manifest, 'at').repository_commit).toBeNull();
  });

  it('refuses a text that is not one, and says what a published set declares', () => {
    const at = 'https://lab.example/base/vendor.json';
    expect(() => readPublishedManifest('not json', at)).toThrow(/is not a published set's manifest/);
    expect(() => readPublishedManifest('[]', at)).toThrow(/declares no files/);
    expect(() => readPublishedManifest('{"files": {}}', at)).toThrow(/pnpm manifest/);
    expect(() => readPublishedManifest('{"files": [{"path": "a.json"}]}', at)).toThrow(
      /a.json has no sha256/,
    );
  });

  it('refuses a path that would leave the set, which is where a manifest is data and not a friend', () => {
    // A manifest comes from another origin. Its paths are appended to the set's root to make a URL
    // and become places in a workspace, so a set may name neither a place above itself nor one
    // somewhere else entirely — and the refusal comes before anything is fetched.
    const at = 'https://lab.example/base/vendor.json';
    for (const path of ['../secret.json', '/etc/passwd', 'https://elsewhere.example/x.json', 'a//b.json', './a.json']) {
      const text = JSON.stringify({ files: [{ path, sha256: '0'.repeat(64), bytes: 1 }] });
      expect(() => readPublishedManifest(text, at), path).toThrow(/is not a path inside the set/);
    }
  });
});

describe('a set fetched from an address', () => {
  it('is the manifest and then every file, each checked against its own digest', async () => {
    const { served } = published(UNITS, { bundle: false });
    const { retrieve, asked } = host(served);
    const set = await fetchPublishedSet({ address: 'https://lab.example/base/', retrieve, digest });
    expect(Object.keys(set.texts).sort()).toEqual(Object.keys(UNITS).sort());
    expect(set.texts['primitives/a/1.0.0.json']).toBe(UNITS['primitives/a/1.0.0.json']);
    expect(set.cached).toBe(false);
    // One request for the manifest and one per file: the arithmetic a bundle exists to change.
    expect(asked).toHaveLength(1 + Object.keys(UNITS).length);
    expect(set.requests).toBe(1 + Object.keys(UNITS).length);
  });

  it('reads through the bundle the manifest declares: one request for the whole set', async () => {
    const { served } = published(UNITS);
    const { retrieve, asked } = host(served);
    const set = await fetchPublishedSet({ address: 'https://lab.example/base/', retrieve, digest });
    expect(Object.keys(set.texts).sort()).toEqual(
      [...Object.keys(UNITS), 'bundles/all.json'].sort(),
    );
    expect(asked).toEqual([
      'https://lab.example/base/vendor.json',
      'https://lab.example/base/bundles/all.json',
    ]);
    expect(set.requests).toBe(2);
  });

  it('reads the files themselves when the caller says so — the measurement’s other half', async () => {
    const { served } = published(UNITS);
    const { retrieve, asked } = host(served);
    const set = await fetchPublishedSet({
      address: 'https://lab.example/base/',
      retrieve,
      digest,
      bundles: false,
    });
    expect(set.requests).toBe(1 + Object.keys(UNITS).length + 1);
    expect(asked).toContain('https://lab.example/base/primitives/a/1.0.0.json');
  });

  it('fetches on its own whatever the bundle does not carry, and asks for the bundle once', async () => {
    const { served } = published(UNITS, { omit: ['primitives/b/1.0.0.json'] });
    const { retrieve, asked } = host(served);
    const set = await fetchPublishedSet({ address: 'https://lab.example/base/', retrieve, digest });
    for (const path of Object.keys(UNITS)) expect(set.texts[path]).toBe(UNITS[path]);
    // The manifest, the bundle, and the one file the bundle did not carry. The bundle itself is
    // **not** fetched a second time as a file: it is already in hand, and its own digest is
    // checked against the bytes that came back.
    expect(asked).toEqual([
      'https://lab.example/base/vendor.json',
      'https://lab.example/base/bundles/all.json',
      'https://lab.example/base/primitives/b/1.0.0.json',
    ]);
  });

  it('refuses a set whose bundle the host does not serve: a declared file is a declared file', async () => {
    const { served } = published(UNITS);
    const without = { ...served };
    delete without['bundles/all.json'];
    const { retrieve } = host(without);
    await expect(
      fetchPublishedSet({ address: 'https://lab.example/base/', retrieve, digest }),
    ).rejects.toMatchObject({ reason: 'not-found' });
  });

  it('catches a bundle that disagrees exactly as it catches a file: the digests are the files’', async () => {
    const { served } = published(UNITS);
    const lying = { ...UNITS, 'primitives/b/1.0.0.json': '{"b": 99}\n' };
    const changed = { ...served, 'bundles/all.json': `${JSON.stringify(lying)}\n` };
    const { retrieve } = host(changed);
    // Two digests disagree at once — the bundle's own, and the file it carried — and either is the
    // whole set refused, with a path in the message. Which of the two is named is a race the
    // reader does not settle and this test does not pretend to.
    const refusal = await fetchPublishedSet({
      address: 'https://lab.example/base/',
      retrieve,
      digest,
    }).catch((error: unknown) => error);
    expect(refusal).toBeInstanceOf(PlatformError);
    expect((refusal as PlatformError).reason).toBe('corrupt');
    expect((refusal as PlatformError).message).toMatch(/ is not what .* declares: sha256 /);
  });

  it('hands the retrieve’s own refusal back when the manifest is not there at all', async () => {
    const { retrieve } = host({});
    await expect(
      fetchPublishedSet({ address: 'https://lab.example/base/', retrieve, digest }),
    ).rejects.toMatchObject({ reason: 'not-found' });
  });

  it('hands back what a browser gives for a host that permits nothing — a refusal, not a spinner', async () => {
    // What a cross-origin read refused by the browser looks like from script: a `TypeError`, no
    // status, no headers. The deployment's own `Retrieve` is what turns it into a refusal naming
    // the header; this states that nothing here swallows one.
    const retrieve: Retrieve = () => Promise.reject(new TypeError('Failed to fetch'));
    await expect(
      fetchPublishedSet({ address: 'https://lab.example/base/', retrieve, digest }),
    ).rejects.toThrow(/Failed to fetch/);
  });
});

describe('a set in hand', () => {
  async function set(): Promise<PublishedSet> {
    const { served } = published(UNITS);
    const { retrieve } = host(served);
    return fetchPublishedSet({
      address: 'https://lab.example/base/',
      retrieve,
      digest,
      now: () => '2026-09-16T08:00:00.000Z',
    });
  }

  it('is a FileSet whose revision is the manifest’s own digest', async () => {
    const one = await set();
    const files = publishedFileSet(one, 'remote', publishedId(one), publishedName(one));
    expect(files.name).toBe('base');
    expect(files.id).toBe('remote:https://lab.example/base/vendor.json');
    expect(files.paths()).toContain('primitives/a/1.0.0.json');
    expect(await files.read('primitives/a/1.0.0.json')).toEqual({
      text: UNITS['primitives/a/1.0.0.json'],
      revision: sha256(UNITS['primitives/a/1.0.0.json'] as string),
    });
    expect(files.revision('primitives/a/1.0.0.json')).toBe(
      sha256(UNITS['primitives/a/1.0.0.json'] as string),
    );
    expect(files.revision('nothing.json')).toBeNull();
    // A bundle is a transport and not a file of the set: it is fetched, its own digest is checked
    // — it is declared, after all — and it is not one of the things the set holds.
    expect(files.paths()).not.toContain('bundles/all.json');
    expect(one.texts['bundles/all.json']).toBeDefined();
  });

  it('opens as the read-only workspace the snapshot and the Examples workspace already are', async () => {
    const one = await set();
    const delivered: string[] = [];
    const workspace = publishedWorkspace(one, (name) => {
      delivered.push(name);
      return Promise.resolve();
    });
    expect(workspace).toBeInstanceOf(ReadOnlyWorkspace);
    expect(workspace.root()).toEqual({
      kind: 'remote',
      id: 'remote:https://lab.example/base/vendor.json',
      name: 'base',
      writable: false,
    });
    expect((await workspace.list('primitives')).map((entry) => entry.name)).toEqual(['a', 'b']);
    // Save is a download and the address is left exactly as it is — nobody writes back to a set
    // they fetched.
    await workspace.write('primitives/a/1.0.0.json', '{"a": 2}\n');
    expect(delivered).toEqual(['1.0.0.json']);
    await expect(workspace.mkdir('primitives/c')).rejects.toMatchObject({ reason: 'read-only' });
  });

  it('answers its files under a prefix, relative to it — what a mounted base hands over', async () => {
    const one = await set();
    expect(Object.keys(textsUnder(one, 'primitives')).sort()).toEqual([
      'a/1.0.0.json',
      'b/1.0.0.json',
    ]);
    expect(Object.keys(textsUnder(one)).sort()).toEqual(Object.keys(UNITS).sort());
  });

  it('can be built from what a cache holds, with the generator’s digests untouched', async () => {
    const one = await set();
    const back = heldSet(one.address, one.root, one.manifest, one.texts, one.fetchedAt, true);
    expect(back.cached).toBe(true);
    expect(back.requests).toBe(0);
    const files = publishedFileSet(back, 'remote', publishedId(back), publishedName(back));
    expect(files.revision('primitives/a/1.0.0.json')).toBe(
      sha256(UNITS['primitives/a/1.0.0.json'] as string),
    );
  });
});

describe('how old a cached set is', () => {
  it('is said in the words a banner says it in', () => {
    const now = Date.parse('2026-09-16T12:00:00.000Z');
    expect(ageOf('2026-09-16T11:59:30.000Z', now)).toBe('30 second(s) ago');
    expect(ageOf('2026-09-16T11:30:00.000Z', now)).toBe('30 minute(s) ago');
    expect(ageOf('2026-09-16T02:00:00.000Z', now)).toBe('10 hour(s) ago');
    expect(ageOf('2026-09-10T12:00:00.000Z', now)).toBe('6 day(s) ago');
    // A moment nobody can read is shown as it stands rather than as a figure invented for it.
    expect(ageOf('whenever', now)).toBe('whenever');
  });
});

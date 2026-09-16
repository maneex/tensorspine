import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, describe, expect, it } from 'vitest';

import {
  PlatformError,
  fetchPublishedSet,
  readPublishedManifest,
  PUBLISHED_MANIFEST,
  type PublishedManifest,
  type Retrieve,
} from '@tensorspine/store/platform';

import { MANIFEST } from '../../scripts/manifest.js';
import { BUNDLES, PublishError, publish, WHOLE_SET } from '../../scripts/publish.js';
import { editorRoot, readEditorFile } from './tree.js';

// **The manifest is generated, never written by hand** — feature 2.20, and the user's own
// instruction with it. `pnpm manifest <dir>` is the generator a laboratory runs over the directory
// it serves its primitive library base from; `pnpm vendor` is the same artifact over the
// repository's own material (feature 2.18). One writer (`scripts/manifest.ts`), one walk
// (`scripts/publish.ts`), and **one reader** — `@tensorspine/store/platform`, which is what the
// editor opens a fetched base and a fetched workspace with.
//
// So this file holds the reader to the generator's own output rather than to a shape written down
// twice: the generator runs here, over a real base of this repository, and the reader reads what
// it wrote. A digest either generator computed differently, a path either spelled differently, a
// bundle either wrote differently, would fail here instead of at somebody's address.
//
// (The vendor's own manifest is read by the same reader in `vendor.test.ts`, where the vendor
// already runs: the two generators are held to the one reader, each where it is generated.)

const temporary: string[] = [];

afterAll(() => {
  for (const path of temporary) rmSync(path, { recursive: true, force: true });
});

/** A copy of a directory of this repository, to publish without touching the tree. */
function copyOf(from: string): string {
  const out = mkdtempSync(join(tmpdir(), 'tensorspine-publish-'));
  temporary.push(out);
  cpSync(join(editorRoot, from), out, { recursive: true });
  return out;
}

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

/** Every file below a directory, as `/`-separated paths, sorted — the walk read back. */
function walked(root: string, at = ''): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(at === '' ? root : join(root, at), { withFileTypes: true })) {
    const path = at === '' ? entry.name : `${at}/${entry.name}`;
    if (entry.isDirectory()) found.push(...walked(root, path));
    else found.push(path);
  }
  return found.sort();
}

/** A host that serves a directory, so the reader reads what the generator wrote. */
function serving(root: string, at = 'https://lab.example/base/'): Retrieve {
  return (url) => {
    if (!url.startsWith(at)) return Promise.reject(new Error(`${url} is not on this host`));
    const path = join(root, url.slice(at.length));
    if (!existsSync(path)) {
      return Promise.reject(new PlatformError(`${url} answered 404 Not Found`, 'not-found'));
    }
    return Promise.resolve(readFileSync(path, 'utf8'));
  };
}

const digest = (text: string): Promise<string> =>
  Promise.resolve(sha256(Buffer.from(text, 'utf8')));

describe('the manifest generator', () => {
  it('declares every file of the directory, with its own length and digest', () => {
    const out = copyOf('tests/fixtures/base');
    const { manifest } = publish({ directory: out, quiet: true });

    const written = walked(out).filter((path) => path !== MANIFEST);
    expect(manifest.files.map((file) => file.path)).toEqual(written);
    for (const file of manifest.files) {
      const bytes = readFileSync(join(out, file.path));
      expect(file.bytes, file.path).toBe(bytes.length);
      expect(file.sha256, file.path).toBe(sha256(bytes));
    }
    // No checkout behind this one — a directory somebody serves need not be in one — and that is
    // an answer rather than a refusal, which is what lets a laboratory publish from anywhere. The
    // commit *is* recorded where there is one, which `vendor.test.ts` holds the other generator to.
    expect(manifest.repository_commit).toBeNull();
    expect(manifest.repository_dirty).toBeNull();
    expect(manifest.generated_by).toBe('editor/scripts/publish.ts');
  });

  it('writes the set a second time as one bundle, and names it like every other file', () => {
    const out = copyOf('tests/fixtures/base');
    const { manifest } = publish({ directory: out, quiet: true });
    const bundle = manifest.bundles?.[0];
    expect(bundle).toBeDefined();
    expect(bundle?.path).toBe(`${BUNDLES}/${WHOLE_SET}.json`);
    // `covers: ''` is the whole set: this generator publishes one directory, not several sets.
    expect(bundle?.covers).toBe('');
    const carried = JSON.parse(readFileSync(join(out, bundle?.path ?? ''), 'utf8')) as Record<string, string>;
    const named = manifest.files.filter((file) => file.path !== bundle?.path);
    expect(Object.keys(carried).sort()).toEqual(named.map((file) => file.path).sort());
    for (const file of named) {
      expect(sha256(Buffer.from(carried[file.path] ?? '', 'utf8')), file.path).toBe(file.sha256);
    }
    // And the bundle is a file of the set like any other, so a reader can check it too.
    expect(manifest.files.some((file) => file.path === bundle?.path)).toBe(true);
  });

  it('publishes twice to the same set: its own output is never its input', () => {
    const out = copyOf('tests/fixtures/base');
    const first = publish({ directory: out, quiet: true }).manifest;
    const again = publish({ directory: out, quiet: true }).manifest;
    expect(again.files).toEqual(first.files);
    expect(again.bundles).toEqual(first.bundles);
  });

  it('declines the bundle where a caller says so, and where there is nothing to gain', () => {
    const out = copyOf('tests/fixtures/base');
    expect(publish({ directory: out, quiet: true, bundle: false }).manifest.bundles).toBeUndefined();
    expect(existsSync(join(out, BUNDLES))).toBe(false);

    const one = mkdtempSync(join(tmpdir(), 'tensorspine-publish-'));
    temporary.push(one);
    writeFileSync(join(one, 'primitive-library.json'), '{"schema": "x"}\n');
    const single = publish({ directory: one, quiet: true }).manifest;
    expect(single.bundles).toBeUndefined();
    expect(single.files).toHaveLength(1);
  });

  it('removes the one file it owns under `bundles/`, and leaves whatever else is there', () => {
    // A directory called `bundles/` may be a laboratory's own, and a generator that cleared it
    // would destroy what it was asked to publish.
    const out = copyOf('tests/fixtures/base');
    mkdirSync(join(out, BUNDLES), { recursive: true });
    writeFileSync(join(out, BUNDLES, 'theirs.json'), '{"kept": true}\n');
    const first = publish({ directory: out, quiet: true }).manifest;
    expect(first.files.map((file) => file.path)).toContain(`${BUNDLES}/theirs.json`);
    const again = publish({ directory: out, quiet: true }).manifest;
    expect(existsSync(join(out, BUNDLES, 'theirs.json'))).toBe(true);
    expect(again.files).toEqual(first.files);
  });

  it('skips a name beginning with a dot, as the loader’s own walk does', () => {
    const out = copyOf('tests/fixtures/base');
    mkdirSync(join(out, '.git'), { recursive: true });
    writeFileSync(join(out, '.git', 'HEAD'), 'ref: refs/heads/main\n');
    writeFileSync(join(out, '.notes.json'), '{"mine": true}\n');
    const { manifest } = publish({ directory: out, quiet: true });
    const paths = manifest.files.map((file) => file.path);
    expect(paths.filter((path) => path.startsWith('.'))).toEqual([]);
    expect(paths).toContain('primitive-library.json');
  });

  it('refuses what is not a published set, rather than declaring an empty one', () => {
    const empty = mkdtempSync(join(tmpdir(), 'tensorspine-publish-'));
    temporary.push(empty);
    expect(() => publish({ directory: empty, quiet: true })).toThrow(PublishError);
    expect(() => publish({ directory: empty, quiet: true })).toThrow(/nothing to declare/);
    expect(() => publish({ directory: join(empty, 'nowhere'), quiet: true })).toThrow(/is not there/);
  });

  it('is `pnpm manifest <dir>`, and the command is what a laboratory runs', () => {
    const out = copyOf('tests/fixtures/base');
    const scripts = (JSON.parse(readEditorFile('package.json')) as { scripts: Record<string, string> })
      .scripts;
    expect(scripts['manifest']).toBe('node --experimental-strip-types scripts/publish.ts');
    // Run the command itself, not the function it calls: what a laboratory types is what is held
    // to account here, argument parsing and all.
    execFileSync('node', ['--experimental-strip-types', join(editorRoot, 'scripts', 'publish.ts'), out, '--quiet'], {
      encoding: 'utf8',
    });
    const written = JSON.parse(readFileSync(join(out, MANIFEST), 'utf8')) as PublishedManifest;
    expect(written.files.length).toBeGreaterThan(1);
  });
});

describe('the reader and the generator', () => {
  it('call the manifest by one name', () => {
    // The package may not import a build script, so the name is written twice; this is the only
    // thing that keeps the two copies together.
    expect(PUBLISHED_MANIFEST).toBe(MANIFEST);
  });

  it('reads what the generator wrote: the same files, the same digests, the same commit', () => {
    const out = copyOf('tests/fixtures/base');
    const { manifest } = publish({ directory: out, quiet: true });
    const read = readPublishedManifest(readFileSync(join(out, MANIFEST), 'utf8'), 'at');
    expect(read.files).toEqual(manifest.files);
    expect(read.bundles).toEqual(manifest.bundles);
    expect(read.repository_commit).toBe(manifest.repository_commit);
  });

  it('fetches the whole set through the bundle the generator wrote — two requests', async () => {
    const out = copyOf('tests/fixtures/base');
    const { manifest } = publish({ directory: out, quiet: true });
    const set = await fetchPublishedSet({
      address: 'https://lab.example/base/',
      retrieve: serving(out),
      digest,
    });
    expect(Object.keys(set.texts).sort()).toEqual(manifest.files.map((file) => file.path).sort());
    for (const file of manifest.files) {
      expect(set.texts[file.path], file.path).toBe(readFileSync(join(out, file.path), 'utf8'));
    }
    expect(set.requests).toBe(2);
  });

  it('refuses a file the generator declared and the host serves differently, naming the path', async () => {
    const out = copyOf('tests/fixtures/base');
    publish({ directory: out, quiet: true, bundle: false });
    // What a base edited after it was published looks like from an address: the bytes moved and
    // the manifest did not.
    const unit = 'primitives/fixture/position_bias/1.0.0.json';
    const text = readFileSync(join(out, unit), 'utf8');
    writeFileSync(join(out, unit), text.replace('"version": "1.0.0"', '"version": "1.0.1"'));
    await expect(
      fetchPublishedSet({ address: 'https://lab.example/base/', retrieve: serving(out), digest }),
    ).rejects.toThrow(new RegExp(`${unit.replace(/[.]/g, '\\.')} is not what`));
  });
});

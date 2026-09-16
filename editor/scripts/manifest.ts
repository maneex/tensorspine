/**
 * What a **published file set** declares about itself, and the writer that declares it.
 *
 * Plain HTTP has no directory listing, so a set of files served over one cannot be discovered —
 * it has to be *declared*. That is why the static application can open the corpus at all
 * (feature 2.4's Examples workspace reads `vendor.json` before it fetches anything), and it is
 * the shape any other published set needs too: a base a lab puts at an address, a workspace
 * fetched rather than opened.
 *
 * So the manifest's own vocabulary and its writer live here, apart from `vendor.ts`'s directory
 * walk, its invocations of `tools/` and its bundles. What a generator has to do to publish a set
 * is three calls — {@link record} each file, {@link sorted} them, {@link writeManifest} last —
 * and none of them knows where the files came from or whether anything was copied at all.
 *
 * **The manifest is written last, and that is the contract.** A directory without one is not a
 * published set: a reader that fetched a manifest naming files a half-finished run had not
 * written yet would fail on the files rather than on the manifest, which is the wrong place to
 * find out. Every writer here therefore assumes its caller has already written what it names.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

/**
 * The name a published set's manifest is written under, and the one a reader asks for first.
 *
 * `vendor.json` because that is what feature 0.2 named it and what feature 2.4's `Vendor.open`
 * fetches; the name travels with the shape rather than being restated by each generator.
 */
export const MANIFEST = 'vendor.json';

/** One file of a published set: where it is, how long it is, and what it is. */
export interface ManifestFile {
  /** The path a reader appends to the set's own address. */
  path: string;
  bytes: number;
  sha256: string;
}

/** What the checkout a set was generated from says about itself. */
export interface Provenance {
  /** The commit the bytes come from, or `null` outside a checkout. */
  repository_commit: string | null;
  /** Whether the working tree still matched that commit, or `null` where it cannot be told. */
  repository_dirty: boolean | null;
}

/**
 * What every published file set declares, whatever else it adds.
 *
 * A generator extends this with what is its own — the vendor adds its sets, its bundles and the
 * artifacts the tools wrote — and a reader that knows only this can still fetch the whole set and
 * check every file it gets.
 */
export interface PublishedManifest extends Provenance {
  /** The script that wrote it, named as the repository names it. */
  generated_by: string;
  files: ManifestFile[];
}

/** One file's record: its length and its digest, taken from the bytes themselves. */
export function record(path: string, bytes: Buffer): ManifestFile {
  return { path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}

/**
 * The records in the order a manifest states them: by path, so two runs over one set agree.
 *
 * A copy, not a sort in place: a caller that kept its own order keeps it.
 */
export function sorted(files: readonly ManifestFile[]): ManifestFile[] {
  return [...files].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

/**
 * The commit a set was generated from, and whether the tree still matched it.
 *
 * Both `null` where `root` is no checkout — a published set is still a published set, and saying
 * "no commit" is more use than refusing to write one.
 */
export function provenance(root: string): Provenance {
  const rev = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
  if (rev.error || rev.status !== 0) return { repository_commit: null, repository_dirty: null };
  const status = spawnSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' });
  const dirty = status.error || status.status !== 0 ? null : status.stdout.trim() !== '';
  return { repository_commit: rev.stdout.trim(), repository_dirty: dirty };
}

/**
 * Write the manifest into a set's own directory — **last**, when everything it names is there.
 *
 * Two spaces and a trailing newline, as every JSON this repository writes has (D12's rule for
 * documents, kept here so that a manifest reads like the rest of the tree in a diff).
 */
export function writeManifest<T extends PublishedManifest>(out: string, manifest: T, name = MANIFEST): T {
  const path = join(out, name);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

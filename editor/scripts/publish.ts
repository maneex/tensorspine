#!/usr/bin/env node
/**
 * `pnpm manifest <dir>` — declare a directory as a **published file set** (feature 2.20).
 *
 * Plain HTTP has no directory listing, so a set of files served over one cannot be discovered: it
 * has to be *declared*. That is why the static application can open the corpus at all (feature
 * 2.4's Examples workspace reads `vendor.json` before it fetches anything), and it is what a
 * laboratory publishing a primitive library base at an address needs too — an address is the
 * address of a **manifest**, and a base that publishes none cannot be opened.
 *
 * So this is the second generator of the one artifact `scripts/manifest.ts` declares. It walks a
 * directory, records every file with its length and its sha256, writes the set a second time as
 * one bundle where that is worth doing, and declares the lot in `vendor.json` — the same writer,
 * the same shape, the same reader (`@tensorspine/store/platform`'s `readPublishedManifest`), over
 * a root that is nobody's checkout in particular.
 *
 * **Where the walk lives, and why here.** Feature 2.18 separated the manifest's *format and
 * writer* into `scripts/manifest.ts` and left "whether 2.20 shares a walk" to this feature. It is
 * shared, and it lives here rather than there: `manifest.ts` is what a published set **declares**
 * — it names no repository path, walks no directory and is audited for both, which is what lets a
 * generator publish a set of files nothing copied — and this module is what **gathers** one from a
 * directory. `vendor.ts` is the other caller of the walk; there is one of it.
 *
 * **The manifest is generated, never written by hand.** A digest somebody typed is a digest
 * nobody checked, and a file list somebody maintained is a file list that goes stale on the next
 * `cp`. The reader checks every file it fetches against this, so what this writes is what the
 * integrity of a fetched base rests on.
 *
 * Usage:  pnpm manifest <dir> [--no-bundle] [--quiet]
 */
import { readFileSync, readdirSync, rmSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  MANIFEST,
  provenance,
  record,
  sorted,
  writeManifest,
  type ManifestFile,
  type PublishedBundle,
  type PublishedManifest,
} from './manifest.ts';

/** A refusal with its cause: what could not be published, and what to do about it. */
export class PublishError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PublishError';
  }
}

/**
 * Every regular file below `root`, as `/`-separated paths, sorted, directories descended.
 *
 * Shared with `vendor.ts`, which is the other generator of this artifact. Anything that is neither
 * a file nor a directory — a symbolic link, a socket, a device — is a refusal rather than a
 * silent omission: a manifest that does not name a file the server will serve is worse than no
 * manifest at all.
 */
export function walk(root: string, at = ''): string[] {
  const here = at === '' ? root : join(root, at);
  const entries = readdirSync(here, { withFileTypes: true });
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const found: string[] = [];
  for (const entry of entries) {
    // A name beginning with a dot is skipped at every level, which is `glob`'s `**` rule and the
    // one the library loader itself follows over a base's sections (feature 1.3). It is also what
    // keeps `.git` and a laboratory's own working files out of a set it meant to publish.
    if (entry.name.startsWith('.')) continue;
    const path = at === '' ? entry.name : `${at}/${entry.name}`;
    if (entry.isDirectory()) found.push(...walk(root, path));
    else if (entry.isFile()) found.push(path);
    else {
      throw new PublishError(
        `${join(root, path)} is neither a file nor a directory — a published set is regular files`,
      );
    }
  }
  return found;
}

/** Where a set written a second time as one file lands, under the set's own root. */
export const BUNDLES = 'bundles';

/** What the whole-set bundle of {@link publish} is called. */
export const WHOLE_SET = 'all';

/** Where that bundle lands — the one file under `bundles/` this generator owns. */
export const WHOLE_SET_PATH = `${BUNDLES}/${WHOLE_SET}.json`;

/**
 * Write the files under `covers` a second time, whole, as one JSON object from path to text.
 *
 * The measurement this exists for is feature 2.18's, and a fetched set faces the same arithmetic
 * one origin further away: **151 files cost 257–275 ms one at a time against 20–41 ms in four
 * bundles**. The bundle's keys are the manifest's own paths, so a reader that has the manifest
 * needs no second naming convention, and the digests stay the *files'* — a bundle is a transport,
 * never a second source, and the reader checks what comes out of it against the manifest exactly
 * as it checks a file it fetched on its own.
 *
 * `null` where there is nothing to gain (fewer than two files) or nothing safe to do: a file whose
 * bytes are not UTF-8 cannot travel through a JSON string and would come back changed, so a set
 * holding one is published without a bundle rather than with one that lies.
 */
export function writeBundle(
  out: string,
  name: string,
  covers: string,
  paths: readonly string[],
): PublishedBundle | null {
  if (paths.length < 2) return null;
  const bundled: Record<string, string> = {};
  for (const path of paths) {
    const bytes = readFileSync(join(out, path));
    const text = bytes.toString('utf8');
    if (!Buffer.from(text, 'utf8').equals(bytes)) return null;
    bundled[path] = text;
  }
  const at = `${BUNDLES}/${name}.json`;
  const bytes = Buffer.from(`${JSON.stringify(bundled)}\n`, 'utf8');
  const file = join(out, at);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, bytes);
  return { name, path: at, covers, files: paths.length, bytes: bytes.length };
}

/** What {@link publish} is given. */
export interface PublishOptions {
  /** The directory to declare. Every regular file below it is published. */
  readonly directory: string;
  /** What the manifest says wrote it; this script by default. */
  readonly generatedBy?: string | undefined;
  /** Write the whole set a second time as one file (the default). */
  readonly bundle?: boolean | undefined;
  /** Say nothing on success. */
  readonly quiet?: boolean | undefined;
}

export interface PublishResult {
  /** The directory declared. */
  readonly out: string;
  /** The manifest, as it was written to {@link MANIFEST}. */
  readonly manifest: PublishedManifest;
}

/**
 * Declare a directory as a published file set: walk it, record it, bundle it, write the manifest.
 *
 * The manifest is written **last**, when everything it names is there — a directory without one is
 * not a published set, and a reader that fetched a manifest naming files a half-finished run had
 * not written would fail on the files rather than on the manifest, which is the wrong place to
 * find out.
 *
 * A previous run's own output is not published a second time: the manifest and the **one bundle
 * this generator writes** are removed before the walk, so publishing twice gives the same set both
 * times. Nothing else under `bundles/` is touched — a directory of that name may be a laboratory's
 * own, and a generator that removed it would destroy what it was asked to publish.
 */
export function publish(options: PublishOptions): PublishResult {
  const out = resolve(options.directory);
  let where;
  try {
    where = statSync(out);
  } catch {
    throw new PublishError(`${out} is not there: a published set is a directory of files`);
  }
  if (!where.isDirectory()) {
    throw new PublishError(`${out} is a file: a published set is a directory of files`);
  }

  // A previous run's manifest and its own bundle are this run's to write, never its input. Only
  // that one file: everything else the directory holds is what it was asked to publish.
  rmSync(join(out, MANIFEST), { force: true });
  rmSync(join(out, WHOLE_SET_PATH), { force: true });

  const names = walk(out);
  if (names.length === 0) {
    throw new PublishError(
      `${out} holds no file: there is nothing to declare, and an empty manifest would say a set is ` +
        'there when none is',
    );
  }

  const files: ManifestFile[] = names.map((name) => record(name, readFileSync(join(out, name))));
  const bundles: PublishedBundle[] = [];
  if (options.bundle !== false) {
    const made = writeBundle(out, WHOLE_SET, '', names);
    if (made !== null) {
      bundles.push(made);
      files.push(record(made.path, readFileSync(join(out, made.path))));
    }
  }

  const manifest: PublishedManifest = {
    generated_by: options.generatedBy ?? 'editor/scripts/publish.ts',
    ...provenance(out),
    ...(bundles.length === 0 ? {} : { bundles }),
    files: sorted(files),
  };
  writeManifest(out, manifest);

  if (options.quiet !== true) {
    const bytes = manifest.files.reduce((total, file) => total + file.bytes, 0);
    console.log(`manifest: ${String(manifest.files.length)} file(s), ${String(bytes)} byte(s) -> ${join(out, MANIFEST)}`);
    for (const one of bundles) {
      console.log(`  ${one.path.padEnd(24)} ${String(one.files).padStart(4)} file(s), ${String(one.bytes)} byte(s)`);
    }
    if (manifest.repository_commit === null) {
      console.log('  (no checkout behind it: the manifest records no commit)');
    }
  }
  return { out, manifest };
}

function main(argv: string[]): number {
  const options: { directory?: string; bundle?: boolean; quiet?: boolean } = {};
  for (const argument of argv) {
    if (argument === '--quiet') options.quiet = true;
    else if (argument === '--no-bundle') options.bundle = false;
    else if (argument.startsWith('--')) {
      console.error(`manifest: unknown option ${argument}`);
      console.error('usage: node scripts/publish.ts <dir> [--no-bundle] [--quiet]');
      return 2;
    } else if (options.directory === undefined) options.directory = argument;
    else {
      console.error('manifest: one directory at a time');
      return 2;
    }
  }
  if (options.directory === undefined) {
    console.error('manifest: which directory? usage: node scripts/publish.ts <dir> [--no-bundle] [--quiet]');
    return 2;
  }
  try {
    publish({
      directory: options.directory,
      ...(options.bundle === undefined ? {} : { bundle: options.bundle }),
      ...(options.quiet === undefined ? {} : { quiet: options.quiet }),
    });
    return 0;
  } catch (error) {
    if (error instanceof PublishError) {
      console.error(`manifest: ${error.message}`);
      return 2;
    }
    throw error;
  }
}

const invoked = process.argv[1];
if (invoked !== undefined && resolve(invoked) === fileURLToPath(import.meta.url)) {
  process.exit(main(process.argv.slice(2)));
}

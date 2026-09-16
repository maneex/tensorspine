/**
 * A **published file set**, read from an address — feature 2.20.
 *
 * > Plain HTTP has no directory listing, so a file **set** cannot be discovered — it must be
 * > declared. … So an address is the address of a **manifest**, and a base that publishes none
 * > cannot be opened. Say so in the refusal rather than guessing at a listing.
 *
 * This is the reading half of `editor/scripts/manifest.ts`: what a manifest declares, how the
 * files it names are gathered, and what each refusal says. It answers a {@link FileSet}, which is
 * the shape {@link ReadOnlyWorkspace} already reads — the third user of it after the folder
 * snapshot and the vendored Examples workspace, not a fourth kind of workspace.
 *
 * **Nothing here reaches for a platform.** The bytes are the caller's: {@link Retrieve} is how a
 * deployment gets a text from a URL and {@link Digest} is how it takes a sha256, both handed in,
 * exactly as `ReadOnlyWorkspace` is handed the shell's download. That is the same line feature 1.9
 * drew for a checkpoint's header — the bytes are the caller's, the format is ours — and it is what
 * lets the whole of this module be tested without a browser and without a server.
 *
 * **What is stated rather than assumed**, because the feature's own block asks for each of them:
 *
 *  - **Integrity.** The manifest carries a sha256 per file and every file is checked against it as
 *    it arrives, whichever way it arrived — on its own, or out of a bundle. A file that does not
 *    match is refused **with its path**, and the set is not opened at all: a set that was not
 *    served whole is not a set.
 *  - **Trust.** What comes back is *data*. Nothing here parses a document, nothing here executes
 *    anything, and nothing here decides what a unit means: the texts are handed to the core, which
 *    puts every one of them through the unit schema and the loader's cross-references, and V1
 *    refuses a redefinition of an identity another base holds (§9 Q6). The reference base's lock is
 *    untouched — a fetched base is another base beside it, never a replacement for it.
 *  - **The whole set, at once.** The set is known from the manifest before the first file is
 *    fetched, so it is fetched whole and checked whole when it is opened, rather than file by file
 *    as somebody reads. Three things follow: a digest that disagrees is named when the set is
 *    opened and not at some later read; a set held in hand can be cached, which is what makes the
 *    offline answer honest; and a read costs nothing afterwards.
 *  - **The transport.** Feature 2.18 measured the vendored material at **151 files, 257–275 ms one
 *    at a time against 20–41 ms in four bundles**, and a fetched set pays the same arithmetic one
 *    origin further away. So a manifest may declare {@link PublishedBundle}s — the same files a
 *    second time, whole, as one JSON object — and this reads through them when it can. A bundle is
 *    a transport and never a second source: what comes out of one is checked against the same
 *    digest as what comes off the wire, and what a bundle does not carry is fetched on its own. A
 *    bundle the host does not serve **at all** is a different thing — a file the manifest declares
 *    and the host has not — and that refuses the set, like any other.
 */
import { PlatformError } from './errors.js';
import { normalise, type WorkspacePath } from './paths.js';
import { ReadOnlyWorkspace, type Deliver, type FileSet } from './readonly.js';
import type { WorkspaceKind } from './types.js';

/**
 * The name a published set's manifest is written under, and the one a reader asks for first.
 *
 * It is `editor/scripts/manifest.ts`'s `MANIFEST`, written out here because a package may not
 * import a build script; an audit holds the two to each other so they cannot part company.
 */
export const PUBLISHED_MANIFEST = 'vendor.json';

/** One file of a published set, as the manifest records it. */
export interface PublishedFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

/** A group of the set's files written a second time, whole; `covers` `''` is the whole set. */
export interface PublishedBundle {
  readonly name: string;
  readonly path: string;
  readonly covers: string;
  readonly files: number;
  readonly bytes: number;
}

/** What every published file set declares, of what a reader needs. */
export interface PublishedManifest {
  readonly generated_by?: string;
  /** The commit the bytes come from, or `null` where the generator ran outside a checkout. */
  readonly repository_commit?: string | null;
  readonly repository_dirty?: boolean | null;
  readonly bundles?: readonly PublishedBundle[];
  readonly files: readonly PublishedFile[];
}

/** How a deployment gets the text of one URL. Refuses by rejecting; never answers a partial text. */
export type Retrieve = (url: string) => Promise<string>;

/** How a deployment takes the sha256 of a text, as lower-case hexadecimal. */
export type Digest = (text: string) => Promise<string>;

/** An address taken apart: where the manifest is, and what a file's path is appended to. */
export interface Addressed {
  /** The manifest's own address. */
  readonly manifest: string;
  /** The set's root, ending in a separator. */
  readonly root: string;
}

/**
 * Where the manifest is, and where the files are, for an address somebody typed.
 *
 * Both spellings are accepted because both are what a person has in hand: the address of the
 * manifest itself (`https://lab.example/base/vendor.json`) and the address of the directory it is
 * in (`https://lab.example/base/`, and `https://lab.example/base` with the separator forgotten,
 * which is a directory as every server writes it). A file's path is appended to the directory,
 * which is the manifest's own — that is what makes a manifest's paths mean the same thing wherever
 * the set is served from.
 */
export function addressOf(address: string): Addressed {
  const trimmed = address.trim();
  if (trimmed === '') {
    throw new PlatformError('no address: a published set is opened by the address of its manifest', 'bad-path');
  }
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new PlatformError(
      `${trimmed} is not an address: a published set is opened by a whole URL, scheme and all ` +
        '(https://…/vendor.json, or the directory it is in)',
      'bad-path',
    );
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    throw new PlatformError(
      `${url.href} is not fetched over the web (${url.protocol}) — a published set is served over http or https`,
      'unsupported',
    );
  }
  // The query and the fragment are the address of a *page*, not of a set: a file's path is
  // appended to the root, and neither survives that.
  url.hash = '';
  url.search = '';
  const last = url.pathname.slice(url.pathname.lastIndexOf('/') + 1);
  if (last === PUBLISHED_MANIFEST) {
    return { manifest: url.href, root: url.href.slice(0, url.href.length - last.length) };
  }
  const root = url.href.endsWith('/') ? url.href : `${url.href}/`;
  return { manifest: `${root}${PUBLISHED_MANIFEST}`, root };
}

/**
 * The manifest a text declares, refused where it does not declare one.
 *
 * The refusal is the feature's own sentence: a directory that publishes no manifest is not a
 * published set, and saying so is the answer — there is no listing to fall back on and guessing at
 * one is what this whole shape exists to avoid.
 */
export function readPublishedManifest(text: string, at: string): PublishedManifest {
  let read: unknown;
  try {
    read = JSON.parse(text);
  } catch (error) {
    throw new PlatformError(
      `${at} is not a published set's manifest: it is not JSON (${String(error)}) — a set served ` +
        'over plain HTTP declares what it holds, since there is no directory listing to read',
      'corrupt',
    );
  }
  if (typeof read !== 'object' || read === null || Array.isArray(read)) {
    throw new PlatformError(
      `${at} is not a published set's manifest: it declares no files`,
      'corrupt',
    );
  }
  const held = read as { files?: unknown; bundles?: unknown };
  if (!Array.isArray(held.files)) {
    throw new PlatformError(
      `${at} is not a published set's manifest: it declares no files — run \`pnpm manifest <dir>\` ` +
        'over the directory it is served from',
      'corrupt',
    );
  }
  const files: PublishedFile[] = [];
  for (const [index, one] of held.files.entries()) {
    const file = one as { path?: unknown; bytes?: unknown; sha256?: unknown };
    if (typeof file.path !== 'string' || file.path === '') {
      throw new PlatformError(`${at}: the file at ${String(index)} has no path`, 'corrupt');
    }
    // A manifest is data from another origin, so a path in it is checked before it is *used*: it
    // is appended to the set's own root to make a URL and it becomes a place in a workspace, and
    // a set may name neither a place above itself nor one somewhere else entirely.
    if (!insideSet(file.path)) {
      throw new PlatformError(
        `${at}: ${file.path} is not a path inside the set — a published set names its own files, ` +
          'relative to where its manifest is',
        'corrupt',
      );
    }
    if (typeof file.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(file.sha256)) {
      throw new PlatformError(`${at}: ${file.path} has no sha256, so nothing it holds could be checked`, 'corrupt');
    }
    files.push({
      path: file.path,
      bytes: typeof file.bytes === 'number' ? file.bytes : 0,
      sha256: file.sha256,
    });
  }
  const bundles: PublishedBundle[] = [];
  if (Array.isArray(held.bundles)) {
    for (const one of held.bundles) {
      const bundle = one as { name?: unknown; path?: unknown; covers?: unknown; files?: unknown; bytes?: unknown };
      if (typeof bundle.path !== 'string' || typeof bundle.covers !== 'string') continue;
      bundles.push({
        name: typeof bundle.name === 'string' ? bundle.name : bundle.path,
        path: bundle.path,
        covers: bundle.covers,
        files: typeof bundle.files === 'number' ? bundle.files : 0,
        bytes: typeof bundle.bytes === 'number' ? bundle.bytes : 0,
      });
    }
  }
  const manifest: PublishedManifest = {
    ...(typeof (read as { generated_by?: unknown }).generated_by === 'string'
      ? { generated_by: (read as { generated_by: string }).generated_by }
      : {}),
    repository_commit: commitOf(read),
    repository_dirty: dirtyOf(read),
    ...(bundles.length === 0 ? {} : { bundles }),
    files,
  };
  return manifest;
}

/**
 * Whether a path a manifest declares names a file **of the set** and nothing else.
 *
 * Relative, with no empty, `.` or `..` segment and no scheme: the path is appended to the set's
 * root to make a URL and is used as a place in a workspace, and neither may leave the set.
 */
function insideSet(path: string): boolean {
  if (path.startsWith('/') || path.startsWith('\\') || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(path)) return false;
  return path.split('/').every((segment) => segment !== '' && segment !== '.' && segment !== '..');
}

/** The commit a manifest records, `null` where it records none or records something else. */
function commitOf(read: unknown): string | null {
  const said = (read as { repository_commit?: unknown }).repository_commit;
  return typeof said === 'string' && said !== '' ? said : null;
}

/** Whether the tree was dirty, `null` where the manifest does not say. */
function dirtyOf(read: unknown): boolean | null {
  const said = (read as { repository_dirty?: unknown }).repository_dirty;
  return said === true || said === false ? said : null;
}

/** A set fetched from an address: what it declares, what it holds, and what it cost. */
export interface PublishedSet {
  /** The manifest's own address, as it was asked for. */
  readonly address: string;
  /** The set's root, ending in a separator: what a file's path is appended to. */
  readonly root: string;
  readonly manifest: PublishedManifest;
  /** Every file the manifest names, by its own path, checked against its digest. */
  readonly texts: Readonly<Record<string, string>>;
  /** When it was gathered, ISO 8601 — the cache's own moment where it came from there. */
  readonly fetchedAt: string;
  /** True where nothing was reached and this came out of the cache (the offline answer). */
  readonly cached: boolean;
  /** How many requests it cost, and how many bytes it holds — the transport, measured. */
  readonly requests: number;
  readonly bytes: number;
}

/** What {@link fetchPublishedSet} is given. */
export interface FetchOptions {
  readonly address: string;
  readonly retrieve: Retrieve;
  readonly digest: Digest;
  /** Read the files one at a time even where the manifest declares a bundle — a measurement's. */
  readonly bundles?: boolean | undefined;
  /** What "now" is, for a set that records when it was gathered. */
  readonly now?: (() => string) | undefined;
}

/**
 * Fetch a published set whole: its manifest, then every file it names, each checked as it arrives.
 *
 * The bundle a manifest declares is read first where there is one, and what it does not carry is
 * fetched on its own — so a set that declares one costs **two** requests, the manifest and the
 * bundle, where a set that declares none costs the manifest and one per file. That is feature
 * 2.18's measurement stated as an arithmetic, one origin further away.
 */
export async function fetchPublishedSet(options: FetchOptions): Promise<PublishedSet> {
  const { manifest: at, root } = addressOf(options.address);
  let requests = 1;
  const manifest = readPublishedManifest(await options.retrieve(at), at);

  /** Each bundle, read once: the text that came back, and what it carries. */
  const held = new Map<string, { text: string; carries: Readonly<Record<string, string>> }>();
  const bundles = options.bundles === false ? [] : (manifest.bundles ?? []);
  const bundleAt = new Set(bundles.map((one) => one.path));
  /** The bundle a file would come out of, a bundle never coming out of itself. */
  const bundleFor = (path: string): PublishedBundle | null =>
    bundleAt.has(path)
      ? null
      : (bundles.find((one) => one.covers === '' || path.startsWith(`${one.covers}/`)) ?? null);

  const wanted = new Set<string>();
  for (const file of manifest.files) {
    const bundle = bundleFor(file.path);
    if (bundle !== null) wanted.add(bundle.path);
  }
  for (const path of wanted) {
    requests += 1;
    // A bundle that does not arrive, or does not read as one, is a **transport** that failed and
    // not a set that lied: every file it would have carried is still there on its own, and that is
    // what the fall-back below reads. Its own entry in `files` is checked like any other, so a
    // bundle the host does not serve at all is still a set that was not served whole.
    const text = await options.retrieve(`${root}${path}`).catch(() => null);
    if (text === null) continue;
    try {
      const read = JSON.parse(text) as unknown;
      if (typeof read === 'object' && read !== null && !Array.isArray(read)) {
        held.set(path, { text, carries: read as Readonly<Record<string, string>> });
      }
    } catch {
      // Not JSON: the same answer, and the text is kept so that its own digest is still checked
      // against what the manifest says it should have been.
      held.set(path, { text, carries: {} });
    }
  }

  const texts: Record<string, string> = {};
  // The lengths the **generator** recorded, which are bytes; a text's own length is characters,
  // and the two part company on the first non-ASCII one (the corpus writes fifteen).
  let bytes = 0;
  for (const file of manifest.files) bytes += file.bytes;
  const reading = manifest.files.map(async (file) => {
    const bundle = bundleFor(file.path);
    const carried = bundle === null ? held.get(file.path)?.text : held.get(bundle.path)?.carries[file.path];
    let text: string;
    if (carried === undefined) {
      requests += 1;
      text = await options.retrieve(`${root}${file.path}`);
    } else text = carried;
    const found = await options.digest(text);
    if (found !== file.sha256) {
      throw new PlatformError(
        `${file.path} is not what ${at} declares: sha256 ${found}, the manifest says ${file.sha256}` +
          ' — the set was not served whole, so none of it is opened',
        'corrupt',
      );
    }
    return { path: file.path, text };
  });
  for (const one of await Promise.all(reading)) texts[one.path] = one.text;

  return {
    address: at,
    root,
    manifest,
    texts,
    fetchedAt: (options.now ?? (() => new Date().toISOString()))(),
    cached: false,
    requests,
    bytes,
  };
}

/**
 * The paths of the set that are **bundles** — a transport, and not a file of the set's own.
 *
 * A bundle is the set's files a second time, so a reader that handed one to the editor would hand
 * over every unit twice: once as itself and once as a JSON object of them all. Its digest is
 * checked like any other file's when it arrives — it is declared, and a set that was not served
 * whole is not a set — and it is left out of what the set *holds*.
 */
function bundlePaths(manifest: PublishedManifest): Set<string> {
  return new Set((manifest.bundles ?? []).map((one) => one.path));
}

/**
 * The set as a {@link FileSet}, with the manifest's digest as each file's revision.
 *
 * The digest is stronger than the modification time and size a folder answers with, and it is what
 * the Examples workspace already uses for the same reason: a published set cannot change under the
 * page, so two reads of one file answer one revision and a Save As is never in conflict with
 * itself. A file the manifest names under a bundle is the same file either way — the digests are
 * the files' own.
 */
export function publishedFileSet(
  set: PublishedSet,
  kind: WorkspaceKind,
  id: string,
  name: string,
): FileSet {
  const held = new Map<WorkspacePath, { text: string; revision: string }>();
  const bundles = bundlePaths(set.manifest);
  for (const file of set.manifest.files) {
    const text = set.texts[file.path];
    if (text === undefined || bundles.has(file.path)) continue;
    const path = normalise(file.path);
    if (path !== '') held.set(path, { text, revision: file.sha256 });
  }
  return {
    kind,
    id,
    name,
    paths: () => [...held.keys()].sort((a, b) => a.localeCompare(b)),
    read: (path) => Promise.resolve(held.get(path) ?? null),
    readMany: (paths) => {
      const found: Record<WorkspacePath, { text: string; revision: string }> = {};
      for (const path of paths) {
        const one = held.get(path);
        if (one !== undefined) found[path] = one;
      }
      return Promise.resolve(found);
    },
    revision: (path) => held.get(path)?.revision ?? null,
  };
}

/**
 * What a set opened as the workspace is called, and what it is known by across reloads.
 *
 * The name is the last segment of the set's own root — `base` of `https://lab.example/base/` —
 * failing that the host, because a set served at the root of a site has no other name. The
 * identity is the manifest's address: a draft, a remembered tab and a mounted base all find the
 * set again by the address they were opened from, which is the one thing about it that does not
 * change between two readers.
 */
export function publishedName(set: PublishedSet): string {
  return nameOfRoot(set.root);
}

/** The name a set's own root gives it, for a caller that has an address and not a set yet. */
export function nameOfRoot(root: string): string {
  let url: URL;
  try {
    url = new URL(root);
  } catch {
    return root;
  }
  const segments = url.pathname.split('/').filter((segment) => segment !== '');
  return segments.at(-1) ?? url.host;
}

/** The workspace identity of a set opened from an address: the address itself. */
export function publishedId(set: PublishedSet): string {
  return `remote:${set.address}`;
}

/**
 * A published set opened as the workspace: read-only, with Save As to copy a document out.
 *
 * The third user of {@link ReadOnlyWorkspace} after the folder snapshot and the vendored Examples
 * workspace, and not a fourth kind of workspace: what differs is where the files came from, which
 * is the {@link FileSet}'s business and not the workspace's. Save hands the bytes to the user as
 * a download, exactly as it does on a snapshot — a fetched folder is nobody's to write back to.
 */
export function publishedWorkspace(set: PublishedSet, deliver: Deliver): ReadOnlyWorkspace {
  return new ReadOnlyWorkspace(
    publishedFileSet(set, 'remote', publishedId(set), publishedName(set)),
    deliver,
  );
}

/**
 * The set's files under a prefix, by their path **relative to it** — what a mounted base hands over.
 *
 * A base fetched from an address stands somewhere in the open workspace's own path space (a
 * document pins a base by the path it resolves, §5.2 rule and V1), so what the gathering layer
 * wants is the files under the set's root keyed the way they will be seen there, not the way they
 * are served.
 */
export function textsUnder(
  set: PublishedSet,
  prefix = '',
): Record<string, string> {
  const at = prefix === '' ? '' : `${normalise(prefix)}/`;
  const bundles = bundlePaths(set.manifest);
  const found: Record<string, string> = {};
  for (const [path, text] of Object.entries(set.texts)) {
    if (!path.startsWith(at) || bundles.has(path)) continue;
    const rest = path.slice(at.length);
    if (rest !== '') found[rest] = text;
  }
  return found;
}

/**
 * A set built from texts already in hand — what a cache answers with, and what a stub platform is.
 *
 * The manifest is the set's own, so the digests are still the generator's: nothing here recomputes
 * one, because a digest a reader computed from what it already holds says nothing about what was
 * served.
 */
export function heldSet(
  address: string,
  root: string,
  manifest: PublishedManifest,
  texts: Readonly<Record<string, string>>,
  fetchedAt: string,
  cached: boolean,
): PublishedSet {
  let bytes = 0;
  for (const file of manifest.files) bytes += file.bytes;
  return { address, root, manifest, texts, fetchedAt, cached, requests: 0, bytes };
}

/** How long ago something was gathered, in the words a banner says it in. */
export function ageOf(fetchedAt: string, now = Date.now()): string {
  const taken = Date.parse(fetchedAt);
  if (Number.isNaN(taken)) return fetchedAt;
  const seconds = Math.max(0, Math.round((now - taken) / 1000));
  if (seconds < 90) return `${String(seconds)} second(s) ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${String(minutes)} minute(s) ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 36) return `${String(hours)} hour(s) ago`;
  return `${String(Math.round(hours / 24))} day(s) ago`;
}

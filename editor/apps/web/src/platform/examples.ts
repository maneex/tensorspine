/**
 * The **Examples** workspace — feature 2.4, D11 and §4.3.
 *
 * > The static build also vendors `data/models/` as a read-only **Examples** workspace, so the
 * > corpus opens in the editor with nothing to download and the editor takes over the site's model
 * > views once `view.py` is removed (F6).
 *
 * The vendor is `pnpm vendor`'s output, served beside the application as static files: the
 * repository's `schemas/`, `data/models/`, `data/primitive-library/` and the tools' generated
 * artifacts, with `vendor.json` naming every one of them with its length and its sha256 (feature
 * 0.2). That manifest is what makes a workspace possible over plain HTTP with no directory
 * listing: the file **set** is known before anything is fetched, which is exactly what a
 * {@link FileSet} is.
 *
 * The workspace's root is the vendor's `data/`, so its paths are the repository's own below that
 * point — `models/llama3-8b.json`, `primitive-library/primitive-library.json` — and a document
 * resolves its bases exactly as it does in `data/`: `"base": "../primitive-library/"` from
 * `models/…` is `primitive-library`, and the base manifest's `"templates": "../models/"` is
 * `models`. Feature 0.2 kept that layout for this reason; nothing here may flatten it.
 *
 * Read-only, with Save As to copy a document out (§4.3): a write goes to the shell as a download,
 * through the same {@link ReadOnlyWorkspace} the snapshot path uses.
 */
import {
  PlatformError,
  ReadOnlyWorkspace,
  normalise,
  type Deliver,
  type FileSet,
  type WorkspacePath,
} from '@tensorspine/store/platform';

/** One vendored file, as `vendor.json` records it. */
export interface VendorFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

/**
 * One set of vendored files written a second time as a single file — feature 2.18.
 *
 * Feature 2.6 measured the first document of a session at 417–449 ms against §5.6's 300 ms for a
 * library load, of which about 130 ms was the reference base's 131 requests. A set that is always
 * read whole is therefore fetched whole, once, and every read below it is answered from what came
 * back. The per-file copies are still there and are still what the manifest's digests are of: a
 * bundle is a transport, not a second source.
 */
export interface VendorBundle {
  readonly name: string;
  readonly path: string;
  /** The vendored path prefix every file in it is under. */
  readonly covers: string;
  readonly files: number;
  readonly bytes: number;
}

/** What `vendor.json` says, of what this module needs. */
export interface VendorManifest {
  readonly repository_commit: string | null;
  readonly examples: { readonly root: string; readonly models: string; readonly primitive_library: string };
  readonly schemas: readonly { readonly path: string; readonly id: string | null }[];
  /**
   * The tools' generated argument schemas, by `<name>@<version>` — `--document primitive-schema`.
   *
   * Consumed as built and never regenerated (plan §1, F5): the argument sheet of §4.12 reads the
   * literal-mode widget, its bounds, its options and its unit from the file this names.
   */
  readonly primitive_schemas?: Readonly<Record<string, string>>;
  /**
   * The sets written whole, for the readers that read a set whole.
   *
   * Absent from a vendor older than feature 2.18, and a build that serves none is a build that
   * reads its files one at a time — slower, and right.
   */
  readonly bundles?: readonly VendorBundle[];
  readonly files: readonly VendorFile[];
}

/** Where the vendor is served from, relative to the page, when nothing says otherwise. */
export const VENDOR = 'vendor/';

/** The material the build vendored, read over HTTP. */
export class Vendor {
  /**
   * Read the manifest.
   *
   * A build that was not vendored has none, and the refusal says what to run: the alternative is
   * an editor with no schemas, no library and no examples, failing later and less clearly.
   */
  static async open(root: string): Promise<Vendor> {
    const at = root.endsWith('/') ? root : `${root}/`;
    const url = `${at}vendor.json`;
    let response: Response;
    try {
      response = await fetch(url);
    } catch (error) {
      throw new PlatformError(
        `the vendored material could not be read from ${url} (${String(error)}) — run \`pnpm vendor\``,
        'not-found',
      );
    }
    if (!response.ok) {
      throw new PlatformError(
        `the vendored material is not at ${url} (${String(response.status)}) — run \`pnpm vendor\``,
        'not-found',
      );
    }
    return new Vendor(at, (await response.json()) as VendorManifest);
  }

  private constructor(
    /** The vendor root, ending in a separator. */
    readonly root: string,
    readonly manifest: VendorManifest,
  ) {}

  /** Each bundle's contents once it has been asked for — one request per set, per session. */
  private readonly held = new Map<string, Promise<Readonly<Record<string, string>>>>();

  /** Every vendored path under a prefix, in the manifest's order (which is sorted). */
  paths(prefix: string): string[] {
    const at = prefix === '' ? '' : `${normalise(prefix)}/`;
    return this.manifest.files.map((one) => one.path).filter((path) => path.startsWith(at));
  }

  /** The bundle a vendored path is in, where the build wrote one. */
  bundleFor(path: string): VendorBundle | null {
    return (this.manifest.bundles ?? []).find((one) => path.startsWith(`${one.covers}/`)) ?? null;
  }

  /**
   * One bundle's files, fetched once.
   *
   * The promise is what is held, not its value, so two readers that ask at the same moment — the
   * schemas and the base of one document, the fifteen documents of a corpus round trip — make one
   * request between them. A bundle that does not come back is not an error: the caller falls back
   * to the file itself, which is still there.
   */
  private bundled(bundle: VendorBundle): Promise<Readonly<Record<string, string>>> {
    const already = this.held.get(bundle.path);
    if (already !== undefined) return already;
    const reading = (async (): Promise<Readonly<Record<string, string>>> => {
      const response = await fetch(`${this.root}${bundle.path}`);
      if (!response.ok) return {};
      return (await response.json()) as Readonly<Record<string, string>>;
    })().catch(() => ({}));
    this.held.set(bundle.path, reading);
    return reading;
  }

  /**
   * The text of one vendored file, by its path under the vendor root.
   *
   * Through the bundle its set was written into where there is one — one request for a hundred
   * and thirty files instead of a hundred and thirty (feature 2.18) — and from the file itself
   * otherwise, which is also what a bundle that did not arrive falls back to.
   */
  async read(path: string): Promise<string> {
    const bundle = this.bundleFor(path);
    if (bundle !== null) {
      const held = (await this.bundled(bundle))[path];
      if (held !== undefined) return held;
    }
    const response = await fetch(`${this.root}${path}`);
    if (!response.ok) {
      throw new PlatformError(`no vendored file ${path} (${String(response.status)})`, 'not-found');
    }
    return response.text();
  }

  /**
   * Every vendored file under a prefix, by its path, with its text — what the core is handed.
   *
   * Read at once rather than one after another: the set is known from the manifest before the
   * first request, there are a hundred and thirty of them under the reference base, and the
   * library load has three hundred milliseconds to its name (§5.6). Where the build wrote the set
   * as a bundle, "at once" is one request rather than a hundred and thirty in flight.
   */
  async readAll(prefix: string): Promise<Record<string, string>> {
    const paths = this.paths(prefix);
    const texts = await Promise.all(paths.map((path) => this.read(path)));
    const found: Record<string, string> = {};
    for (const [index, path] of paths.entries()) found[path] = texts[index] ?? '';
    return found;
  }

  /**
   * The Examples workspace: the corpus and the reference base, rooted at the vendor's `data/`.
   *
   * The revision of a file is the **digest** the manifest records, which is stronger than the
   * modification time and size a folder answers with: vendored material cannot change while the
   * page is open, and two reads of one file give one revision.
   */
  files(): FileSet {
    const root = this.manifest.examples.root;
    const prefix = `${root}/`;
    const held = new Map<WorkspacePath, VendorFile>();
    for (const file of this.manifest.files) {
      if (file.path.startsWith(prefix)) held.set(file.path.slice(prefix.length), file);
    }
    return {
      kind: 'examples',
      id: 'examples',
      name: 'Examples',
      paths: () => [...held.keys()].sort((a, b) => a.localeCompare(b)),
      read: async (path) => {
        const file = held.get(path);
        if (file === undefined) return null;
        return { text: await this.read(file.path), revision: file.sha256 };
      },
      // Together, not one after another: feature 2.4 measured the reference base's 131 files at
      // **167–215 ms** read in sequence and **82–90 ms** read at once, and a library load has
      // three hundred milliseconds to its name (§5.6). The set is known from the manifest before
      // the first request, which is what makes the bulk read possible at all.
      readMany: async (paths) => {
        const found: Record<WorkspacePath, { text: string; revision: string }> = {};
        const read = await Promise.all(
          paths.map(async (path) => {
            const file = held.get(path);
            if (file === undefined) return null;
            return { path, text: await this.read(file.path), revision: file.sha256 };
          }),
        );
        for (const one of read) if (one !== null) found[one.path] = { text: one.text, revision: one.revision };
        return found;
      },
      revision: (path) => held.get(path)?.sha256 ?? null,
    };
  }

  /**
   * One primitive's generated argument schema, by its identity; `null` where the build has none.
   *
   * A primitive declared in the editor has no artifact, and neither has one a build predates — F5:
   * "a unit the build did not see … is served by the generic walker over its declaration, and the
   * artifact is marked *not generated*". The absence is an answer, so it is not an error here.
   */
  async primitiveSchema(id: string): Promise<string | null> {
    const path = this.manifest.primitive_schemas?.[id];
    if (path === undefined) return null;
    return this.read(path);
  }

  /** The Examples workspace, saving through the shell (§4.3's "Save As to copy a document out"). */
  workspace(deliver: Deliver): ReadOnlyWorkspace {
    return new ReadOnlyWorkspace(this.files(), deliver);
  }
}

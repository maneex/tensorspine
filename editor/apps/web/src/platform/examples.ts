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

/** What `vendor.json` says, of what this module needs. */
export interface VendorManifest {
  readonly repository_commit: string | null;
  readonly examples: { readonly root: string; readonly models: string; readonly primitive_library: string };
  readonly schemas: readonly { readonly path: string; readonly id: string | null }[];
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

  /** Every vendored path under a prefix, in the manifest's order (which is sorted). */
  paths(prefix: string): string[] {
    const at = prefix === '' ? '' : `${normalise(prefix)}/`;
    return this.manifest.files.map((one) => one.path).filter((path) => path.startsWith(at));
  }

  /** The text of one vendored file, by its path under the vendor root. */
  async read(path: string): Promise<string> {
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
   * library load has three hundred milliseconds to its name (§5.6).
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
      revision: (path) => held.get(path)?.sha256 ?? null,
    };
  }

  /** The Examples workspace, saving through the shell (§4.3's "Save As to copy a document out"). */
  workspace(deliver: Deliver): ReadOnlyWorkspace {
    return new ReadOnlyWorkspace(this.files(), deliver);
  }
}

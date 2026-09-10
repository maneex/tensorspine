/**
 * Where the loader's bytes come from.
 *
 * `tools/primitive_library.py` reads the filesystem directly — `os.path.isdir`, `glob.glob`,
 * `open` — and the core may not: it is pure, it runs in a Web Worker, and the plan's §5.3 says
 * "the UI reads files through `Platform` and hands the core texts and trees". So the four calls
 * the loader makes are an interface, and the caller decides what stands behind it: an in-memory
 * snapshot of a workspace folder ({@link memorySource}), the vendored base of the static build,
 * or the repository itself in a Node test.
 *
 * It is synchronous on purpose. `validateUnit` answers while a unit is typed (§4.22) and
 * `loadLibrary` is one call in a worker; making the loader asynchronous would make every one of
 * its callers asynchronous for the sake of reads the caller has already done.
 *
 * A base's templates location is resolved against the base and may leave it — the reference base
 * says `"templates": "../models/"`, and the rejection fixtures reach four directories up — so a
 * source answers for whole paths, not for one directory's contents.
 */
import { normalise } from './paths.js';

/** The reads the loader makes, as `os` makes them. */
export interface LibrarySource {
  /** `os.path.isdir`: whether an exploded base, or a section of one, is there. */
  isDirectory(path: string): boolean;
  /** `os.path.isfile`: whether a manifest or a pinned template document is there. */
  isFile(path: string): boolean;
  /** `os.path.exists`: whether a declared base is there at all (V1). */
  exists(path: string): boolean;
  /**
   * Every `*.json` at any depth under a directory — `glob.glob('<dir>/**\/*.json',
   * recursive=True)`, whose `**` skips names that begin with a dot, as `glob` does. The order is
   * the caller's business: the loader sorts what it is given, exactly as `_units` does.
   */
  find(directory: string): readonly string[];
  /** The file's text, decoded as UTF-8, as `open(path, encoding='utf-8')` reads it. */
  read(path: string): string;
}

/** Raised when a source is asked for a file it does not hold: a caller's mistake, not a refusal. */
export class LibrarySourceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LibrarySourceError';
  }
}

/**
 * A source over texts already in memory, keyed by path.
 *
 * A directory exists when some key lies under it, which is what a folder read into a map looks
 * like; an empty directory is therefore invisible, exactly as it is to a loader that only ever
 * globs for files. Paths are normalised on the way in and on the way out, so a caller may write
 * `data/primitive-library/` or `data/./primitive-library` for the same base.
 */
export function memorySource(files: Readonly<Record<string, string>>): LibrarySource {
  const byPath = new Map<string, string>();
  for (const [path, text] of Object.entries(files)) byPath.set(normaliseKey(path), text);
  const directories = new Set<string>();
  for (const path of byPath.keys()) {
    const parts = path.split('/');
    for (let depth = 1; depth < parts.length; depth += 1) {
      directories.add(parts.slice(0, depth).join('/'));
    }
  }
  return {
    isDirectory: (path) => directories.has(normaliseKey(path)),
    isFile: (path) => byPath.has(normaliseKey(path)),
    exists: (path) => byPath.has(normaliseKey(path)) || directories.has(normaliseKey(path)),
    find(directory) {
      const prefix = `${normaliseKey(directory)}/`;
      return [...byPath.keys()].filter(
        (path) =>
          path.startsWith(prefix) &&
          path.endsWith('.json') &&
          !path
            .slice(prefix.length)
            .split('/')
            .some((segment) => segment.startsWith('.')),
      );
    },
    read(path) {
      const text = byPath.get(normaliseKey(path));
      if (text === undefined) throw new LibrarySourceError(`${path}: no such file`);
      return text;
    },
  };
}

/** A key of the map: {@link normalise}, so that `a/./b/` and `a/b` name one file. */
function normaliseKey(path: string): string {
  return normalise(path);
}

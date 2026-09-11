/**
 * Reading a whole subtree of a workspace — feature 2.4.
 *
 * The core opens nothing: "the UI reads files through `Platform` and hands the core texts and
 * trees" (§5.3). `loadSchemas` takes the schema files, `loadLibrary` takes each base as a map of
 * path to text, and this is what produces one.
 *
 * **The trap this leaves open, stated.** A base's templates location is resolved against the base
 * and *may leave it* — the reference base says `"templates": "../models/"` — so the files under a
 * base are not always all the loader needs: it opens the template document a template primitive
 * pins, to check its file, name, version and id. A base handed over without those is refused for a
 * reason the workspace, not the base, is responsible for. Two ways to close it, and feature 2.6
 * chooses:
 *
 *   - read the manifest and add the directory its templates member names (what
 *     `packages/lang/test/api/source.ts` does by hand); or
 *   - hand over the base's own files, read `Library.bases[].templates` off the load — the core
 *     computes it whether or not the documents were there — and load again with them.
 *
 * The second names nothing of the unit schema in interface source, which is what §1 asks for; the
 * first costs one load instead of two. Neither is decided here, because neither belongs to the
 * workspace: this module reads what it is told to read.
 */
import { isUnder, normalise, type WorkspacePath } from './paths.js';
import { PlatformError, type Workspace } from './types.js';

/** What {@link readTree} keeps and how deep it goes. */
export interface TreeOptions {
  /** Keep only the files whose name ends this way — `.json` for a base, for the schemas. */
  readonly suffix?: string;
  /** Stop below this many directories; unlimited by default. */
  readonly depth?: number;
  /** Skip a name beginning with a dot, as `glob`'s `**` does. True by default. */
  readonly hidden?: boolean;
}

/**
 * Every file under a directory, by workspace path, with its text.
 *
 * The paths are the workspace's own and are **absolute within the workspace** — `data/models/…`,
 * not `models/…` — because that is the space a document's bases are resolved in. A base at the
 * workspace root is the one place the two spellings differ, and `toPosix` is the conversion.
 */
export async function readTree(
  workspace: Workspace,
  directory: WorkspacePath,
  options: TreeOptions = {},
): Promise<Record<WorkspacePath, string>> {
  const root = normalise(directory);
  const suffix = options.suffix ?? '';
  const hidden = options.hidden ?? true;
  const found: Record<WorkspacePath, string> = {};
  const walk = async (at: WorkspacePath, depth: number): Promise<void> => {
    for (const entry of await workspace.list(at)) {
      if (hidden && entry.name.startsWith('.')) continue;
      if (entry.kind === 'directory') {
        if (options.depth === undefined || depth < options.depth) await walk(entry.path, depth + 1);
        continue;
      }
      if (!entry.name.endsWith(suffix)) continue;
      found[entry.path] = (await workspace.read(entry.path)).text;
    }
  };
  await walk(root, 0);
  return found;
}

/**
 * Every file under a directory, deepest last, without reading any of them — the explorer's walk
 * and what a poll of a snapshot would list.
 */
export async function listTree(
  workspace: Workspace,
  directory: WorkspacePath,
  options: TreeOptions = {},
): Promise<WorkspacePath[]> {
  const suffix = options.suffix ?? '';
  const hidden = options.hidden ?? true;
  const found: WorkspacePath[] = [];
  const walk = async (at: WorkspacePath, depth: number): Promise<void> => {
    for (const entry of await workspace.list(at)) {
      if (hidden && entry.name.startsWith('.')) continue;
      if (entry.kind === 'directory') {
        if (options.depth === undefined || depth < options.depth) await walk(entry.path, depth + 1);
      } else if (entry.name.endsWith(suffix)) found.push(entry.path);
    }
  };
  await walk(normalise(directory), 0);
  return found;
}

/**
 * The directory a path names, refusing one that leaves the workspace.
 *
 * `resolve` already refuses a path that climbs above the root; this is the check a caller makes
 * before handing a path it did not compute itself — a base a document declares, a template
 * location a manifest gives — to a read.
 */
export function within(path: WorkspacePath, directory: WorkspacePath = ''): WorkspacePath {
  const at = normalise(path);
  if (!isUnder(at, directory)) {
    throw new PlatformError(`${at} is outside ${normalise(directory) || 'the workspace'}`, 'bad-path');
  }
  return at;
}

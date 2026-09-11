/**
 * Reading a whole subtree of a workspace — feature 2.4, with feature 2.6's two answers on it.
 *
 * The core opens nothing: "the UI reads files through `Platform` and hands the core texts and
 * trees" (§5.3). `loadSchemas` takes the schema files, `loadLibrary` takes each base as a map of
 * path to text, and this is what produces one.
 *
 * **The templates hand-over, decided (feature 2.6).** A base's templates location is resolved
 * against the base and *may leave it* — the reference base says `"templates": "../models/"` — so
 * the files under a base are not always all the loader needs: it opens the template document a
 * template primitive pins, to check its file, name, version and id. Feature 2.4 named two ways
 * out and left the choice here: read the manifest's member, or hand over the base's own files,
 * read `Library.bases[].templates` off the load and load again.
 *
 * **Neither, as they were stated. The core is asked, once, and nothing is loaded twice.** The
 * first costs one load but writes `templates` — a member of the unit schema — into interface
 * source, which is what §1 exists to prevent; the second writes nothing but pays the library load
 * twice, against a budget of 300 ms for one (§5.6). So `packages/lang` gained `baseTemplates` —
 * `readDirectoryBase`'s own reading of the manifest, factored out of it, with the load left off —
 * and the workspace layer asks for it between reading a base's files and handing them over
 * (`gatherBases`, `../documents/gather.ts`). One manifest read, one load, and the member is named
 * where every other member of the language is named: in the core.
 *
 * **Reading a set costs less than reading it file by file.** Feature 2.4 measured the reference
 * base's 131 files at 167–215 ms one after another and 82–90 ms at once; a workspace that can do
 * better in bulk says so with {@link Workspace.readMany}, and this uses it when it is there.
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
  const paths = await listTree(workspace, directory, options);
  const found: Record<WorkspacePath, string> = {};
  if (workspace.readMany !== undefined) {
    for (const [path, one] of Object.entries(await workspace.readMany(paths))) found[path] = one.text;
    return found;
  }
  for (const path of paths) found[path] = (await workspace.read(path)).text;
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

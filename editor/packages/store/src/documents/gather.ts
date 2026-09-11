/**
 * What a document needs read before it can be judged — feature 2.6.
 *
 * The core is pure and opens nothing (§5.3): `loadSchemas` takes the schema files and
 * `loadLibrary` takes each base as a map of path to text. This is the half that reads them, out
 * of the {@link Workspace} of §5.2, and it is the only place in the editor that decides *what* to
 * read for a document.
 *
 * Three readings, in the order they have to happen:
 *
 *  1. **the schemas** — the ones the build vendored, unless the workspace carries its own, which
 *     §1 admits ("a workspace that carries its own `schemas/` overrides them, and a mismatch of
 *     `$id` or content between the two is a warning in the log");
 *  2. **the bases the document declares** — `documentBases`, which is `bases_of` in the core, so
 *     that `primitive_libraries` is read where every other member of the grammar is read;
 *  3. **the template documents those bases pin** — `baseTemplates`, which is the manifest read
 *     by the core, for the reason `../platform/tree.ts` states at length. The manifest is fetched
 *     on its own first, under the name the *core* gives it (`BASE_MANIFEST`), because handing the
 *     core a base's whole hundred and thirty files to read one of them costs a clone of the lot.
 */
import { BASE_MANIFEST, type JsonValue } from '@tensorspine/lang';
import type { Lang, LibraryBaseFiles, Problem, SchemasHandle } from '@tensorspine/lang/api';

import { fromPosix, join, toPosix, type WorkspacePath } from '../platform/paths.js';
import { readTree } from '../platform/tree.js';
import type { Workspace } from '../platform/types.js';

/** Where a workspace keeps the schemas that override the vendored ones (§4.3's layout). */
export const WORKSPACE_SCHEMAS: WorkspacePath = 'schemas';

/** What {@link gatherSchemas} answers: the files, and where they came from. */
export interface GatheredSchemas {
  /** The files, by path, as `loadSchemas` takes them. */
  readonly files: Readonly<Record<string, string>>;
  /** What the "no schema with $id ending in …" line names — the directory, or the build. */
  readonly origin: string;
  /** True where the workspace's own directory was used instead of the vendored copy. */
  readonly fromWorkspace: boolean;
}

/**
 * The schemas for a workspace: its own `schemas/` where it has one, the vendored copy otherwise.
 *
 * A workspace that carries none is the ordinary case — the corpus's `data/` does — and the
 * vendored files are what the build was made from. A workspace that carries some is the case §1
 * admits, and the difference between the two is the caller's to report in the log
 * ({@link schemaDifferences}).
 */
export async function gatherSchemas(
  workspace: Workspace,
  vendored: Readonly<Record<string, string>>,
): Promise<GatheredSchemas> {
  // A workspace with no `schemas/` refuses the listing, which is not a failure of anything: the
  // build's own copy is what §1 calls the fallback.
  const own = await readTree(workspace, WORKSPACE_SCHEMAS, { suffix: '.json' }).catch(() => ({}));
  if (Object.keys(own).length === 0) {
    return { files: vendored, origin: WORKSPACE_SCHEMAS, fromWorkspace: false };
  }
  return { files: own, origin: WORKSPACE_SCHEMAS, fromWorkspace: true };
}

/** One schema the workspace and the build disagree about (§1's warning in the log). */
export interface SchemaDifference {
  readonly path: string;
  /** The line the log shows. */
  readonly message: string;
}

/**
 * What the workspace's schemas say that the vendored ones do not — §1's mismatch warning.
 *
 * Compared by *file name* and by text, because that is all this layer can compare without
 * parsing: the `$id` half of §1's sentence is the registry's, and the registry reports a schema it
 * cannot index. A difference is a warning and never a refusal — the workspace's copy is the one
 * in force, which is what "overrides them" means.
 */
export function schemaDifferences(
  own: Readonly<Record<string, string>>,
  vendored: Readonly<Record<string, string>>,
): SchemaDifference[] {
  const nameOf = (path: string): string => path.slice(path.lastIndexOf('/') + 1);
  const built = new Map(Object.entries(vendored).map(([path, text]) => [nameOf(path), text]));
  const found: SchemaDifference[] = [];
  for (const [path, text] of Object.entries(own)) {
    const name = nameOf(path);
    const before = built.get(name);
    if (before === undefined) {
      found.push({ path, message: `${path} is the workspace's own; the build vendors no ${name}` });
      continue;
    }
    if (before !== text) {
      found.push({ path, message: `${path} differs from the ${name} the build vendors` });
    }
  }
  for (const name of built.keys()) {
    if (![...Object.keys(own)].some((path) => nameOf(path) === name)) {
      found.push({
        path: `${WORKSPACE_SCHEMAS}/${name}`,
        message: `the workspace's schemas carry no ${name}; the build's copy is not used`,
      });
    }
  }
  return found;
}

/** What {@link gatherBases} answers. */
export interface GatheredBases {
  /** Each base with every file the loader will open for it, ready for `loadLibrary`. */
  readonly bases: readonly LibraryBaseFiles[];
  /** `bases_of`'s own refusal, where the document cannot be resolved from at all. */
  readonly problems: readonly Problem[];
  /** How many files were read, for the log and for the timings. */
  readonly files: number;
}

/**
 * Every file the loader will open for a document: each declared base, and the templates it pins.
 *
 * The bases come from the core (`documentBases`) in `os.path`'s spelling, where the workspace
 * root is `.`; {@link fromPosix} and {@link toPosix} are the seam, and the base is handed back in
 * the spelling the loader's refusals should name — which is the one the core computed.
 *
 * A base the workspace does not hold is handed over empty rather than refused here: the loader
 * says what a missing base is (V1) in its own words, and a reading layer that refused first would
 * be a second implementation of that verdict.
 */
export async function gatherBases(
  lang: Lang,
  schemas: SchemasHandle,
  workspace: Workspace,
  tree: JsonValue,
  path: WorkspacePath,
): Promise<GatheredBases> {
  const declared = await lang.documentBases(tree, toPosix(path));
  // The manifest first, and it alone: it is what says where the templates are, and there is no
  // reason to send a base's whole hundred and thirty files across to read one of them.
  const manifests = await Promise.all(
    declared.bases.map(async (base) => {
      const at = inside(base);
      return {
        base,
        files: at === null ? {} : await readOne(workspace, join(at, BASE_MANIFEST)),
      };
    }),
  );
  const templates =
    manifests.length === 0 ? [] : await lang.baseTemplates(manifests, schemas);
  // Then everything, at once: the base's own files and the documents its manifest points at,
  // which for the reference base is `../models/` and so lies outside the base entirely.
  const wanted = declared.bases.map((base, index) => ({ base, templates: templates[index] ?? null }));
  const read = await Promise.all(
    wanted.flatMap((one) => [
      readUnder(workspace, one.base),
      one.templates === null ? Promise.resolve({}) : readUnder(workspace, one.templates),
    ]),
  );
  const bases: LibraryBaseFiles[] = wanted.map((one, index) => ({
    base: one.base,
    files: { ...read[index * 2], ...read[index * 2 + 1] },
  }));
  return {
    bases,
    problems: declared.problems,
    files: bases.reduce((count, base) => count + Object.keys(base.files).length, 0),
  };
}

/**
 * A path the core computed, in the workspace's spelling — `null` where it leaves the workspace.
 *
 * A document declaring `"base": "../primitive-library/"` at the *root* of its workspace resolves
 * to a place above it, which no workspace can be asked for: a folder of one dropped file is the
 * ordinary way to meet it. Nothing is read for it and nothing is refused here — the loader is what
 * says a base that is not there is a V1, in its own words.
 */
function inside(path: string): WorkspacePath | null {
  try {
    return fromPosix(path);
  } catch {
    return null;
  }
}

/** One file, by the path space the core resolves against; nothing where it is not there. */
async function readOne(
  workspace: Workspace,
  path: WorkspacePath,
): Promise<Record<string, string>> {
  try {
    return { [toPosix(path)]: (await workspace.read(path)).text };
  } catch {
    return {};
  }
}

/** Every `.json` under a directory the core named, in the paths the core will resolve against. */
async function readUnder(
  workspace: Workspace,
  directory: string,
): Promise<Record<string, string>> {
  const at = inside(directory);
  if (at === null) return {};
  // A base the workspace does not hold answers nothing: the loader is what says a missing base is
  // a V1, in its own words, and a reading layer that refused first would be a second one. A base
  // that is a *file* rather than a folder is the monolithic form the loader still accepts, and it
  // is what a failed listing means here.
  const read: Record<WorkspacePath, string> | null = await readTree(workspace, at, {
    suffix: '.json',
  }).catch(() => null);
  if (read === null) return readOne(workspace, at);
  const found: Record<string, string> = {};
  for (const [file, text] of Object.entries(read)) found[toPosix(file)] = text;
  return found;
}

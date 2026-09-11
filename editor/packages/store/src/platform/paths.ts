/**
 * The path arithmetic of a workspace — feature 2.4.
 *
 * A workspace path is the editor's own spelling of a place inside the open folder: relative to
 * its root, `/`-separated, no leading and no trailing separator, no `.` and no `..` left in it.
 * The root itself is the empty string. Every {@link Workspace} answers and accepts that spelling
 * and no other, so a listing, a `read`, a `watch` event and a sidecar's name are comparable as
 * they stand.
 *
 * **Two spellings, and the seam between them.** The language core computes a base's path with
 * `os.path`'s rules (`packages/lang`'s `library/paths.ts`, the port of `primitive_library.py`),
 * where the root is written `.` and never `''`. A document at the root of its workspace declaring
 * the folder it sits in therefore resolves to `.` there and to `''` here — the two are the same
 * place, and comparing them as text is the defect the review repair `0232559` closed. So nothing
 * here ever compares a path with one the core computed: {@link fromPosix} and {@link toPosix} are
 * the one conversion, and they are used at the seam rather than at every call.
 */

import { PlatformError } from './errors.js';

/** A place inside the workspace, relative to its root; `''` is the root itself. */
export type WorkspacePath = string;

/** The root of a workspace, as this module writes it. */
export const WORKSPACE_ROOT: WorkspacePath = '';

/** The root as `os.path` writes a relative path with nothing in it, which is what the core uses. */
export const POSIX_ROOT = '.';

/**
 * The workspace's own spelling of a path: empty segments and `.` dropped, `..` applied.
 *
 * A path that climbs above the root is refused rather than clamped, because a workspace path is
 * user input — a sidecar's name, a base a document declares, a file a drop carried — and a
 * silently clamped `../../etc/passwd` is a path the caller never wrote.
 */
export function normalise(path: WorkspacePath): WorkspacePath {
  const segments: string[] = [];
  for (const segment of path.split('/')) {
    if (segment === '' || segment === '.') continue;
    if (segment !== '..') {
      segments.push(segment);
      continue;
    }
    if (segments.length === 0) {
      throw new PlatformError(`the path ${JSON.stringify(path)} leaves the workspace`, 'bad-path');
    }
    segments.pop();
  }
  return segments.join('/');
}

/** The segments of a path, the root having none. */
export function segmentsOf(path: WorkspacePath): string[] {
  const normalised = normalise(path);
  return normalised === WORKSPACE_ROOT ? [] : normalised.split('/');
}

/** The directory a path is in — the root for a file at the root. */
export function parentOf(path: WorkspacePath): WorkspacePath {
  const segments = segmentsOf(path);
  segments.pop();
  return segments.join('/');
}

/** The last segment of a path: the file's own name, or `''` for the root. */
export function nameOf(path: WorkspacePath): string {
  return segmentsOf(path).at(-1) ?? '';
}

/** Join a directory and what is under it, either of which may be the root. */
export function join(directory: WorkspacePath, name: string): WorkspacePath {
  return normalise(directory === WORKSPACE_ROOT ? name : `${directory}/${name}`);
}

/** Whether `path` is `directory` itself or lies under it. */
export function isUnder(path: WorkspacePath, directory: WorkspacePath): boolean {
  const under = normalise(path);
  const root = normalise(directory);
  return root === WORKSPACE_ROOT || under === root || under.startsWith(`${root}/`);
}

/**
 * What a relative reference written **in** `from` denotes.
 *
 * `from` is the file the reference is written in — a document's `primitive_libraries[].base`, a
 * base manifest's templates location, a sidecar — so the reference resolves against the directory
 * that holds it. Feature 0.6 took that reading because §5.2 does not say, and it is the one the
 * corpus needs: `../primitive-library/`, written in `models/llama3-8b.json`, is
 * `primitive-library`.
 */
export function resolveFrom(from: WorkspacePath, relative: string): WorkspacePath {
  if (relative.startsWith('/')) return normalise(relative);
  return normalise(`${parentOf(from)}/${relative}`);
}

/**
 * A path the core computed, in the workspace's spelling.
 *
 * `os.path.normpath` writes a relative path with nothing left in it as `.`; a workspace writes it
 * `''`. Everything else is already the same text.
 */
export function fromPosix(path: string): WorkspacePath {
  if (path.startsWith('/')) {
    throw new PlatformError(`${path} is absolute: a workspace path is relative to its root`, 'bad-path');
  }
  return normalise(path);
}

/**
 * A workspace path in the spelling the core computes with.
 *
 * The one place it matters is a base handed to `loadLibrary`: the loader resolves what a document
 * declares with `os.path`'s rules and compares it with the paths the caller gave, so a base at the
 * workspace root must be given to it as `.` and not as `''`.
 */
export function toPosix(path: WorkspacePath): string {
  const normalised = normalise(path);
  return normalised === WORKSPACE_ROOT ? POSIX_ROOT : normalised;
}

/** Order a listing the way a tree shows it: by name, as the site sorts. */
export function byName(a: { name: string }, b: { name: string }): number {
  return a.name.localeCompare(b.name);
}

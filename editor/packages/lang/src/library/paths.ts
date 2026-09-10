/**
 * The path arithmetic `tools/primitive_library.py` does with `os.path`.
 *
 * The loader computes paths in four places, and each of them is part of what it refuses: a unit's
 * identity is read from its path relative to its section root, a base's templates location is
 * `templates` resolved against the base, a pinned template is `<location>/<name>/<version>.json`,
 * and every refusal names the file it was computed for. So the arithmetic is the port's, not the
 * host's — `node:path` is not importable here (the core runs in a worker) and would in any case
 * answer Windows' separators on Windows, where the tools would answer POSIX's.
 *
 * These are `os.path`'s POSIX rules, which is what the repository is read on and what a browser
 * workspace's paths look like (0.6: `resolve(from, relative)` reads `from` as the file the
 * reference is written in). A backslash is an ordinary character here, as it is to `posixpath`.
 */

/** `os.path.isabs`. */
export function isAbsolute(path: string): boolean {
  return path.startsWith('/');
}

/**
 * `os.path.join`: an absolute component restarts the path, and a component is separated from the
 * one before it by exactly one `/` unless that one already ends in a separator.
 */
export function join(...parts: readonly string[]): string {
  let out = parts[0] ?? '';
  for (const part of parts.slice(1)) {
    if (part.startsWith('/')) out = part;
    else if (out === '' || out.endsWith('/')) out += part;
    else out += `/${part}`;
  }
  return out;
}

/**
 * `os.path.normpath`: `.` dropped, `..` resolved lexically, repeated separators collapsed.
 *
 * Two details are POSIX's own and are reproduced: a path of exactly two leading separators keeps
 * both (three or more collapse to one), and an empty path is `.`. Symbolic links are not resolved
 * — `normpath` does not resolve them either.
 */
export function normalise(path: string): string {
  if (path === '') return '.';
  const absolute = path.startsWith('/');
  const doubled = absolute && path.startsWith('//') && !path.startsWith('///');
  const out: string[] = [];
  for (const segment of path.split('/')) {
    if (segment === '' || segment === '.') continue;
    if (segment !== '..') {
      out.push(segment);
      continue;
    }
    const last = out[out.length - 1];
    if (out.length > 0 && last !== '..') out.pop();
    else if (!absolute) out.push('..');
  }
  const body = out.join('/');
  if (absolute) return (doubled ? '//' : '/') + body;
  return body === '' ? '.' : body;
}

/**
 * `os.path.dirname`: what stands before the last separator.
 *
 * `posixpath` keeps the separator when what precedes it is made of separators alone, which is why
 * `dirname('/a')` is `/` and `dirname('a')` is the empty string.
 */
export function dirname(path: string): string {
  const cut = path.lastIndexOf('/') + 1;
  const head = path.slice(0, cut);
  if (head === '' || head === '/'.repeat(head.length)) return head;
  return head.replace(/\/+$/, '');
}

/** `os.path.basename`: what stands after the last separator. */
export function basename(path: string): string {
  const cut = path.lastIndexOf('/');
  return cut < 0 ? path : path.slice(cut + 1);
}

/**
 * `os.path.relpath`, lexically.
 *
 * The tools call it on a globbed path and the section root it was globbed under, so `to` always
 * begins with `from`; the general case is implemented all the same, since `..` is what `relpath`
 * answers there and a caller that reached it would otherwise get a silently wrong identity. Both
 * paths must be of the same kind: `relpath` resolves a relative path against the working
 * directory, and the core has none.
 */
export function relative(to: string, from: string): string {
  if (isAbsolute(to) !== isAbsolute(from)) {
    throw new TypeError(`relative('${to}', '${from}'): one path is absolute and the other is not`);
  }
  const target = segments(normalise(to));
  const start = segments(normalise(from));
  let shared = 0;
  while (shared < target.length && shared < start.length && target[shared] === start[shared]) {
    shared += 1;
  }
  const up = Array.from({ length: start.length - shared }, () => '..');
  const down = target.slice(shared);
  const out = [...up, ...down];
  return out.length === 0 ? '.' : out.join('/');
}

/** The meaningful segments of a normalised path, the root's empty one dropped. */
function segments(path: string): string[] {
  if (path === '.') return [];
  return path.split('/').filter((segment) => segment !== '');
}

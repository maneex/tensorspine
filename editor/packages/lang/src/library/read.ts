/**
 * `primitive_library.read_json`: a unit's text, read as the tools read it.
 *
 * Four refusals live in that one function, and all four end the same way — wrapped as
 * `<path>: <reason> (V12)`, the file first, because "a unit outside the vocabulary is a load error
 * naming the file" (§1 of the unit guide):
 *
 * 1. a text that is not JSON at all;
 * 2. a duplicate member name, which V12 refuses rather than resolving to the last value;
 * 3. the earlier untagged field layout (`legacy.ts`);
 * 4. a `schema` revision this reading does not support.
 *
 * The order the *loader* runs them in is not this order, and that is deliberate on the tools'
 * side: `_units` checks the unit against the schema before calling `read_json`, and the schema
 * check reads the file with a plain `json.load` — no duplicate hook. So a unit that is both
 * off-schema and holds a duplicate is refused for being off-schema. {@link readText} reproduces
 * that by handing back the reading `json.load` would have given (last value wins) together with
 * the duplicate it saw, and {@link readRefusal} states the refusal only once the caller has run
 * the schema stage.
 *
 * One divergence, stated: a unit whose text is not JSON at all makes `schema.check`'s own
 * `json.load` raise a `JSONDecodeError` **out of** `load`, uncaught — `read_json`'s wrapper never
 * sees it, and the rejection runner, which catches `PrimitiveLibraryError` alone, would crash.
 * The port states it as a refusal under V12 in CPython's own words instead, which is the reading
 * feature 1.1 took for the same hole on the model side: such a text is refused either way, and
 * the reader is told why.
 */
import { JsonParseError, parse } from '../json/parse.js';
import type { JsonValue } from '../json/tree.js';
import { hasKey, isRecord, member, toPython, type PyValue } from '../expr/value.js';
import { legacyLayout } from './legacy.js';
import { libraryProblem, type LibraryProblem } from './problems.js';
import { pyRepr } from './repr.js';

/**
 * The document revisions `read_json` accepts.
 *
 * This is the tools' own list, not a schema's: it names revisions of formats `schemas/` does not
 * carry (`tensorspine-capabilities/1`), and its purpose is to tell a supported document from one
 * that needs `tools/migrate.py`. A revision that does not begin with `tensorspine` is not judged
 * at all — a file that is not ours is refused by the schema stage, not by this list.
 */
const SUPPORTED_REVISIONS: ReadonlySet<string> = new Set([
  'tensorspine/2.0',
  'tensorspine-primitive-library-unit/2.0',
  'tensorspine-derived/2.1',
  'tensorspine-capabilities/1',
  'tensorspine-fixture/1',
]);

/** A file's text as the loader reads it: the tree, its value, and the duplicate it forgave. */
export interface ReadText {
  /** The ordered tree, with every number's lexeme (D12). */
  readonly tree: JsonValue;
  /** The same document as the evaluators hold one: Python's `int` and `float` kept apart. */
  readonly value: PyValue;
  /**
   * The duplicate member name `json.load(object_pairs_hook=…)` would have refused, when the text
   * holds one. The tree is then the reading a plain `json.load` gives — the first occurrence's
   * place, the last occurrence's value — because that is what the schema stage is handed.
   */
  readonly duplicate: JsonParseError | null;
}

/** The text read, or the refusal for a text that is not JSON at all. */
export interface TextReading {
  readonly read: ReadText | null;
  readonly problem: LibraryProblem | null;
}

/** The text of one file, parsed the way the loader's two stages need it. */
export function readText(path: string, text: string): TextReading {
  let tree: JsonValue;
  let duplicate: JsonParseError | null = null;
  try {
    tree = parse(text);
  } catch (error) {
    if (!(error instanceof JsonParseError)) throw error;
    if (error.duplicateMember === null) {
      return { read: null, problem: v12(path, error.message) };
    }
    duplicate = error;
    tree = parse(text, { duplicates: 'last' });
  }
  return { read: { tree, value: toPython(tree), duplicate }, problem: null };
}

/**
 * The refusals `read_json` states about a document it managed to parse: the duplicate it saw, the
 * legacy field layout, the unsupported revision. `null` when the file is readable.
 */
export function readRefusal(path: string, read: ReadText): LibraryProblem | null {
  if (read.duplicate !== null) return v12(path, read.duplicate.reason);
  if (legacyLayout(read.value)) {
    return v12(
      path,
      'legacy field layout; convert it with python3 tools/migrate.py INPUT -o OUTPUT',
      'revision',
    );
  }
  const revision = isRecord(read.value) ? member(read.value, 'schema') : undefined;
  if (
    typeof revision === 'string' &&
    revision.startsWith('tensorspine') &&
    !SUPPORTED_REVISIONS.has(revision)
  ) {
    return v12(
      path,
      `unsupported revision ${pyRepr(revision)}; convert supported legacy input with ` +
        'python3 tools/migrate.py INPUT -o OUTPUT',
      'revision',
    );
  }
  return null;
}

/** `raise PrimitiveLibraryError(f"{path}: {e} (V12)")`, whichever of the four raised it. */
function v12(path: string, reason: string, kind: 'json' | 'revision' = 'json'): LibraryProblem {
  return libraryProblem(kind, path, `${path}: ${reason} (V12)`, { code: 'V12' });
}

/** A member of a document read as a record, `undefined` when there is none or it is not one. */
export function memberOf(value: PyValue, name: string): PyValue | undefined {
  if (!isRecord(value) || !hasKey(value, name)) return undefined;
  return member(value, name);
}

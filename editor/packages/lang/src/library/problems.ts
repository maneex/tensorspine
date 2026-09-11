/**
 * What the loader refuses with.
 *
 * `tools/primitive_library.py` raises one `PrimitiveLibraryError` and stops — "a unit outside the
 * vocabulary is a load error naming the file, never a document read with a guess (I7)". The
 * editor cannot stop: the Problems panel wants every row, and §4.22's live judgement wants a row
 * per cross-reference with somewhere to click. So the port keeps the tools' *staging* and adds the
 * continuation the plan's §3 asks for — "extra rows, never fewer" — which makes the parity
 * contract precise:
 *
 * > the first problem `loadLibrary` reports, rendered by {@link formatLibraryProblem}, is the text
 * > `str(PrimitiveLibraryError)` the tools would have raised.
 *
 * A refusal is one problem, not one per line, because that is the unit the tools raise and the
 * unit `tests/rejections/primitive-library.json` matches against: a `match` may span the head and
 * a line under it. The lines under the head are in {@link LibraryProblem.detail}, each with the
 * place in the unit it speaks of, which is the pointer §3 asks the core to emit natively beside
 * the tools' wording.
 */
import { pointerOf, type PathSegment } from '../schema/types.js';

/**
 * The exception the tools raise where they do not collect.
 *
 * The loader itself collects (above), but `primitive_library.template_path` raises out of
 * `validate.analyse` on a template primitive whose document the load did not resolve — the one
 * place a caller of a *gathered* library still meets a `PrimitiveLibraryError`. The port raises it
 * there too, with the tools' words: where they raise, it raises (feature 1.5's rule).
 */
export class PrimitiveLibraryError extends Error {
  override name = 'PrimitiveLibraryError';
}

/** The rule a refusal names, when its own text names one. */
export type LibraryProblemCode = 'V1' | 'V12' | null;

/**
 * What a refusal is about. Not part of the wording — the tools have no such field — but what the
 * panel groups by and what a fix action keys on ("Create axis …", "Add argument …").
 */
export type LibraryProblemKind =
  /** The text could not be read as JSON, or holds a duplicate member name (V12). */
  | 'json'
  /** The unit is off the primitive-library-unit schema, or a template off the model schema. */
  | 'schema'
  /** A legacy field layout, or a `schema` revision this reading does not support. */
  | 'revision'
  /** The unit's `schema`, `kind`, `name` or `version` disagrees with its path (§8.2). */
  | 'identity'
  /** Two bases carry one identity with different contents (V1). */
  | 'conflict'
  /** A declared base is not there (V1). */
  | 'base'
  /** A precision role whose default is outside its own admissible set. */
  | 'precision'
  /** A primitive citing what the gathered bases do not hold. */
  | 'reference'
  /** A template primitive whose pinned document is missing, misversioned or misidentified. */
  | 'template';

/** One line under a refusal's head, with the place in the unit it names. */
export interface LibraryDetail {
  /** The tools' words for this line, unindented. */
  readonly message: string;
  /** Where in the unit, as a JSON pointer (RFC 6901); `''` when the line names no place. */
  readonly path: string;
  /** The same place as its segments, for a store that walks the tree. */
  readonly segments: readonly PathSegment[];
}

/** One refusal of the loader: what a `PrimitiveLibraryError` carries, with the places beside it. */
export interface LibraryProblem {
  readonly code: LibraryProblemCode;
  readonly kind: LibraryProblemKind;
  /** The file the refusal names — the error "carries the file path". */
  readonly file: string;
  /** The head line, exactly as the tools raise it, the file included. */
  readonly message: string;
  /** The lines under the head, in the tools' order; empty for a one-line refusal. */
  readonly detail: readonly LibraryDetail[];
  /**
   * True when the tools would already have stopped before reaching this problem — the
   * continuation of plan §3. The first problem of a load is never marked; every later one is.
   */
  readonly afterRefusal: boolean;
}

/** `str(PrimitiveLibraryError)`: the head, then each line under it indented by four spaces. */
export function formatLibraryProblem(problem: LibraryProblem): string {
  if (problem.detail.length === 0) return problem.message;
  return [problem.message, ...problem.detail.map((one) => `    ${one.message}`)].join('\n');
}

/** {@link formatLibraryProblem} over each, which is what a load would have printed line by line. */
export function formatLibraryProblems(problems: readonly LibraryProblem[]): string[] {
  return problems.map((problem) => formatLibraryProblem(problem));
}

/** A problem with no lines under it: the shape most of the loader's refusals have. */
export function libraryProblem(
  kind: LibraryProblemKind,
  file: string,
  message: string,
  options: { code?: LibraryProblemCode; detail?: readonly LibraryDetail[] } = {},
): LibraryProblem {
  return {
    code: options.code ?? null,
    kind,
    file,
    message,
    detail: options.detail ?? [],
    afterRefusal: false,
  };
}

/** The same problem, marked as one the tools would never have reached (plan §3). */
export function afterRefusal(problem: LibraryProblem): LibraryProblem {
  return problem.afterRefusal ? problem : { ...problem, afterRefusal: true };
}

/** One line under a refusal, from its message and the place in the unit it names. */
export function detailAt(message: string, segments: readonly PathSegment[]): LibraryDetail {
  return { message, path: pointerOf(segments), segments };
}

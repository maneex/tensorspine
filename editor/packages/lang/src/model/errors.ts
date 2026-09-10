/**
 * `tools/model.py`'s `ModelError`: "a document that cannot be read as written".
 *
 * Four texts are raised under it, and every one of them is a parity contract — the rejection
 * suite matches two of them word for word:
 *
 * | text | raised by |
 * |---|---|
 * | `duplicate member name 'x' (V12)` | the JSON layer, `_pairs` (V12; here the parser of 0.3) |
 * | `binding 'C.R' is declared both in composition 'C' and at the top level` | a scoped rule whose expanded name is a top-level rule's (§5.2 rule 7, V12) |
 * | `composition 'C', binding 'R': no site named 'S'` | an endpoint selecting no site of its composition (§5.2 rule 3, V1) |
 * | `composition 'C', binding 'R': 'i' is not an index of the composition` | an index override naming what the composition does not range over |
 *
 * The rule each one carries is **not** decided here: `validate.analyse` catches the error and
 * reads its own text — `V12` when it holds `duplicate` or `declared both`, `V1` otherwise — and
 * that line is `validate.py`'s, so the port of it owns the mapping (feature 1.6). {@link
 * ModelError.kind} is what a caller reads instead of the text, and it says the same thing without
 * a substring test.
 *
 * Python's `ModelError` is a `ValueError`, which matters in exactly one place: `d1.main` catches
 * `(ValueError, KeyError, OSError, ModelError)` and would catch it either way. Nothing here
 * inherits from another refusal of the core, so a caller that wants it names it.
 */

/** Which of the four refusals this is. */
export type ModelErrorKind =
  /** A duplicate member name: the JSON layer's refusal, V12. */
  | 'duplicate'
  /** A scoped rule whose expanded name is already a top-level rule's, V12. */
  | 'collision'
  /** An endpoint naming no site of its composition. */
  | 'site'
  /** An index override naming what is not an index of the composition. */
  | 'index';

/** A document that cannot be read as written. `message` is the tools' text, unchanged. */
export class ModelError extends Error {
  /** What was refused, for a caller that would otherwise test the text for a substring. */
  readonly kind: ModelErrorKind;

  constructor(kind: ModelErrorKind, message: string) {
    super(message);
    this.name = 'ModelError';
    this.kind = kind;
  }
}

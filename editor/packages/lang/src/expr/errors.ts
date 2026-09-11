/**
 * The refusals the evaluators raise, where CPython raises one.
 *
 * `expr.py` decides between two answers to what it cannot compute. An operand of the wrong kind
 * or a division by zero is a *value* it will not decide — `_apply` catches `TypeError` and
 * `ZeroDivisionError` and answers `UNRESOLVED` — while everything else propagates: an argument
 * list too short for its operator, a comparison operator the tables do not carry, a `compare`
 * without its members, a document without its `quantities`. Those are documents off the grammar,
 * which the tools never reach because meaning assumes grammar, and the port raises there too
 * rather than inventing an answer the tools do not give.
 *
 * They are their own classes rather than the JavaScript built-ins so that `apply`'s catch takes
 * the refusals this package raises deliberately and never a genuine defect of its own.
 *
 * Each declares its `name` as a string rather than leaving `Error`'s. Two places read it and
 * neither can read the class: a structured clone strips an `Error` of its prototype and answers a
 * plain `Error` named `Error` (feature 1.11's worker boundary), and a production bundle may
 * rename the class itself, so `constructor.name` is not the identity either. The name is what
 * tells a caller that a derivation stopped on the tools' own `TypeError` rather than on a defect
 * of the port.
 */

/** The base of every refusal raised here. */
export class PyError extends Error {
  override name = 'PyError';
}

/** Python's `TypeError`: an operand of a kind the operator does not take. */
export class PyTypeError extends PyError {
  override name = 'PyTypeError';
}

/** Python's `ZeroDivisionError`, raised by `/`, `//` and `%` alike, integer or float. */
export class PyZeroDivisionError extends PyError {
  override name = 'PyZeroDivisionError';
}

/** Python's `ValueError`: `min` of an empty sequence, a range of step zero. */
export class PyValueError extends PyError {
  override name = 'PyValueError';
}

/** Python's `IndexError`: an operator read past the end of its argument list. */
export class PyIndexError extends PyError {
  override name = 'PyIndexError';
}

/** Python's `KeyError`: a member the grammar requires and the document does not carry. */
export class PyKeyError extends PyError {
  override name = 'PyKeyError';
}

/**
 * Python's `OverflowError`: `int()` of an infinity, and `float()` of an integer past the double
 * range.
 *
 * One line of the tools reaches it — V3's whole-number test on a physical value,
 * `float(v) != int(v)` — and only for a value CPython's `json` accepts and JavaScript's would
 * not: `Infinity` and `NaN` as bare tokens (feature 0.3), or an integer of more than 308 digits.
 * The tools raise there rather than answering, so the port raises too.
 */
export class PyOverflowError extends PyError {
  override name = 'PyOverflowError';
}

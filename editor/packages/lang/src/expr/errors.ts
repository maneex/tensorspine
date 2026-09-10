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
 */

/** The base of every refusal raised here. */
export class PyError extends Error {}

/** Python's `TypeError`: an operand of a kind the operator does not take. */
export class PyTypeError extends PyError {}

/** Python's `ZeroDivisionError`, raised by `/`, `//` and `%` alike, integer or float. */
export class PyZeroDivisionError extends PyError {}

/** Python's `ValueError`: `min` of an empty sequence, a range of step zero. */
export class PyValueError extends PyError {}

/** Python's `IndexError`: an operator read past the end of its argument list. */
export class PyIndexError extends PyError {}

/** Python's `KeyError`: a member the grammar requires and the document does not carry. */
export class PyKeyError extends PyError {}

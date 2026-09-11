import { JsonParseError, ModelError, PyError } from '../../src/index.js';

/**
 * What the tools raised, by CPython's own name for it.
 *
 * A fixture records the exception where the tools raise rather than answer — `analyse` runs after
 * the schema stage, and a document that never crossed it reaches a `KeyError` or a `TypeError`
 * (features 1.5 and 1.6a both meet several). The port raises the same, with the same words, and
 * the classes carry Python's names with a `Py` prefix.
 */
export interface Raised {
  type: string;
  message: string;
}

/** The name and words of a refusal the port raised; anything else is re-thrown. */
export function raisedAs(error: unknown): Raised {
  if (error instanceof ModelError) return { type: 'ModelError', message: error.message };
  // A text that is not JSON at all: CPython's `json` raises `JSONDecodeError`, and the parser
  // reproduces its words and its position (feature 0.3). A *duplicate* member name is not one —
  // it is `model.py`'s `ModelError` or `primitive_library`'s `ValueError`, depending on which
  // hook read the text — so only the syntax case is named here.
  if (error instanceof JsonParseError && error.duplicateMember === null) {
    return { type: 'JSONDecodeError', message: error.message };
  }
  if (error instanceof PyError) {
    // `PyKeyError` is `KeyError`, and so on.
    return { type: error.constructor.name.replace(/^Py/, ''), message: error.message };
  }
  throw error;
}

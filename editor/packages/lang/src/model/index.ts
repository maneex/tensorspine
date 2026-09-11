/**
 * Model normalisation: the port of `tools/model.py` (plan §5.3, `parse`/`serialize`'s companion).
 *
 * One rule of the specification lives here — §5.2 rule 7, scoped bindings — and one refusal:
 * a document that cannot be read as written. Every later stage of the core reads what
 * {@link loadModel} answers, because that is what the tools read: "validation, D1, the viewer and
 * the linter read one form and the denotation has one definition".
 */
export { ModelError, type ModelErrorKind } from './errors.js';
export { hoistingOf, loadModel, normalise } from './normalise.js';
export {
  hoisted,
  HoistRecorder,
  NO_HOISTING,
  writtenPlace,
  type HoistedPlace,
  type Hoisting,
  type HoistKind,
  type WrittenPlace,
} from './hoisting.js';

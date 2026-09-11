/**
 * The generic schema walker and the form model (plan §1, D7, §4.11, §4.12).
 *
 * `formOf(context, {anchor, value})` is the whole interface: a place of a schema and a value
 * become a flat list of rows keyed by JSON pointer, indented by the depth of the path, each row
 * carrying what the schema asserts about its place, what `presentation.json` binds to it, and the
 * problem the core raised there. Every sheet of §4.11, the argument sheet of §4.12, the location
 * editor and the primitive editor of §4.22 are this function with a different anchor.
 */
export { alternationAt, chosenOf, type Alternation, type Alternative } from './alternatives.js';
export type {
  Form,
  FormBounds,
  FormCondition,
  FormKeys,
  FormMode,
  FormNote,
  FormOption,
  FormProblem,
  FormRow,
  Widget,
} from './types.js';
export { formLines, lineOf, noteLines } from './render.js';
export {
  AMBIGUOUS,
  formContext,
  formOf,
  memberOrder,
  UNDECLARED,
  UNDISCRIMINATED,
  UNREADABLE,
  UNTAGGED,
  type FormContext,
  type FormRequest,
} from './walk.js';
export {
  CHOOSER,
  editsOneValue,
  FIXED,
  JSON_EDITOR,
  LIST,
  MAP,
  NOTHING,
  NUMBER,
  SCALAR,
  SECTION,
  SELECT,
  TEXT,
  TOGGLE,
  WHOLE,
  widgetOf,
  WIDGETS,
} from './widget.js';

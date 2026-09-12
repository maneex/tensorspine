/**
 * `@tensorspine/ui` — the interface.
 *
 * React: the shell and its activities, the canvas, the generic schema walker and the forms it
 * generates, the property sheets, the panels, the primitive editor, the expression editors and
 * the source view. It imports no platform: the application passes it a `Platform`. It hard-codes
 * no vocabulary of the schemas — presentation is bound to schema anchors in `presentation.json`,
 * and every semantic verdict comes from `@tensorspine/lang`.
 */

/** The package's own name, as the workspace declares it. */
export const packageName = '@tensorspine/ui';

/**
 * The presentation bindings (plan §1, Appendix B) and the startup audit of catching rule (a).
 *
 * `presentation.json` is the one data file the interface carries; everything else about the
 * schemas is read from them. `startPresentation` resolves every binding against the loaded
 * registry and writes the report to the log.
 */
export {
  auditPresentation,
  genericLines,
  presentation,
  PresentationError,
  presentationLines,
  readPresentation,
  referenceSelectors,
  SCOPE,
  startPresentation,
  type Binding,
  type GenericConstruct,
  type Presentation,
  type PresentationAudit,
  type PresentationProblem,
  type ReferenceRule,
  type Scope,
  type StartedPresentation,
  type SymbolBinding,
} from './presentation/index.js';

/**
 * The generic schema walker and the form model (plan §1, D7, §4.11, §4.12).
 *
 * "Every form is generated": one walker renders any `$def` of any of the schemas, and every
 * sheet, dialog and editor of the interface is that walker with a different anchor.
 */
export {
  type Alternation,
  alternationAt,
  type Alternative,
  AMBIGUOUS,
  CHOOSER,
  chosenOf,
  editsOneValue,
  FIXED,
  type Form,
  type FormBounds,
  type FormCondition,
  formContext,
  type FormContext,
  type FormKeys,
  formLines,
  type FormMode,
  type FormNote,
  formOf,
  type FormOption,
  type FormProblem,
  type FormRequest,
  type FormRow,
  JSON_EDITOR,
  lineOf,
  LIST,
  MAP,
  memberOrder,
  noteLines,
  NOTHING,
  NUMBER,
  SCALAR,
  SECTION,
  SELECT,
  TEXT,
  TOGGLE,
  UNDECLARED,
  UNDISCRIMINATED,
  UNREADABLE,
  UNTAGGED,
  WHOLE,
  type Widget,
  widgetOf,
  WIDGETS,
} from './forms/index.js';

/**
 * The shell (plan §4.2, §4.4, §4.21) is the package's third entry point,
 * `@tensorspine/ui/shell` (`./src/shell/index.ts`), with its stylesheet at
 * `@tensorspine/ui/style.css`. It is deliberately not re-exported here for the reason the layout
 * is not: what a module imports says what it depends on, and the walker, the forms and the
 * presentation bindings above are pure readings of the schemas that a worker, a script or a Node
 * suite can use without React.
 */

/**
 * The canvas's automatic layout (plan §4.7) — ELK's layered algorithm, top to bottom,
 * compositions as compound nodes — is the package's second entry point, `@tensorspine/ui/layout`
 * (`./src/layout/elk.ts`). It is deliberately not re-exported here: importing it pulls in a
 * megabyte and a half of compiled ELK, which belongs in the worker that lays a canvas out (§5.6),
 * never in the shell's first bundle.
 */

/**
 * The expanded graph (plan §4.9) — the read-only, filtered, virtualised tab over D1 — is the
 * package's sixth entry point, `@tensorspine/ui/expanded` (`./src/expanded/index.ts`). It is its
 * own for the opposite of the canvas's reason: it must *not* reach the layout, since a whole
 * expanded layout costs 1.5 s on the largest corpus graph (feature 0.4) and §4.9's view is
 * D1's own topological order, virtualised.
 */

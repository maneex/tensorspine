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
 * The canvas's automatic layout (plan §4.7, §4.9) — ELK's layered algorithm, top to bottom,
 * compositions as compound nodes — is the package's second entry point, `@tensorspine/ui/layout`
 * (`./src/layout/elk.ts`). It is deliberately not re-exported here: importing it pulls in a
 * megabyte and a half of compiled ELK, which belongs in the worker that lays a canvas out (§5.6),
 * never in the shell's first bundle.
 */

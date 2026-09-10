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

/**
 * `@tensorspine/store` — the document store.
 *
 * The document is the model (the plan's D1): this package holds the `tensorspine/2.0` JSON as
 * an ordered tree with its number lexemes, the commands that patch it, undo and redo, the
 * reference index, the layout and preview sidecars and the drafts. It carries no rule of the
 * language: it asks `@tensorspine/lang`.
 */

/** The package's own name, as the workspace declares it. */
export const packageName = '@tensorspine/store';

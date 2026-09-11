/**
 * `@tensorspine/lang` — the language core.
 *
 * The TypeScript port of the repository's `tools/`, held to parity with them (the plan's D2):
 * lexeme-preserving JSON and the serializer, the schema registry, expressions and conditions,
 * the primitive library loader and its refusals, model normalisation, validation V1–V20,
 * expansion (D1), derivation (D2–D6), the checkpoint check (V17), lint, and the editor's
 * `describe` and `check`. Every rule of the language lives here once and nowhere else; the
 * interface packages display what this package answers.
 *
 * The core is pure: it reads no file and touches no DOM, so that it runs unchanged in a Web
 * Worker, in Node and in a test.
 */

/** The package's own name, as the workspace declares it. */
export const packageName = '@tensorspine/lang';

export * from './json/index.js';
export * from './schema/index.js';
export * from './expr/index.js';
export * from './library/index.js';
export * from './model/index.js';
export * from './validate/index.js';
export * from './d1/index.js';
export * from './describe/index.js';
export * from './derive/index.js';

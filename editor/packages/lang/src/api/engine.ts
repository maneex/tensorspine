/**
 * `@tensorspine/lang/api/engine` — the side that **runs** the language core.
 *
 * `createLang` runs it in this thread (Node, a test, a shell with no worker) and `serveLang` runs
 * it behind a port (a Web Worker's global scope, a `MessagePort`). Both drive one
 * {@link LangSession}, so the two deployments cannot drift, and both pull the whole core: the
 * schema registry with its compiled Ajv validators, the library loader, the validator, the
 * expansion, the derivation, the checkpoint check and lint.
 *
 * That is why it is a separate entry point from `@tensorspine/lang/api`. A page that *talks* to a
 * worker imports the proxy alone; a page that imported this would bundle the engine it is talking
 * to, which is feature 0.4's lesson (ELK out of the shell's first chunk) and feature 0.5's (the
 * Hub client out of it) in their third instance. The application's worker entry imports this one;
 * nothing else on the page does.
 */
export { createLang } from './core.js';
export type { Lang } from './core.js';
export { serveLang, type LangHost } from './host.js';
export { LangSession, UNWATCHED, yieldToTasks, type Control } from './session.js';
export type {
  DescribeArguments,
  DocumentOptions,
  ValidateArguments,
} from './session.js';

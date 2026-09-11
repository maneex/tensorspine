/**
 * `@tensorspine/lang/api` — the `Lang` interface of plan Appendix C, as the editor holds it.
 *
 * This is the seam. Everything from feature 2.6 on reaches the language core through it (plan §8,
 * "everything from 2.6 on waits for 1.11"), so what it answers is what the store, the canvas, the
 * sheets, the Problems panel, the Derived panel and the primitive editor are written against.
 *
 * **What is here, and what is one subpath away.** This entry carries the *caller's* half: the
 * interface's types, the problem every panel reads, the proxy that talks to a worker, and the
 * encoding of what a structured clone would drop. It carries **no rule of the language** — no
 * schema registry, no Ajv, no validator, no derivation — because a page that talks to a worker
 * must not bundle the engine it is talking to (feature 0.4's lesson, in its third instance:
 * measured, importing the engine's entry put a hundred kilobytes of Ajv into the page's chunk
 * though nothing on the page used it). The side that *runs* the core is
 * `@tensorspine/lang/api/engine`: {@link createLang} for this thread, `serveLang` for a worker's.
 *
 * **Why a second entry point rather than the package root.** The root star-exports every module
 * of the core, and two star exports of one name are *silently excluded* from a module's exports
 * (feature 1.4 found that with `normalise`). `Verdict` is already the name of `check`'s answer
 * there, while Appendix C calls a validation's answer `Verdict` and a candidate's
 * `CandidateVerdict`; keeping the API's own namespace is what lets the seam carry the sketch's
 * names exactly.
 *
 * **The three ways in.**
 *
 * - `createLang` (`/api/engine`) — the core in this thread. Node, a test, a shell with no worker.
 * - {@link connectLang} — the core in a worker, reached over a port. The same `Lang`.
 * - `serveLang` (`/api/engine`) — the worker's own side, given its global scope or a port.
 *
 * A worker entry point is two lines and belongs to the application, because it is the one place a
 * platform is named: `serveLang(globalThis as unknown as LangPort)` in a Web Worker,
 * `serveLang(parentPort)` in a Node thread. `packages/lang` names no platform, and its own
 * imports are written with the `.js` specifiers a browser bundler resolves and plain Node does
 * not — so the browser's worker is where a *thread* boundary is exercised, and the unit layer
 * exercises the same protocol over a `MessageChannel`, which clones exactly as a thread does.
 */
export type { Lang } from './core.js';
export { connectLang, type ConnectOptions } from './client.js';
export {
  decode,
  encode,
  encodeFailure,
  LangFailure,
  type EncodedFailure,
} from './codec.js';
export {
  CALLS,
  isRequest,
  isResponse,
  listen,
  type CallName,
  type LangEmitterPort,
  type LangEventPort,
  type LangMessageEvent,
  type LangPort,
  type LangRequest,
  type LangResponse,
} from './protocol.js';
export {
  checkpointRow,
  derivationRow,
  libraryRow,
  lintRow,
  schemaProblem,
  semanticRow,
} from './problems.js';
export { stripped, strippedDescribe } from './options.js';
export {
  LangCancelled,
  LangHandleError,
  PROBLEM_SEVERITIES,
  PROBLEM_SEVERITY,
  PROBLEM_SOURCE,
  PROBLEM_SOURCES,
  STAGE,
  STAGES,
  type CallOptions,
  type CancelReason,
  type CheckpointReport,
  type DescribeOptions,
  type DocumentBases,
  type Facts,
  type FixAction,
  type Handle,
  type LibraryBaseFiles,
  type LibraryData,
  type LibraryHandle,
  type LibraryLoaded,
  type LintDocuments,
  type LoadedSchemaInfo,
  type Problem,
  type ProblemDetail,
  type ProblemSeverity,
  type ProblemSource,
  type Progress,
  type SchemasHandle,
  type SchemasLoaded,
  type Stage,
  type ValidateOptions,
  type Verdict,
} from './types.js';

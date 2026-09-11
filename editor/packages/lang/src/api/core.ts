/**
 * The `Lang` interface of plan Appendix C, and the in-process implementation of it.
 *
 * One interface, two implementations: this one calls the session directly — for Node, for the
 * tests and for a deployment that has no worker — and `client.ts` posts to a worker running
 * `host.ts`. Both drive the same {@link LangSession}, so a conformance suite run against the two
 * is a test of the boundary rather than of a second implementation.
 *
 * **Why every call answers a promise, when the core answers synchronously.** D3 says `describe`,
 * `check` and `validateUnit` "answer synchronously" and they do: the functions of the core are
 * synchronous and a component reading a held answer pays nothing. What is asynchronous is the
 * *proxy* — a worker round trip is a message and a message is a task — and the interface is the
 * one seam every feature of phases 2, 3 and 4 reaches the core through (plan §8), so it has the
 * shape the worker forces. A caller that wants the synchronous functions imports them from
 * `@tensorspine/lang` and holds its own registry; the whole point of this seam is that it does
 * not have to.
 *
 * **What diverges from Appendix C's sketch, and why.** Three signatures:
 *
 * - `loadSchemas` and `loadLibrary` answer a **handle** beside their data. A `SchemaRegistry`
 *   answers through methods and holds compiled Ajv validators; a gathered library is read through
 *   a `LibrarySource`. A function is a `DataCloneError`, so neither can be sent, and every later
 *   call names them instead of carrying them. `validateUnit(unit, path, bases)` therefore takes
 *   the handle rather than the bases again — re-gathering per keystroke would cost the load's
 *   261 ms.
 * - `check(path, candidate)` takes the document's path where the sketch writes the tree. Feature
 *   1.6d measured the reason: `analyse` alone is 10–93 ms on the corpus and a drag asks per
 *   hovered handle, so the verdict reads the analysis the session already holds for that document.
 * - `derive(tree, path, …)` takes the path, because §5.3's "one in-flight derivation per document"
 *   needs the document's identity to say which derivation supersedes which. `expand` does not:
 *   nothing about D1 is per-document state.
 */
import type { CheckpointHeaders, HeaderRead } from '../artifact/index.js';
import type { Candidate, Verdict as CandidateVerdict } from '../describe/index.js';
import type { Env, Quantities } from '../expr/model.js';
import type { PyRecord, PyValue } from '../expr/value.js';
import type { JsonValue } from '../json/tree.js';

import { stripped, strippedDescribe } from './options.js';
import { LangSession, UNWATCHED, type Control } from './session.js';
import {
  LangCancelled,
  type CallOptions,
  type CheckpointReport,
  type DescribeOptions,
  type Facts,
  type LibraryBaseFiles,
  type LibraryHandle,
  type LibraryLoaded,
  type Problem,
  type SchemasHandle,
  type SchemasLoaded,
  type ValidateOptions,
  type Verdict,
} from './types.js';

/**
 * The language core as the editor reaches it (plan Appendix C, §5.3).
 *
 * Pure: it reads no file and touches no DOM. Files reach it as texts — the workspace is
 * `Platform`'s (§5.2) — and trees reach it as the ordered trees the store holds (D1).
 */
export interface Lang {
  /** The registry by `$id`, with the enums and `oneOf` tags the forms and the audits read. */
  loadSchemas(
    files: Readonly<Record<string, string>>,
    options?: { readonly origin?: string },
  ): Promise<SchemasLoaded>;

  /** The bases gathered: units, `byId`, axes, roles, template interfaces, and every refusal. */
  loadLibrary(
    bases: readonly LibraryBaseFiles[],
    schemas: SchemasHandle,
    options?: { readonly modelsBase?: string | null },
  ): Promise<LibraryLoaded>;

  /** The loader's checks on one unit, live while it is typed (D15). */
  validateUnit(unit: JsonValue, path: string, library: LibraryHandle): Promise<Problem[]>;

  /** An ordered tree with its number lexemes kept (D12). */
  parse(text: string): Promise<JsonValue>;

  /** The corpus's bytes (D12). */
  serialize(tree: JsonValue): Promise<string>;

  /** The stages, the problems with their pointers, the counters (plan §5.3). */
  validate(tree: JsonValue, path: string, options: ValidateOptions): Promise<Verdict>;

  /** Per site: ports, slots, states, shapes, applicability, rules, entries (plan §5.3). */
  describe(tree: JsonValue, path: string, options: DescribeOptions): Promise<Facts>;

  /** The verdict on one candidate edit, over the reading the session holds for that document. */
  check(path: string, candidate: Candidate): Promise<CandidateVerdict>;

  /** D1. */
  expand(tree: JsonValue, options: CallOptions): Promise<PyRecord>;

  /** The derived document, validated against the derived schema. */
  derive(tree: JsonValue, path: string, options: CallOptions): Promise<PyRecord>;

  /** V17 against a checkpoint's headers: errors, warnings, stats. */
  checkCheckpoint(derived: PyValue, headers: CheckpointHeaders): Promise<CheckpointReport>;

  /** A safetensors header from its bytes, in the language's dtype names. */
  readHeader(bytes: Uint8Array, file: string): Promise<HeaderRead>;

  /** A model expression against a document's quantities and the indices in scope. */
  evaluate(expression: PyValue, quantities: Quantities, env?: Env): Promise<PyValue>;

  /** Drop what is held for a document: a closed tab, a file reverted. */
  forget(path: string): Promise<void>;

  /** Drop a registry or a gathered library. */
  release(handle: SchemasHandle | LibraryHandle): Promise<void>;

  /** Stop: every outstanding call is cancelled, and a worker-backed proxy ends its worker. */
  close(): void;
}

/**
 * The `Lang` that runs here, in this thread — for Node, for the tests, for a desktop shell.
 *
 * Every call answers a promise and **none of them throws**: a caller must be able to write one
 * `catch` whichever implementation it holds, and a worker-backed call can only ever reject.
 * {@link settled} is what turns the core's synchronous refusals into rejections.
 */
export function createLang(): Lang {
  const session = new LangSession();
  let closed = false;
  const open = (call: string): void => {
    if (closed) throw new LangCancelled(call, 'closed');
  };

  return {
    loadSchemas: (files, options) =>
      settled(() => {
        open('loadSchemas');
        return session.loadSchemas(files, options ?? {});
      }),
    loadLibrary: (bases, schemas, options) =>
      settled(() => {
        open('loadLibrary');
        return session.loadLibrary(bases, schemas, options ?? {});
      }),
    validateUnit: (unit, path, library) =>
      settled(() => {
        open('validateUnit');
        return session.validateUnit(unit, path, library);
      }),
    parse: (text) =>
      settled(() => {
        open('parse');
        return session.parse(text);
      }),
    serialize: (tree) =>
      settled(() => {
        open('serialize');
        return session.serialize(tree);
      }),
    validate: (tree, path, options) =>
      settled(() => {
        open('validate');
        return session.validate(tree, path, stripped(options), controlOf('validate', options));
      }),
    describe: (tree, path, options) =>
      settled(() => {
        open('describe');
        return session.describe(tree, path, strippedDescribe(options), controlOf('describe', options));
      }),
    check: (path, candidate) =>
      settled(() => {
        open('check');
        return session.check(path, candidate);
      }),
    expand: (tree, options) =>
      settled(() => {
        open('expand');
        return session.expand(tree, stripped(options), controlOf('expand', options));
      }),
    derive: (tree, path, options) =>
      settled(() => {
        open('derive');
        return session.derive(tree, path, stripped(options), controlOf('derive', options));
      }),
    checkCheckpoint: (derived, headers) =>
      settled(() => {
        open('checkCheckpoint');
        return session.checkCheckpoint(derived, headers);
      }),
    readHeader: (bytes, file) =>
      settled(() => {
        open('readHeader');
        return session.readHeader(bytes, file);
      }),
    evaluate: (expression, quantities, env) =>
      settled(() => {
        open('evaluate');
        return session.evaluate(expression, quantities, env);
      }),
    forget: (path) =>
      settled(() => {
        open('forget');
        session.forget(path);
      }),
    release: (handle) =>
      settled(() => {
        open('release');
        session.release(handle);
      }),
    close: () => {
      closed = true;
      session.clear();
    },
  };
}

/**
 * A call's answer as a promise, whatever the core did.
 *
 * The core is synchronous and refuses by raising; the interface answers a promise and refuses by
 * rejecting, because that is the only thing a worker-backed call can do. Without this the two
 * implementations would differ in the one place a caller cannot paper over — where the `catch`
 * goes.
 */
function settled<T>(run: () => T | Promise<T>): Promise<T> {
  try {
    return Promise.resolve(run());
  } catch (error) {
    return Promise.reject(error instanceof Error ? error : new Error(String(error)));
  }
}

/** The caller's signal and progress callback, as the session watches them. */
function controlOf(call: string, options: CallOptions): Control {
  const { signal, onProgress } = options;
  if (signal === undefined && onProgress === undefined) return UNWATCHED;
  return {
    cancelled: () => (signal?.aborted === true ? new LangCancelled(call, 'requested') : null),
    report: onProgress ?? (() => undefined),
  };
}


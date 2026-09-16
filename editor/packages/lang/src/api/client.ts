/**
 * The editor's side of the seam: the `Lang` interface over a port.
 *
 * It is the same interface `createLang` answers, so a caller cannot tell which it holds — which
 * is what lets the tests run one conformance suite against both and what lets a deployment
 * without a worker (Node, a service) use the core through the same calls.
 *
 * What it does beyond posting: it keeps the two things that cannot cross. An `AbortSignal` stays
 * here and its abort becomes a `cancel` message that overtakes the work; an `onProgress` callback
 * stays here and the host's progress messages are dispatched to it. Both are stripped off the
 * arguments before they are encoded, because a function is a `DataCloneError` and sending the
 * options whole would fail the call rather than the callback.
 */
import type { CheckpointHeaders, HeaderRead } from '../artifact/index.js';
import type { Candidate, Verdict as CandidateVerdict } from '../describe/index.js';
import type { Env, Quantities } from '../expr/model.js';
import type { PyRecord, PyValue } from '../expr/value.js';
import type { JsonValue } from '../json/tree.js';

import { decode, encode, LangFailure, type EncodedFailure } from './codec.js';
import type { Lang } from './core.js';
import { stripped, strippedDescribe } from './options.js';
import {
  describePortFailure,
  isResponse,
  listen,
  listenForFailure,
  type CallName,
  type LangPort,
  type LangResponse,
  type PortFailure,
} from './protocol.js';
import {
  LangCancelled,
  type CallOptions,
  type CheckpointReport,
  type DescribeOptions,
  type DocumentBases,
  type Facts,
  type LibraryBaseFiles,
  type LibraryHandle,
  type LibraryLoaded,
  type Problem,
  type Progress,
  type SchemasHandle,
  type SchemasLoaded,
  type ValidateOptions,
  type Verdict,
} from './types.js';

/** What {@link connectLang} takes beside the port. */
export interface ConnectOptions {
  /**
   * What to do when the proxy is closed: terminate the worker, close the port.
   *
   * The core names no platform, so it does not know how the port it was handed is ended; the
   * application that created it does (`apps/web` terminates its `Worker`).
   */
  readonly onClose?: () => void;
}

/** One outstanding request. */
interface Pending {
  readonly call: string;
  readonly settle: (response: LangResponse) => void;
  readonly progress: ((progress: Progress) => void) | undefined;
  /** Stop listening to the caller's signal once the request has settled. */
  readonly release: () => void;
}

/** The `Lang` that runs in a worker, reached over a port. */
export function connectLang(port: LangPort, options: ConnectOptions = {}): Lang {
  const pending = new Map<number, Pending>();
  /** The derivation in flight for each document: "one in-flight derivation per document" (§5.3). */
  const deriving = new Map<string, number>();
  let next = 0;
  let closed = false;

  const listener = (message: unknown): void => {
    const response: unknown = message;
    if (!isResponse(response)) return;
    const held = pending.get(response.id);
    if (held === undefined) return;
    if (response.kind === 'progress') {
      held.progress?.(response.progress);
      return;
    }
    pending.delete(response.id);
    held.release();
    held.settle(response);
  };

  const stopListening = listen(port, listener);

  /**
   * Reject everything outstanding, because the port reported a failure instead of a reply.
   *
   * `LangFailure`, not `LangCancelled`: nothing was cancelled — a cancel means the caller gave up,
   * or a later request replaced this one — and a caller that catches a cancellation quietly would
   * swallow a worker that never loaded. `raised` names the failure the way the worker's own
   * refusals name theirs, so one `catch` reads both.
   */
  const failAll = (kind: PortFailure, event: unknown): void => {
    const failure: EncodedFailure = {
      name: kind === 'error' ? 'WorkerError' : 'WorkerMessageError',
      message: describePortFailure(kind, event),
    };
    // An `error` is the worker itself: it is gone, or it never ran, and nothing asked of it after
    // this can answer either. A `messageerror` is one reply that could not be deserialised — the
    // port is still there, so the outstanding requests are refused and the proxy stays open.
    if (kind === 'error') closed = true;
    for (const [id, held] of pending) {
      pending.delete(id);
      held.release();
      held.settle({ id, kind: 'failure', failure });
    }
    if (kind === 'error') {
      stopListening();
      stopFailures();
      options.onClose?.();
    }
  };

  const stopFailures = listenForFailure(port, failAll);

  const request = <T>(
    call: CallName,
    args: readonly unknown[],
    watched: Pick<CallOptions, 'signal' | 'onProgress'> = {},
    /** Told the id this request was given, for a caller that must name it later. */
    track?: (id: number) => void,
  ): Promise<T> => {
    if (closed) return Promise.reject(new LangCancelled(call, 'closed'));
    next += 1;
    const id = next;
    track?.(id);
    const { signal, onProgress } = watched;
    if (signal?.aborted === true) return Promise.reject(new LangCancelled(call, 'requested'));

    return new Promise<T>((resolve, reject) => {
      // The caller gave up: the request is settled **here**, and the cancel is posted all the
      // same so the worker drops the work whenever it is in time to. Whether it is in time is the
      // scheduler's business and not the caller's — the call is suspended on a timer task at its
      // yield and the cancel is a message task, and nothing orders the two (see `derive` below
      // for the measurement; the abort is the same race one message over). A result that arrives
      // for a request no longer pending is discarded, as the listener discards every answer it
      // did not ask for. The in-process proxy needs none of this: its signal is shared memory,
      // read at the yield.
      const abort = (): void => {
        const held = pending.get(id);
        if (held === undefined) return;
        pending.delete(id);
        held.release();
        port.postMessage({ id: 0, call: 'cancel', args: [id] });
        held.settle({ id, kind: 'cancelled', call, reason: 'requested' });
      };
      signal?.addEventListener('abort', abort, { once: true });
      pending.set(id, {
        call,
        progress: onProgress,
        release: () => signal?.removeEventListener('abort', abort),
        settle: (response) => {
          if (response.kind === 'value') resolve(decode(response.value) as T);
          else if (response.kind === 'cancelled') {
            reject(new LangCancelled(response.call, response.reason));
          } else if (response.kind === 'failure') reject(new LangFailure(response.failure));
        },
      });
      port.postMessage({ id, call, args: args.map((one) => encode(one)) });
    });
  };

  /**
   * `derive(tree, path)`: the newer derivation of a document supersedes the older **here**, and
   * not only in the worker.
   *
   * §5.3 promises the caller that "a second `derive` for a path cancels the first, whose result is
   * dropped whether or not it had already been computed". The session keeps that promise when it
   * can, and that is where the *saving* comes from: a request that reaches the host while the
   * older call is suspended at its yield cancels it before the heavy stage runs at all. But
   * whether it arrives inside that window is the scheduler's business — the older call is waiting
   * on a timer task and the newer one is a message task, and no specification orders the two.
   * Measured: Chromium delivered the message first on 200 idle trials and the timer first under a
   * loaded machine, where the older derivation then ran to the end and answered its caller.
   *
   * What the caller is promised need not rest on that. Both requests are known here, in one turn,
   * so the older promise is settled `superseded` the moment the newer one is made, and the cancel
   * is posted all the same so the worker still drops the work whenever it is in time to. A result
   * that arrives for a request no longer pending is discarded, which is what the listener already
   * does with every answer it did not ask for.
   */
  const derive = (tree: JsonValue, path: string, callOptions: CallOptions): Promise<PyRecord> => {
    const older = deriving.get(path);
    const held = older === undefined ? undefined : pending.get(older);
    if (older !== undefined && held !== undefined) {
      pending.delete(older);
      held.release();
      port.postMessage({ id: 0, call: 'cancel', args: [older] });
      held.settle({ id: older, kind: 'cancelled', call: 'derive', reason: 'superseded' });
    }
    let mine = 0;
    const answer = request<PyRecord>(
      'derive',
      [tree, path, stripped(callOptions)],
      callOptions,
      (id) => {
        mine = id;
        deriving.set(path, id);
      },
    );
    const forget = (): void => {
      if (deriving.get(path) === mine) deriving.delete(path);
    };
    return answer.then(
      (value) => {
        forget();
        return value;
      },
      (error: unknown) => {
        forget();
        throw error;
      },
    );
  };

  const close = (): void => {
    if (closed) return;
    closed = true;
    port.postMessage({ id: 0, call: 'close', args: [] });
    for (const [id, held] of pending) {
      pending.delete(id);
      held.release();
      held.settle({ id, kind: 'cancelled', call: held.call, reason: 'closed' });
    }
    deriving.clear();
    stopListening();
    stopFailures();
    options.onClose?.();
  };

  return {
    loadSchemas: (files, schemaOptions) =>
      request<SchemasLoaded>('loadSchemas', [files, schemaOptions]),
    loadLibrary: (bases: readonly LibraryBaseFiles[], schemas: SchemasHandle, libraryOptions) =>
      request<LibraryLoaded>('loadLibrary', [bases, schemas, libraryOptions]),
    documentBases: (tree: JsonValue, path: string) =>
      request<DocumentBases>('documentBases', [tree, path]),
    baseTemplates: (bases: readonly LibraryBaseFiles[], schemas: SchemasHandle) =>
      request<(string | null)[]>('baseTemplates', [bases, schemas]),
    validateUnit: (unit: JsonValue, path: string, library: LibraryHandle) =>
      request<Problem[]>('validateUnit', [unit, path, library]),
    parse: (text: string) => request<JsonValue>('parse', [text]),
    serialize: (tree: JsonValue) => request<string>('serialize', [tree]),
    validate: (tree: JsonValue, path: string, callOptions: ValidateOptions) =>
      request<Verdict>('validate', [tree, path, stripped(callOptions)], callOptions),
    describe: (tree: JsonValue, path: string, callOptions: DescribeOptions) =>
      request<Facts>('describe', [tree, path, strippedDescribe(callOptions)], callOptions),
    check: (path: string, candidate: Candidate) =>
      request<CandidateVerdict>('check', [path, candidate]),
    expand: (tree: JsonValue, callOptions: CallOptions) =>
      request<PyRecord>('expand', [tree, stripped(callOptions)], callOptions),
    derive,
    checkCheckpoint: (derived: PyValue, headers: CheckpointHeaders) =>
      request<CheckpointReport>('checkCheckpoint', [derived, headers]),
    readHeader: (bytes: Uint8Array, file: string) => request<HeaderRead>('readHeader', [bytes, file]),
    evaluate: (expression: PyValue, quantities: Quantities, env?: Env) =>
      request<PyValue>('evaluate', [expression, quantities, env]),
    forget: (path: string) => request<void>('forget', [path]),
    release: (handle) => request<void>('release', [handle]),
    close,
  };
}


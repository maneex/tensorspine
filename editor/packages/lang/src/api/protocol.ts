/**
 * The wire between the editor and the worker: request ids, cancellation and progress (§5.3).
 *
 * A hand protocol rather than Comlink, for two reasons the plan's own words make necessary.
 * §5.3 asks for "request ids, cancellation and progress", and cancellation is not a remote call
 * at all: it is a message that must *overtake* the work it cancels, which a call-and-return proxy
 * has no shape for. And what crosses has to be encoded (`codec.ts`): `UNRESOLVED` is a symbol and
 * a refusal is a class, and a generic proxy would drop both silently.
 *
 * The port is structural, so that one host serves a `DedicatedWorkerGlobalScope`, a browser
 * `Worker`, a `MessagePort` of either platform and anything else that posts and listens — which
 * is what lets the boundary be tested in Node without a bundler (`packages/lang` names its own
 * imports with the `.js` specifiers a browser bundler resolves and Node does not, so a Node
 * worker thread cannot load the core at all; the `MessageChannel` of one thread is a true
 * structured-clone boundary and is what the unit layer uses).
 */
import type { EncodedFailure } from './codec.js';
import type { CancelReason, Progress } from './types.js';

/** The message a port delivers. `MessageEvent` is the DOM's; the core names no DOM type. */
export interface LangMessageEvent {
  readonly data: unknown;
}

/**
 * A port that dispatches events: a browser `Worker`, a `MessagePort`, a worker's own global scope.
 */
export interface LangEventPort {
  postMessage(message: unknown): void;
  addEventListener(type: 'message', listener: (event: LangMessageEvent) => void): void;
  removeEventListener(type: 'message', listener: (event: LangMessageEvent) => void): void;
  /** A `MessagePort` delivers nothing until it is started; a `Worker` has no such method. */
  start?: () => void;
}

/**
 * A port that *emits* messages: `node:worker_threads`' `parentPort` and `MessagePort`.
 *
 * Both shapes are admitted because §5.3 asks the core to run "in a worker or in Node", and Node's
 * ports are `EventEmitter`s whose `addEventListener` is typed for the DOM's generic listener and
 * therefore fits no per-event signature. The two differ in what the listener is handed — an event
 * with a `data` member, or the message itself — which {@link listen} is what normalises.
 */
export interface LangEmitterPort {
  postMessage(message: unknown): void;
  on(event: 'message', listener: (data: unknown) => void): unknown;
  off(event: 'message', listener: (data: unknown) => void): unknown;
  start?: () => void;
}

/** Anything that can carry the protocol. */
export type LangPort = LangEventPort | LangEmitterPort;

/**
 * Listen for messages on a port of either shape, and answer how to stop.
 *
 * A port that has `addEventListener` is driven through it — Node's has one too, and it delivers an
 * event with a `data` member exactly as the DOM's does (measured) — so the emitter branch is for
 * a port that has only `on`.
 */
export function listen(port: LangPort, handler: (data: unknown) => void): () => void {
  if ('addEventListener' in port) {
    const listener = (event: LangMessageEvent): void => {
      handler(event.data);
    };
    port.addEventListener('message', listener);
    port.start?.();
    return () => {
      port.removeEventListener('message', listener);
    };
  }
  const listener = (data: unknown): void => {
    handler(data);
  };
  port.on('message', listener);
  port.start?.();
  return () => {
    port.off('message', listener);
  };
}

/** Every call of the `Lang` interface, by the name the wire carries. */
export const CALLS = [
  'loadSchemas',
  'loadLibrary',
  'validateUnit',
  'parse',
  'serialize',
  'validate',
  'describe',
  'check',
  'expand',
  'derive',
  'checkCheckpoint',
  'readHeader',
  'evaluate',
  'forget',
  'release',
] as const;

/** One of {@link CALLS}. */
export type CallName = (typeof CALLS)[number];

/** What the client sends. `cancel` and `close` are the protocol's own, not calls of the API. */
export interface LangRequest {
  readonly id: number;
  readonly call: CallName | 'cancel' | 'close';
  /** The call's arguments, encoded; `[targetId]` for a cancel, `[]` for a close. */
  readonly args: readonly unknown[];
}

/** What the host sends back. Only the first three settle a request; `progress` does not. */
export type LangResponse =
  | { readonly id: number; readonly kind: 'value'; readonly value: unknown }
  | { readonly id: number; readonly kind: 'failure'; readonly failure: EncodedFailure }
  | {
      readonly id: number;
      readonly kind: 'cancelled';
      readonly call: string;
      readonly reason: CancelReason;
    }
  | { readonly id: number; readonly kind: 'progress'; readonly progress: Progress };

/** Whether a message off the wire is a request this host understands. */
export function isRequest(data: unknown): data is LangRequest {
  if (typeof data !== 'object' || data === null) return false;
  const message = data as Partial<LangRequest>;
  return (
    typeof message.id === 'number' && typeof message.call === 'string' && Array.isArray(message.args)
  );
}

/** Whether a message off the wire is a response this client understands. */
export function isResponse(data: unknown): data is LangResponse {
  if (typeof data !== 'object' || data === null) return false;
  const message = data as Partial<LangResponse>;
  return typeof message.id === 'number' && typeof message.kind === 'string';
}

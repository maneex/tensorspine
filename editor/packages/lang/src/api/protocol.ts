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
/**
 * Listen for the two failures a port reports *instead of* a message, and answer how to stop.
 *
 * A worker that never loaded — a 404 under a mis-set base, a chunk that throws while it is
 * evaluated, the browser killing it — delivers `error` and nothing else, ever; a reply that could
 * not be deserialised delivers `messageerror` and no `message`. Both are silent to
 * {@link listen}, and silence is the worst answer a request can get: every promise stays pending,
 * the page never becomes ready, and nothing anywhere says why.
 *
 * **Why this reads the port structurally.** {@link LangEventPort} declares exactly the `message`
 * listener the protocol needs, and widening it to a second event type would make every port the
 * core accepts — a browser `Worker`, a `MessagePort`, a worker's own global scope, Node's
 * `parentPort` — have to satisfy a signature for events some of them do not have. So the two
 * extra registrations are made through a structural reading of the port, and a port that has
 * neither shape gets nothing and loses nothing: this is the only place in the protocol that asks
 * a port for something it may not have.
 */
export function listenForFailure(
  port: LangPort,
  handler: (kind: PortFailure, event: unknown) => void,
): () => void {
  const any = port as unknown as {
    addEventListener?: (type: string, listener: (event: unknown) => void) => void;
    removeEventListener?: (type: string, listener: (event: unknown) => void) => void;
    on?: (event: string, listener: (value: unknown) => void) => unknown;
    off?: (event: string, listener: (value: unknown) => void) => unknown;
  };
  const stops: (() => void)[] = [];
  for (const kind of PORT_FAILURES) {
    const listener = (event: unknown): void => {
      handler(kind, event);
    };
    if (typeof any.addEventListener === 'function') {
      any.addEventListener(kind, listener);
      const remove = any.removeEventListener;
      if (typeof remove === 'function') {
        stops.push(() => {
          remove.call(any, kind, listener);
        });
      }
      continue;
    }
    if (typeof any.on === 'function') {
      any.on(kind, listener);
      const off = any.off;
      if (typeof off === 'function') {
        stops.push(() => {
          off.call(any, kind, listener);
        });
      }
    }
  }
  return () => {
    for (const stop of stops) stop();
  };
}

/** How a port can fail instead of delivering a message. */
export type PortFailure = 'error' | 'messageerror';

/** Both of them, in the order they are registered. */
export const PORT_FAILURES: readonly PortFailure[] = ['error', 'messageerror'];

/**
 * What such an event says, as far as anything can be read from it without naming a DOM type.
 *
 * An `ErrorEvent` carries `message`, `filename` and `lineno`; Node hands its `'error'` listener
 * the `Error` itself. Every member is read defensively, because the one thing that must not
 * happen here is a throw inside the listener that reports a failure.
 */
export function describePortFailure(kind: PortFailure, event: unknown): string {
  if (kind === 'messageerror') return 'a reply from the worker could not be deserialised';
  if (event instanceof Error) return event.message;
  const fields = event as { message?: unknown; filename?: unknown; lineno?: unknown } | null;
  const message = typeof fields?.message === 'string' ? fields.message : 'the worker failed';
  const file = typeof fields?.filename === 'string' && fields.filename !== '' ? fields.filename : null;
  const line = typeof fields?.lineno === 'number' ? String(fields.lineno) : null;
  if (file === null) return message;
  return `${message} (${file}${line === null ? '' : `:${line}`})`;
}

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

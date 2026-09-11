/**
 * The worker's side of the seam: one {@link LangSession} driven from the messages a port delivers.
 *
 * **How a cancellation can be noticed at all.** The core is synchronous: once `derive` is entered
 * it runs to the end, and a worker's thread cannot read its own queue while it does. So the host
 * does not interrupt work — it *checks* between stages, and the session yields to the task queue
 * there so that the cancel message has a turn to arrive (`session.ts`, `yieldToTasks`). What
 * follows from that, and is what the editor gets:
 *
 * - a request cancelled before its heavy stage begins never runs;
 * - a request cancelled while its heavy stage runs has its **result dropped** — the stage
 *   finishes, and the answer is not sent;
 * - a cheap call (`check` during a drag, `parse`, `evaluate`) posted behind a queued derivation is
 *   answered *first*, because the derivation is waiting at its yield and the cheap call does not
 *   yield at all. §5.6 gives `check` twenty milliseconds and a derivation two seconds; a strict
 *   first-in-first-out would have made the first unmeetable behind the second.
 *
 * Nothing here decides anything about a document. Every answer is the session's, and the session's
 * is the core's.
 */
import { encode, decode, encodeFailure } from './codec.js';
import { LangSession, type Control } from './session.js';
import {
  isRequest,
  listen,
  type CallName,
  type LangPort,
  type LangRequest,
  type LangResponse,
} from './protocol.js';
import { LangCancelled, type CancelReason } from './types.js';

/** What {@link serveLang} answers, so that a host can be stopped in a test. */
export interface LangHost {
  /** The session it drives, for a caller that wants it in the same thread. */
  readonly session: LangSession;
  /** Stop listening and drop everything held. */
  stop(): void;
}

/**
 * Serve the `Lang` API over a port.
 *
 * The port is whatever carries messages: a worker's own global scope (the browser's entry point),
 * one end of a `MessageChannel` (what the unit suite uses, which is a true structured-clone
 * boundary), or a `MessagePort` handed over by a shell.
 */
export function serveLang(port: LangPort, session: LangSession = new LangSession()): LangHost {
  /** The requests that have been asked to stop, by id, with the reason. */
  const cancelled = new Map<number, CancelReason>();
  /** The requests still running, so that a close can cancel them all. */
  const running = new Set<number>();

  const send = (response: LangResponse): void => {
    port.postMessage(response);
  };

  const listener = (message: unknown): void => {
    const request: unknown = message;
    if (!isRequest(request)) return;
    if (request.call === 'cancel') {
      const target = request.args[0];
      // Only a request still running can be cancelled. A cancel that arrives after its request
      // settled — the race a caller cannot avoid — is dropped rather than remembered, or the map
      // would grow one entry per abort for the life of the session.
      if (typeof target === 'number' && running.has(target)) cancelled.set(target, 'requested');
      return;
    }
    if (request.call === 'close') {
      for (const id of running) cancelled.set(id, 'closed');
      session.clear();
      return;
    }
    running.add(request.id);
    void answer(request).finally(() => {
      running.delete(request.id);
      cancelled.delete(request.id);
    });
  };

  const answer = async (request: LangRequest): Promise<void> => {
    const control: Control = {
      cancelled: () => {
        const reason = cancelled.get(request.id);
        return reason === undefined ? null : new LangCancelled(request.call, reason);
      },
      report: (progress) => {
        send({ id: request.id, kind: 'progress', progress });
      },
    };
    try {
      const value = await dispatch(session, request.call as CallName, request.args, control);
      // The last check before the answer leaves: a request cancelled while its stage was running
      // has its result **dropped**, which is what "a superseded derivation is cancelled and its
      // result dropped" asks for.
      const late = control.cancelled();
      if (late !== null) throw late;
      send({ id: request.id, kind: 'value', value: encode(value) });
    } catch (error) {
      if (error instanceof LangCancelled) {
        send({ id: request.id, kind: 'cancelled', call: error.call, reason: error.reason });
        return;
      }
      send({ id: request.id, kind: 'failure', failure: encodeFailure(error) });
    }
  };

  const stopListening = listen(port, listener);

  return {
    session,
    stop: () => {
      stopListening();
      session.clear();
    },
  };
}

/**
 * One call, from the arguments the wire carried.
 *
 * The wire is untyped by construction — a structured clone answers `unknown` — and this is the
 * one place the types are put back on. The producer is `client.ts`, whose surface is the typed
 * `Lang` interface, so what a cast asserts here is what that interface guaranteed there; a
 * message from anywhere else is a caller's mistake and comes back as the refusal the session
 * raises on it.
 */
async function dispatch(
  session: LangSession,
  call: CallName,
  encoded: readonly unknown[],
  control: Control,
): Promise<unknown> {
  const args = encoded.map((one) => decode(one));
  const at = <T>(index: number): T => args[index] as T;
  switch (call) {
    case 'loadSchemas':
      return session.loadSchemas(at(0), at(1) ?? {});
    case 'loadLibrary':
      return session.loadLibrary(at(0), at(1), at(2) ?? {});
    case 'validateUnit':
      return session.validateUnit(at(0), at(1), at(2));
    case 'parse':
      return session.parse(at(0));
    case 'serialize':
      return session.serialize(at(0));
    case 'validate':
      return session.validate(at(0), at(1), at(2), control);
    case 'describe':
      return session.describe(at(0), at(1), at(2), control);
    case 'check':
      return session.check(at(0), at(1));
    case 'expand':
      return session.expand(at(0), at(1), control);
    case 'derive':
      return session.derive(at(0), at(1), at(2), control);
    case 'checkCheckpoint':
      return session.checkCheckpoint(at(0), at(1));
    case 'readHeader':
      return session.readHeader(at(0), at(1));
    case 'evaluate':
      return session.evaluate(at(0), at(1), at(2));
    case 'forget':
      session.forget(at(0));
      return undefined;
    case 'release':
      session.release(at(0));
      return undefined;
  }
}

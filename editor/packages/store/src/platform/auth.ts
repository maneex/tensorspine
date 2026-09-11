/**
 * `NoAuth` — feature 2.4, §5.2.
 *
 * The static web application and the desktop have no accounts: everything computes in the client,
 * there is nothing to sign in to, and Q8 defers the SaaS. So `current()` is always `null`, the
 * chrome shows no avatar (the inventory's rule: "no avatar without a session"), and `signIn`
 * refuses with `unsupported` in the deployment's own words rather than pretending.
 *
 * It is written here, beside the interfaces, rather than in a deployment: §5.2 lists it under the
 * static web application and under the desktop, and it is the same object in both. The SaaS
 * increment adds `OidcAuth` or `SessionAuth` beside it and changes nothing else.
 */
import { PlatformError, type AuthProvider, type Session, type Unsubscribe } from './types.js';

/** The provider for a deployment with no accounts. `reason` is what a sign-in is refused with. */
export function noAuth(reason: string): AuthProvider {
  return {
    current: (): Session | null => null,
    signIn: (): Promise<Session> => Promise.reject(new PlatformError(reason, 'unsupported')),
    // Signing out of nothing is done: a caller that offers the command unconditionally is right.
    signOut: (): Promise<void> => Promise.resolve(),
    // The session never changes, so the callback is never called and there is nothing to cancel.
    onChange: (): Unsubscribe => () => undefined,
  };
}

/**
 * A provider that holds a session in memory — the stub's, where a suite needs one.
 *
 * The chrome's rule is "no avatar without a session", which is two claims and not one: nothing
 * shows where there is no session, and something shows where there is. `NoAuth` can only be asked
 * the first, so this answers the second. It is not a deployment — there is nothing to sign in to
 * and `signIn` answers the session it already holds — and it names no platform, so it belongs
 * beside `noAuth` rather than in one.
 */
export function memoryAuth(session: Session | null = null): AuthProvider {
  let held = session;
  const listeners = new Set<(session: Session | null) => void>();
  const announce = (): void => {
    for (const listener of listeners) listener(held);
  };
  return {
    current: (): Session | null => held,
    signIn: (): Promise<Session> => {
      if (held === null) {
        return Promise.reject(new PlatformError('this provider was given no session', 'unsupported'));
      }
      announce();
      return Promise.resolve(held);
    },
    signOut: (): Promise<void> => {
      held = null;
      announce();
      return Promise.resolve();
    },
    onChange: (callback): Unsubscribe => {
      listeners.add(callback);
      return () => {
        listeners.delete(callback);
      };
    },
  };
}

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

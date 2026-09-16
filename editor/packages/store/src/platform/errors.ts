/**
 * What the platform refuses with — feature 2.4.
 *
 * One class for every refusal of §5.2, in a module of its own so that the path arithmetic and the
 * interfaces can both raise it without importing each other: a caller writes one `catch` and reads
 * {@link PlatformError.reason} to know what happened, and never meets an error class it has no
 * name for.
 *
 * A refusal of the *language* is not one of these: the core answers a `Problem` with the tools'
 * wording (§5.3), and nothing here knows what a document is.
 */

/**
 * Why the platform refused.
 *
 * `not-found` — no such file or directory; `read-only` — the workspace cannot be written at all
 * (a snapshot asked to create a folder); `conflict` — the file moved on since it was read;
 * `bad-path` — the path is not one this workspace can name, `..` above its root among them;
 * `unsupported` — the deployment has no such facility (a browser asked to sign in where there are
 * no accounts, or to open a folder where it has no writable picker); `denied` — the user or the
 * browser said no.
 *
 * Two are a fetched set's (feature 2.20), and they are apart because the chrome answers them
 * differently: `unreachable` — nothing came back from the address, which is where a cache with its
 * age is the honest offer; `corrupt` — something came back and it is not what the set declares,
 * which is never a cache and never a retry.
 */
export type PlatformRefusal =
  | 'not-found'
  | 'read-only'
  | 'conflict'
  | 'bad-path'
  | 'unsupported'
  | 'denied'
  | 'unreachable'
  | 'corrupt';

/** A refusal of the platform, as opposed to one of the language. */
export class PlatformError extends Error {
  constructor(
    message: string,
    /** What kind of refusal it is, for the chrome that has to answer it. */
    readonly reason: PlatformRefusal,
    /** The revisions a conflict compared: what the file holds, and what the caller expected. */
    readonly revisions?: { readonly found: string; readonly expected: string },
  ) {
    super(message);
    this.name = 'PlatformError';
  }
}

/**
 * Published file sets fetched from an address, in a browser — feature 2.20.
 *
 * The format, the integrity check and every refusal's wording are
 * `@tensorspine/store/platform`'s (`remote.ts`); what is here is the three things only a
 * deployment can do — get a text from a URL, take a sha256, and remember what was fetched — and
 * the policy between them.
 *
 * **CORS, stated rather than assumed.** A static page on GitHub Pages fetching another origin
 * needs that origin's permission, and the answer is the *server's*, not the engine's: feature
 * 0.5 measured exactly that against the Hub, where `huggingface.co` answers a preflight with
 * `access-control-allow-headers: range`, `access-control-allow-methods: GET` and an
 * `access-control-allow-origin` on the response. A host that answers none of it leaves the page
 * with a `TypeError` carrying no status and no headers — the browser refuses the read before any
 * content arrives, and script cannot see why. So the refusal **names the header the host has to
 * send**, and says in the same breath that an unreachable host looks the same from here, because
 * it does. A spinner would be the alternative, and the feature's block forbids it.
 *
 * **The cache, and why it holds whole sets.** A set is fetched whole and checked whole when it is
 * opened (the reader's rule), so what there is to cache is the set — the manifest and every file,
 * each already checked against its digest. That is what makes the offline answer honest: a
 * workspace whose address cannot be reached opens from the cache **with its age shown**, and a
 * set that was never fetched refuses with the address in the message. Nothing opens silently
 * stale, and nothing that was served wrong is ever cached: a digest that disagrees refuses the
 * whole set and never reaches the store.
 *
 * **Integrity needs `crypto.subtle`, which needs a secure context.** `https://…` and `localhost`
 * have one; plain `http://` on a LAN address does not. Where it is missing the set is **refused**,
 * not opened unchecked: a manifest whose digests nobody compares is a file list, and a file list
 * is not what the feature's Integrity paragraph asks for.
 */
import {
  PlatformError,
  fetchPublishedSet,
  heldSet,
  addressOf,
  type PublishedManifest,
  type PublishedSet,
  type RemoteSets,
  type Retrieve,
} from '@tensorspine/store/platform';

import { REMOTE, memoryStore, openStore, type Keyed, type Store } from './idb.js';

/** One cached set, as the store holds it. */
interface CachedSet extends Keyed {
  /** The manifest's own address — the key. */
  readonly id: string;
  readonly root: string;
  readonly fetchedAt: string;
  readonly manifest: PublishedManifest;
  readonly texts: Readonly<Record<string, string>>;
}

/** The one header a host must answer with for another origin's page to read it at all. */
export const ALLOW_ORIGIN = 'Access-Control-Allow-Origin';

/**
 * The text of one URL, or a refusal that says which of the two things went wrong.
 *
 * A status is a host that answered; a `TypeError` is a host that did not, or did not permit it.
 * The two are apart because they need different answers: the first is an address to check, the
 * second a header to add.
 */
export const browserRetrieve: Retrieve = async (url: string): Promise<string> => {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new PlatformError(
      `nothing came back from ${url} (${describe(error)}): either the host is not reachable from ` +
        `here, or it did not permit this page to read it — a set served from another origin must ` +
        `answer with ${ALLOW_ORIGIN} naming this page's origin (or *). A browser does not tell a ` +
        'refused read from an unreachable host, so both are named.',
      'unreachable',
    );
  }
  if (!response.ok) {
    throw new PlatformError(
      `${url} answered ${String(response.status)} ${response.statusText}`,
      response.status === 404 ? 'not-found' : 'unreachable',
    );
  }
  return response.text();
};

/** The sha256 of a text, as the manifest writes one. */
export async function browserDigest(text: string): Promise<string> {
  const subtle = globalThis.crypto.subtle as SubtleCrypto | undefined;
  if (subtle === undefined) {
    throw new PlatformError(
      'this page cannot take a sha256 (crypto.subtle is absent, which is what an insecure context ' +
        'means): a published set is checked against the digests its manifest declares, and an ' +
        'unchecked set is not opened. Serve the editor over https, or over localhost.',
      'unsupported',
    );
  }
  const bytes = await subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** What {@link browserRemote} is given — a suite's seams, and nothing a page needs to supply. */
export interface RemoteOptions {
  /** How a text is fetched; the browser's own by default. */
  readonly retrieve?: Retrieve;
  /** Where the cache lives; IndexedDB by default, memory where it refused. */
  readonly store?: Store<CachedSet> | null;
  /** Read the files one at a time even where the manifest declares a bundle — a measurement's. */
  readonly bundles?: boolean;
}

/**
 * The published sets of the static application: fetched, checked, cached, and offered back.
 *
 * Asynchronous because the cache is: the store is probed once, here, so that a browser which
 * refuses IndexedDB is known before the first set is opened rather than at it.
 */
export async function browserRemote(options: RemoteOptions = {}): Promise<RemoteSets> {
  const store = options.store === undefined ? await openStore<CachedSet>(REMOTE) : options.store;
  const cache = store ?? memoryStore<CachedSet>();
  const retrieve = options.retrieve ?? browserRetrieve;
  const opened: PublishedSet[] = [];

  const remember = (set: PublishedSet): PublishedSet => {
    const already = opened.findIndex((one) => one.address === set.address);
    if (already >= 0) opened.splice(already, 1);
    opened.unshift(set);
    return set;
  };

  return {
    persistent: store !== null,

    async open(address: string): Promise<PublishedSet> {
      const at = addressOf(address).manifest;
      try {
        const set = await fetchPublishedSet({
          address,
          retrieve,
          digest: browserDigest,
          ...(options.bundles === undefined ? {} : { bundles: options.bundles }),
        });
        // Cached after it was checked, never before: a set that was served wrong is refused above
        // and never reaches the store, so nothing here can hand back bytes the manifest refused.
        await cache
          .put({
            id: set.address,
            root: set.root,
            fetchedAt: set.fetchedAt,
            manifest: set.manifest,
            texts: set.texts,
          })
          .catch(() => undefined);
        return remember(set);
      } catch (error) {
        // The cache answers for **one** thing: nothing came back from the address. A set that was
        // served and did not match what it declares is a refusal and nothing else — answering it
        // with bytes from another day, for an address that is answering wrongly now, is exactly
        // the silent staleness the feature's Offline paragraph forbids — and anything else that
        // went wrong is not evidence about the host at all, so it is raised as it stands.
        const unreached =
          error instanceof PlatformError &&
          (error.reason === 'unreachable' || error.reason === 'not-found');
        if (!unreached) throw error;
        const held = await cache.get(at).catch(() => null);
        if (held === null) {
          throw new PlatformError(
            `${error.message} — and nothing of ${at} is in this browser's cache, so there is ` +
              'nothing to open',
            error.reason,
          );
        }
        return remember(
          heldSet(at, held.root, held.manifest, held.texts, held.fetchedAt, true),
        );
      }
    },

    held: () => opened,

    async forget(address?: string): Promise<void> {
      if (address === undefined) {
        opened.length = 0;
        await cache.clear().catch(() => undefined);
        return;
      }
      const at = addressOf(address).manifest;
      const index = opened.findIndex((one) => one.address === at);
      if (index >= 0) opened.splice(index, 1);
      await cache.remove(at).catch(() => undefined);
    },
  };
}

/** A refusal in the words whatever raised it gave it. */
function describe(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

/**
 * Settings in `localStorage` — feature 2.4, §5.2.
 *
 * "User settings, layouts, recents, unlocked bases": which theme, which panels are where, which
 * base a maintainer unlocked (Q6). Small, stringifiable, read while a component renders — which
 * is why the store is loaded once when the platform is built and answers from memory afterwards,
 * writing through on every change.
 *
 * **It can refuse, and not merely come back empty.** In a private window, with site data cleared,
 * or in a browser set to block it, reading `window.localStorage` *itself* throws. So every access
 * here is guarded and the store keeps working with nothing behind it: the settings live for the
 * session, {@link SettingsStore.persistent} is false, and the chrome is entitled to say so.
 *
 * Keys are prefixed so that the editor shares an origin politely — the documentation site is
 * served from the same one (D11) — and a key whose stored text is not JSON is dropped rather
 * than raised, because whatever wrote it was not this.
 *
 * Recents are **not** here though §5.2 lists them under settings: a remembered folder is a
 * `FileSystemDirectoryHandle`, which IndexedDB stores and `localStorage` cannot (`recents.ts`).
 */
import type { SettingValue, SettingsStore, Unsubscribe } from '@tensorspine/store/platform';

/** What the editor's own keys begin with. */
export const PREFIX = 'tensorspine.editor.';

/** The backing store, where the browser has one that works. */
function backing(): Storage | null {
  try {
    const store = window.localStorage;
    const probe = `${PREFIX}probe`;
    store.setItem(probe, '1');
    store.removeItem(probe);
    return store;
  } catch {
    return null;
  }
}

/** The settings of the static application, loaded once and written through. */
export function browserSettings(): SettingsStore {
  const store = backing();
  const held = new Map<string, SettingValue>();
  const listeners = new Set<(key: string) => void>();

  const load = (): void => {
    held.clear();
    if (store === null) return;
    for (let index = 0; index < store.length; index += 1) {
      const key = store.key(index);
      if (key === null || !key.startsWith(PREFIX)) continue;
      const text = store.getItem(key);
      if (text === null) continue;
      try {
        held.set(key.slice(PREFIX.length), JSON.parse(text) as SettingValue);
      } catch {
        // Whatever wrote this was not the editor, or was an older one; it is not a refusal.
      }
    }
  };
  load();

  const announce = (key: string): void => {
    for (const listener of listeners) listener(key);
  };

  // Another tab of the editor wrote: the same origin, the same keys, so the store keeps up.
  //
  // `window.onstorage` rather than `addEventListener`, and the reason is worth writing down: the
  // event's name is `storage`, which is a value of the unit schema's axis `space`, and catching
  // rule (b) of §1 forbids the interface to write a vocabulary item as a string literal. The
  // property is the same event and this module is its one owner — the last store built takes the
  // handler, and the application builds one. The collision is a finding of feature 2.4.
  if (store !== null) {
    window.onstorage = (event: StorageEvent) => {
      if (event.storageArea !== store) return;
      if (event.key === null) {
        load();
        announce('');
        return;
      }
      if (!event.key.startsWith(PREFIX)) return;
      const key = event.key.slice(PREFIX.length);
      if (event.newValue === null) held.delete(key);
      else {
        try {
          held.set(key, JSON.parse(event.newValue) as SettingValue);
        } catch {
          return;
        }
      }
      announce(key);
    };
  }

  return {
    get: <T extends SettingValue>(key: string, fallback: T): T => {
      const found = held.get(key);
      return found === undefined ? fallback : (found as T);
    },
    peek: (key) => held.get(key),
    set: (key, value) => {
      held.set(key, value);
      try {
        store?.setItem(`${PREFIX}${key}`, JSON.stringify(value));
      } catch {
        // A quota or a browser that changed its mind: the setting still holds for this session.
      }
      announce(key);
    },
    remove: (key) => {
      held.delete(key);
      try {
        store?.removeItem(`${PREFIX}${key}`);
      } catch {
        // As above.
      }
      announce(key);
    },
    keys: () => [...held.keys()].sort((a, b) => a.localeCompare(b)),
    onChange: (callback): Unsubscribe => {
      listeners.add(callback);
      return () => {
        listeners.delete(callback);
      };
    },
    persistent: store !== null,
  };
}

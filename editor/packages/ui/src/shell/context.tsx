/**
 * How a component reaches the shell's store and the platform.
 *
 * Two contexts and no more. The store is the chrome's state (`store.ts`); the platform is
 * everything the editor cannot do for itself (§5.2). Nothing else is ambient: a component that
 * needs a document will be given one by the feature that opens it, because there is no second
 * model to reach for (D1).
 *
 * `useShell` takes a selector, which is the reason Zustand is here at all: a status bar that
 * reads the zoom does not re-render when a panel moves, and a canvas does not re-render when the
 * Log gains a line. A selector must answer a value that compares equal between renders — a
 * primitive, or a slice the store itself holds — because the subscription is `Object.is` over
 * what it answers; a selector that built a new object each time would re-render for ever.
 */
import { createContext, useContext, type JSX, type ReactNode } from 'react';
import { useStore } from 'zustand';

import type { Platform } from '@tensorspine/store/platform';

import type { Shell, ShellStore } from './store.js';

const StoreContext = createContext<ShellStore | null>(null);
const PlatformContext = createContext<Platform | null>(null);

/** Put the store and the platform in reach of everything below. */
export function ShellProvider({
  store,
  platform,
  children,
}: {
  store: ShellStore;
  platform: Platform;
  children: ReactNode;
}): JSX.Element {
  return (
    <StoreContext.Provider value={store}>
      <PlatformContext.Provider value={platform}>{children}</PlatformContext.Provider>
    </StoreContext.Provider>
  );
}

/** The store itself, for a gesture that needs to read the state at the moment it fires. */
export function useShellStore(): ShellStore {
  const store = useContext(StoreContext);
  if (store === null) throw new Error('a component of the shell was rendered outside it');
  return store;
}

/** One value of the shell's state, subscribed to on its own. */
export function useShell<T>(selector: (state: Shell) => T): T {
  return useStore(useShellStore(), selector);
}

/** The platform (§5.2) — the workspace, the settings, the shell's own few powers. */
export function usePlatform(): Platform {
  const platform = useContext(PlatformContext);
  if (platform === null) throw new Error('a component of the shell was rendered outside it');
  return platform;
}

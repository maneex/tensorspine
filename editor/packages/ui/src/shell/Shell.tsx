/**
 * The shell — plan §4.2 as drawn on S1, assembled.
 *
 * ```
 * ┌──────────────────────────────────────────────────────────────────────────────┐
 * │ Menu bar                                                          ⟳ ✓ user  │
 * ├───┬──────────────────┬──────────────────────────────────────┬───────────────┤
 * │ A │ Side bar         │ Editor area (tabs)                   │ Properties    │
 * │ c │                  │                                      │               │
 * │ t │                  ├──────────────────────────────────────┤               │
 * │ y │                  │ Bottom panel: Problems │ Derived │ Log│               │
 * ├───┴──────────────────┴──────────────────────────────────────┴───────────────┤
 * │ Status bar                                                                  │
 * └──────────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * Three things are this component's own rather than any region's.
 *
 * **The accelerators.** A key event that reaches nothing focused reaches the window, so the
 * chords are bound there and not on an element; `commandFor` decides which command a stroke is,
 * under this platform's modifier and under whether a text field has the focus (§4.4: an editing
 * chord belongs to whatever is being typed in while it is being typed in). Every command it can
 * fire is also in a menu, which is the rule that keeps a shortcut an accelerator.
 *
 * **The theme.** `tokens.css` puts the dark block on `:root` and the light one on
 * `.theme-light`, so which of the two is in force is one class on the frame — and the frame is
 * what the palette's scrim and every pop-up sit inside, so there is no corner of the page that
 * keeps the other theme.
 *
 * **The native menu.** §4.4's commands are offered to the platform on mount: a browser has no
 * menu and ignores them, and Electron will build its own from them. The interface's own menu bar
 * is not that; it is `Bar.tsx`, and it draws the same table.
 */
import { useEffect, type JSX, type ReactNode } from 'react';

import type { MenuCommand, Platform } from '@tensorspine/store/platform';

import { Bar } from './Bar.js';
import { acceleratorText, COMMANDS, commandFor, matches, PALETTE_CHORD } from './commands.js';
import { ShellProvider, useShell, useShellStore, usePlatform } from './context.js';
import { EditorArea, type TabViews } from './EditorArea.js';
import { Palette } from './Palette.js';
import { PanelRegion } from './Panels.js';
import { Rail, Side } from './Rail.js';
import { StatusBar } from './StatusBar.js';
import type { ShellStore } from './store.js';
import { text } from './strings.js';
import { LIGHT_CLASS, resolveTheme } from './theme.js';
import { SHELL_VIEWS } from './views.js';

/** Whether the focus is somewhere a key belongs to what is being typed rather than to a command. */
export function isTyping(element: Element | null): boolean {
  if (element === null) return false;
  if (element instanceof HTMLElement && element.isContentEditable) return true;
  const tag = element.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

/** The frame, inside the provider. */
function Frame({ views, children }: { views: TabViews; children?: ReactNode }): JSX.Element {
  const store = useShellStore();
  const platform = usePlatform();
  const theme = useShell((state) => state.theme);
  const scheme = useShell((state) => state.scheme);
  const side = useShell((state) => state.side);
  const palette = useShell((state) => state.palette);
  const resolved = resolveTheme(theme, scheme);
  const modifier = platform.shell.modifier;

  // §4.4's shortcuts, bound where a key with nothing focused still arrives.
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (matches(PALETTE_CHORD, event, modifier)) {
        event.preventDefault();
        store.getState().setPalette(true);
        return;
      }
      if (event.key === 'Escape') {
        const state = store.getState();
        if (state.palette) state.setPalette(false);
        else if (state.menu !== null) state.openMenu(null);
        return;
      }
      const command = commandFor(event, modifier, isTyping(document.activeElement));
      if (command === undefined) return;
      // A browser keeps some of these for itself (Ctrl+W, Ctrl+N); asking is all a page can do.
      event.preventDefault();
      store.getState().run(command.id);
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
    };
  }, [store, modifier]);

  // §4.4's commands, offered to a native menu that may not exist.
  useEffect(() => {
    const menu: MenuCommand[] = COMMANDS.map((command) => ({
      id: command.id,
      label: text(command.label),
      enabled: true,
      ...(command.accelerator === undefined
        ? {}
        : { accelerator: acceleratorText(command.accelerator, modifier) }),
    }));
    platform.shell.setMenu(menu);
  }, [platform, modifier]);

  return (
    <div className={resolved === 'light' ? `app ${LIGHT_CLASS}` : 'app'} data-scheme={resolved}>
      <Bar />
      <div className="mid">
        <Rail />
        {side.open ? <Side /> : null}
        <EditorArea views={views} below={<PanelRegion region="region.bottom" />} />
        <PanelRegion region="region.right" />
      </div>
      <StatusBar />
      {palette ? <Palette /> : null}
      {children}
    </div>
  );
}

/** What {@link Shell} is given. */
export interface ShellProps {
  /**
   * The store, built by whoever owns the page.
   *
   * Not built here, and that is a decision rather than an omission: the store subscribes to the
   * platform's session, workspace and colour scheme, so its life is the application's and not a
   * component's. A store built during a render is a store React may build twice and throw one
   * away, subscriptions and all — which is the defect feature 2.4's review repair closed one
   * level down. The application calls `createShell`, keeps the store, and takes it down with the
   * page; it is also what a later feature binds its command handlers through.
   */
  readonly store: ShellStore;
  readonly platform: Platform;
  /** Views for tab kinds a later feature opens; the shell's own two are always there. */
  readonly views?: TabViews;
  /**
   * What a later feature puts inside the frame beside the regions — a dialog, a toast.
   *
   * Inside, and not beside: the theme is one class on the frame (`tokens.css` puts the dark block
   * on `:root` and the light one on `.theme-light`), so anything rendered outside it would keep
   * the other theme. Feature 2.6's dialogs and its toast arrive this way.
   */
  readonly children?: ReactNode;
}

/** The shell, over a store and the platform that store was built on. */
export function Shell({ store, platform, views, children }: ShellProps): JSX.Element {
  return (
    <ShellProvider store={store} platform={platform}>
      <Frame views={{ ...SHELL_VIEWS, ...views }}>{children}</Frame>
    </ShellProvider>
  );
}

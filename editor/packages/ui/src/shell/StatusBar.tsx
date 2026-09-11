/**
 * The status bar — S1's `.status`, §4.2's last row.
 *
 * > model id · `tensorspine/2.0` · validation state (checking / ok / N problems) · derivation
 * > state (fresh / stale / failed) · D3 total bytes · D5 operations per element · D4 append bytes
 * > per cached position · D2 peak live bytes per element · zoom · theme
 *
 * Eight of those ten are a document's, and **every one of them is a figure the core computes** —
 * the component inventory's rule ("Every figure — D1–D6. No component adds, converts or rounds a
 * byte count"). There is no document here yet and no core reading yet (2.6, 2.8, 2.15), so the
 * bar shows the two that are answerable today and says plainly that it is showing nothing else:
 * the workspace it is looking at, the zoom, and the theme. Writing `tensorspine/2.0` into it
 * would be writing a `const` of the schema into the interface, which catching rule (b) forbids
 * and which feature 2.1 already refused to do for the same reason.
 *
 * Both fields on the right are **clickable**, which is §4.4's rule rather than a flourish: "a
 * value is edited in the sheet's row, or by clicking the element itself". The zoom runs `View ▸
 * Zoom to Fit`; the theme steps through `Theme: Light`, `Theme: Dark`, `Theme: System` — the
 * three commands of §4.4, so nothing here is reachable only from the status bar either.
 */
import type { JSX } from 'react';

import { useShell, useShellStore, usePlatform } from './context.js';
import { text, textWith } from './strings.js';
import { resolveTheme, type ColourScheme, type ThemeChoice } from './theme.js';

/** What the theme field says: the choice, and what it resolves to when it is the machine's. */
export function themeLabel(choice: ThemeChoice, scheme: ColourScheme): string {
  const resolved = resolveTheme(choice, scheme);
  const name = (one: ThemeChoice | 'light' | 'dark'): string =>
    one === 'light' ? text('Light') : one === 'dark' ? text('Dark') : text('System');
  return choice === 'system'
    ? `${text('Theme')}: ${name('system')} (${name(resolved)})`
    : `${text('Theme')}: ${name(choice)}`;
}

/** Which command the theme field runs next: the three of §4.4, in the menu's order. */
const NEXT_THEME: Readonly<Record<ThemeChoice, string>> = {
  light: 'view.theme-dark',
  dark: 'view.theme-system',
  system: 'view.theme-light',
};

export function StatusBar(): JSX.Element {
  const store = useShellStore();
  const platform = usePlatform();
  const workspace = useShell((state) => state.workspace);
  const zoom = useShell((state) => state.zoom);
  const theme = useShell((state) => state.theme);
  const scheme = useShell((state) => state.scheme);

  return (
    <footer className="status" aria-label={text('Status')}>
      <span className="mono" data-workspace={workspace.kind}>
        {workspace.kind === 'empty' ? text('No workspace') : workspace.name}
      </span>
      {workspace.kind === 'empty' || workspace.writable ? null : (
        <span className="mono dimf">{text('read-only — Save hands you the file')}</span>
      )}
      <span className="mono dimf">{textWith('platform: {}', platform.describe())}</span>
      <span className="right">
        <button
          type="button"
          className="fig"
          data-zoom={zoom}
          title={text('Zoom to Fit')}
          onClick={() => {
            store.getState().run('view.zoom-fit');
          }}
        >
          {`${String(Math.round(zoom * 100))}%`}
        </button>
        <button
          type="button"
          data-theme={theme}
          data-scheme={scheme}
          onClick={() => {
            store.getState().run(NEXT_THEME[theme]);
          }}
        >
          {themeLabel(theme, scheme)}
        </button>
      </span>
    </footer>
  );
}

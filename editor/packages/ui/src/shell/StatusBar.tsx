/**
 * The status bar — S1's `.status`, §4.2's last row.
 *
 * > model id · `tensorspine/2.0` · validation state (checking / ok / N problems) · derivation
 * > state (fresh / stale / failed) · D3 total bytes · D5 operations per element · D4 append bytes
 * > per cached position · D2 peak live bytes per element · zoom · theme
 *
 * Eight of those ten are a document's, and **every one of them is a figure the core computes** —
 * the component inventory's rule ("Every figure — D1–D6. No component adds, converts or rounds a
 * byte count"). Feature 2.5 left them out because there was no document; feature 2.6 opens one
 * and `../documents/Pills.tsx` answers all eight, each from the core or from the document's own
 * tree and none from a literal: the model id and the revision tag are read at the two members the
 * *schema* names (the one it requires as free text, the one it fixes), the two states are
 * `validate` and `derive`, and the four figures are the places `presentation.json` marks in the
 * derived document. With nothing open they are not shown at all, and what is left is what this
 * component has always said: the workspace, the zoom, the theme.
 *
 * Both fields on the right are **clickable**, which is §4.4's rule rather than a flourish: "a
 * value is edited in the sheet's row, or by clicking the element itself". The zoom runs `View ▸
 * Zoom to Fit`; the theme steps through `Theme: Light`, `Theme: Dark`, `Theme: System` — the
 * three commands of §4.4, so nothing here is reachable only from the status bar either.
 */
import type { JSX } from 'react';

import { StatusFields, useOpenDocument } from '../documents/Pills.js';
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
  const showing = useOpenDocument();
  const zoom = useShell((state) => state.zoom);
  const theme = useShell((state) => state.theme);
  const scheme = useShell((state) => state.scheme);

  return (
    <footer className="status" aria-label={text('Status')}>
      <span className="mono" data-workspace={workspace.kind}>
        {workspace.kind === 'empty' ? text('No workspace') : workspace.name}
      </span>
      {showing || workspace.kind === 'empty' || workspace.writable ? null : (
        <span className="mono dimf">{text('read-only — Save hands you the file')}</span>
      )}
      <StatusFields />
      {showing ? null : (
        <span className="mono dimf">{textWith('platform: {}', platform.describe())}</span>
      )}
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

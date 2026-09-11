/**
 * The Help menu's two in-app screens — §4.4's `Keyboard Shortcuts` and `About`.
 *
 * They are tabs of the editor area rather than dialogs, for two reasons. The first is that they
 * are documents: a table of ninety commands and a list of what this build is, both worth leaving
 * open beside the work. The second is that they are what proves the tab strip works — until a
 * document can be opened (2.6) they are the only tabs there are, so opening, switching and
 * closing a tab is exercised by the shell's own suite rather than owed to a later one.
 *
 * **Keyboard Shortcuts is where §4.21's "every command reachable from the keyboard" is answered**
 * to the person who has to find them: every command of §4.4, its menu, and its accelerator where
 * it has one — read from the same table the menus and the palette read, so it cannot fall behind
 * them.
 */
import { Fragment, type JSX } from 'react';

import { acceleratorText, COMMANDS, MENUS, PALETTE_CHORD } from './commands.js';
import { usePlatform, useShell } from './context.js';
import { currentLocale, text, textWith } from './strings.js';
import { themeLabel } from './StatusBar.js';
import type { Tab } from './store.js';

/** Every command of §4.4, by menu, with the accelerator this platform writes. */
export function KeyboardShortcuts(): JSX.Element {
  const platform = usePlatform();
  const modifier = platform.shell.modifier;
  return (
    <div className="doc" role="region" tabIndex={0} aria-label={text('Keyboard Shortcuts')}>
      <div className="doc-body">
        <h1>{text('Keyboard Shortcuts')}</h1>
        <p className="note-line">
          {text(
            'Every command is in a menu and in the command palette; a shortcut is an accelerator for the ones you use often, and never the only way to reach anything.',
          )}
        </p>
        <table>
          <caption>
            {textWith('{} commands.', String(COMMANDS.length))}{' '}
            {textWith('The command palette is {}.', acceleratorText(PALETTE_CHORD, modifier))}
          </caption>
          <thead>
            <tr>
              <th scope="col">{text('Menu')}</th>
              <th scope="col">{text('Command')}</th>
              <th scope="col">{text('Shortcut')}</th>
            </tr>
          </thead>
          <tbody>
            {COMMANDS.map((command) => (
              <tr key={command.id}>
                <td>{text(MENUS.find((menu) => menu.id === command.menu)?.label ?? '')}</td>
                <td>{text(command.label)}</td>
                <td className="mono">
                  {command.accelerator === undefined ? '' : acceleratorText(command.accelerator, modifier)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * What this build is — §4.4's "About (versions: editor, schemas' `$id`s, the core's parity
 * commit, Electron)".
 *
 * Three of those four are read from things this feature does not load: the schemas arrive with
 * the workspace (2.6) and the core's parity commit with the vendored manifest. What is here is
 * what is known, stated as it stands rather than as a blank labelled with a promise.
 */
export function About(): JSX.Element {
  const platform = usePlatform();
  const workspace = useShell((state) => state.workspace);
  const theme = useShell((state) => state.theme);
  const scheme = useShell((state) => state.scheme);
  const rows: [string, string][] = [
    [text('Platform'), platform.describe()],
    [text('Workspace'), `${workspace.kind} · ${workspace.name}`],
    [text('Writable'), workspace.writable ? text('yes') : text('no')],
    [text('Settings'), platform.settings.persistent ? text('remembered') : text('this session only')],
    [text('Drafts'), platform.drafts.persistent ? text('remembered') : text('this session only')],
    [text('Accelerators'), acceleratorText({ mod: true, key: 'S' }, platform.shell.modifier)],
    [text('Theme'), themeLabel(theme, scheme)],
    [text('Language'), currentLocale()],
  ];
  return (
    <div className="doc" role="region" tabIndex={0} aria-label={text('About')}>
      <div className="doc-body">
        <h1>{text('TensorSpine Editor')}</h1>
        <p className="note-line">
          {text(
            'A graph editor for tensorspine documents and primitive-library units. Everything it validates and derives, it computes here, in this page.',
          )}
        </p>
        <dl>
          {rows.map(([name, value]) => (
            <Fragment key={name}>
              <dt>{name}</dt>
              <dd>{value}</dd>
            </Fragment>
          ))}
        </dl>
      </div>
    </div>
  );
}

/** The views the shell brings with it, by the `kind` their tabs carry. */
export const SHELL_VIEWS: Readonly<Record<string, (tab: Tab) => JSX.Element>> = {
  'view.shortcuts': () => <KeyboardShortcuts />,
  'view.about': () => <About />,
};

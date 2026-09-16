/**
 * The menu bar — S1's `.bar`, and §4.4's menus inside it.
 *
 * The lockup, the kind, the seven menus, and on the right the palette's way in and the avatar.
 * Two rules of §4.2 and §4.4 are structural here rather than remembered:
 *
 *  - **every command of §4.4 has a menu entry**, because the entries are `commands.ts` read —
 *    there is no list of "the ones we built", and a command nobody has wired says so in the Log
 *    when it is chosen rather than being absent from the menu that promises it;
 *  - **no avatar without a session** (component inventory §2): `NoAuth` answers `null`, which is
 *    every deployment of this plan (Q8 defers the SaaS), so the bar ends at the palette button.
 *
 * The two pills on the right are **the core's validation and derivation state** (inventory §2),
 * shown only with a document open — feature 2.6 answers them, and with nothing open there is
 * nothing to be the state of.
 */
import { useCallback, useEffect, useRef, useState, type JSX, type KeyboardEvent } from 'react';

import { BarPills } from '../documents/Pills.js';
import { acceleratorText, commandsOf, MENUS, PALETTE_CHORD, type MenuId } from './commands.js';
import { useShell, useShellStore, usePlatform } from './context.js';
import { FilterIcon } from './icons.js';
import { MONOGRAM } from './logo.js';
import { text } from './strings.js';

/** The lockup: the monogram of `docs/tensorspine.svg` with the name beside it. */
export function Wordmark(): JSX.Element {
  return (
    // One image with one name. The drawing is the file's and the lettering beside it is the
    // brand's, so nothing inside is read out twice — and WCAG's own logotype exemption is why
    // the name keeps the mark's colours rather than the interface's ink.
    <div className="wordmark" role="img" aria-label="TensorSpine">
      <span className="mark" dangerouslySetInnerHTML={{ __html: MONOGRAM }} />
      <span className="name" aria-hidden="true">
        <span>Tensor</span>
        <span>Spine</span>
      </span>
    </div>
  );
}

/** A person's initials, for the avatar S1 draws as `PL`. */
export function initialsOf(name: string): string {
  const words = name.split(/\s+/).filter((word) => word.length > 0);
  const letters = (words.length > 1 ? [words[0], words[words.length - 1]] : words)
    .map((word) => word?.[0] ?? '')
    .join('');
  return (letters.length > 0 ? letters : name.slice(0, 2)).toUpperCase();
}

/** One menu of the bar, with its pop-up. */
function BarMenu({ id, label }: { id: MenuId; label: string }): JSX.Element {
  const store = useShellStore();
  const open = useShell((state) => state.menu === id);
  const platform = usePlatform();
  const modifier = platform.shell.modifier;
  const button = useRef<HTMLButtonElement>(null);
  const items = useRef<HTMLDivElement>(null);
  const commands = commandsOf(id);

  useEffect(() => {
    if (!open) return;
    items.current?.querySelector<HTMLButtonElement>('button')?.focus();
  }, [open]);

  const close = useCallback(
    (refocus: boolean): void => {
      store.getState().openMenu(null);
      if (refocus) button.current?.focus();
    },
    [store],
  );

  const onItemKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.stopPropagation();
      close(true);
      return;
    }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    const all = [...(items.current?.querySelectorAll<HTMLButtonElement>('button') ?? [])];
    const at = all.indexOf(document.activeElement as HTMLButtonElement);
    const step = event.key === 'ArrowDown' ? 1 : -1;
    all[(at + step + all.length) % all.length]?.focus();
  };

  return (
    <div
      className="menu-anchor"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) close(false);
      }}
    >
      <button
        ref={button}
        type="button"
        role="menuitem"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => {
          store.getState().openMenu(open ? null : id);
        }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowDown' && !open) {
            event.preventDefault();
            store.getState().openMenu(id);
          }
        }}
        onMouseEnter={() => {
          // A bar whose menus swap under the pointer once one is open: the desktop convention,
          // and the reason a menu bar is quicker to read than a stack of separate buttons.
          if (store.getState().menu !== null) store.getState().openMenu(id);
        }}
      >
        {text(label)}
      </button>
      {open ? (
        <div className="menu-pop" role="menu" aria-label={text(label)} ref={items} onKeyDown={onItemKey}>
          {commands.map((command) => (
            <button
              key={command.id}
              type="button"
              role="menuitem"
              data-command={command.id}
              onClick={() => {
                store.getState().run(command.id);
              }}
            >
              <span>{text(command.label)}</span>
              {command.accelerator === undefined ? null : (
                <span className="kbd">{acceleratorText(command.accelerator, modifier)}</span>
              )}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** What {@link Bar} is given. */
export interface BarProps {
  /**
   * Whether this bar carries the lockup (default: it does).
   *
   * `false` where the page already has one above it — the documentation site's bar of S18, which
   * the static build is served under (D11). One lockup to a page: the kind ("Editor") goes with
   * it, the site's bar marking the editor as the page being read.
   */
  readonly lockup?: boolean;
}

/** The bar. */
export function Bar({ lockup = true }: BarProps = {}): JSX.Element {
  const store = useShellStore();
  const platform = usePlatform();
  const session = useShell((state) => state.session);
  const [bar, setBar] = useState<HTMLElement | null>(null);

  // Left and right walk the menu titles, which is what a menu bar is for; the pop-up's own arrows
  // walk its entries.
  const onBarKey = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const titles = [...(bar?.querySelectorAll<HTMLButtonElement>(':scope > div > button') ?? [])];
    const at = titles.indexOf(document.activeElement as HTMLButtonElement);
    if (at < 0) return;
    event.preventDefault();
    const step = event.key === 'ArrowRight' ? 1 : -1;
    const next = titles[(at + step + titles.length) % titles.length];
    next?.focus();
    if (store.getState().menu !== null) next?.click();
  };

  return (
    <header className="bar">
      {lockup ? (
        <>
          <Wordmark />
          <span className="sitekind">{text('Editor')}</span>
        </>
      ) : null}
      <nav className="menu" role="menubar" aria-label={text('Main menu')} ref={setBar} onKeyDown={onBarKey}>
        {MENUS.map((menu) => (
          <BarMenu key={menu.id} id={menu.id} label={menu.label} />
        ))}
      </nav>
      <div className="bar-right">
        <BarPills />
        <button
          type="button"
          className="palette-open"
          onClick={() => {
            store.getState().setPalette(true);
          }}
        >
          <FilterIcon />
          <span>{text('Commands')}</span>
          <span className="kbd">{acceleratorText(PALETTE_CHORD, platform.shell.modifier)}</span>
        </button>
        {session === null ? null : (
          <span className="avatar" title={session.name}>
            {initialsOf(session.name)}
          </span>
        )}
      </div>
    </header>
  );
}

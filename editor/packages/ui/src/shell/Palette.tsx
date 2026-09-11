/**
 * The command palette — §4.2's own region: "Ctrl/⌘ + Shift + P: every command of §4.4 by name".
 *
 * Every command, and the word is the test: the list is `COMMANDS` filtered, never a curated
 * subset, so a command that exists is findable by name whether or not a feature has wired it yet.
 * A query matches the command's own label first and its menu second, which is what makes
 * `derive` find `Model ▸ Derive Now` and `model` find all fifteen of that menu.
 *
 * It is a dialog rather than a drop-down because it takes the keyboard whole while it is up:
 * Escape closes it, the arrows walk it, Enter runs what is highlighted, and nothing behind it is
 * reachable until it goes — which is the only way a list of ninety commands can be driven from
 * the keyboard without fighting whatever had focus.
 */
import { useEffect, useMemo, useRef, useState, type JSX } from 'react';

import { acceleratorText, matching } from './commands.js';
import { useShellStore, usePlatform } from './context.js';
import { FilterIcon } from './icons.js';
import { text, textWith } from './strings.js';

export function Palette(): JSX.Element {
  const store = useShellStore();
  const platform = usePlatform();
  const [query, setQuery] = useState('');
  const [at, setAt] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const found = useMemo(() => matching(query), [query]);
  const here = Math.min(at, Math.max(0, found.length - 1));

  useEffect(() => {
    input.current?.focus();
  }, []);

  useEffect(() => {
    list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
  }, [here, found.length]);

  const close = (): void => {
    store.getState().setPalette(false);
  };

  return (
    <div
      className="scrim"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div
        className="dlg"
        role="dialog"
        aria-modal="true"
        aria-label={text('Commands')}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            close();
            return;
          }
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            const step = event.key === 'ArrowDown' ? 1 : -1;
            setAt((was) => {
              const now = Math.min(was, Math.max(0, found.length - 1));
              return found.length === 0 ? 0 : (now + step + found.length) % found.length;
            });
            return;
          }
          if (event.key === 'Enter') {
            event.preventDefault();
            const chosen = found[here];
            if (chosen !== undefined) store.getState().run(chosen.command.id);
          }
        }}
      >
        <div className="dlg-head">
          <FilterIcon />
          <input
            ref={input}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={found[here] === undefined ? undefined : `cmd-${found[here].command.id}`}
            aria-label={text('Search every command')}
            placeholder={text('Search every command')}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setAt(0);
            }}
          />
        </div>
        <div className="dlg-body" id="palette-list" role="listbox" aria-label={text('Commands')} ref={list}>
          {found.map(({ command, menu }, index) => (
            <div
              key={command.id}
              id={`cmd-${command.id}`}
              className="cmd"
              role="option"
              aria-selected={index === here}
              data-command={command.id}
              onPointerDown={(event) => {
                event.preventDefault();
                store.getState().run(command.id);
              }}
              onPointerEnter={() => {
                setAt(index);
              }}
            >
              <span className="in">{text(menu)}</span>
              <span>{text(command.label)}</span>
              {command.accelerator === undefined ? null : (
                <span className="kbd">{acceleratorText(command.accelerator, platform.shell.modifier)}</span>
              )}
            </div>
          ))}
        </div>
        <div className="dlg-foot">{textWith('{} commands.', String(found.length))}</div>
      </div>
    </div>
  );
}

/**
 * The Model explorer — plan §4.5, S1's left panel.
 *
 * > Selecting an item selects it on the canvas and in Properties; double-click drills in; the
 * > tree supports rename by clicking the name (F2 as an accelerator), delete, drag of a quantity
 * > onto an argument row (sets its source to that quantity), and a filter box.
 *
 * The rows are `outline.ts`'s and every gesture is a command of the store (2.1) computed with the
 * reference rules of `presentation.json` (2.2): a rename rewrites every occurrence of the name, a
 * delete cascades over everything that names the thing and says what it kept, and both are one
 * entry of the Edit menu's undo (D13). Nothing here decides what a name refers to and nothing
 * here refuses a gesture for a semantic reason (Q5) — a refusal is the core's, in Problems.
 *
 * **The tree is one ARIA widget, not a stack of buttons.** Feature 2.5 measured what that costs:
 * a focusable element inside a widget is a *serious* axe violation, and a `role="treeitem"` with
 * a chevron button and a name button inside it is exactly that. So the row itself takes the
 * keyboard — arrows move and open, Enter opens the sheet, F2 renames, Delete deletes — and the
 * chevron and the name are parts of the row that a pointer can hit. Every one of those gestures
 * is also a command of §4.4, which is the rule that keeps a shortcut an accelerator.
 *
 * **Clicking the name of a selected row renames it.** §4.5 asks for rename by clicking the name;
 * a name that renamed on the first click would rename on the click that only meant to select, so
 * the first click selects and the second — on the name of the row that is already selected —
 * opens the editor. The same edit is in the Properties sheet's name row and on F2 (§4.4's rule:
 * a shortcut is never the only way, nor the main way).
 */
import { useEffect, useMemo, useRef, useState, type JSX, type KeyboardEvent } from 'react';

import { compositionAt, foldedGraph } from '@tensorspine/lang';
import { EditError, pointerOf, remove, rename, type EditContext, type Path } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import type { OpenDocument } from '../documents/store.js';
import { presentation, referenceSelectors } from '../presentation/index.js';
import { text, textWith } from '../shell/strings.js';
import { COMPUTED, GUARDED, outlineOf, SHARED, TEMPLATE, type OutlineRow } from './outline.js';

/** What a row dragged out of the tree carries — §4.5's "drag of a quantity onto an argument row". */
export const DECLARATION_TRANSFER = 'application/x-tensorspine-declaration';

/** The declaration a drag carries: what it is, what it is called, and where it is written. */
export interface DeclarationTransfer {
  /** What the map it is declared in declares, as `presentation.json` names it. */
  readonly declares?: string;
  readonly name: string;
  /** The place, as an RFC 6901 pointer — so a drop can read the declaration itself. */
  readonly pointer: string;
}

/** What a drag put on the clipboard, or `null` where it carried something else. */
export function declarationOf(transfer: DataTransfer): DeclarationTransfer | null {
  const held = transfer.getData(DECLARATION_TRANSFER);
  if (held === '') return null;
  try {
    const parsed: unknown = JSON.parse(held);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const one = parsed as DeclarationTransfer;
    return typeof one.name === 'string' && typeof one.pointer === 'string' ? one : null;
  } catch {
    return null;
  }
}

/** The command a rename makes, with the occurrences `presentation.json` says name the thing. */
export function renameCommand(row: OutlineRow, to: string): (context: EditContext) => ReturnType<typeof rename> {
  const binding = row.declaredAt === undefined ? undefined : presentation().at(row.declaredAt);
  const references = binding === undefined ? [] : referenceSelectors(binding, row.scope);
  const what = row.declares ?? '';
  return (context) =>
    rename(context, {
      path: row.path,
      to,
      references,
      label: `Rename ${what === '' ? row.label : `${what} ${row.label}`} to ${to}`,
    });
}

/** The command a delete makes: the declaration, and everything whose rule names it. */
export function removeCommand(row: OutlineRow): (context: EditContext) => ReturnType<typeof remove> {
  const binding = row.declaredAt === undefined ? undefined : presentation().at(row.declaredAt);
  const references = binding === undefined ? [] : referenceSelectors(binding, row.scope);
  const what = row.declares ?? '';
  return (context) =>
    remove(context, {
      path: row.path,
      references,
      label: `Delete ${what === '' ? row.label : `${what} ${row.label}`}`,
    });
}

/**
 * What each mark of §4.5 means, in words, for whoever cannot see the glyph.
 *
 * Keyed by the marks themselves, which are the outline's own constants and no vocabulary of the
 * language: a guard is a condition the row writes, a sharing is a list of several members, and a
 * computed figure is one the document says how to obtain rather than only what it is.
 */
const MARKS: Readonly<Record<string, string>> = {
  [TEMPLATE]: 'an instance of a template primitive',
  [GUARDED]: 'guarded',
  [SHARED]: 'shared by several members',
  [COMPUTED]: 'computed from other quantities',
};

/** The keys the tree answers itself, as an ARIA tree does. */
const HANDLED: readonly string[] = [
  'ArrowDown',
  'ArrowUp',
  'ArrowRight',
  'ArrowLeft',
  'Enter',
  ' ',
  'F2',
  'Delete',
  'Backspace',
];

/** The rows of the outline for a document, recomputed when the document or the reading moves. */
function useOutline(
  one: OpenDocument | undefined,
  filter: string,
  templates: ReadonlySet<string>,
): OutlineRow[] {
  const revision = one?.session.store.revision ?? -1;
  const toggled = one?.toggled ?? [];
  const problems = one?.reading.verdict?.problems;
  return useMemo(() => {
    if (one === undefined) return [];
    return outlineOf({
      tree: one.session.store.tree,
      shapes: one.session.store.shapes,
      bindings: presentation(),
      role: one.session.store.role,
      toggled: new Set(toggled),
      filter,
      problems: (problems ?? []).map((problem) => problem.path),
      templates,
    });
    // The tree is immutable and the revision counts its states (2.1), so the revision is what
    // says the outline has to be built again — reading `store.tree` here would build it on every
    // render of the side bar instead.
  }, [one?.id, revision, toggled, filter, problems, templates]);
}

/** The Model explorer: the filter box of §4.5 and the outline under it. */
export function ModelExplorer(): JSX.Element {
  const store = useDocumentsStore();
  const current = useDocuments((state) => state.current);
  const open = useDocuments((state) => state.open);
  const filter = useDocuments((state) => state.filter);
  const templates = useDocuments((state) => state.library.templates);
  const one = open.find((each) => each.id === current);
  const rows = useOutline(one, filter, templates);
  const selection = one?.selection === undefined ? null : pointerOf(one.selection);
  const [editing, setEditing] = useState<string | null>(null);
  const [focused, setFocused] = useState<string | null>(null);
  const tree = useRef<HTMLDivElement>(null);

  // A document that closed, or a place that went: the tree stops editing what is not there.
  useEffect(() => {
    if (editing !== null && !rows.some((row) => row.pointer === editing)) setEditing(null);
  }, [rows, editing]);

  // One row of the tree takes the tab stop, as an ARIA tree has it: the one the reader last
  // moved to, failing that the selected one, failing that the first.
  const at = focused ?? selection ?? rows[0]?.pointer ?? null;

  const select = (row: OutlineRow): void => {
    setFocused(row.pointer);
    // The document's own row selects the document: §4.11 gives it a sheet ("`model` id ·
    // `version` · `primitive_libraries` … · counts"), which is the sheet of the root place, and
    // feature 2.7 left the row clearing the selection because that sheet did not exist yet.
    store.getState().selectPlace(row.path);
  };

  const toggle = (row: OutlineRow): void => {
    if (!row.container) return;
    store.getState().togglePlace(row.pointer);
  };

  /**
   * A double-click on a row: §4.8's drill-in where the row is a composition, the fold elsewhere.
   *
   * Feature 2.7 left the double-click toggling every row "(2.14 owns the drill-in tab)"; this is
   * that hand-over. A composition is the one row of the outline that has a canvas of its own, and
   * opening the thing you double-click is what the gesture means everywhere else in the editor
   * (S3's own caption: "double-click, or Ctrl+Enter to drill in").
   */
  const openRow = (row: OutlineRow): void => {
    // Whether the row is a composition is the **core's** answer, not a reading of the path: the
    // interface names no map of the grammar (§1), and the folded graph already knows which box
    // holds sites.
    const tree = one?.session.store.tree;
    const composition = tree === undefined ? null : compositionAt(foldedGraph(tree), row.pointer);
    if (composition !== null) {
      store.getState().drillInto(composition);
      return;
    }
    toggle(row);
  };

  const move = (from: OutlineRow, by: number): void => {
    // The line that says what a row holds is passed over: it is a row of the tree and not a
    // place of the document, so there is nothing to select, rename or delete on it.
    let position = rows.indexOf(from) + by;
    while (rows[position]?.kind === 'note') position += by;
    const next = rows[Math.min(Math.max(position, 0), rows.length - 1)];
    if (next === undefined || next.kind === 'note') return;
    setFocused(next.pointer);
    tree.current?.querySelector<HTMLElement>(`[data-row="${cssEscape(next.pointer)}"]`)?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>, row: OutlineRow): void => {
    const key = event.key;
    // What the tree answers, the window's accelerators must not answer again (§4.4's chords are
    // bound on the window, and a `treeitem` is not a field `isTyping` would except).
    if (HANDLED.includes(key)) event.stopPropagation();
    if (key === 'ArrowDown' || key === 'ArrowUp') {
      event.preventDefault();
      move(row, key === 'ArrowDown' ? 1 : -1);
    } else if (key === 'ArrowRight') {
      if (row.container && !row.open) toggle(row);
      else move(row, 1);
    } else if (key === 'ArrowLeft') {
      if (row.container && row.open) toggle(row);
      else move(row, -1);
    } else if (key === 'Enter' || key === ' ') {
      event.preventDefault();
      select(row);
    } else if (key === 'F2') {
      event.preventDefault();
      if (row.named) setEditing(row.pointer);
    } else if (key === 'Delete' || key === 'Backspace') {
      event.preventDefault();
      if (row.named) store.getState().offer(removeCommand(row));
    }
  };

  if (one === undefined) {
    return (
      <>
        <FilterBox />
        <div className="side-body" tabIndex={0}>
          <p className="empty-sub">{text('No document is open.')}</p>
        </div>
      </>
    );
  }

  return (
    <>
      <FilterBox />
      <div className="side-body" ref={tree}>
        <div className="tree" role="tree" aria-label={textWith('The outline of {}', one.title)}>
          {rows.map((row) => {
            const chosen = row.pointer === selection;
            const editable = editing === row.pointer;
            return (
              <div
                key={row.pointer}
                data-row={row.pointer}
                data-kind={row.kind}
                className={[
                  'row',
                  `lvl${String(Math.min(row.depth, 4))}`,
                  row.kind === 'entry' && !row.container ? 'leaf' : '',
                  row.kind === 'note' ? 'quiet' : '',
                  chosen ? 'sel' : '',
                ]
                  .filter((part) => part !== '')
                  .join(' ')}
                // Every child of a `tree` is a `treeitem`; the line that says what a row holds
                // is one nothing can act on, which is what `aria-disabled` says. A row with a
                // role and an `aria-` attribute its role does not admit is what the axe pass
                // reports, and the shape below is what it accepts.
                role="treeitem"
                aria-level={row.depth + 1}
                aria-disabled={row.kind === 'note' ? true : undefined}
                aria-selected={row.kind === 'note' ? undefined : chosen}
                aria-expanded={row.container ? row.open : undefined}
                tabIndex={row.kind === 'note' ? -1 : row.pointer === at ? 0 : -1}
                draggable={row.named && !editable}
                title={row.kind === 'note' ? row.label : row.tail}
                onFocus={() => {
                  setFocused(row.pointer);
                }}
                onKeyDown={(event) => {
                  if (row.kind !== 'note' && !editable) onKeyDown(event, row);
                }}
                onClick={() => {
                  if (row.kind !== 'note' && !editable) select(row);
                }}
                onDoubleClick={() => {
                  openRow(row);
                }}
                onDragStart={(event) => {
                  const held: DeclarationTransfer = {
                    ...(row.declares === undefined ? {} : { declares: row.declares }),
                    name: row.label,
                    pointer: row.pointer,
                  };
                  event.dataTransfer.setData(DECLARATION_TRANSFER, JSON.stringify(held));
                  event.dataTransfer.effectAllowed = 'copy';
                }}
              >
                {row.kind === 'note' ? (
                  <span className="held">{row.label}</span>
                ) : (
                  <>
                    <span
                      className="chev"
                      aria-hidden="true"
                      data-chevron={row.container ? row.pointer : undefined}
                      onClick={(event) => {
                        event.stopPropagation();
                        toggle(row);
                      }}
                    >
                      {row.container ? (row.open ? '▾' : '▸') : ''}
                    </span>
                    {row.kind === 'entry' ? (
                      <span className={`sw ${row.tint ?? ''}`} aria-hidden="true" />
                    ) : null}
                    {row.problem ? (
                      <span className="dotbad" role="img" aria-label={text('has a problem')} />
                    ) : null}
                    {editable ? (
                      <NameEditor
                        row={row}
                        onDone={() => {
                          setEditing(null);
                          tree.current
                            ?.querySelector<HTMLElement>(`[data-row="${cssEscape(row.pointer)}"]`)
                            ?.focus();
                        }}
                      />
                    ) : (
                      <span
                        className="n-name"
                        data-name={row.pointer}
                        onClick={(event) => {
                          // The **second** click of a double-click is not the click that renames:
                          // a double-click opens the row (§4.8's drill-in), and the rename the
                          // name offers is the *slow* second click. `detail` is the click count,
                          // which is the one place a browser tells the two apart — and without
                          // this the row is replaced by its editor before the double-click can be
                          // dispatched, so the gesture would simply be lost.
                          if (event.detail >= 2) {
                            event.stopPropagation();
                            openRow(row);
                            return;
                          }
                          if (!row.named || !chosen) return;
                          event.stopPropagation();
                          setEditing(row.pointer);
                        }}
                      >
                        {row.kind === 'document' ? <b>{row.label}</b> : row.label}
                      </span>
                    )}
                    {row.count === undefined ? null : <span className="n">{row.count}</span>}
                    {row.marks.map((mark) => (
                      <span
                        className={mark.length === 1 ? 'mk' : 'drv'}
                        key={mark}
                        role="img"
                        aria-label={text(MARKS[mark] ?? mark)}
                        title={text(MARKS[mark] ?? mark)}
                      >
                        {mark}
                      </span>
                    ))}
                    {row.figure === undefined ? null : <span className="fig">{row.figure}</span>}
                    {row.tail === undefined || editable ? null : (
                      <span className="tail mono">{row.tail}</span>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

/** §4.5's filter box, which is the side bar's own `.search` (S1). */
function FilterBox(): JSX.Element {
  const store = useDocumentsStore();
  const filter = useDocuments((state) => state.filter);
  return (
    <label className="search">
      <svg viewBox="0 0 16 16" className="ico" aria-hidden="true">
        <circle cx="7" cy="7" r="4.2" />
        <path d="m10.1 10.1 3.2 3.2" />
      </svg>
      <input
        type="search"
        value={filter}
        placeholder={text('Filter the document…')}
        aria-label={text('Filter the document…')}
        onChange={(event) => {
          store.getState().setFilter(event.target.value);
        }}
      />
    </label>
  );
}

/** The name of a row, while it is being edited (§4.5's rename by clicking the name). */
function NameEditor({ row, onDone }: { row: OutlineRow; onDone: () => void }): JSX.Element {
  const store = useDocumentsStore();
  const [value, setValue] = useState(row.label);
  const field = useRef<HTMLInputElement>(null);
  useEffect(() => {
    field.current?.select();
  }, []);

  const commit = (): void => {
    const to = value.trim();
    if (to === '' || to === row.label) {
      onDone();
      return;
    }
    try {
      const applied = store.getState().edit(renameCommand(row, to));
      if (applied !== null) store.getState().selectPlace([...row.path.slice(0, -1), to]);
    } catch (error) {
      // A name the map already has is a tree off the *grammar*, which is the one thing a command
      // refuses (D5); the reason is said rather than swallowed, and nothing was written.
      store.getState().setToast({
        text: error instanceof EditError ? error.message : String(error),
      });
    }
    onDone();
  };

  return (
    <input
      ref={field}
      className="n-edit mono"
      data-rename={row.pointer}
      value={value}
      aria-label={textWith('Rename {}', row.label)}
      onChange={(event) => {
        setValue(event.target.value);
      }}
      onClick={(event) => {
        event.stopPropagation();
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Enter') commit();
        if (event.key === 'Escape') onDone();
      }}
    />
  );
}

/**
 * A pointer as a CSS attribute value.
 *
 * `CSS.escape` is the browser's and this module runs in a suite as well, so the two characters a
 * JSON pointer can carry into an attribute selector are escaped by hand.
 */
function cssEscape(pointer: string): string {
  return pointer.replace(/["\\]/g, '\\$&');
}

/** The place a command of §4.4's Edit menu is about: what is selected, as a row of the outline. */
export function selectedRow(store: ReturnType<typeof useDocumentsStore>): OutlineRow | null {
  const state = store.getState();
  const one = state.open.find((each) => each.id === state.current);
  if (one?.selection === undefined) return null;
  const rows = outlineOf({
    tree: one.session.store.tree,
    shapes: one.session.store.shapes,
    bindings: presentation(),
    role: one.session.store.role,
    // Every row, whatever the reader has open: a selection is a place of the *document*, not of
    // the tree's own state, so a command must find it even under a closed group.
    openAll: true,
  });
  return rows.find((each) => each.pointer === pointerOf(one.selection as Path)) ?? null;
}

/** Where a name is typed when a command asks for a rename — the Properties sheet's own row. */
export const NAME_FIELD = 'data-name-field';

/**
 * §4.4's `Delete` and `Rename` on the current selection, for the application to bind.
 *
 * `Delete` offers the cascade and `Rename` puts the caret in the sheet's name row — because a
 * command may not be the only way to reach an edit (§4.4) and the sheet's row is where the name
 * is edited without a mouse. The tree answers the same two keys itself while it has the focus,
 * which is what makes them accelerators there rather than the only way anywhere.
 */
export function selectionHandlers(
  store: ReturnType<typeof useDocumentsStore>,
  showProperties: () => void,
): { readonly rename: () => void; readonly remove: () => void } {
  /** What a command about the selection says when there is none (feature 2.5's own rule). */
  const nothing = (what: string): void => {
    store.getState().note(`${what}: nothing is selected`);
  };
  return {
    rename: () => {
      const row = selectedRow(store);
      if (row === null || !row.named) {
        nothing('Rename');
        return;
      }
      showProperties();
      // After the panel has been drawn, which is the frame after the state changed.
      requestAnimationFrame(() => {
        document.querySelector<HTMLInputElement>(`[${NAME_FIELD}]`)?.select();
      });
    },
    remove: () => {
      const row = selectedRow(store);
      if (row === null || !row.named) {
        nothing('Delete');
        return;
      }
      store.getState().offer(removeCommand(row));
    },
  };
}

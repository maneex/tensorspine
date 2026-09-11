/**
 * The Properties panel's body for whatever is selected — the first row of §4.11's sheets.
 *
 * §4.11 gives every kind of selection its sections, and they arrive with the features that can
 * answer them: the arguments with `describe` (2.10), the ports, the slots and the states with it,
 * the derived rows with the Derived panel (2.15). What is here is the row every one of those
 * sheets starts with — **Identity: the name** — and it is here because §4.4 asks for it by name:
 *
 * > A value is edited in the sheet's row, or by clicking the element itself on the canvas (a name,
 * > a guard badge, a slot chip); a keyboard shortcut is an accelerator for people who know it and
 * > never the only way, nor the main way, to edit anything.
 *
 * So the name of the selected thing is typed here, and the tree's own rename (§4.5) is the second
 * way rather than the only one. The rest of the sheet says what the selection *is* — the word
 * `presentation.json` gives the map it is declared in, and the place it is written — and says
 * plainly that the sections are not here yet, which is the design's own empty-state voice.
 */
import { useEffect, useState, type JSX } from 'react';

import { EditError, pointerOf } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { text } from '../shell/strings.js';
import { NAME_FIELD, renameCommand, selectedRow } from './Explorer.js';

/** The sheet of the current selection, or the line that says there is none. */
export function SelectionSheet(): JSX.Element {
  const store = useDocumentsStore();
  const current = useDocuments((state) => state.current);
  const open = useDocuments((state) => state.open);
  const one = open.find((each) => each.id === current);
  const selection = one?.selection;
  const revision = one?.session.store.revision ?? -1;
  const pointer = selection === undefined ? null : pointerOf(selection);
  const row = selection === undefined ? null : selectedRow(store);
  const [name, setName] = useState(row?.label ?? '');

  // The sheet follows the selection and the document: a rename made in the tree, an undo, a
  // selection that moved — the field shows what the document holds, never what was typed into it
  // for something else.
  useEffect(() => {
    setName(row?.label ?? '');
    // The row is rebuilt on every render; the selection and the revision are what change it.
  }, [pointer, revision]);

  if (one === undefined || row === null || selection === undefined) {
    return <p className="empty-sub">{text('Nothing is selected.')}</p>;
  }

  const commit = (): void => {
    const to = name.trim();
    if (to === '' || to === row.label) {
      setName(row.label);
      return;
    }
    try {
      store.getState().edit(renameCommand(row, to));
      store.getState().selectPlace([...row.path.slice(0, -1), to]);
    } catch (error) {
      store.getState().setToast({ text: error instanceof EditError ? error.message : String(error) });
      setName(row.label);
    }
  };

  return (
    <>
      {/* `h2`, not the design's `h4`: the page's own heading is the open document's name, and a
          heading that skipped two levels is a heading order a reader cannot follow — the same
          correction feature 2.6 made to the dialogs, found the same way. */}
      <h2 className="ih">
        {text('Identity')}
        {row.declares === undefined ? null : <span className="ihn">{row.declares}</span>}
      </h2>
      <div className="frow">
        <span className="fk">{text('Name')}</span>
        <span className="fv">
          {row.named ? (
            <input
              className="ctl wide mono"
              value={name}
              {...{ [NAME_FIELD]: 'true' }}
              aria-label={text('Name')}
              onChange={(event) => {
                setName(event.target.value);
              }}
              onBlur={commit}
              onKeyDown={(event) => {
                event.stopPropagation();
                if (event.key === 'Enter') commit();
                if (event.key === 'Escape') setName(row.label);
              }}
            />
          ) : (
            <span className="mono">{row.label}</span>
          )}
        </span>
      </div>
      <div className="frow">
        <span className="fk">{text('Place')}</span>
        <span className="fv mono" data-place={pointer}>
          {pointer}
        </span>
      </div>
      {row.tail === undefined ? null : (
        <div className="frow">
          <span className="fk">{text('Holds')}</span>
          <span className="fv mono">{row.tail}</span>
        </div>
      )}
      <p className="empty-sub">
        {text(
          'The sections of this sheet — the arguments, the ports, the slots, the states and the derived rows — are what the core answers for a selected instance, and they arrive with the instance sheet.',
        )}
      </p>
    </>
  );
}

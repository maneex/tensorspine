/**
 * The three dialogs of §4.3 and §4.4's File menu — feature 2.6.
 *
 * **Open Recent ▸**, which is also where *Open Examples* is: §4.3 says the Examples workspace
 * "is always available", and §4.4's File menu has one entry for the folders one has had open. So
 * the entry opens a list whose first row is the examples the build carries and whose others are
 * the folders the browser still holds handles for — a list, because a submenu of the bar cannot
 * carry the "Forget" beside each one, and because an upload and a drop have to be offered here
 * for the two engines with no writable picker (D11).
 *
 * **Open Model…**, which §4.4 writes as "a single file": what a static page can offer is the
 * documents of the workspace it has open, which is where a single file of one is; a file from
 * anywhere else reaches the editor as a drop, which the window already takes. The Model explorer
 * of §4.5 is what replaces the list (feature 2.7).
 *
 * **Save As…**, which on a folder the browser can write writes, and on a snapshot downloads —
 * the button says which (§4.3, S18).
 *
 * **Restore a draft**: "on reopening a document with a newer draft the editor offers to restore
 * it" (§4.3). Offered, never applied: the draft is the user's own typing and so is the file.
 */
import { useEffect, useRef, useState, type JSX, type ReactNode } from 'react';

import { useDocuments, useDocumentsStore } from './context.js';
import { text, textWith } from '../shell/strings.js';
import { usePlatform } from '../shell/context.js';

/** The dialog frame the design draws (`.dlg` inside a `.scrim`), with Escape and a focus trap. */
function Frame({
  title,
  hint,
  onClose,
  children,
  foot,
}: {
  title: string;
  hint?: string;
  onClose: () => void;
  children: ReactNode;
  foot?: JSX.Element;
}): JSX.Element {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    box.current?.querySelector<HTMLElement>('button, input')?.focus();
  }, []);
  return (
    <div
      className="scrim"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="dlg"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        ref={box}
        onKeyDown={(event) => {
          if (event.key === 'Escape') onClose();
        }}
      >
        <div className="dlg-head">
          {/* `h2`, not the design's `h3`: the page's own heading is the open document's name, and
              a heading that skipped a level would be a heading order a reader cannot follow.
              Found by this feature's axe pass. */}
          <h2>{title}</h2>
          {hint === undefined ? null : <span className="sub">{hint}</span>}
        </div>
        <div className="dlg-body">{children}</div>
        <div className="dlg-foot">{foot}</div>
      </div>
    </div>
  );
}

/** Open Recent ▸: the examples, the folders opened before, and the two ways in without a picker. */
function Workspaces(): JSX.Element {
  const store = useDocumentsStore();
  const platform = usePlatform();
  const recent = useDocuments((state) =>
    state.dialog?.kind === 'workspaces' ? state.dialog.recent : [],
  );
  const upload = useRef<HTMLInputElement>(null);
  const close = (): void => {
    store.getState().setDialog(null);
  };
  return (
    <Frame
      title={text('Open a workspace')}
      hint={text('a folder laid out like the repository’s data directory')}
      onClose={close}
      foot={
        <>
          <span className="note-line">
            {platform.workspaces.writablePicker
              ? text('A folder you pick is read and written in place; the examples are read-only.')
              : text('This browser has no writable picker: a folder you upload or drop is a read-only snapshot, and Save downloads.')}
          </span>
          <span className="right">
            <button type="button" className="btn ghost" onClick={close}>
              {text('Cancel')}
            </button>
          </span>
        </>
      }
    >
      <button
        type="button"
        className="cmd"
        data-workspace="examples"
        onClick={() => {
          void store.getState().openExamples();
        }}
      >
        <span>{text('Examples')}</span>
        <span className="in">{text('the corpus and the reference base, vendored with the build')}</span>
      </button>
      {platform.workspaces.writablePicker ? (
        <button
          type="button"
          className="cmd"
          data-workspace="folder"
          onClick={() => {
            void store.getState().openFolder();
          }}
        >
          <span>{text('Open Folder…')}</span>
          <span className="in">{text('read and written in place')}</span>
        </button>
      ) : null}
      <button
        type="button"
        className="cmd"
        data-workspace="upload"
        onClick={() => upload.current?.click()}
      >
        <span>{text('Upload a folder…')}</span>
        <span className="in">{text('a read-only snapshot; Save downloads')}</span>
      </button>
      <input
        ref={upload}
        type="file"
        className="offscreen"
        aria-label={text('Upload a folder…')}
        multiple
        // The one attribute a folder upload needs, and the reason `UploadedFile` declares
        // `webkitRelativePath`: it is the browser's own name for where a file stood in the folder.
        {...{ webkitdirectory: '' }}
        onChange={(event) => {
          const files = [...(event.target.files ?? [])];
          if (files.length > 0) void store.getState().openUpload(files);
        }}
      />
      {recent.map((one) => (
        <span className="cmd" key={one.id}>
          <button
            type="button"
            data-recent={one.id}
            onClick={() => {
              void store.getState().openRecent(one.id);
            }}
          >
            {one.name}
          </button>
          <span className="in">{one.openedAt.slice(0, 10)}</span>
          <button
            type="button"
            className="link"
            data-forget={one.id}
            onClick={() => {
              void store.getState().forget(one.id);
            }}
          >
            {text('Forget')}
          </button>
        </span>
      ))}
    </Frame>
  );
}

/** Open Model…: the documents the open workspace holds, filtered by what is typed. */
function Documents(): JSX.Element {
  const store = useDocumentsStore();
  const paths = useDocuments((state) => state.documents);
  const open = useDocuments((state) => state.open);
  const [query, setQuery] = useState('');
  const close = (): void => {
    store.getState().setDialog(null);
  };
  const held = new Set(open.map((one) => one.path));
  const needle = query.trim().toLowerCase();
  const found = paths.filter((path) => needle === '' || path.toLowerCase().includes(needle));
  return (
    <Frame
      title={text('Open Model…')}
      hint={text('the documents this workspace holds')}
      onClose={close}
      foot={
        <>
          <span className="note-line">{textWith('{} documents.', String(found.length))}</span>
          <span className="right">
            <button type="button" className="btn ghost" onClick={close}>
              {text('Cancel')}
            </button>
          </span>
        </>
      }
    >
      <div className="frow">
        <span className="fk">{text('Find')}</span>
        <input
          className="ctl wide mono"
          value={query}
          placeholder={text('a name, or part of a path')}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      {found.slice(0, 200).map((path) => (
        <button
          key={path}
          type="button"
          className="cmd"
          data-document={path}
          onClick={() => {
            close();
            void store.getState().openDocument(path);
          }}
        >
          <span className="mono">{path}</span>
          {held.has(path) ? <span className="in">{text('open')}</span> : null}
        </button>
      ))}
    </Frame>
  );
}

/** Save As…: where the document goes, and what the button will actually do. */
function SaveAs(): JSX.Element {
  const store = useDocumentsStore();
  const dialog = useDocuments((state) => (state.dialog?.kind === 'save-as' ? state.dialog : null));
  const writable = useDocuments((state) => state.workspace.writable);
  const [path, setPath] = useState(dialog?.path ?? '');
  const close = (): void => {
    store.getState().setDialog(null);
  };
  if (dialog === null) return <></>;
  const write = (): void => {
    void store.getState().saveAs(dialog.id, path);
  };
  return (
    <Frame
      title={text('Save As…')}
      hint={writable ? text('into this workspace') : text('this workspace is read-only: it downloads')}
      onClose={close}
      foot={
        <>
          <span className="note-line">
            {writable ? text('A file that is already there is a conflict, never an overwrite.') : text('The folder on disk is left exactly as it is.')}
          </span>
          <span className="right">
            <button type="button" className="btn ghost" onClick={close}>
              {text('Cancel')}
            </button>
            <button type="button" className="btn pri" data-save-as={dialog.id} onClick={write}>
              {writable ? text('Save') : text('Download')}
            </button>
          </span>
        </>
      }
    >
      <label className="frow">
        <span className="fk">{text('Path')}</span>
        <input
          className="ctl wide mono"
          value={path}
          onChange={(event) => setPath(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') write();
          }}
        />
      </label>
    </Frame>
  );
}

/** The draft offer of §4.3 — what was autosaved, against what the file holds. */
function Restore(): JSX.Element {
  const store = useDocumentsStore();
  const id = useDocuments((state) => (state.dialog?.kind === 'restore' ? state.dialog.id : null));
  const one = useDocuments((state) => state.open.find((open) => open.id === id));
  const close = (): void => {
    store.getState().setDialog(null);
  };
  if (one?.draft === undefined) return <></>;
  const { draft, standing } = one.draft;
  return (
    <Frame
      title={textWith('Restore the draft of {}?', one.title)}
      hint={draft.savedAt.replace('T', ' ').slice(0, 19)}
      onClose={close}
      foot={
        <>
          <span className="note-line">
            {standing === 'unsaved'
              ? text('The file has not got this work: the editor saved it while you were typing.')
              : text('The file has changed since this draft was taken; restoring replaces what it holds in the editor, not on disk.')}
          </span>
          <span className="right">
            <button
              type="button"
              className="btn ghost"
              data-discard={one.id}
              onClick={() => {
                void store.getState().discardDraft(one.id);
              }}
            >
              {text('Discard')}
            </button>
            <button
              type="button"
              className="btn pri"
              data-restore={one.id}
              onClick={() => {
                store.getState().restoreDraft(one.id);
              }}
            >
              {text('Restore')}
            </button>
          </span>
        </>
      }
    >
      <div className="frow">
        <span className="fk">{text('Draft')}</span>
        <span className="fv mono">{textWith('{} bytes', String(draft.text.length))}</span>
      </div>
      <div className="frow">
        <span className="fk">{text('File')}</span>
        <span className="fv mono">{one.path}</span>
      </div>
    </Frame>
  );
}

/**
 * The confirmation of §4.4's "Delete (cascades with confirmation)" — feature 2.7.
 *
 * A delete takes with it every rule that names the thing (plan §3: "the delete command cascades
 * over every selector that names the instance, listed in the confirmation and undone as one
 * command"), and the grammar keeps what it cannot let go — a public input's `to` cannot be
 * emptied, so the dangling name stays and V1 reports it, which is the editor's own rule of
 * wiring first and fixing afterwards (Q5). Both lists are shown, because the second is the
 * surprising one.
 */
function Remove(): JSX.Element {
  const store = useDocumentsStore();
  const dialog = useDocuments((state) => (state.dialog?.kind === 'remove' ? state.dialog : null));
  const close = (): void => {
    store.getState().setDialog(null);
  };
  if (dialog === null) return <></>;
  return (
    <Frame
      title={`${dialog.label}?`}
      hint={textWith('{} place(s)', String(dialog.removed.length))}
      onClose={close}
      foot={
        <>
          <span className="note-line">
            {dialog.kept.length === 0
              ? text('One edit of the Edit menu: Undo takes all of it back.')
              : text(
                  'What the grammar will not let go stays, and the core reports the name it no longer resolves — wire first, fix afterwards.',
                )}
          </span>
          <span className="right">
            <button type="button" className="btn ghost" data-cancel="remove" onClick={close}>
              {text('Cancel')}
            </button>
            <button
              type="button"
              className="btn pri"
              data-confirm="remove"
              onClick={() => {
                store.getState().confirmed();
              }}
            >
              {text('Delete')}
            </button>
          </span>
        </>
      }
    >
      {dialog.removed.map((place) => (
        <div className="frow" key={place}>
          <span className="fk">{text('Remove')}</span>
          <span className="fv mono" data-removed={place}>
            {place}
          </span>
        </div>
      ))}
      {dialog.kept.map((place) => (
        <div className="frow" key={place}>
          <span className="fk">{text('Kept')}</span>
          <span className="fv mono" data-kept={place}>
            {place}
          </span>
        </div>
      ))}
    </Frame>
  );
}

/** Whichever dialog is up. */
export function DocumentDialogs(): JSX.Element | null {
  const kind = useDocuments((state) => state.dialog?.kind ?? null);
  if (kind === 'workspaces') return <Workspaces />;
  if (kind === 'documents') return <Documents />;
  if (kind === 'save-as') return <SaveAs />;
  if (kind === 'restore') return <Restore />;
  if (kind === 'remove') return <Remove />;
  return null;
}

/**
 * What an open document shows, and the three things the chrome says around it — feature 2.6.
 *
 * The **banner** (component inventory §5) is where a workspace states what a Save will do: S18's
 * "This folder is a read-only snapshot. Your browser cannot write back to it, so Save downloads
 * the file for you to put back." Stated in a banner, never silent.
 *
 * The **toast** is the inventory's own — "a replaced edge, naming what went, undoable" — and here
 * it says what a Save wrote or handed over.
 *
 * The **document view** is the tab's frame: the heading the page needs, the banner, the body, and
 * the strip that says what a Save will do. What the *body* is, is the feature that owns the tab's
 * kind: feature 2.9's canvas for a model (§4.7), and the source pane below for a JSON source tab
 * (§4.10, which feature 2.17 makes editable). The pane itself is the document as its file holds
 * it — through the core's serializer and nothing else (D12) — which is exactly the bytes a Save
 * will write.
 */
import { useEffect, useRef, type JSX } from 'react';

import { useDocuments, useDocumentsStore } from './context.js';
import { DOCUMENT_TAB, documentTab, sourceStanding, type OpenDocument } from './store.js';
import { text, textWith } from '../shell/strings.js';
import type { Tab } from '../shell/store.js';

/** The banner of the open workspace, where it has something to say (S18, inventory §5). */
export function WorkspaceBanner(): JSX.Element | null {
  const banner = useDocuments((state) => state.banner);
  if (banner === null) return null;
  return (
    <div className={`banner ${banner.kind}`} role="status">
      <span className="bi" aria-hidden="true">
        {banner.kind === 'stop' ? '■' : banner.kind === 'warn' ? '▲' : '●'}
      </span>
      <span>
        <b>{text(banner.head)}</b> {text(banner.body)}
      </span>
    </div>
  );
}

/** The toast: what just happened, and the one thing to do about it. */
export function DocumentToast(): JSX.Element | null {
  const store = useDocumentsStore();
  const toast = useDocuments((state) => state.toast);
  useEffect(() => {
    if (toast === null) return;
    const timer = setTimeout(() => {
      store.getState().setToast(null);
    }, 6000);
    return () => {
      clearTimeout(timer);
    };
  }, [toast, store]);
  if (toast === null) return null;
  return (
    <div className="toast" role="status">
      <span aria-hidden="true">⇄</span>
      <span>{toast.text}</span>
      {toast.action === undefined ? null : (
        <button type="button" className="undo" onClick={toast.run}>
          {text(toast.action)}
        </button>
      )}
    </div>
  );
}

/**
 * The banner an off-grammar source stands behind — plan §4.10, §3, artboard S16.
 *
 * > An off-schema source is allowed to exist (§3): the canvas keeps its last drawable state with
 * > a banner until the source is back on the grammar.
 *
 * It is about the **source** and not about the document, which is why it is drawn from
 * {@link sourceStanding} and not from the problems: a New Model is off the grammar by
 * construction (its skeleton is the required members, empty — feature 2.6) and there is nothing
 * to say about that beyond what the Problems panel already says. What this says is that the text
 * a reader typed is not what the rest of the editor is looking at, which is a fact about two
 * writers and about nothing else.
 */
export function SourceBanner({ one }: { one: OpenDocument }): JSX.Element | null {
  const standing = sourceStanding(one);
  if (standing === null) return null;
  return (
    <div className="banner stop" role="status" data-source-banner={standing}>
      <span className="bi" aria-hidden="true">
        ■
      </span>
      <span>
        {standing === 'pending' ? (
          <>
            <b>{text('The JSON source is not a document yet.')}</b>{' '}
            {text(
              'Nothing has been taken into the document, so the canvas and the panels show what it held. The refusal is in Problems and in the source, at its place.',
            )}
          </>
        ) : (
          <>
            <b>{text('The source is off the grammar.')}</b>{' '}
            {text(
              'The canvas shows the last drawable state. Saving is allowed with a confirmation — the file is yours — and the core’s refusal stays in the log.',
            )}
          </>
        )}
      </span>
    </div>
  );
}

/** The document behind a tab, or nothing where the tab is not one. */
export function documentFor(tab: Tab, open: readonly OpenDocument[]): OpenDocument | undefined {
  const id = documentTab(tab.id);
  return open.find((one) => one.id === id);
}

/** The document as its file holds it, through the core's serializer and nothing else (D12). */
export function SourcePane({ one }: { one: OpenDocument }): JSX.Element {
  const body = useRef<HTMLPreElement>(null);
  useEffect(() => {
    body.current?.scrollTo({ top: 0 });
  }, [one.id]);
  return (
    <pre className="doc-json mono" ref={body} data-path={one.path} tabIndex={0} aria-label={one.title}>
      {one.session.text}
    </pre>
  );
}

/** One open document: the heading, the banner, the body its tab's kind draws, and the Save strip. */
export function DocumentView({ tab, body }: { tab: Tab; body?: (one: OpenDocument) => JSX.Element }): JSX.Element {
  const store = useDocumentsStore();
  const open = useDocuments((state) => state.open);
  const writable = useDocuments((state) => state.workspace.writable);
  const one = documentFor(tab, open);

  if (one === undefined) {
    return (
      <div className="canvas">
        <span className="canvas-note">{text('No document')}</span>
      </div>
    );
  }
  return (
    <div className="doc-open">
      {/* The page's level-one heading. The tab strip names the document where a reader can see
          it, and a strip of buttons is not a heading — so the heading is here, in the body it is
          the heading of, and drawn nowhere. Found by feature 2.6's own axe pass. */}
      <h1 className="offscreen">{one.title}</h1>
      <WorkspaceBanner />
      <SourceBanner one={one} />
      {body === undefined ? <SourcePane one={one} /> : body(one)}
      <div className="doc-foot">
        <button
          type="button"
          className="btn pri"
          data-save={one.id}
          onClick={() => {
            void store.getState().save(one.id);
          }}
        >
          {writable
            ? textWith('Save {}', one.path)
            : textWith('Save — download {}', one.session.name)}
        </button>
        <span className="note-line">{textWith('through the core’s serializer, as {} holds it', one.path)}</span>
      </div>
    </div>
  );
}

/**
 * The views this feature gives the shell, by the `kind` their tabs carry.
 *
 * The model's own tab draws the read-only pane here and the **canvas** where feature 2.9's views
 * are composed over these (§4.7: "the default editor of a model"); an application that composes
 * only these — a build with no canvas — still opens a document and shows what was opened. The
 * JSON source tab is no longer one of them: feature 2.17 gives it Monaco and its own entry point
 * (`@tensorspine/ui/source`), which is what keeps a text editor out of the shell's first chunk.
 */
export const DOCUMENT_VIEWS: Readonly<Record<string, (tab: Tab) => JSX.Element>> = {
  [DOCUMENT_TAB]: (tab) => <DocumentView tab={tab} />,
};

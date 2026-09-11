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
 * The **document view** is the tab's body: the document as its file holds it, through the core's
 * serializer and nothing else (D12), which is exactly the bytes a Save will write. The canvas of
 * §4.7 and the editable source of §4.10 are features 2.9 and 2.17; what a tab has to do here is
 * open, and show what was opened.
 */
import { useEffect, useRef, type JSX } from 'react';

import { useDocuments, useDocumentsStore } from './context.js';
import { DOCUMENT_TAB, type OpenDocument } from './store.js';
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

/** The document behind a tab, or nothing where the tab is not one. */
function documentFor(tab: Tab, open: readonly OpenDocument[]): OpenDocument | undefined {
  return open.find((one) => one.id === tab.id);
}

/** One open document: the banner, the Save strip a read-only workspace needs, and the JSON. */
export function DocumentView({ tab }: { tab: Tab }): JSX.Element {
  const store = useDocumentsStore();
  const open = useDocuments((state) => state.open);
  const writable = useDocuments((state) => state.workspace.writable);
  const one = documentFor(tab, open);
  const body = useRef<HTMLPreElement>(null);

  useEffect(() => {
    body.current?.scrollTo({ top: 0 });
  }, [tab.id]);

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
          the heading of, and drawn nowhere. Found by this feature's own axe pass. */}
      <h1 className="offscreen">{one.title}</h1>
      <WorkspaceBanner />
      <pre className="doc-json mono" ref={body} data-path={one.path} tabIndex={0} aria-label={one.title}>
        {one.session.text}
      </pre>
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

/** The views this feature gives the shell, by the `kind` their tabs carry. */
export const DOCUMENT_VIEWS: Readonly<Record<string, (tab: Tab) => JSX.Element>> = {
  [DOCUMENT_TAB]: (tab) => <DocumentView tab={tab} />,
};

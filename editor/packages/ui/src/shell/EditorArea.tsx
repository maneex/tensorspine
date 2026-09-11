/**
 * The editor area — S1's `.ed`: the tab strip and what the selected tab draws.
 *
 * §4.2: "tabs: one per open model, one per drill-in (composition, template instance), one per
 * JSON source view, one per unit". None of those exists yet, and the strip is built so that none
 * of them has to change it: a tab carries a `kind`, and the shell is given a view per kind. Its
 * own two — the Help menu's Keyboard Shortcuts and About — are registered here, which is what
 * lets the strip be opened, switched and closed today instead of being taken on trust.
 *
 * With no tab open it is S17's empty state on the canvas ground: what the editor shows before it
 * has anything to show, with the three commands §4.3 names as the ways out of it and the recent
 * folders beneath them.
 *
 * **Why the strip is not a `tablist`.** An editor tab carries a close button, and a close button
 * inside a `tab` is a focusable element inside a widget — which is a serious accessibility
 * defect (a control assistive technology cannot reach separately), and taking the close button
 * out to satisfy the role would take away the one visible way to close a tab. So the strip is a
 * set of buttons that say which one is current: the same keyboard, the same reading, and the
 * close button beside its tab where it belongs. The panel strips, which have nothing beside their
 * tabs, are proper tab lists.
 *
 * The column's own rule is the component inventory's, and it is load-bearing: `.ed` must be
 * `flex:1; min-height:0; display:flex; flex-direction:column`, "without it the canvas and panel
 * never size themselves".
 */
import { useEffect, useState, type JSX, type ReactNode } from 'react';

import type { RecentWorkspace } from '@tensorspine/store/platform';

import { useShell, useShellStore, usePlatform } from './context.js';
import { text } from './strings.js';
import type { Tab } from './store.js';

/** What draws a tab's body, by the `kind` the tab carries. */
export type TabViews = Readonly<Record<string, (tab: Tab) => ReactNode>>;

/** The empty state of S17: no workspace, no document, and the ways out of both. */
function Nothing(): JSX.Element {
  const store = useShellStore();
  const platform = usePlatform();
  const workspace = useShell((state) => state.workspace);
  const [recent, setRecent] = useState<readonly RecentWorkspace[]>([]);

  useEffect(() => {
    let live = true;
    void platform.workspaces.recent().then((found) => {
      if (live) setRecent(found);
    });
    return () => {
      live = false;
    };
  }, [platform, workspace]);

  const run = (id: string) => () => {
    store.getState().run(id);
  };

  return (
    <div className="flow">
      <div className="nothing">
        <h1>{text('No workspace is open.')}</h1>
        <p className="note-line">
          {text(
            'A workspace is a folder laid out like the repository’s data directory: models anywhere below it, library bases resolved from each document’s primitive libraries, the schemas from the repository the editor was built with.',
          )}
        </p>
        <div className="actions">
          <button type="button" className="btn pri" onClick={run('file.open-folder')}>
            {text('Open Folder…')}
          </button>
          <button type="button" className="btn" onClick={run('file.new-model')}>
            {text('New Model')}
          </button>
          <button type="button" className="btn" onClick={run('file.new-base')}>
            {text('New Base…')}
          </button>
        </div>
        <div className="recent">
          <span className="note-line">{text('Recent')}</span>
          <span className="note-line">
            {/* The examples the build carries are always available (§4.3) and are the one
                workspace that needs no folder, no permission and no upload — so they are offered
                beside the folders that have been open, rather than as a fourth button S17 does
                not draw. */}
            <button type="button" className="link" data-open="examples" onClick={run('file.open-recent')}>
              {text('Examples')}
            </button>
            {recent.map((one) => (
              <button key={one.id} type="button" className="link" data-open={one.id} onClick={run('file.open-recent')}>
                <code>{one.name}</code>
              </button>
            ))}
          </span>
        </div>
      </div>
    </div>
  );
}

/** One tab: the button that selects it, and the button that closes it. */
function TabSlot({ tab, current }: { tab: Tab; current: boolean }): JSX.Element {
  const store = useShellStore();
  return (
    <span className={current ? 'tab-slot on' : 'tab-slot'}>
      <button
        type="button"
        className="tab"
        data-tab={tab.id}
        aria-current={current}
        onClick={() => {
          store.getState().selectTab(tab.id);
        }}
        onAuxClick={(event) => {
          // The middle click every editor closes a tab with — an accelerator, beside the button.
          if (event.button !== 1) return;
          event.preventDefault();
          store.getState().closeTab(tab.id);
        }}
      >
        {tab.title}
        {tab.badge === undefined ? null : <span className="bdg">{tab.badge}</span>}
        {tab.dirty === true ? <span className="dot" role="img" aria-label={text('Unsaved')} /> : null}
      </button>
      <button
        type="button"
        className="shut"
        data-close={tab.id}
        aria-label={`${text('Close Tab')}: ${tab.title}`}
        onClick={() => {
          store.getState().closeTab(tab.id);
        }}
      >
        ×
      </button>
    </span>
  );
}

/** The tab strip and the body below it. */
export function EditorArea({ views, below }: { views: TabViews; below?: ReactNode }): JSX.Element {
  const tabs = useShell((state) => state.tabs);
  const activeTab = useShell((state) => state.activeTab);
  const zoom = useShell((state) => state.zoom);
  const shown = tabs.find((tab) => tab.id === activeTab);
  const view = shown === undefined ? undefined : views[shown.kind];

  return (
    <main className="ed">
      {tabs.length === 0 ? null : (
        // No strip where there is nothing to strip: S17's empty state has none, and an empty
        // band of chrome would be the first thing a new workspace shows.
        <nav className="tabs" aria-label={text('Open tabs')}>
          {tabs.map((tab) => (
            <TabSlot key={tab.id} tab={tab} current={tab.id === activeTab} />
          ))}
        </nav>
      )}
      <div className="ed-body" id="editor-body">
        {view === undefined || shown === undefined ? (
          <div className="canvas">
            <span className="canvas-note">{text('No document')}</span>
            <span className="canvas-note right fig">{`${String(Math.round(zoom * 100))}%`}</span>
            <Nothing />
          </div>
        ) : (
          view(shown)
        )}
      </div>
      {below}
    </main>
  );
}

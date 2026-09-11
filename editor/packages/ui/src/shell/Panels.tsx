/**
 * The two panel regions — S1's `.panel` (bottom) and `.insp` (right), §4.2's 200 px and 340 px.
 *
 * > Every panel is a tab that can be dragged to another region (Properties to the bottom, Derived
 * > to the right) — the VS Code convention; layouts are saved in settings.
 *
 * So one component draws both: a strip of tabs and a body, the tabs draggable and the region a
 * drop target. The bottom region opens with the component inventory's **exactly three** —
 * Problems, Derived, Log — and Properties opens on the right; there is no fourth panel to invent,
 * the design pass's proposal of one having been rejected (inventory §6).
 *
 * **The drag is not the only way.** A pointer gesture that has no other form is a gesture some
 * people cannot make, so each region carries a button that sends its showing panel to the other
 * one — the same edit, visible, and reachable from the keyboard (§4.21).
 *
 * **What the panels show today.** Problems and Derived are the core's, and the core has nothing
 * to say until a document is open (2.6, 2.8, 2.15); Properties is the selection's, and feature
 * 2.7 gave it the row every sheet of §4.11 starts with. Each says so in the design's own
 * empty-state voice. The Log is this feature's own and is full from
 * the first frame: what the platform is, what the workspace is, and every command chosen that
 * nobody has wired yet.
 */
import type { DragEvent, JSX } from 'react';

import { useDocumentState } from '../documents/Pills.js';
import { SelectionSheet } from '../explorer/Sheet.js';
import { useShell, useShellStore } from './context.js';
import { BOUNDS, PANELS, panelById, type PanelId, type RegionId } from './regions.js';
import { Splitter } from './Splitter.js';
import { text, textWith } from './strings.js';

/**
 * What a dragged tab carries.
 *
 * A type of the editor's own, not `text/plain`: a drop of a panel is not a drop of text, and the
 * region has to be able to tell them apart in `dragover`, where a browser lets a page read the
 * *types* on offer and nothing else.
 */
export const PANEL_TRANSFER = 'application/x-tensorspine-panel';

/**
 * What a region is called.
 *
 * By its place and not by what it holds: §4.2 calls the right-hand one "Properties" because
 * Properties is what opens there, but a panel can be dragged across, and a region labelled after
 * a panel that has left it would be a name that lies.
 */
function regionLabel(region: RegionId): string {
  return region === 'region.bottom' ? text('Bottom panel') : text('Right panel');
}

/** The other region — where a panel goes when it is sent away from this one. */
function otherThan(region: RegionId): RegionId {
  return region === 'region.bottom' ? 'region.right' : 'region.bottom';
}

/** What the button that sends a panel across says. */
function sendLabel(to: RegionId): string {
  return to === 'region.right' ? text('Move to the right') : text('Move to the bottom');
}

/** The body of one panel. */
function PanelBody({ panel }: { panel: PanelId }): JSX.Element {
  const log = useShell((state) => state.log);
  if (panel === 'panel.log') {
    return (
      <>
        {log.map((line) => (
          <div className="logline" key={`${line.at}-${line.text}`}>
            <time dateTime={line.at}>{line.at.slice(11, 19)}</time>
            <span>{line.text}</span>
          </div>
        ))}
      </>
    );
  }
  if (panel === 'panel.properties') {
    // §4.11's sheet of the current selection. The sections each kind of selection gets are the
    // features that can answer them (2.10 onwards); the row every one of them starts with — the
    // name, edited here — is feature 2.7's, because §4.4 asks that a name be editable in the
    // sheet as well as by clicking the thing.
    return <SelectionSheet />;
  }
  // With a document open, the panel says what the core has said about it and nothing more: the
  // rows of §4.17 are feature 2.8's and the six products of §4.18 are feature 2.15's, and a panel
  // that went on saying "no document is open" over an open one would be saying something false.
  return <PanelState panel={panel} />;
}

/** What a panel with nothing of its own to draw yet says, truthfully. */
function PanelState({ panel }: { panel: PanelId }): JSX.Element {
  const summary = useDocumentState();
  const problems = panel === 'panel.problems';
  if (summary === null) {
    return (
      <>
        <div className="empty-line">
          {problems
            ? text('Nothing to check — no document is open.')
            : text('Nothing to show — no document is open.')}
        </div>
        <p className="empty-sub">
          {text('Open a folder or a model, and the core validates and derives it as you edit.')}
        </p>
      </>
    );
  }
  return (
    <>
      <div className="empty-line">{problems ? summary.problems : summary.derivation}</div>
      <p className="empty-sub">{summary.where}</p>
    </>
  );
}

/** One region: its tabs, its body, and the drop it accepts. */
export function PanelRegion({ region }: { region: RegionId }): JSX.Element | null {
  const store = useShellStore();
  const state = useShell((one) => one.regions[region]);
  const dragging = useShell((one) => one.draggingPanel);
  if (!state.open || state.panels.length === 0) return null;

  const onDrop = (event: DragEvent<HTMLElement>): void => {
    // Read the transfer here, in the handler, before anything is awaited — a `DataTransfer` is
    // disabled the moment the synchronous part returns (feature 2.4's finding, one level up).
    const panel = event.dataTransfer.getData(PANEL_TRANSFER);
    event.preventDefault();
    store.getState().setDraggingPanel(null);
    if (panel === '') return;
    if (!PANELS.some((one) => one.id === panel)) return;
    store.getState().dockPanel(panel as PanelId, region);
  };

  const accepts = dragging !== null && !state.panels.includes(dragging);
  const active = state.active;

  return (
    <section
      className={`${region === 'region.bottom' ? 'panel' : 'insp'}${accepts ? ' dropping' : ''}`}
      style={region === 'region.bottom' ? { height: state.size } : { width: state.size }}
      data-region={region}
      data-size={state.size}
      aria-label={regionLabel(region)}
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes(PANEL_TRANSFER)) return;
        event.preventDefault();
      }}
      onDrop={onDrop}
    >
      <div className="panel-tabs">
        <div className="ptabs" role="tablist" aria-label={text('Panels')}>
          {state.panels.map((id) => {
            const panel = panelById(id);
            if (panel === undefined) return null;
            return (
              <button
                key={id}
                type="button"
                role="tab"
                className="ptab"
                data-panel={id}
                id={`ptab-${id}`}
                aria-controls={`pbody-${region}`}
                aria-selected={id === active}
                tabIndex={id === active ? 0 : -1}
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.setData(PANEL_TRANSFER, id);
                  event.dataTransfer.effectAllowed = 'move';
                  store.getState().setDraggingPanel(id);
                }}
                onDragEnd={() => {
                  store.getState().setDraggingPanel(null);
                }}
                onClick={() => {
                  store.getState().selectPanel(region, id);
                }}
              >
                {text(panel.label)}
              </button>
            );
          })}
        </div>
        <span className="panel-right">
          {active === null ? null : (
            <button
              type="button"
              className="tbtn"
              data-send={active}
              // The visible text says where it goes; the name says what goes, because a button
              // read out of its context has to say both.
              aria-label={`${sendLabel(otherThan(region))}: ${text(panelById(active)?.label ?? '')}`}
              title={textWith('Move {} to the other region.', text(panelById(active)?.label ?? ''))}
              onClick={() => {
                store.getState().dockPanel(active, otherThan(region));
              }}
            >
              {sendLabel(otherThan(region))}
            </button>
          )}
        </span>
      </div>
      <div
        className="panel-body"
        id={`pbody-${region}`}
        role="tabpanel"
        tabIndex={0}
        aria-labelledby={active === null ? undefined : `ptab-${active}`}
      >
        {active === null ? null : <PanelBody panel={active} />}
      </div>
      <Splitter
        orientation={region === 'region.bottom' ? 'horizontal' : 'vertical'}
        at={region === 'region.bottom' ? 'top' : 'left'}
        size={state.size}
        min={BOUNDS[region].min}
        max={BOUNDS[region].max}
        grows={-1}
        label={regionLabel(region)}
        onSize={(size) => {
          store.getState().setRegionSize(region, size);
        }}
      />
    </section>
  );
}

/** How far each region may be dragged — the splitters read it. */
export const REGION_BOUNDS = BOUNDS;

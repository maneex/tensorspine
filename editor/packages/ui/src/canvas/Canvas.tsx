/**
 * The folded canvas — plan §4.7, artboards S1, S2 and S3.
 *
 * Everything drawn is answered elsewhere: `model.ts` says what the boxes and the wires are,
 * `layout.ts` says where they go, `gestures.ts` says what each gesture writes, `Box.tsx` draws one
 * box. What this file owns is the **pointer and the keyboard**: the connection drag, the palette
 * drop, the move, the selection, the pan, the zoom and the context menu.
 *
 * **Q5 is the spine, and it shows in three places.** A connection asks the core's `check` while it
 * is in flight and shows the answer at the pointer; the drop is made *whatever it said*; an input
 * already fed has its older edge replaced and the toast names it with an Undo. Nothing on this
 * canvas is disabled, greyed or refused for a semantic reason — the verdict lands in Problems.
 *
 * **A connection is one gesture with two shapes.** Press an output handle and release over an
 * input handle: the drag. Click an output handle and then an input handle: the same connection,
 * made without holding anything down — which is the keyboard's form of it, and what §4.4's rule
 * that a shortcut is never the only way asks for. The two are one state machine, so there is one
 * place where a connection is decided.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
  type JSX,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';

import {
  foldedGraph,
  formatSemanticProblem,
  literalIndex,
  rootSite,
  type FoldedGraph,
  type FoldedHandle,
  type PortRef,
  type Verdict as CandidateVerdict,
} from '@tensorspine/lang';
import { pointerOf, type Path } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import type { OpenDocument } from '../documents/store.js';
import { presentation } from '../presentation/index.js';
import { useShell, useShellStore } from '../shell/context.js';
import { text, textWith } from '../shell/strings.js';

import { Box } from './Box.js';
import {
  addInstance,
  connectHandles,
  duplicateAt,
  removeAt,
  renameAt,
  type GestureContext,
} from './gestures.js';
import { fit, NO_PLACEMENT, place, routes, type Placement, type WireRoute } from './layout.js';
import { entriesFor, type MenuEntry } from './menu.js';
import { canvasModel, ROLE, type CanvasBox, type CanvasModel, type CanvasWire } from './model.js';

/** What a palette drag carries onto the canvas — §4.7's "Drop a primitive from the palette". */
export const PRIMITIVE_TRANSFER = 'application/x-tensorspine-primitive';

/** The primitive a drag carries: what to instantiate, and at which version. */
export interface PrimitiveTransfer {
  readonly primitive: string;
  readonly version: string;
}

/** What a drag put on the clipboard, or `null` where it carried something else. */
export function primitiveOf(transfer: DataTransfer): PrimitiveTransfer | null {
  const held = transfer.getData(PRIMITIVE_TRANSFER);
  if (held === '') return null;
  try {
    const parsed: unknown = JSON.parse(held);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const one = parsed as PrimitiveTransfer;
    return typeof one.primitive === 'string' && typeof one.version === 'string' ? one : null;
  } catch {
    return null;
  }
}

/** A connection in flight: where it started, where the pointer is, and what the core said. */
interface Flight {
  readonly from: FoldedHandle;
  /** The `data-port` the flight left, which is what `check` is asked about. */
  readonly port: string;
  /** The box the flight left, so the wire can be drawn from it. */
  readonly box: string;
  readonly x: number;
  readonly y: number;
  /** Whether the pointer is still down: a press-drag-release rather than a click-then-click. */
  readonly pressed: boolean;
  /** The `data-port` of the handle the pointer is over. */
  readonly over: string | null;
  readonly verdict: CandidateVerdict | null;
}

/** A box being moved: which, and where it is now. */
interface Move {
  readonly pointer: string;
  readonly path: Path;
  readonly dx: number;
  readonly dy: number;
  readonly x: number;
  readonly y: number;
}

/** Where the canvas is looking. */
interface Viewport {
  readonly x: number;
  readonly y: number;
  readonly zoom: number;
}

/** The folded graph of a document, rebuilt when the document moves and not before. */
function useFolded(one: OpenDocument): FoldedGraph {
  const revision = one.session.store.revision;
  const id = one.id;
  const tree = one.session.store.tree;
  return useMemo(() => foldedGraph(tree), [id, revision]);
}

/** The whole canvas of one open document. */
export function Canvas({ one }: { one: OpenDocument }): JSX.Element {
  const store = useDocumentsStore();
  const shell = useShellStore();
  const registry = useDocuments((state) => state.registry);
  const templates = useDocuments((state) => state.library.templates);
  const toggles = useShell((state) => state.canvas);
  const zoom = useShell((state) => state.zoom);
  const shapes = one.session.store.shapes;
  const folded = useFolded(one);
  const surface = useRef<HTMLDivElement>(null);
  const elements = useRef(new Map<string, HTMLElement>());
  const [placement, setPlacement] = useState<Placement>(NO_PLACEMENT);
  const [measured, setMeasured] = useState<ReadonlyMap<string, { width: number; height: number }>>(
    new Map(),
  );
  const [viewport, setViewport] = useState<Viewport>({ x: 0, y: 0, zoom: 1 });
  const [flight, setFlight] = useState<Flight | null>(null);
  const [move, setMove] = useState<Move | null>(null);
  const [pan, setPan] = useState<{ x: number; y: number } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [menu, setMenu] = useState<{ box: CanvasBox; x: number; y: number } | null>(null);
  const [announced, setAnnounced] = useState('');

  const bindings = presentation();
  const expanded = useMemo(() => new Set(one.expanded), [one.expanded]);
  const problems = useMemo(
    () => [
      ...(one.reading.structural ?? []),
      ...(one.reading.verdict?.problems ?? []),
      ...one.reading.notices,
    ],
    [one.reading],
  );

  const model: CanvasModel | null = useMemo(() => {
    if (registry === null) return null;
    return canvasModel({
      folded,
      facts: one.reading.facts,
      derived: one.reading.derived,
      stale: one.reading.derivedAt !== one.reading.revision,
      problems,
      collapsed: collapsedOf(folded, expanded),
      view: toggles,
      registry,
      shapes,
      bindings,
      role: one.session.store.role,
      templates,
    });
  }, [folded, registry, one.reading, problems, expanded, toggles, shapes, bindings, one.session.store.role, templates]);

  // The layout runs when the *shape* of the drawing changes, and never while a pointer is down
  // (feature 0.4's own reservation about what a layout costs).
  const shape =
    model === null
      ? ''
      : model.boxes.map((box) => `${box.pointer}:${box.width}x${box.height}`).join('|');
  const positions = one.session.layout.layout.positions;
  useEffect(() => {
    if (model === null) return;
    let live = true;
    void place({ model, positions, measured }).then((answer) => {
      if (live) setPlacement(answer);
    });
    return () => {
      live = false;
    };
  }, [shape, measured, positions]);

  // What the browser makes of a card, fed back so the next layout uses the real sizes — feature
  // 0.4: "the real canvas measures the rendered card and lays out again when a card's size
  // changes; a spike cannot, and does not pretend to."
  useEffect(() => {
    if (model === null) return;
    const found = new Map<string, { width: number; height: number }>();
    let moved = false;
    for (const box of model.boxes) {
      const element = elements.current.get(box.pointer);
      if (element === undefined) continue;
      const width = element.offsetWidth;
      const height = element.offsetHeight;
      if (width <= 0 || height <= 0) continue;
      found.set(box.pointer, { width, height });
      const before = measured.get(box.pointer);
      if (before === undefined || before.width !== width || before.height !== height) moved = true;
    }
    if (moved) setMeasured(found);
  });

  const fitNow = useCallback(() => {
    const element = surface.current;
    if (element === null) return;
    const answer = fit(placement, { width: element.clientWidth, height: element.clientHeight });
    setViewport(answer);
    shell.getState().setZoom(answer.zoom);
  }, [placement, shell]);

  const extent = `${String(placement.width)}x${String(placement.height)}`;
  useEffect(() => {
    fitNow();
  }, [one.id, extent]);

  useEffect(() => {
    setViewport((before) => (before.zoom === zoom ? before : { ...before, zoom }));
  }, [zoom]);

  // §4.4's Auto-layout, Reset Layout and Zoom to Fit are the canvas's to perform, so the canvas
  // binds them — the seam feature 2.5 built for exactly this.
  useEffect(() => {
    shell.getState().bind({
      'view.auto-layout': () => {
        store.getState().resetLayout(one.id);
        setMeasured(new Map());
      },
      'view.reset-layout': () => {
        store.getState().resetLayout(one.id);
      },
      'view.zoom-fit': () => {
        fitNow();
      },
    });
  }, [shell, store, one.id, fitNow]);

  const gestures: GestureContext = { shapes, bindings, role: one.session.store.role };
  const selection = one.selection === undefined ? null : pointerOf(one.selection);
  const announce = (line: string): void => {
    setAnnounced(line);
  };

  const select = (box: CanvasBox): void => {
    store.getState().selectPlace(box.path, one.id);
  };

  /** The connection, made whatever the verdict said (Q5). */
  const connect = (from: FoldedHandle, to: FoldedHandle): void => {
    const applied = store.getState().edit((edit) => connectHandles(edit, from, to), one.id);
    if (applied === null) return;
    if (applied.replaced !== undefined) {
      const gone = pointerOf(applied.replaced);
      store.getState().setToast({
        text: textWith('{} replaced the edge that fed that input.', `${applied.label} —`) + ` ${gone}`,
        action: text('Undo'),
        run: () => {
          one.session.store.undo();
          store.getState().setToast(null);
        },
      });
    }
    announce(applied.label);
  };

  /**
   * Press or click on a port handle: the one place a connection is decided.
   *
   * The two shapes of the gesture meet here. A **press** on an output starts the flight and a
   * release over an input finishes it (the canvas's own `onPointerUp`); a **click** on an output
   * starts it and a click on an input finishes it, with nothing held down — which is the
   * keyboard's form and what §4.4's rule about accelerators asks for. The click that follows the
   * press that started a flight is the same gesture, so it is ignored rather than cancelling it.
   */
  const onPort = (box: CanvasBox, port: string, side: 'inputs' | 'outputs', pressed: boolean): void => {
    const handle = handleFor(folded, box, port);
    if (handle === null) return;
    const attribute = `${box.pointer}:${port}`;
    if (flight === null) {
      // A connection runs from a producer to a consumer, and a drag that starts on an input is
      // §4.7's "drag from an input handle to empty space" — the library picker of feature 3.1.
      if (side !== 'outputs') return;
      const at = placement.boxes.get(box.pointer);
      setFlight({
        from: handle,
        port: attribute,
        box: box.pointer,
        x: at === undefined ? 0 : at.x + at.width / 2,
        y: at === undefined ? 0 : at.y + at.height,
        pressed,
        over: null,
        verdict: null,
      });
      announce(textWith('Connecting from {}', `${handle.name}.${handle.port}`));
      return;
    }
    // The click that follows the press that armed this flight: the same gesture, not a second one.
    if (flight.port === attribute) return;
    if (side !== 'inputs') return;
    const from = flight.from;
    setFlight(null);
    connect(from, handle);
  };

  /** What the core says about the candidate the pointer is over — shown *during* the drag (§4.7). */
  const ask = useCallback(
    (from: string, over: string | null): void => {
      if (over === null) return;
      const source = portRef(folded, from);
      const target = portRef(folded, over);
      if (source === null || target === null) return;
      void store
        .getState()
        .checkEdge(source, target, one.id)
        .then((verdict: CandidateVerdict | null) => {
          setFlight((before) =>
            before === null || before.over !== over ? before : { ...before, verdict },
          );
        });
    },
    [folded, store, one.id],
  );

  const onDrop = (event: ReactDragEvent<HTMLDivElement>): void => {
    const carried = primitiveOf(event.dataTransfer);
    if (carried === null) return;
    event.preventDefault();
    const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
    let added: string | null = null;
    const applied = store.getState().edit((edit) => {
      const command = addInstance(edit, { primitive: carried.primitive, version: carried.version });
      added = command.name;
      return command;
    }, one.id);
    if (applied === null || added === null) return;
    const path = ['instances', added] as Path;
    // Where it was dropped is where it goes: a manual place, which is D6's own override.
    store.getState().moveBox(path, { x: Math.round(at.x), y: Math.round(at.y) }, one.id);
    store.getState().selectPlace(path, one.id);
    // §4.7: the sheet opens on the new instance, so the author fills in what V2 is about to ask
    // for. The Properties region is where §4.11's sheet is.
    shell.getState().revealPanel('panel.properties');
    announce(applied.label);
  };

  const run = (entry: MenuEntry, box: CanvasBox): void => {
    setMenu(null);
    select(box);
    if (entry.id === 'canvas.open-sheet') {
      shell.getState().revealPanel('panel.properties');
      return;
    }
    if (entry.id === 'canvas.show-in-explorer') {
      store.getState().revealPlace(box.path, one.id);
      shell.getState().setActivity('activity.explorer');
      return;
    }
    if (entry.id === 'canvas.copy-identifier') {
      const identifier = box.where ?? box.name;
      announce(textWith('Copied {}', identifier));
      void navigator.clipboard?.writeText(identifier).catch(() => undefined);
      return;
    }
    if (entry.id === 'edit.rename') {
      setRenaming(box.pointer);
      return;
    }
    if (entry.id === 'edit.duplicate') {
      const applied = store.getState().edit((edit) => duplicateAt(gestures, edit, box.path), one.id);
      if (applied !== null) announce(applied.label);
      return;
    }
    if (entry.id === 'edit.delete') {
      store.getState().offer((edit) => removeAt(gestures, edit, box.path), one.id);
      return;
    }
    shell.getState().run(entry.id);
  };

  /**
   * §4.4's canvas shortcuts, the ones this feature owns.
   *
   * > Enter opens the sheet of the selection; Escape clears; arrows nudge; Tab cycles the
   * > selection.
   *
   * **The arrows nudge**, which is the keyboard's form of the manual move — a box can be placed
   * without a pointer, and the override it writes is the same override a drag writes (D6). **Tab
   * is the browser's**: every control of every box is a real button, so the tab order already
   * walks the boxes, in the document's own order rather than the topological one §4.4 names (a
   * finding, recorded). An arrow with nothing selected selects the first box, which is the way in.
   */
  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>): void => {
    if (model === null) return;
    if (event.key === 'Escape') {
      if (flight !== null) setFlight(null);
      else if (menu !== null) setMenu(null);
      else store.getState().selectPlace(null, one.id);
      return;
    }
    // Enter and the arrows belong to whatever has the focus while a control has it: Enter on the
    // name button is what opens the rename, and a `preventDefault` here would suppress the click
    // a button synthesises from it. Escape is the exception, because cancelling a drag has to work
    // wherever the focus happens to be.
    if ((event.target as HTMLElement).closest('button, input') !== null) return;
    const chosen = model.boxes.find((box) => box.pointer === (selection ?? ''));
    if (event.key === 'Enter' && chosen !== undefined) {
      event.preventDefault();
      shell.getState().revealPanel('panel.properties');
      return;
    }
    if (!NUDGES.includes(event.key)) return;
    event.preventDefault();
    if (chosen === undefined) {
      const first = model.boxes.find((box) => placement.boxes.has(box.pointer));
      if (first !== undefined) select(first);
      return;
    }
    const at = placement.boxes.get(chosen.pointer);
    if (at === undefined) return;
    const step = event.shiftKey ? 1 : 8;
    const dx = (event.key === 'ArrowRight' ? step : 0) - (event.key === 'ArrowLeft' ? step : 0);
    const dy = (event.key === 'ArrowDown' ? step : 0) - (event.key === 'ArrowUp' ? step : 0);
    store.getState().moveBox(chosen.path, { x: Math.round(at.x + dx), y: Math.round(at.y + dy) }, one.id);
  };

  const wires = useMemo(() => (model === null ? [] : routes(model, placement)), [model, placement]);

  if (model === null) {
    return (
      <div className="canvas graph">
        <span className="canvas-note">{text('The schemas are not loaded.')}</span>
      </div>
    );
  }

  const drawn = model.boxes.filter((box) => placement.boxes.has(box.pointer));

  return (
    <div
      className={`canvas graph${pan !== null ? ' panning' : ''}`}
      ref={surface}
      data-path={one.path}
      tabIndex={0}
      role="group"
      aria-label={textWith('The folded document of {}', one.title)}
      onKeyDown={onKeyDown}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes(PRIMITIVE_TRANSFER)) event.preventDefault();
      }}
      onDrop={onDrop}
      onPointerDown={(event) => {
        const target = event.target as HTMLElement;
        // A press inside a box belongs to the box, and one inside the context menu to the menu:
        // closing it here would close it before the entry's own click could land on it.
        if (target.closest('[data-box]') !== null || target.closest('.ctxmenu') !== null) return;
        setMenu(null);
        if (event.button !== 0 && event.button !== 1) return;
        setPan({ x: event.clientX - viewport.x, y: event.clientY - viewport.y });
      }}
      onPointerMove={(event) => {
        if (pan !== null) {
          setViewport((before) => ({ ...before, x: event.clientX - pan.x, y: event.clientY - pan.y }));
          return;
        }
        if (move !== null) {
          const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
          setMove({ ...move, x: at.x - move.dx, y: at.y - move.dy });
          return;
        }
        if (flight === null) return;
        const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
        const over = portUnder(event.clientX, event.clientY);
        setFlight((before) =>
          before === null
            ? before
            : { ...before, x: at.x, y: at.y, over, verdict: over === before.over ? before.verdict : null },
        );
        if (over !== null && over !== flight.over) ask(flight.port, over);
      }}
      onPointerUp={(event) => {
        setPan(null);
        if (move !== null) {
          store.getState().moveBox(move.path, { x: Math.round(move.x), y: Math.round(move.y) }, one.id);
          setMove(null);
          return;
        }
        if (flight === null || !flight.pressed) return;
        const over = portUnder(event.clientX, event.clientY);
        const side = sideUnder(event.clientX, event.clientY);
        const [pointer, port] = over === null ? ['', ''] : splitPort(over);
        const target = model.byPointer.get(pointer);
        if (over === null || target === undefined || side !== 'inputs') {
          // A release over nothing leaves the connection **armed**, not cancelled: the click that
          // follows is the second half of the same gesture, and Escape is what puts it down. A
          // drag that ended somewhere unexpected is not a reason to make the author start again.
          setFlight((before) => (before === null ? null : { ...before, pressed: false }));
          return;
        }
        const to = handleFor(folded, target, port);
        const from = flight.from;
        setFlight(null);
        if (to !== null) connect(from, to);
      }}
      onWheel={(event) => {
        // §4.4's own: "wheel zooms". The canvas has nothing to scroll — `.canvas` is
        // `overflow: hidden` — so the default is harmless and is not prevented: React registers
        // `wheel` at the root as a passive listener, where `preventDefault` is a no-op and a
        // warning.
        shell.getState().setZoom(viewport.zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1));
      }}
    >
      <span className="canvas-note">{textWith('folded document · {}', model.note)}</span>
      <span className="canvas-note right fig">{`${String(Math.round(viewport.zoom * 100))}%`}</span>
      <div className="viewport">
        <div
          className="gcanvas"
          data-canvas={one.path}
          style={{
            transform: `translate(${String(viewport.x)}px, ${String(viewport.y)}px) scale(${String(viewport.zoom)})`,
          }}
        >
          <Wires
            wires={model.wires}
            routes={wires}
            links={model.links}
            placement={placement}
            selection={selection}
            flight={flight}
            onSelect={(wire) => {
              store.getState().selectPlace(wire.path, one.id);
            }}
          />
          {drawn.map((box) => {
            const at = placement.boxes.get(box.pointer);
            if (at === undefined) return null;
            const dragged = move?.pointer === box.pointer;
            const litPort =
              flight?.over?.startsWith(`${box.pointer}:`) === true
                ? splitPort(flight.over)[1]
                : null;
            return (
              <div
                key={box.pointer}
                className="abs"
                style={{
                  left: `${String(dragged ? move.x : at.x)}px`,
                  top: `${String(dragged ? move.y : at.y)}px`,
                  width: `${String(at.width)}px`,
                }}
              >
                <Box
                  box={box}
                  selected={box.pointer === selection}
                  renaming={renaming === box.pointer}
                  litPort={litPort}
                  register={(element) => {
                    if (element === null) elements.current.delete(box.pointer);
                    else elements.current.set(box.pointer, element);
                  }}
                  onSelect={() => {
                    select(box);
                  }}
                  onRename={(to) => {
                    setRenaming(null);
                    if (to === '' || to === box.name) return;
                    const applied = store
                      .getState()
                      .edit((edit) => renameAt(gestures, edit, box.path, to), one.id);
                    if (applied !== null) announce(applied.label);
                  }}
                  onStartRename={() => {
                    setRenaming(box.pointer);
                  }}
                  onFold={() => {
                    store.getState().toggleGroup(box.pointer, box.path, one.id);
                  }}
                  onMenu={(x, y) => {
                    select(box);
                    setMenu({ box, x, y });
                  }}
                  onGrab={(event: ReactPointerEvent<HTMLElement>) => {
                    const point = pointOf(event.clientX, event.clientY, surface.current, viewport);
                    setMove({
                      pointer: box.pointer,
                      path: box.path,
                      dx: point.x - at.x,
                      dy: point.y - at.y,
                      x: at.x,
                      y: at.y,
                    });
                  }}
                  onPort={(port, side, pressed) => {
                    onPort(box, port, side, pressed);
                  }}
                  onProblem={() => {
                    store.getState().revealPlace(box.path, one.id);
                    shell.getState().revealPanel('panel.problems');
                  }}
                  onChip={(what) => {
                    select(box);
                    shell.getState().revealPanel('panel.properties');
                    announce(textWith('Editing {}', `${what} of ${box.name}`));
                  }}
                  onHandle={(site) => {
                    store.getState().revealPlace(pathOfPointer(site), one.id);
                  }}
                />
              </div>
            );
          })}
        </div>
      </div>
      {flight === null ? null : <FlightNote flight={flight} surface={surface.current} viewport={viewport} />}
      {menu === null ? null : (
        <ContextMenu
          box={menu.box}
          x={menu.x}
          y={menu.y}
          onRun={(entry) => {
            run(entry, menu.box);
          }}
        />
      )}
      <p className="offscreen" role="status" aria-live="polite">
        {announced}
      </p>
    </div>
  );
}

/** The keys that nudge the selected box — §4.4's "arrows nudge". */
const NUDGES: readonly string[] = ['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight'];

/** What the drag says about itself while it is in flight: the source, and the core's verdict. */
function FlightNote({
  flight,
  surface,
  viewport,
}: {
  flight: Flight;
  surface: HTMLElement | null;
  viewport: Viewport;
}): JSX.Element {
  const box = surface?.getBoundingClientRect();
  const left = (box?.left ?? 0) + viewport.x + flight.x * viewport.zoom + 14;
  const top = (box?.top ?? 0) + viewport.y + flight.y * viewport.zoom + 14;
  return (
    <div className="dghost" style={{ left: `${String(left)}px`, top: `${String(top)}px` }} role="status">
      <span>{`${flight.from.name}.${flight.from.port} →`}</span>
      {flight.verdict === null ? null : (
        <span className={flight.verdict.ok ? 'verdict ok' : 'verdict'} data-verdict={flight.verdict.ok ? 'ok' : 'bad'}>
          {flight.verdict.ok ? text('No refusal.') : refusalText(flight.verdict)}
        </span>
      )}
    </div>
  );
}

/**
 * What a refusing verdict says, in the tools' own line — `[V5] expects token, receives patch`.
 *
 * S17 draws that line while the drag is in flight, and it is the same line the row will carry on
 * the drop (§4.17): one wording, formatted by the core's own printer, so the ghost and the panel
 * cannot say two different things about one candidate.
 */
function refusalText(verdict: CandidateVerdict): string {
  const first = verdict.problems[0];
  return first === undefined ? (verdict.unknown ?? '') : formatSemanticProblem(first);
}

/** Which compositions are drawn shut: everything the reader has not opened (§4.7). */
function collapsedOf(folded: FoldedGraph, expanded: ReadonlySet<string>): Set<string> {
  const shut = new Set<string>();
  for (const node of folded.nodes) {
    if (node.children.length > 0 && !expanded.has(node.pointer)) shut.add(node.pointer);
  }
  return shut;
}

/** A client point in the drawing's own coordinates. */
function pointOf(
  clientX: number,
  clientY: number,
  surface: HTMLElement | null,
  viewport: Viewport,
): { x: number; y: number } {
  if (surface === null) return { x: clientX, y: clientY };
  const box = surface.getBoundingClientRect();
  return {
    x: (clientX - box.left - viewport.x) / viewport.zoom,
    y: (clientY - box.top - viewport.y) / viewport.zoom,
  };
}

/**
 * The `(site, port)` the core's `check` takes, from a `data-port` of the drawing.
 *
 * `PortRef` names a site of the *expanded* graph — the iteration the card stands for — which is
 * exactly the representative `describe(folded)` described, so a verdict during a drag is about the
 * same site the card's own ports came from.
 */
function portRef(folded: FoldedGraph, attribute: string): PortRef | null {
  const [pointer, port] = splitPort(attribute);
  const node = folded.byPointer.get(pointer);
  if (node === undefined || node.site === null) return null;
  return { site: node.site, port };
}

/** The `data-port` of the handle under the pointer, or `null`. */
function portUnder(clientX: number, clientY: number): string | null {
  const element = document.elementFromPoint(clientX, clientY);
  const handle = element === null ? null : (element as HTMLElement).closest('[data-port]');
  return handle === null ? null : handle.getAttribute('data-port');
}

/** Which side the handle under the pointer is. */
function sideUnder(clientX: number, clientY: number): string | null {
  const element = document.elementFromPoint(clientX, clientY);
  const handle = element === null ? null : (element as HTMLElement).closest('[data-port]');
  return handle === null ? null : handle.getAttribute('data-side');
}

/** A `data-port` split back into the box it is on and the port it names. */
export function splitPort(attribute: string): [string, string] {
  const at = attribute.lastIndexOf(':');
  return at < 0 ? [attribute, ''] : [attribute.slice(0, at), attribute.slice(at + 1)];
}

/** An RFC 6901 pointer back as a path of the document. */
function pathOfPointer(pointer: string): Path {
  return pointer
    .split('/')
    .slice(1)
    .map((step) => step.replace(/~1/g, '/').replace(/~0/g, '~'));
}

/**
 * The handle of a box's port, as the core writes an endpoint from it (`valueEndpoint`).
 *
 * A site of a composition is named at the iteration its card stands for, which is the
 * representative `describe(folded)` described — so a wire drawn onto `attn_n` writes
 * `[layer=0]`, exactly as `decoder.entry` does in the corpus.
 */
export function handleFor(folded: FoldedGraph, box: CanvasBox, port: string): FoldedHandle | null {
  const node = folded.byPointer.get(box.pointer);
  if (node === undefined) return null;
  const site = node.site ?? rootSite(node.name);
  return {
    box: node.parent ?? node.pointer,
    site: node.pointer,
    name: node.name,
    port,
    // The index of a selector is a `scalar_expression`, not a number: a gesture onto the card of
    // `attn_n` writes `{"literal": 0}`, which is what `decoder.entry` writes in the corpus. The
    // core is what knows that, so the core writes it.
    indices: site.indices.map((one) => ({
      name: one.name,
      written: literalIndex(one.value as bigint),
      value: one.value,
    })),
    where: node.where,
    value: node.where === null ? null : `${node.where}.${port}`,
    boundary: node.parent !== null,
  };
}

/** The wires, the identity links and the label of each, over the boxes. */
function Wires({
  wires,
  routes: drawn,
  links,
  placement,
  selection,
  flight,
  onSelect,
}: {
  wires: readonly CanvasWire[];
  routes: readonly WireRoute[];
  links: CanvasModel['links'];
  placement: Placement;
  selection: string | null;
  flight: Flight | null;
  onSelect: (wire: CanvasWire) => void;
}): JSX.Element {
  const byId = new Map(wires.map((wire) => [wire.id, wire]));
  const start = flight === null ? undefined : placement.boxes.get(flight.box);
  return (
    <>
      <svg className="wires" aria-hidden="true">
        {drawn.map((route) => {
          const wire = byId.get(route.id);
          if (wire === undefined) return null;
          const classes = ['w', wire.interface ? 'iface' : '', wire.id === selection ? 'sel' : '']
            .filter((one) => one !== '')
            .join(' ');
          return <path key={route.id} className={classes} d={route.d} />;
        })}
        {links.map((link, at) => {
          const from = placement.boxes.get(link.from.box);
          const to = placement.boxes.get(link.to.box);
          if (from === undefined || to === undefined) return null;
          return (
            <path
              key={`${link.identity}:${String(at)}`}
              className="idlink"
              d={`M${String(from.x + from.width / 2)} ${String(from.y + from.height)} L${String(to.x + to.width / 2)} ${String(to.y)}`}
            />
          );
        })}
        {flight === null || start === undefined ? null : (
          <path
            className={`w flight${flight.verdict === null ? '' : flight.verdict.ok ? ' ok' : ' bad'}`}
            d={`M${String(start.x + start.width / 2)} ${String(start.y + start.height)} L${String(flight.x)} ${String(flight.y)}`}
          />
        )}
      </svg>
      {drawn.map((route) => {
        const wire = byId.get(route.id);
        if (wire === undefined) return null;
        return (
          <button
            key={`${route.id}:label`}
            type="button"
            className="elbl"
            data-wire={wire.id}
            style={{ left: `${String(route.label.x)}px`, top: `${String(route.label.y)}px` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(wire);
            }}
          >
            {wire.label}
            {wire.type === null ? null : <span className="wtype">{wire.type}</span>}
          </button>
        );
      })}
      {links.map((link, at) => {
        const from = placement.boxes.get(link.from.box);
        const to = placement.boxes.get(link.to.box);
        if (from === undefined || to === undefined) return null;
        return (
          <span
            key={`${link.identity}:${String(at)}:label`}
            className="elbl idlbl"
            data-identity={link.identity}
            style={{
              left: `${String((from.x + to.x) / 2 + from.width / 2)}px`,
              top: `${String((from.y + from.height + to.y) / 2)}px`,
            }}
          >
            {link.identity}
          </span>
        );
      })}
    </>
  );
}

/** §4.7's context menu, at the pointer. */
function ContextMenu({
  box,
  x,
  y,
  onRun,
}: {
  box: CanvasBox;
  x: number;
  y: number;
  onRun: (entry: MenuEntry) => void;
}): JSX.Element {
  return (
    <div className="ctxmenu" style={{ left: `${String(x)}px`, top: `${String(y)}px` }} role="menu">
      {entriesFor(box.role, ROLE).map((entry) => (
        <button
          key={entry.id}
          type="button"
          role="menuitem"
          className={entry.separated === true ? 'sep' : ''}
          data-entry={entry.id}
          onClick={(event) => {
            event.stopPropagation();
            onRun(entry);
          }}
        >
          {text(entry.label)}
        </button>
      ))}
    </div>
  );
}

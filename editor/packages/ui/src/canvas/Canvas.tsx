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
  identityReadings,
  literalIndex,
  PROPOSED_COMPOSITION,
  PROPOSED_INDEX,
  rootSite,
  toPython,
  type FoldedGraph,
  type FoldedHandle,
  type IdentityReading,
  type PortRef,
  type Verdict as CandidateVerdict,
} from '@tensorspine/lang';
import { pointerOf, type Path, type Position } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { carriedIdentity, offersOf } from '../library/primitives.js';
import { sourceStanding, type OpenDocument } from '../documents/store.js';
import { presentation } from '../presentation/index.js';
import { bindPrivately, tieTo, type HeldMember, type SlotTarget } from '../sheet/bindings.js';
import { useShell, useShellStore } from '../shell/context.js';
import { text, textWith } from '../shell/strings.js';

import { Box } from './Box.js';
import type { DrillGhostBox, DrillModel } from './drill.js';
import {
  addInstance,
  addSite,
  connectFromPreviousIteration,
  connectHandles,
  connectScoped,
  duplicateAt,
  duplicateWithComplementaryGuard,
  moveIntoComposition,
  proposeGuard,
  removeAt,
  renameAt,
  scopedValuesOf,
  sitesOf,
  type GestureContext,
} from './gestures.js';
import {
  fit,
  NO_PLACEMENT,
  place,
  routes,
  type PlacedBox,
  type Placement,
  type WireRoute,
} from './layout.js';
import {
  CARRY,
  compositionEntries,
  drillEntriesFor,
  entriesFor,
  portEntries,
  type MenuEntry,
} from './menu.js';
import {
  canvasModel,
  ROLE,
  SIDE,
  type CanvasBox,
  type CanvasModel,
  type CanvasSlot,
  type CanvasWire,
} from './model.js';

/** What a palette drag carries onto the canvas — §4.7's "Drop a primitive from the palette". */
export const PRIMITIVE_TRANSFER = 'application/x-tensorspine-primitive';

/**
 * The primitive a drag carries: what to instantiate, and at which version.
 *
 * The version is **optional** since feature 2.21: which versions a name has is a fact about the
 * library, so a palette that carries a name alone is answered from the catalog (`carriedIdentity`)
 * rather than by a version written into the drag.
 */
export interface PrimitiveTransfer {
  readonly primitive: string;
  readonly version?: string;
}

/** What a drag put on the clipboard, or `null` where it carried something else. */
export function primitiveOf(transfer: DataTransfer): PrimitiveTransfer | null {
  const held = transfer.getData(PRIMITIVE_TRANSFER);
  if (held === '') return null;
  try {
    const parsed: unknown = JSON.parse(held);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const one = parsed as PrimitiveTransfer;
    if (typeof one.primitive !== 'string') return null;
    return one.version === undefined || typeof one.version === 'string' ? one : null;
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
  /**
   * The index this connection takes its source at the previous value of — §4.8's carry.
   *
   * Absent for the ordinary connection, which is every connection the folded canvas makes.
   */
  readonly carry?: string;
}

/**
 * A tie in flight: a slot chip pressed, and the chip the pointer is over.
 *
 * The same state machine as a connection (§4.7's two shapes of one gesture): a press arms it and a
 * release over another chip makes it, or a click arms it and a second click makes it — which is
 * the keyboard's form. A release over the chip it started on is the *click*, and the chip's own
 * gesture (select the identity) is what happens instead.
 */
interface Tie {
  /** The chip it started on, as `data-slot` writes it — which is all {@link slotAt} needs. */
  readonly chip: string;
  /** Whether a pointer is down: a click-armed tie waits for a second click. */
  readonly pressed: boolean;
  /** The chip the pointer is over, or `null`. */
  readonly over: string | null;
  /** What the core says about the candidate, shown during the drag (§4.7). */
  readonly verdict: CandidateVerdict | null;
}

/** What a chip stands for: the slot, where its member is written, and the rule that holds it. */
interface TiedSlot {
  readonly target: SlotTarget;
  readonly held: HeldMember | null;
  /** The place the rule is written at, or `null` where nothing binds the slot (V7's chip). */
  readonly rule: Path | null;
}

/**
 * The identities a document declares, or none where the reading refuses it.
 *
 * `identityReadings` normalises the document to read §5.2 rule 7's names, and `normalise` refuses
 * a document being edited between two keystrokes (a duplicate rule name) — the canvas draws
 * either way, as feature 2.9's folded reading does.
 */
function readIdentities(tree: Parameters<typeof foldedGraph>[0]): readonly IdentityReading[] {
  try {
    return identityReadings(toPython(tree));
  } catch {
    return [];
  }
}

/**
 * What tells the canvas it is drawing a **drill-in** rather than the folded document (§4.8).
 *
 * The drill-in is the same canvas one level down: the same cards, the same connection gesture, the
 * same context menu, the same keyboard. What changes is the *model* it draws — one composition's
 * sites, its scoped edges, its ghost columns and its pinned terminals — and what a gesture writes:
 * a connection inside a composition is a scoped rule, a dropped primitive is a site. So the drill
 * is a parameter of this component and not a second one; a second would be a second drawing to
 * keep in step (feature 2.9's own reason for not wrapping React Flow).
 */
export interface DrillContext {
  /** The composition the tab is over, by name. */
  readonly composition: string;
  readonly model: DrillModel;
  /** The point the scrubber stands on, or `null` while it is unset. */
  readonly scrub: string | null;
}

/** A box being moved: which, and where it is now. */
interface Move {
  readonly pointer: string;
  readonly path: Path;
  readonly dx: number;
  readonly dy: number;
  readonly x: number;
  readonly y: number;
  /**
   * Whether the drag makes a **copy** — §4.4's "Alt+drag duplicates".
   *
   * The copy is made where the drag ends rather than where it starts: a duplicate that appeared
   * under the pointer at the first pixel of movement would be a gesture the author could not
   * abandon, and the original stays where it was either way.
   */
  readonly copy?: boolean;
}

/** The rubber band of §4.4's "Shift+drag rubber-bands": where it started and where it is now. */
interface Band {
  readonly x: number;
  readonly y: number;
  readonly toX: number;
  readonly toY: number;
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

/** The whole canvas of one open document, folded (§4.7) or drilled into (§4.8). */
export function Canvas({ one, drill }: { one: OpenDocument; drill?: DrillContext }): JSX.Element {
  const store = useDocumentsStore();
  const shell = useShellStore();
  const registry = useDocuments((state) => state.registry);
  // The one catalog every chooser of a primitive reads (feature 2.21): the drop's identity and the
  // sheet's suggest list are the same list, so a base that was not gathered is missing from both.
  const catalog = useDocuments((state) => state.library.catalog);
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
  /** §4.7's other drag: a slot chip onto another slot chip, which ties or shares (2.13). */
  const [tying, setTying] = useState<Tie | null>(null);
  const [move, setMove] = useState<Move | null>(null);
  const [pan, setPan] = useState<{ x: number; y: number } | null>(null);
  /** §4.4's rubber band, and §4.7's "Add to Composition…" chooser, both at the pointer. */
  const [band, setBand] = useState<Band | null>(null);
  const [choosing, setChoosing] = useState<{ box: CanvasBox; x: number; y: number } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [menu, setMenu] = useState<{ box: CanvasBox; x: number; y: number } | null>(null);
  /** §4.15's port menu: which port it was opened on, and where. */
  const [portMenu, setPortMenu] = useState<
    { box: CanvasBox; port: string; side: string; x: number; y: number } | null
  >(null);
  const [announced, setAnnounced] = useState('');
  /**
   * Whether the click that closes the current gesture belongs to it.
   *
   * A pointer press on a card's ground starts a move; the browser sends the `click` after the
   * release, and a copy made by an Alt+drag would be selected and then unselected by that click
   * landing back on the original. The click is part of the drag, so it is skipped once.
   */
  const consumed = useRef(false);

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
    if (drill !== undefined) return drill.model.canvas;
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
      tree: one.session.store.tree,
    });
  }, [drill, folded, registry, one.reading, problems, expanded, toggles, shapes, bindings, one.session.store.role, templates]);

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

  /**
   * The placement the drawing uses: ELK's, with the ghost columns put beside it (§4.8).
   *
   * A ghost is a *copy* of a card that is already drawn, so it is no node of the graph: letting
   * ELK route around one would move the real chain to make room for a picture of it. It is placed
   * on the side its override is on, level with the wires that reach it, and the drawing is shifted
   * to make the column's width room rather than letting it fall off the left edge.
   */
  const placed: Placement = useMemo(() => {
    if (drill === undefined || drill.model.ghosts.length === 0) return placement;
    return withGhosts(placement, drill.model.ghosts);
  }, [placement, drill]);

  const fitNow = useCallback(() => {
    const element = surface.current;
    if (element === null) return;
    const answer = fit(placed, { width: element.clientWidth, height: element.clientHeight });
    setViewport(answer);
    shell.getState().setZoom(answer.zoom);
  }, [placed, shell]);

  const extent = `${String(placed.width)}x${String(placed.height)}`;
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

  /**
   * Where the pointer last was on this canvas — §4.4's `Add Instance… (opens the library picker
   * **at the cursor**)`.
   *
   * A ref and not state, because it is written on every pointer move and read once: a canvas that
   * re-rendered per move would be the one thing §5.6's 60 fps forbids. The *last* place and not
   * the current one, because reaching the command through the menu bar takes the pointer off the
   * canvas — a person pointing at where they want it and then opening Model is the gesture.
   */
  const cursor = useRef<Position | null>(null);

  // The command the picker of feature 2.21 opens, bound while a canvas is mounted so that the
  // point is this canvas's; `apps/web` binds the same store call without one, which is what the
  // cleanup puts back — so an unmounted canvas cannot answer with a point it no longer has.
  useEffect(() => {
    const bar = shell.getState();
    bar.bind({
      'model.add-instance': () => {
        store.getState().offerPrimitives(cursor.current ?? undefined);
      },
    });
    return () => {
      bar.bind({
        'model.add-instance': () => {
          store.getState().offerPrimitives();
        },
      });
    };
  }, [shell, store]);

  /**
   * What the scrubber dims — §4.8: "set to a value it dims every site and edge absent at that
   * index". Absence is D1's answer ({@link DrillModel.presence}); the drawing only reads it.
   */
  const dimmed = useMemo(() => {
    const boxes = new Set<string>();
    const edges = new Set<string>();
    const ghosts = new Set<string>();
    const scrub = drill?.scrub ?? null;
    if (drill === undefined || scrub === null) return { boxes, edges, ghosts };
    const { presence, terminals, canvas, ghosts: columns } = drill.model;
    for (const box of canvas.boxes) {
      const present = presence.sites.get(box.name);
      if (present !== undefined && !present.has(scrub)) boxes.add(box.pointer);
    }
    for (const wire of canvas.wires) {
      const present = presence.edges.get(wire.id);
      if (present !== undefined && !present.has(scrub)) edges.add(wire.id);
    }
    for (const [id, terminal] of terminals) {
      if (terminal.links.every((link) => edges.has(link.at.join('/') === '' ? '' : pointerOf(link.at)))) {
        boxes.add(id);
      }
    }
    for (const column of columns) {
      if (canvas.wires.filter((wire) => wire.from === column.id || wire.to === column.id).every((wire) => edges.has(wire.id))) {
        ghosts.add(column.id);
      }
    }
    return { boxes, edges, ghosts };
  }, [drill]);
  const dimmedWires = dimmed.edges;

  /**
   * What each guard evaluates to where the scrubber stands — S4's `when layer ≥ 1 — false`.
   *
   * §4.8 asks the scrubber to "show the guard's evaluation", which D1 cannot say: an absent node
   * says that a site is not there, never why. The truth is the core's ({@link DrillPresence}), read
   * at the point the scrubber is on.
   */
  const truths = useMemo(() => {
    const found = new Map<string, boolean | null>();
    const scrub = drill?.scrub ?? null;
    if (drill === undefined || scrub === null) return found;
    for (const [what, values] of drill.model.presence.guards) {
      found.set(what, values.get(scrub) ?? null);
    }
    return found;
  }, [drill]);

  const gestures: GestureContext = { shapes, bindings, role: one.session.store.role };
  const selection = one.selection === undefined ? null : pointerOf(one.selection);
  /** Every place the rubber band holds — §4.4's Shift+drag, and §4.11's intersection. */
  const marked = useMemo(
    () => new Set((one.marked ?? []).map((path) => pointerOf(path))),
    [one.marked],
  );
  const announce = (line: string): void => {
    setAnnounced(line);
  };

  const select = (box: CanvasBox): void => {
    store.getState().selectPlace(box.path, one.id);
  };

  /**
   * The guard "Connect from previous iteration…" proposes — §4.8's "a proposal the user confirms".
   *
   * The edge is made first, because Q5 says a gesture is never held back by what the core will
   * say about it; the guard is the *fact of its own* the model guide asks the author to state, so
   * it is offered beside the edge with the one action that writes it. Refusing it leaves a
   * document the validator refuses at the first iteration, which is what the Problems panel is
   * for — and the row says so in the core's own words.
   */
  const proposeCarryGuard = (
    from: FoldedHandle,
    to: FoldedHandle,
    context: DrillContext,
    index: string,
  ): void => {
    const rule = [...scopedValuesOf(context.composition), `${to.name}.${to.port}`] as Path;
    store.getState().setToast({
      text: textWith(
        'Connected {} from the previous iteration. It fires at every index; the guard states where it may.',
        `${from.name}.${from.port}`,
      ),
      action: textWith('Add guard {}', `${index} >= 1`),
      run: () => {
        const applied = store.getState().edit((edit) => proposeGuard(edit, rule, index), one.id);
        store.getState().setToast(null);
        if (applied !== null) announce(applied.label);
      },
    });
  };

  /**
   * The connection, made whatever the verdict said (Q5).
   *
   * §4.7's own table: "the new binding is named `<to>.<port>` (uniquified), **or is a scoped rule
   * in a drill-in**". Inside a composition both ends are its sites, so the rule is written in the
   * composition's own `bindings/values` with site endpoints — and `carry` is §4.8's
   * "Connect from previous iteration…", which is the same rule with the override on its source
   * and the guard proposed after it.
   */
  const connect = (from: FoldedHandle, to: FoldedHandle, carry?: string): void => {
    const applied = store.getState().edit((edit) => {
      if (drill === undefined) return connectHandles(edit, from, to);
      const ends = {
        composition: drill.composition,
        from: { site: from.name, port: from.port },
        to: { site: to.name, port: to.port },
      };
      if (carry === undefined) return connectScoped(edit, ends);
      return connectFromPreviousIteration(edit, { ...ends, index: carry });
    }, one.id);
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
    // The carry's **proposal** is the last word, because it is the one thing left to decide: the
    // replacement is already made and `Edit ▸ Undo` takes the whole gesture back either way (the
    // connection and its replacement are one command, D13).
    if (carry !== undefined && drill !== undefined) proposeCarryGuard(from, to, drill, carry);
    announce(applied.label);
  };

  /** The identities the document declares, read once per revision (§5.2 rule 7's own names). */
  const identities = useMemo(
    () => readIdentities(one.session.store.tree),
    [one.id, one.session.store.revision],
  );

  /**
   * The slot a chip stands for: the site the card is drawn over, and where its member is written.
   *
   * Everything in it is the core's — the site is the representative iteration `describe(folded)`
   * answered, the rule and the position are what bound the slot (`boundBy`, `boundAt`), and the
   * place the rule is written at is the reading of §5.2 rule 7. The canvas reads none of it.
   */
  const slotAt = (chip: string): TiedSlot | null => {
    const [pointer, slot] = splitPort(chip);
    const node = folded.byPointer.get(pointer);
    const described = node?.where === null || node?.where === undefined
      ? undefined
      : one.reading.facts?.sites.get(node.where);
    if (node?.site === undefined || node.site === null || described === undefined) return null;
    const state = described.states.find((each) => each.name === slot);
    const parameter = described.parameters.find((each) => each.name === slot);
    const bound = state ?? parameter;
    if (bound === undefined) return null;
    const rule = bound.boundBy;
    const reading = rule === null ? undefined : identities.find((each) => each.rule === rule);
    return {
      target: { site: node.site, slot, state: state !== undefined },
      held:
        rule === null || reading === undefined
          ? null
          : { rule: pathOfPointer(reading.pointer), at: bound.boundAt },
      rule: reading === undefined ? null : pathOfPointer(reading.pointer),
    };
  };

  /** The tie, made whatever the verdict said (Q5) — "creates or extends the identity" (§4.7). */
  const tie = (from: string, onto: string): void => {
    const source = slotAt(from);
    const target = slotAt(onto);
    if (source === null || target === null) return;
    // Asked before the gesture is made, for the reason §4.7 gives the drag its verdict at all:
    // `check` reads the analysis the session holds, and after the edit that analysis is of a
    // document already carrying the member.
    const asked = store.getState().checkMember(
      {
        kind: source.target.state ? 'state' : 'parameter',
        slot: { site: source.target.site, name: source.target.slot },
        into: { slot: { site: target.target.site, name: target.target.slot } },
      },
      one.id,
    );
    const applied = store.getState().edit((edit) => {
      // The chip dropped on carries the identity: its rule takes the member. Where it carries
      // none — a document already refused by V7 — the identity is created holding both.
      if (target.rule !== null) {
        return tieTo(edit, { target: source.target, held: source.held, into: target.rule });
      }
      return bindPrivately(edit, {
        target: target.target,
        held: target.held,
        joining: { target: source.target, held: source.held },
      });
    }, one.id);
    if (applied === null) return;
    announce(applied.label);
    void asked.then((verdict) => {
      const first = verdict?.problems[0];
      if (first !== undefined) {
        store.getState().setToast({ text: `[${first.code}] ${first.message}` });
      }
    });
  };

  /**
   * Press or click on a slot chip: the tie of §4.7, and the chip's own gesture.
   *
   * A press arms the tie; a release or a click over **another** chip makes it; a click on the chip
   * it started on is the chip's own gesture — the identity is selected and the sheet opens, which
   * is what §4.7 gives a click on a chip.
   */
  const onSlot = (box: CanvasBox, slot: CanvasSlot, pressed: boolean): void => {
    const chip = `${box.pointer}:${slot.name}`;
    if (tying === null) {
      setTying({ chip, pressed, over: null, verdict: null });
      announce(textWith('Tying from {}', `${box.name}.${slot.name}`));
      return;
    }
    if (tying.chip === chip) {
      // The click that follows the press that armed this tie: the same gesture, and the chip's own
      // when nothing else happened.
      if (pressed) return;
      setTying(null);
      select(box);
      shell.getState().revealPanel('panel.properties');
      announce(textWith('Editing {}', `${slot.name} of ${box.name}`));
      return;
    }
    if (pressed) return;
    const from = tying.chip;
    setTying(null);
    tie(from, chip);
  };

  /** What the core says about the tie the pointer is over — shown *during* the drag (§4.7). */
  const askTie = (from: string, over: string | null): void => {
    if (over === null) return;
    const source = slotAt(from);
    const target = slotAt(over);
    if (source === null || target === null) return;
    void store
      .getState()
      .checkMember(
        {
          kind: source.target.state ? 'state' : 'parameter',
          slot: { site: source.target.site, name: source.target.slot },
          into: { slot: { site: target.target.site, name: target.target.slot } },
        },
        one.id,
      )
      .then((verdict) => {
        setTying((before) => (before === null || before.over !== over ? before : { ...before, verdict }));
      });
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
      const at = placed.boxes.get(box.pointer);
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
    const carry = flight.carry;
    setFlight(null);
    connect(from, handle, carry);
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
    const held = primitiveOf(event.dataTransfer);
    if (held === null) return;
    event.preventDefault();
    // The identity is the **catalog's**, which is the one list every chooser of a primitive reads
    // (feature 2.21): a drag that names a version keeps it, one that names only a name is pinned
    // at the version the library carries, and one the catalog does not carry still lands (Q5).
    const carried = carriedIdentity(offersOf(catalog), held);
    const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
    let added: string | null = null;
    const applied = store.getState().edit((edit) => {
      // §4.7's own table: the drop "adds an instance (root canvas) or a site (drill-in)".
      const command =
        drill === undefined
          ? addInstance(edit, { primitive: carried.name, version: carried.version })
          : addSite(edit, {
              composition: drill.composition,
              primitive: carried.name,
              version: carried.version,
            });
      added = command.name;
      return command;
    }, one.id);
    if (applied === null || added === null) return;
    const path = (drill === undefined ? ['instances', added] : [...sitesOf(drill.composition), added]) as Path;
    // Where it was dropped is where it goes: a manual place, which is D6's own override.
    store.getState().moveBox(path, { x: Math.round(at.x), y: Math.round(at.y) }, one.id);
    store.getState().selectPlace(path, one.id);
    // §4.7: the sheet opens on the new instance, so the author fills in what V2 is about to ask
    // for. The Properties region is where §4.11's sheet is.
    shell.getState().revealPanel('panel.properties');
    announce(applied.label);
  };

  /**
   * §4.15's `Expose as input…` / `Expose as output…`, from the port's own menu.
   *
   * The handle is the folded reading's — the same one a connection is made from — and what is
   * written is the interface's own definition with that endpoint in it. Which map takes it is the
   * *side* the terminal is drawn on, which `presentation.json` says and the store reads.
   */
  const expose = (box: CanvasBox, port: string, side: string): void => {
    const handle = handleFor(folded, box, port);
    if (handle === null) return;
    const name = store
      .getState()
      .expose(handle, side === SIDE_OF.inputs ? SIDE.left : SIDE.right, one.id);
    if (name === null) return;
    shell.getState().revealPanel('panel.properties');
    announce(textWith('Exposed {}', `${handle.name}.${handle.port} as ${name}`));
  };

  const run = (entry: MenuEntry, box: CanvasBox): void => {
    setMenu(null);
    select(box);
    if (entry.id === 'canvas.drill-in') {
      drillInto(box);
      return;
    }
    if (entry.id === 'canvas.add-to-composition') {
      // §4.7 writes the entry with an ellipsis: what it asks is *which* composition, and the
      // answer is the document's own list (`compositionEntries`).
      setChoosing({ box, x: menu?.x ?? 0, y: menu?.y ?? 0 });
      return;
    }
    if (entry.id.startsWith('canvas.add-to-composition:')) {
      moveInto(entry.id.slice('canvas.add-to-composition:'.length), [box]);
      return;
    }
    if (entry.id === 'canvas.extract-to-composition') {
      moveInto(null, selectedBoxes(box));
      return;
    }
    if (entry.id === 'canvas.duplicate-complementary') {
      const applied = store
        .getState()
        .edit((edit) => duplicateWithComplementaryGuard(edit, box.path), one.id);
      if (applied !== null) announce(applied.label);
      return;
    }
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
   * §4.4's `Ctrl+Enter`: "drills into a composition or template" — §4.8's own tab.
   *
   * The tab is the store's (one per drill-in, §4.2), and what is handed over is the composition's
   * *name*, which is the map key the document writes and what a tab is named after.
   */
  const drillInto = (box: CanvasBox): void => {
    if (box.role !== ROLE.group) return;
    const name = box.path[box.path.length - 1];
    if (typeof name !== 'string') return;
    store.getState().drillInto(name, one.id);
  };

  /**
   * §4.20's move, in both its forms: into a composition that exists, or into one it creates.
   *
   * The reading is the core's and the command is `gestures.ts`'s; what is here is what the
   * *interface* owes it — the name a created composition is proposed under (§9 Q4's own kind of
   * convention, changed in the sheet like any other name), the selection it acts on, and the note
   * a move that could not decide something leaves in the toast.
   */
  const moveInto = (composition: string | null, boxes: readonly CanvasBox[]): void => {
    const instances = boxes
      .filter((box) => box.role === ROLE.node && box.parent === null)
      .map((box) => box.name);
    if (instances.length === 0) return;
    const taken = new Set(
      folded.nodes.filter((node) => node.children.length > 0).map((node) => node.name),
    );
    let proposed = composition ?? PROPOSED_COMPOSITION;
    for (let at = 2; composition === null && taken.has(proposed); at += 1) {
      proposed = `${PROPOSED_COMPOSITION}_${String(at)}`;
    }
    let notes: readonly string[] = [];
    const applied = store.getState().edit((edit) => {
      const command = moveIntoComposition(edit, {
        instances,
        composition: proposed,
        ...(composition === null ? { index: PROPOSED_INDEX } : {}),
      });
      if (command === null) throw new Error('nothing to move');
      notes = command.notes;
      return command;
    }, one.id);
    if (applied === null) return;
    announce(applied.label);
    store.getState().selectPlace(['compositions', proposed] as Path, one.id);
    for (const note of notes) store.getState().note(note);
    const first = notes[0];
    if (first !== undefined) store.getState().setToast({ text: first });
  };

  /**
   * What a rubber band selected — §4.4's "Shift+drag rubber-bands", and the multi-selection
   * §4.11 asks for ("several selected instances show the intersection of their editable rows").
   *
   * Every box whose rectangle the band touches, in the drawing's own order; the first of them is
   * the selection the sheets open on, and the whole set is what the intersection is taken over. A
   * band that caught nothing clears the selection, which is what dragging over empty ground means.
   */
  const markBand = (held: Band): void => {
    const left = Math.min(held.x, held.toX);
    const right = Math.max(held.x, held.toX);
    const top = Math.min(held.y, held.toY);
    const bottom = Math.max(held.y, held.toY);
    if (right - left < 3 && bottom - top < 3) {
      store.getState().selectPlaces([], one.id);
      return;
    }
    const caught = (model?.boxes ?? []).filter((box) => {
      const at = placed.boxes.get(box.pointer);
      if (at === undefined) return false;
      return at.x < right && at.x + at.width > left && at.y < bottom && at.y + at.height > top;
    });
    store.getState().selectPlaces(
      caught.filter((box) => box.path.length > 0).map((box) => box.path),
      one.id,
    );
    announce(textWith('{} selected', String(caught.length)));
  };

  /** The boxes a gesture on one box acts on: the whole selection where it holds that box. */
  const selectedBoxes = (box: CanvasBox): CanvasBox[] => {
    const held = new Set((one.marked ?? []).map((path) => pointerOf(path)));
    if (!held.has(box.pointer)) return [box];
    return model?.boxes.filter((each) => held.has(each.pointer)) ?? [box];
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
      else if (tying !== null) setTying(null);
      else if (menu !== null) setMenu(null);
      else if (choosing !== null) setChoosing(null);
      else store.getState().selectPlace(null, one.id);
      return;
    }
    // §4.4's two shortcuts into and out of a drill-in. They carry a modifier, so they are not the
    // plain Enter and Arrow a control inside a box answers, and they work wherever the focus is —
    // which is what makes Ctrl+↑ a way back rather than a thing to aim at.
    const modified = event.ctrlKey || event.metaKey;
    if (modified && event.key === 'Enter') {
      event.preventDefault();
      const chosenBox = model.boxes.find((box) => box.pointer === (selection ?? ''));
      if (chosenBox !== undefined) drillInto(chosenBox);
      return;
    }
    if (modified && event.key === 'ArrowUp') {
      event.preventDefault();
      if (drill !== undefined) store.getState().select(one.id);
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
      const first = model.boxes.find((box) => placed.boxes.has(box.pointer));
      if (first !== undefined) select(first);
      return;
    }
    const at = placed.boxes.get(chosen.pointer);
    if (at === undefined) return;
    const step = event.shiftKey ? 1 : 8;
    const dx = (event.key === 'ArrowRight' ? step : 0) - (event.key === 'ArrowLeft' ? step : 0);
    const dy = (event.key === 'ArrowDown' ? step : 0) - (event.key === 'ArrowUp' ? step : 0);
    store.getState().moveBox(chosen.path, { x: Math.round(at.x + dx), y: Math.round(at.y + dy) }, one.id);
  };

  const wires = useMemo(() => (model === null ? [] : routes(model, placed)), [model, placed]);

  if (model === null) {
    return (
      <div className="canvas graph">
        <span className="canvas-note">{text('The schemas are not loaded.')}</span>
      </div>
    );
  }

  const drawn = model.boxes.filter((box) => placed.boxes.has(box.pointer));

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
        setChoosing(null);
        if (event.button !== 0 && event.button !== 1) return;
        // §4.4: "Shift+drag rubber-bands". A band selects; a plain drag on the ground pans.
        if (event.shiftKey && event.button === 0) {
          const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
          setBand({ x: at.x, y: at.y, toX: at.x, toY: at.y });
          return;
        }
        setPan({ x: event.clientX - viewport.x, y: event.clientY - viewport.y });
      }}
      onPointerMove={(event) => {
        cursor.current = pointOf(event.clientX, event.clientY, surface.current, viewport);
        if (band !== null) {
          const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
          setBand({ ...band, toX: at.x, toY: at.y });
          return;
        }
        if (pan !== null) {
          setViewport((before) => ({ ...before, x: event.clientX - pan.x, y: event.clientY - pan.y }));
          return;
        }
        if (move !== null) {
          const at = pointOf(event.clientX, event.clientY, surface.current, viewport);
          setMove({ ...move, x: at.x - move.dx, y: at.y - move.dy });
          return;
        }
        if (tying !== null && tying.pressed) {
          const overChip = slotUnder(event.clientX, event.clientY);
          setTying((before) =>
            before === null || before.over === overChip
              ? before
              : { ...before, over: overChip, verdict: null },
          );
          if (overChip !== null && overChip !== tying.over && overChip !== tying.chip) {
            askTie(tying.chip, overChip);
          }
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
        if (band !== null) {
          const held = band;
          setBand(null);
          markBand(held);
          return;
        }
        if (move !== null) {
          // §4.4's Alt+drag: the copy is made where the drag ended, and the original stays.
          if (move.copy === true) {
            let added: string | null = null;
            const applied = store.getState().edit((edit) => {
              const command = duplicateAt(gestures, edit, move.path);
              added = command.name;
              return command;
            }, one.id);
            if (applied !== null && added !== null) {
              const path = [...move.path.slice(0, -1), added] as Path;
              store.getState().moveBox(path, { x: Math.round(move.x), y: Math.round(move.y) }, one.id);
              store.getState().selectPlace(path, one.id);
              consumed.current = true;
              announce(applied.label);
            }
            setMove(null);
            return;
          }
          store.getState().moveBox(move.path, { x: Math.round(move.x), y: Math.round(move.y) }, one.id);
          setMove(null);
          return;
        }
        if (tying !== null && tying.pressed) {
          const onto = slotUnder(event.clientX, event.clientY);
          if (onto === null || onto === tying.chip) {
            // A release over nothing, or back on the chip it started on, leaves the tie **armed**:
            // the click that follows is the second half of the gesture, exactly as a connection's
            // is, and the chip's own click is what a release on itself becomes.
            setTying((before) => (before === null ? null : { ...before, pressed: false }));
            return;
          }
          const from = tying.chip;
          setTying(null);
          tie(from, onto);
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
        const carry = flight.carry;
        setFlight(null);
        if (to !== null) connect(from, to, carry);
      }}
      onWheel={(event) => {
        // §4.4's own: "wheel zooms". The canvas has nothing to scroll — `.canvas` is
        // `overflow: hidden` — so the default is harmless and is not prevented: React registers
        // `wheel` at the root as a passive listener, where `preventDefault` is a no-op and a
        // warning.
        shell.getState().setZoom(viewport.zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1));
      }}
    >
      <span className="canvas-note">
        {sourceStanding(one) === null
          ? textWith('folded document · {}', model.note)
          : // §4.10: "the canvas keeps its last drawable state with a banner until the source is
            // back on the grammar". What is drawn is the tree as it now stands — the folded
            // reading is the tolerant one and draws whatever a document still has (feature 2.9)
            // — so the note says so rather than the count changing meaning.
            textWith('last drawable state · {}', model.note)}
      </span>
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
            placement={placed}
            dimmed={dimmedWires}
            truths={truths}
            selection={selection}
            flight={flight}
            onSelect={(wire) => {
              store.getState().selectPlace(wire.path, one.id);
            }}
          />
          {drill?.model.ghosts.map((ghost) => {
            const at = placed.boxes.get(ghost.id);
            if (at === undefined) return null;
            return (
              <div
                key={ghost.id}
                className={dimmed.ghosts.has(ghost.id) ? 'abs ghostcol dim' : 'abs ghostcol'}
                style={{ left: `${String(at.x)}px`, top: `${String(at.y)}px`, width: `${String(at.width)}px` }}
                data-ghost={ghost.id}
              >
                <span className="gc-cap">{text('ghost')}</span>
                <button
                  type="button"
                  className="node tiny ghosted"
                  title={text(
                    'A copy of the site at another iteration, generated from the rule. Edit the rule to change it.',
                  )}
                  onClick={(event) => {
                    event.stopPropagation();
                    store.getState().selectPlace(pathOfPointer(ghost.site), one.id);
                  }}
                >
                  <span className="n-head">
                    <b>{ghost.name}</b>
                    <span className="n-prim">{ghost.label}</span>
                  </span>
                  {ghost.primitive === null ? null : <span className="n-args">{ghost.primitive}</span>}
                </button>
              </div>
            );
          })}
          {drawn.map((box) => {
            const at = placed.boxes.get(box.pointer);
            if (at === undefined) return null;
            const terminal = drill?.model.terminals.get(box.pointer);
            if (terminal !== undefined) {
              return (
                <div
                  key={box.pointer}
                  className="abs"
                  style={{ left: `${String(at.x)}px`, top: `${String(at.y)}px`, width: `${String(at.width)}px` }}
                >
                  <div
                    className={dimmed.boxes.has(box.pointer) ? 'bterm dim' : 'bterm'}
                    data-terminal={terminal.id}
                    ref={(element) => {
                      if (element === null) elements.current.delete(box.pointer);
                      else elements.current.set(box.pointer, element);
                    }}
                  >
                    <b>{terminal.label}</b>
                    <span>
                      {terminal.links.map((link) => (
                        <button
                          key={link.rule}
                          type="button"
                          className="bterm-rule"
                          data-rule={link.rule}
                          onClick={(event) => {
                            event.stopPropagation();
                            store.getState().selectPlace(link.at, one.id);
                            shell.getState().revealPanel('panel.properties');
                          }}
                        >
                          {link.text}
                        </button>
                      ))}
                    </span>
                  </div>
                </div>
              );
            }
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
                  selected={box.pointer === selection || marked.has(box.pointer)}
                  dimmed={dimmed.boxes.has(box.pointer)}
                  mini={box.parent !== null}
                  renaming={renaming === box.pointer}
                  litPort={litPort}
                  register={(element) => {
                    if (element === null) elements.current.delete(box.pointer);
                    else elements.current.set(box.pointer, element);
                  }}
                  onSelect={() => {
                    if (consumed.current) {
                      consumed.current = false;
                      return;
                    }
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
                  onOpen={() => {
                    drillInto(box);
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
                      ...(event.altKey ? { copy: true } : {}),
                    });
                  }}
                  onPort={(port, side, pressed) => {
                    onPort(box, port, side, pressed);
                  }}
                  onPortMenu={(port, side, x, y) => {
                    select(box);
                    setMenu(null);
                    setPortMenu({ box, port, side, x, y });
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
                  onSlot={(slot, pressed) => {
                    onSlot(box, slot, pressed);
                  }}
                  litSlot={tying?.over ?? null}
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
      {tying === null || tying.verdict === null ? null : (
        <span
          className={tying.verdict.ok ? 'canvas-note tieverdict ok' : 'canvas-note tieverdict'}
          data-tie-verdict={tying.verdict.ok ? 'ok' : 'bad'}
          role="status"
        >
          {tying.verdict.ok ? text('No refusal.') : refusalText(tying.verdict)}
        </span>
      )}
      {band === null ? null : (
        <div
          className="band"
          data-band="true"
          style={{
            left: `${String(viewport.x + Math.min(band.x, band.toX) * viewport.zoom)}px`,
            top: `${String(viewport.y + Math.min(band.y, band.toY) * viewport.zoom)}px`,
            width: `${String(Math.abs(band.toX - band.x) * viewport.zoom)}px`,
            height: `${String(Math.abs(band.toY - band.y) * viewport.zoom)}px`,
          }}
        />
      )}
      {choosing === null ? null : (
        <div
          className="ctxmenu"
          style={{ left: `${String(choosing.x)}px`, top: `${String(choosing.y)}px` }}
          role="menu"
        >
          {compositionEntries(
            folded.nodes.filter((node) => node.children.length > 0).map((node) => node.name),
          ).map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="menuitem"
              data-entry={entry.id}
              onClick={(event) => {
                event.stopPropagation();
                const held = choosing;
                setChoosing(null);
                moveInto(entry.label, selectedBoxes(held.box));
              }}
            >
              {entry.label}
            </button>
          ))}
        </div>
      )}
      {menu === null ? null : (
        <ContextMenu
          box={menu.box}
          // A **site** offers the drill-in's own entries wherever it is drawn: in the tab of
          // §4.8, and inside a composition opened in place on the folded canvas (S3).
          drill={drill !== undefined || menu.box.parent !== null}
          x={menu.x}
          y={menu.y}
          onRun={(entry) => {
            run(entry, menu.box);
          }}
        />
      )}
      {portMenu === null ? null : (
        <PortMenu
          side={portMenu.side}
          indices={drill === undefined ? [] : drill.model.ranges.map((range) => range.name)}
          x={portMenu.x}
          y={portMenu.y}
          onRun={(entry) => {
            const held = portMenu;
            setPortMenu(null);
            if (!entry.id.startsWith(`${CARRY}:`)) {
              expose(held.box, held.port, held.side);
              return;
            }
            // §4.8's "Connect from previous iteration…": the same connection, armed with the
            // override. The author then names the consuming port, which is where the rule's own
            // name comes from — one gesture, two clicks, as every connection here is.
            const handle = handleFor(folded, held.box, held.port);
            const at = placed.boxes.get(held.box.pointer);
            if (handle === null) return;
            setFlight({
              from: handle,
              port: `${held.box.pointer}:${held.port}`,
              box: held.box.pointer,
              x: at === undefined ? 0 : at.x + at.width / 2,
              y: at === undefined ? 0 : at.y + at.height,
              pressed: false,
              over: null,
              verdict: null,
              carry: entry.id.slice(CARRY.length + 1),
            });
            announce(
              textWith('Connecting from {} at the previous iteration', `${handle.name}.${handle.port}`),
            );
          }}
        />
      )}
      <p className="offscreen" role="status" aria-live="polite">
        {announced}
      </p>
    </div>
  );
}

/** Which side of the canvas a terminal for a port of each kind sits on (feature 2.9's bindings). */
const SIDE_OF = { inputs: 'inputs', outputs: 'outputs' } as const;



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

/**
 * The placement with the ghost columns beside it — §4.8's "a ghost column on the left".
 *
 * A left column needs room that ELK did not leave, so the whole drawing is shifted by its width
 * and the columns are put in the space that opens; a right column is put past the drawing's own
 * right edge. Each sits level with the wires that reach it, which is what makes the copy read as
 * the source of the edge it feeds.
 */
export function withGhosts(placement: Placement, ghosts: readonly DrillGhostBox[]): Placement {
  if (ghosts.length === 0) return placement;
  const left = ghosts.filter((ghost) => ghost.side === SIDE.left);
  const width = Math.max(0, ...ghosts.map((ghost) => ghost.width));
  const shift = left.length === 0 ? 0 : width + GHOST_GAP;
  const boxes = new Map<string, PlacedBox>();
  for (const [pointer, box] of placement.boxes) boxes.set(pointer, { ...box, x: box.x + shift });
  let right = 0;
  for (const box of boxes.values()) right = Math.max(right, box.x + box.width);
  for (const ghost of ghosts) {
    const levels = ghost.targets
      .map((target) => boxes.get(target))
      .filter((box): box is PlacedBox => box !== undefined);
    const top =
      levels.length === 0
        ? 0
        : levels.reduce((total, box) => total + box.y + box.height / 2, 0) / levels.length -
          ghost.height / 2;
    boxes.set(ghost.id, {
      pointer: ghost.id,
      x: ghost.side === SIDE.left ? 0 : right + GHOST_GAP,
      y: Math.max(0, Math.round(top)),
      width: ghost.width,
      height: ghost.height,
      manual: false,
    });
  }
  let full = 0;
  let height = placement.height;
  for (const box of boxes.values()) {
    full = Math.max(full, box.x + box.width);
    height = Math.max(height, box.y + box.height);
  }
  return { ...placement, boxes, width: full, height };
}

/** The room between a ghost column and the drawing beside it. */
const GHOST_GAP = 44;

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

/** The slot chip under the pointer, by its `data-slot` — §4.7's chip-onto-chip gesture. */
function slotUnder(clientX: number, clientY: number): string | null {
  const element = document.elementFromPoint(clientX, clientY);
  const chip = element === null ? null : (element as HTMLElement).closest('[data-slot]');
  return chip === null ? null : chip.getAttribute('data-slot');
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
  dimmed,
  truths,
  selection,
  flight,
  onSelect,
}: {
  wires: readonly CanvasWire[];
  routes: readonly WireRoute[];
  links: CanvasModel['links'];
  placement: Placement;
  /** The wires the scrubber dimmed — §4.8, S4's `.w.dim` and `.elbl.dim`. */
  dimmed?: ReadonlySet<string>;
  /** What each guard evaluates to where the scrubber stands — S4's `— false` beside it. */
  truths?: ReadonlyMap<string, boolean | null>;
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
          const classes = [
            'w',
            wire.interface ? 'iface' : '',
            wire.id === selection ? 'sel' : '',
            dimmed?.has(wire.id) === true ? 'dim' : '',
          ]
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
            className={dimmed?.has(wire.id) === true ? 'elbl dim' : 'elbl'}
            data-wire={wire.id}
            style={{ left: `${String(route.label.x)}px`, top: `${String(route.label.y)}px` }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(wire);
            }}
          >
            {wire.label}
            {wire.type === null ? null : <span className="wtype">{wire.type}</span>}
            {wire.guard === null ? null : (
              <span className="wguard" data-guard={wire.id}>
                {`⚑ ${wire.guard}`}
                {truths?.has(wire.id) === true ? (
                  <b className={truths.get(wire.id) === true ? 'holds' : 'fails'}>
                    {truths.get(wire.id) === null
                      ? ` — ${text('unresolved')}`
                      : ` — ${truths.get(wire.id) === true ? text('true') : text('false')}`}
                  </b>
                ) : null}
              </span>
            )}
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

/** §4.15's port menu, at the pointer, with §4.8's carry where the canvas is a drill-in. */
function PortMenu({
  side,
  indices,
  x,
  y,
  onRun,
}: {
  side: string;
  indices: readonly string[];
  x: number;
  y: number;
  onRun: (entry: MenuEntry) => void;
}): JSX.Element {
  return (
    <div className="ctxmenu" style={{ left: `${String(x)}px`, top: `${String(y)}px` }} role="menu">
      {portEntries(side, indices).map((entry) => (
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

/** §4.7's context menu, at the pointer. */
function ContextMenu({
  box,
  drill,
  x,
  y,
  onRun,
}: {
  box: CanvasBox;
  /** Whether the canvas is a drill-in: §4.20's entries stand where §4.7's three do not. */
  drill: boolean;
  x: number;
  y: number;
  onRun: (entry: MenuEntry) => void;
}): JSX.Element {
  return (
    <div className="ctxmenu" style={{ left: `${String(x)}px`, top: `${String(y)}px` }} role="menu">
      {(drill ? drillEntriesFor(box.role, ROLE) : entriesFor(box.role, ROLE)).map((entry) => (
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

/**
 * One box of the folded canvas: a node card (S2), a composition box (S3) or a terminal (S1).
 *
 * Everything it shows is a field of {@link CanvasBox}, which `model.ts` filled from the core; what
 * this file decides is which element each field becomes, and that decision is the one feature 2.5
 * measured: **a control is a `<button>`**. An artboard draws a `<span>`; a reader who cannot use a
 * pointer needs the name, the primitive, the guard, every family chip, every port handle, every
 * slot chip, the problem dot and the fold to be reachable and operable, and a `<span>` with a
 * click handler is none of those.
 *
 * The box itself is a `role="group"` and **not** focusable: a focusable element inside a widget is
 * the serious defect feature 2.5's axe pass named on the tab strip, and the answer there — a set
 * of buttons that say which one is current with `aria-current` — is the answer here. The name
 * button is the box's own tab stop and the arrow keys move between them.
 */
import type { JSX, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import { text, textWith } from '../shell/strings.js';

import { ROLE, SIDE, type CanvasBox, type CanvasSlot } from './model.js';

/** What a box is drawn with, and what each gesture on it calls. */
export interface BoxProps {
  readonly box: CanvasBox;
  readonly selected: boolean;
  readonly renaming: boolean;
  /** Whether a connection in flight is over one of this box's handles (S2's lit handle). */
  readonly litPort: string | null;
  /**
   * Whether the box is drawn as absent — §4.8's scrubber: "it dims every site and edge absent at
   * that index". S5 draws it as `.node.ghosted`, which is the same shape a site with no facts is
   * drawn in, for the same reason: the card is there and the iteration is not.
   */
  readonly dimmed?: boolean;
  /**
   * Whether the box is drawn **small** — S3's `.minode`, a composition expanded in place.
   *
   * "Expanded in place the sites and their scoped edges are drawn as in the drill-in, smaller"
   * (S3's own note). The drill-in draws cards; a site inside an open group box draws its name, its
   * primitive and its slots on one line, which is what leaves room for the box around them.
   */
  readonly mini?: boolean;
  /** The chip a tie in flight is over, by its `data-slot`; `null` when none is. */
  readonly litSlot?: string | null;
  readonly register: (element: HTMLElement | null) => void;
  readonly onSelect: () => void;
  readonly onRename: (to: string) => void;
  readonly onStartRename: () => void;
  readonly onFold: () => void;
  readonly onMenu: (x: number, y: number) => void;
  /**
   * The box was opened — a double-click, which §4.8 and S3 make the way into a composition
   * ("double-click or Ctrl+Enter drills in"). Feature 2.7 left the same gesture on the explorer's
   * own rows; here it is the box's.
   */
  readonly onOpen?: () => void;
  readonly onGrab: (event: ReactPointerEvent<HTMLElement>) => void;
  /** A port handle was pressed (the drag's start) or clicked (the keyboard's form of it). */
  readonly onPort: (port: string, side: 'inputs' | 'outputs', pressed: boolean) => void;
  /** A port handle's own menu — §4.15's "Expose as input…" and "Expose as output…". */
  readonly onPortMenu: (port: string, side: 'inputs' | 'outputs', x: number, y: number) => void;
  readonly onProblem: () => void;
  /**
   * A part of the card was clicked: §4.7's "edits that element in place".
   *
   * What is handed over is the thing's **own text** — the primitive it pins, the family, the
   * guard's condition, the slot's name — and not the member of the grammar it is written under.
   * The card knows what it drew; naming the member here would be the interface writing a word the
   * schemas own (§1 b, which caught exactly this).
   */
  readonly onChip: (what: string) => void;
  /**
   * A slot chip was pressed (the tie's start) or clicked (the keyboard's form of it).
   *
   * §4.7: "drag a slot chip onto another node's slot chip → proposes tying/sharing from the core's
   * compatibility list; creates or extends the identity". The chip's own click still selects the
   * identity, so the canvas decides which of the two a gesture was: a release on another chip is a
   * tie, a release on this one is the click.
   */
  readonly onSlot: (slot: CanvasSlot, pressed: boolean) => void;
  /** A boundary handle was clicked: the site it names is selected (§4.7, D8). */
  readonly onHandle: (site: string) => void;
}

/** The classes a slot chip carries, from what the core and D3 say about it (S2's six states). */
export function slotClass(slot: CanvasSlot): string {
  const classes = ['slot'];
  if (slot.kind === 'state') classes.push('st');
  else if (slot.identity === null) classes.push('unbound');
  else classes.push(slot.shared ? 'tied' : 'priv');
  if (slot.located) classes.push('located');
  return classes.join(' ');
}

/** The mark a slot chip is drawn with: a parameter, a constant, a state port (S2). */
export function slotMark(kind: string): string {
  if (kind === 'state') return '▣';
  if (kind === 'constant') return '●';
  return '◆';
}

/** One box, drawn. */
export function Box(props: BoxProps): JSX.Element {
  const { box } = props;
  const classes = [
    box.role === ROLE.group ? 'group' : box.role === ROLE.terminal ? 'term' : 'node',
    props.mini === true && box.role !== ROLE.group && box.role !== ROLE.terminal ? 'minode' : '',
    props.selected ? 'sel' : '',
    box.role === ROLE.group && !box.collapsed ? 'open' : '',
    (box.role === ROLE.node && !box.facts) || props.dimmed === true ? 'ghosted' : '',
    box.template ? 'tmpl' : '',
  ]
    .filter((one) => one !== '')
    .join(' ');

  return (
    <div
      ref={props.register}
      className={classes}
      data-box={box.pointer}
      role="group"
      aria-label={box.name}
      onClick={(event) => {
        event.stopPropagation();
        props.onSelect();
      }}
      onDoubleClick={(event) => {
        event.stopPropagation();
        props.onOpen?.();
      }}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        props.onMenu(event.clientX, event.clientY);
      }}
      onPointerDown={(event) => {
        // The card's own ground moves it; a control inside it belongs to the control.
        if (event.button !== 0) return;
        if ((event.target as HTMLElement).closest('button, input') !== null) return;
        props.onGrab(event);
      }}
    >
      {box.problem === null ? null : (
        <button
          type="button"
          className={`n-dot ${box.problem === 'error' ? '' : box.problem}`}
          data-dot={box.pointer}
          aria-label={textWith('{} problems here', String(box.problems))}
          onClick={(event) => {
            event.stopPropagation();
            props.onProblem();
          }}
        />
      )}
      {box.role === ROLE.terminal ? <Terminal {...props} /> : null}
      {box.role === ROLE.group ? <Group {...props} /> : null}
      {box.role !== ROLE.terminal && box.role !== ROLE.group ? (
        props.mini === true ? <MiniCard {...props} /> : <Card {...props} />
      ) : null}
    </div>
  );
}

/** The name, which is the box's tab stop and its rename affordance (§4.4's Q4 rule). */
function Name(props: BoxProps): JSX.Element {
  const { box } = props;
  if (props.renaming) {
    return (
      <input
        className="n-rename"
        defaultValue={box.name}
        autoFocus
        aria-label={text('Name')}
        onClick={(event) => {
          event.stopPropagation();
        }}
        onBlur={(event) => {
          props.onRename(event.currentTarget.value.trim());
        }}
        onKeyDown={(event: ReactKeyboardEvent<HTMLInputElement>) => {
          event.stopPropagation();
          if (event.key === 'Enter') props.onRename(event.currentTarget.value.trim());
          if (event.key === 'Escape') props.onRename('');
        }}
      />
    );
  }
  return (
    <button
      type="button"
      className="n-name"
      data-name={box.pointer}
      aria-current={props.selected}
      title={box.template ? text('an instance of a template primitive') : undefined}
      onClick={(event) => {
        event.stopPropagation();
        // A double-click **opens** the box — §4.8's drill-in, S3's own caption — and the name's
        // own editor is the slow second click. `detail` is the click count, which is where the
        // browser tells them apart; without it the name is replaced by its input before the
        // double-click can be dispatched and the gesture is lost.
        if (event.detail >= 2 && props.onOpen !== undefined) {
          props.onOpen();
          return;
        }
        // §4.5's rule, taken again: the first click selects, and a click on the name of a row
        // that is already selected opens the editor. A name that renamed on the first click would
        // rename on the click that only meant to select.
        if (props.selected) props.onStartRename();
        else props.onSelect();
      }}
    >
      {box.template ? <span aria-hidden="true">{TEMPLATE} </span> : null}
      {box.name}
    </button>
  );
}

/** The mark a template instance carries — §4.7's own, and the Model explorer's (feature 2.7). */
const TEMPLATE = '▣';

/** The mark an identity with several members carries — S2's `⇄ ◆ weight`. */
const SHARED = '⇄';

/** An interface terminal: `◁ tokens · token` (S1). */
function Terminal(props: BoxProps): JSX.Element {
  const { box } = props;
  return (
    <>
      <span aria-hidden="true">{box.side === SIDE.left ? '◁' : '▷'}</span>
      <Name {...props} />
      {box.badges.map((badge) => (
        <span key={badge} className="tkind">
          {badge}
        </span>
      ))}
    </>
  );
}

/** A composition box: the header, the boundary handles, the summary and the derived line (S3). */
function Group(props: BoxProps): JSX.Element {
  const { box } = props;
  return (
    <>
      <div className="g-head">
        <Name {...props} />
        {box.range === null ? null : <span className="g-range">{box.range}</span>}
        {box.count === null ? null : (
          <span className={box.count.endsWith('?') ? 'g-count unres' : 'g-count'}>{box.count}</span>
        )}
        <button
          type="button"
          className="g-fold"
          data-fold={box.pointer}
          aria-expanded={!box.collapsed}
          aria-label={textWith('Expand {}', box.name)}
          onClick={(event) => {
            event.stopPropagation();
            props.onFold();
          }}
        >
          {box.collapsed ? '▸' : '▾'}
        </button>
      </div>
      {box.families.length === 0 ? null : (
        <div className="g-fams">
          {box.families.map((family) => (
            <button
              key={family}
              type="button"
              className="fchip"
              onClick={(event) => {
                event.stopPropagation();
                props.onChip(family);
              }}
            >
              {family}
            </button>
          ))}
        </div>
      )}
      {!box.collapsed ? null : (
        <div className="g-body">
          {([SIDE.left, SIDE.right] as const).map((side) => (
            <div key={side} className={side === SIDE.left ? 'g-handles left' : 'g-handles right'}>
              {box.handles
                .filter((handle) => handle.side === side)
                .map((handle) => (
                  <button
                    key={handle.label}
                    type="button"
                    className="bh"
                    data-handle={handle.label}
                    onClick={(event) => {
                      event.stopPropagation();
                      props.onHandle(handle.site);
                    }}
                  >
                    {handle.label}
                  </button>
                ))}
            </div>
          ))}
        </div>
      )}
      {box.held === null ? null : <div className="g-sites">{box.held}</div>}
      {box.derived === null ? null : (
        <div className={box.stale ? 'g-der stale' : 'g-der'}>
          {box.derived}
          {box.stale ? <span className="stalebdg">{text('stale')}</span> : null}
        </div>
      )}
    </>
  );
}

/**
 * A site drawn small — S3's `.minode`, a composition expanded in place.
 *
 * The board writes `<b>attn</b><span>attention.dense · ◆ q k v out · ▣ kv</span>`: the name, the
 * primitive and the slot marks on one line, and nothing else. Every one of them is still a
 * control, because §4.4's rule does not stop being true because a card is small — the name
 * renames, the primitive opens its row, a chip selects its identity — and the ports are reached
 * in the drill-in, which is where §4.8 puts them (Ctrl+Enter, or a double-click).
 */
function MiniCard(props: BoxProps): JSX.Element {
  const { box } = props;
  return (
    <>
      <Name {...props} />
      <span className="mi-line">
        {box.primitive === null ? null : (
          <button
            type="button"
            className="mi-prim"
            data-primitive={box.pointer}
            onClick={(event) => {
              event.stopPropagation();
              props.onChip(box.primitive ?? '');
            }}
          >
            {box.primitive}
          </button>
        )}
        {box.guard === null ? null : (
          <button
            type="button"
            className="n-guard"
            data-guard={box.pointer}
            onClick={(event) => {
              event.stopPropagation();
              props.onChip(box.guard ?? '');
            }}
          >
            <span aria-hidden="true">⚑</span>
            {box.guard}
          </button>
        )}
        {box.slots.map((slot) => (
          <button
            key={slot.name}
            type="button"
            className={slotClass(slot)}
            data-slot={`${box.pointer}:${slot.name}`}
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              event.stopPropagation();
              props.onSlot(slot, true);
            }}
            onClick={(event) => {
              event.stopPropagation();
              props.onSlot(slot, false);
            }}
          >
            <span aria-hidden="true">{slotMark(slot.kind)}</span>
            {slot.shared ? <span aria-hidden="true">{SHARED}</span> : null}
            {slot.name}
          </button>
        ))}
      </span>
    </>
  );
}

/** A node card: the whole of S2. */
function Card(props: BoxProps): JSX.Element {
  const { box } = props;
  return (
    <>
      <div className="n-head">
        <Name {...props} />
        {box.primitive === null ? null : (
          <button
            type="button"
            className="n-prim"
            data-primitive={box.pointer}
            onClick={(event) => {
              event.stopPropagation();
              props.onChip(box.primitive ?? '');
            }}
          >
            {box.primitive}
            <em>{box.version === null ? '' : `@${box.version}`}</em>
          </button>
        )}
      </div>
      {box.guard === null && box.families.length === 0 ? null : (
        <div className="n-meta">
          {box.guard === null ? null : (
            <button
              type="button"
              className="n-guard"
              data-guard={box.pointer}
              onClick={(event) => {
                event.stopPropagation();
                props.onChip(box.guard ?? '');
              }}
            >
              <span aria-hidden="true">⚑</span>
              {textWith('when {}', box.guard)}
            </button>
          )}
          {box.families.map((family) => (
            <button
              key={family}
              type="button"
              className="fchip"
              onClick={(event) => {
                event.stopPropagation();
                props.onChip(family);
              }}
            >
              {family}
            </button>
          ))}
        </div>
      )}
      {box.summary === '' ? null : <div className="n-args">{box.summary}</div>}
      {box.inputs.length === 0 && box.outputs.length === 0 ? null : (
        <div className="n-ports">
          <div className="col">
            {box.inputs.map((port) => (
              <Handle key={port.name} {...props} port={port.name} side="inputs" state={port.state} title={port.title} />
            ))}
          </div>
          <div className="col r">
            {box.outputs.map((port) => (
              <Handle key={port.name} {...props} port={port.name} side="outputs" state={port.state} title={port.title} />
            ))}
          </div>
        </div>
      )}
      {box.slots.length === 0 ? null : (
        <div className="n-slots">
          {box.slots.map((slot) => (
            <button
              key={`${slot.kind}:${slot.name}`}
              type="button"
              className={slotClass(slot) + (props.litSlot === `${box.pointer}:${slot.name}` ? ' lit' : '')}
              data-slot={`${box.pointer}:${slot.name}`}
              data-slot-kind={slot.kind}
              title={slot.identity ?? text('bound by nothing')}
              onPointerDown={(event) => {
                event.stopPropagation();
                props.onSlot(slot, true);
              }}
              onClick={(event) => {
                event.stopPropagation();
                props.onSlot(slot, false);
              }}
            >
              {`${slot.shared ? `${SHARED} ` : ''}${slotMark(slot.kind)} ${slot.name}`}
            </button>
          ))}
        </div>
      )}
      {box.derived === null ? null : (
        <div className={box.stale ? 'n-der stale' : 'n-der'}>
          {box.derived}
          {box.stale ? <span className="stalebdg">{text('stale')}</span> : null}
        </div>
      )}
    </>
  );
}

/** One port handle: hollow red when V7 says nothing feeds it, amber when V13 says nothing uses it. */
function Handle(
  props: BoxProps & {
    port: string;
    side: 'inputs' | 'outputs';
    state: string;
    title: string;
  },
): JSX.Element {
  const lit = props.litPort === props.port;
  const classes = ['p', props.side === 'inputs' ? 'in' : 'out', props.state === 'bound' ? '' : props.state, lit ? 'lit' : '']
    .filter((one) => one !== '')
    .join(' ');
  return (
    <button
      type="button"
      className={classes}
      data-port={`${props.box.pointer}:${props.port}`}
      data-side={props.side}
      title={props.title}
      aria-label={`${props.port} — ${props.title}`}
      onPointerDown={(event) => {
        event.stopPropagation();
        if (event.button !== 0) return;
        props.onPort(props.port, props.side, true);
      }}
      onClick={(event) => {
        event.stopPropagation();
        props.onPort(props.port, props.side, false);
      }}
      onContextMenu={(event) => {
        // The port's own menu, not the card's: §4.15's two gestures are about *this* port, and a
        // menu that belonged to the box would have to name the port some other way.
        event.preventDefault();
        event.stopPropagation();
        props.onPortMenu(props.port, props.side, event.clientX, event.clientY);
      }}
    >
      {props.port}
    </button>
  );
}

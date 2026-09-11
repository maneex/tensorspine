/**
 * A region's edge — §4.2's "resizable regions with remembered sizes".
 *
 * No artboard draws one: a printed shell has no drag. What it must be is decided by §4.21 rather
 * than by a picture — "every command reachable from the keyboard" — so it is a `separator` that
 * takes focus and moves on the arrow keys as well as under the pointer, and it reports where it
 * stands so that what it is doing can be heard as well as seen.
 *
 * Pointer capture rather than a window listener: the element keeps the events until the button
 * comes up, so a drag that leaves the window and comes back is still the same drag, and there is
 * nothing to unsubscribe if the shell unmounts mid-gesture.
 */
import { useRef, type JSX, type KeyboardEvent, type PointerEvent } from 'react';

import { text } from './strings.js';

/** How far an arrow key moves an edge, and how far with Shift. */
const STEP = 16;
const LEAP = 64;

export interface SplitterProps {
  /** Which way the edge runs: a vertical edge changes a width, a horizontal one a height. */
  readonly orientation: 'vertical' | 'horizontal';
  /**
   * Which edge of its region it sits on.
   *
   * The splitter is rendered **inside** the region it resizes and pinned to one of its edges,
   * rather than beside it in the row. Two reasons, and the second is the one that decided it:
   * it cannot then disturb the flow of a row whose widths are the point, and it is inside the
   * landmark its region is — a control floating between landmarks is content no landmark
   * contains, which an accessibility audit names and a screen reader loses.
   */
  readonly at: 'left' | 'right' | 'top';
  readonly size: number;
  readonly min: number;
  readonly max: number;
  /** What the person resizing is resizing, for the label they hear. */
  readonly label: string;
  /**
   * Whether a larger size lies towards larger coordinates.
   *
   * The side bar grows as the pointer moves right and the Properties region grows as it moves
   * left; the bottom panel grows as it moves up. One flag rather than three components.
   */
  readonly grows: 1 | -1;
  readonly onSize: (size: number) => void;
}

export function Splitter({ orientation, at, size, min, max, label, grows, onSize }: SplitterProps): JSX.Element {
  const from = useRef<{ at: number; size: number } | null>(null);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0) return;
    from.current = { at: orientation === 'vertical' ? event.clientX : event.clientY, size };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>): void => {
    const start = from.current;
    if (start === null) return;
    const at = orientation === 'vertical' ? event.clientX : event.clientY;
    onSize(start.size + (at - start.at) * grows);
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>): void => {
    from.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const step = event.shiftKey ? LEAP : STEP;
    const less = orientation === 'vertical' ? 'ArrowLeft' : 'ArrowUp';
    const more = orientation === 'vertical' ? 'ArrowRight' : 'ArrowDown';
    const towards = event.key === less ? -1 : event.key === more ? 1 : undefined;
    if (towards !== undefined) {
      event.preventDefault();
      onSize(size + towards * step * grows);
      return;
    }
    if (event.key === 'Home') {
      event.preventDefault();
      onSize(min);
    } else if (event.key === 'End') {
      event.preventDefault();
      onSize(max);
    }
  };

  return (
    <div
      className={`splitter ${orientation} at-${at}`}
      role="separator"
      tabIndex={0}
      aria-orientation={orientation}
      aria-label={`${text('Resize')} ${label}`}
      aria-valuenow={Math.round(size)}
      aria-valuemin={min}
      aria-valuemax={max}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    />
  );
}

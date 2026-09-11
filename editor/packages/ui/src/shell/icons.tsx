/**
 * The shell's icons — the drawings of S1, copied.
 *
 * Every path below is the artboard's own (`plans/graph-editor-design/body_S1.html`, the `.rail`'s
 * five activities and the `.search` magnifier), at the artboard's stroke width and viewBox. They
 * are decorative: an icon-only control carries its name in `aria-label`, so each `<svg>` is
 * hidden from assistive technology rather than given a title nobody would want read twice.
 *
 * The sixth, the palette's, is the same magnifier at the same weight: §4.2's command palette has
 * no artboard, and inventing a mark for it would be inventing chrome.
 */
import type { JSX } from 'react';

/**
 * The stroked line-art every rail icon is drawn in.
 *
 * The board writes `fill`, `stroke` and `stroke-width` as attributes because a board is static
 * HTML; here they are one rule of `shell.css` (`.ico`) and the element carries geometry alone.
 * Better code, and it has to be: `fill="none"` would write `none` — a value of the model schema's
 * `mask` and of `partition_options` — into interface source, which catching rule (b) of §1
 * forbids. Feature 2.4 met the same collision in `style.display = 'none'` and reached the same
 * kind of answer.
 */
function Line({ children, box = '0 0 20 20' }: { children: JSX.Element; box?: string }): JSX.Element {
  return (
    <svg className="ico" viewBox={box} aria-hidden="true">
      {children}
    </svg>
  );
}

/** Model explorer: three sites and the edges between them. */
export function GraphIcon(): JSX.Element {
  return (
    <Line>
      <g>
        <circle cx="4" cy="10" r="2.2" />
        <circle cx="15" cy="4.5" r="2.2" />
        <circle cx="15" cy="15.5" r="2.2" />
        <path d="M6 9 13 5.4M6 11l7 3.6" />
      </g>
    </Line>
  );
}

/** Library: the stacked units of a base. */
export function LibraryIcon(): JSX.Element {
  return (
    <Line>
      <g>
        <path d="M10 2.6 17.4 6.4 10 10.2 2.6 6.4Z" />
        <path d="M2.6 10 10 13.8 17.4 10" />
        <path d="M2.6 13.6 10 17.4 17.4 13.6" />
      </g>
    </Line>
  );
}

/** Weights: a tensor's grid. */
export function WeightsIcon(): JSX.Element {
  return (
    <Line>
      <g>
        <rect x="2.8" y="2.8" width="14.4" height="14.4" rx="1.4" />
        <path d="M7.6 2.8v14.4M12.4 2.8v14.4M2.8 7.6h14.4M2.8 12.4h14.4" />
      </g>
    </Line>
  );
}

/** Search. */
export function SearchIcon(): JSX.Element {
  return (
    <Line>
      <g>
        <circle cx="8.8" cy="8.8" r="5.2" />
        <path d="m12.7 12.7 4 4" />
      </g>
    </Line>
  );
}

/** Settings: the sliders at the foot of the rail. */
export function SettingsIcon(): JSX.Element {
  return (
    <Line>
      <g>
        <path d="M3 6h14M3 14h14" />
        <circle cx="8" cy="6" r="2" />
        <circle cx="13" cy="14" r="2" />
      </g>
    </Line>
  );
}

/** Account: shown only where the deployment has a session (Q8 defers the SaaS). */
export function AccountIcon(): JSX.Element {
  return (
    <Line>
      <g>
        <circle cx="10" cy="7" r="3.2" />
        <path d="M3.6 17c0-3.3 2.9-5.2 6.4-5.2s6.4 1.9 6.4 5.2" />
      </g>
    </Line>
  );
}

/** The magnifier of the side bar's filter and of the palette's query, at the board's own size. */
export function FilterIcon(): JSX.Element {
  return (
    <svg className="ico" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="7" cy="7" r="4.2" />
      <path d="m10.1 10.1 3.2 3.2" />
    </svg>
  );
}

/**
 * The activity rail and the side bar — S1's `.rail` and `.side`, §4.2's 48 px and 264 px.
 *
 * **The rail navigates; the side bar is what it navigates to.** Not a `tablist` and a `tabpanel`,
 * though that is the shape the arrangement suggests: an `aside` cannot take the `tabpanel` role
 * (its own is `complementary`), and a `nav` given `tablist` stops being a landmark — so the pair
 * would cost two of the three landmarks the shell has and gain nothing a screen reader needs. A
 * rail of buttons that say which one is showing, and a complementary region beside them, is the
 * same interface with its own roles rather than borrowed ones. Measured, not reasoned: both were
 * built, and axe named the second.
 *
 * Settings sits at the foot, and Account beside it only where the deployment has a session (§4.2
 * marks it "(SaaS)", and Q8 defers the SaaS, so `NoAuth` never shows it).
 *
 * The activities themselves — the Model explorer's tree, the Library's palette, the Weights
 * panel, Search — are features 2.7, 3.1 and 4.2. What the side bar shows until then is the name
 * of the activity, which is the honest reading of §4.3's "what the editor shows before it has
 * anything to show".
 */
import type { JSX } from 'react';

import { useShell, useShellStore } from './context.js';
import {
  AccountIcon,
  FilterIcon,
  GraphIcon,
  LibraryIcon,
  SearchIcon,
  SettingsIcon,
  WeightsIcon,
} from './icons.js';
import { ACTIVITIES, BOUNDS, type ActivityId } from './regions.js';
import { Splitter } from './Splitter.js';
import { text } from './strings.js';

/** The mark on the rail for each activity. */
const MARKS: Readonly<Record<ActivityId, () => JSX.Element>> = {
  'activity.explorer': GraphIcon,
  'activity.library': LibraryIcon,
  'activity.weights': WeightsIcon,
  'activity.search': SearchIcon,
  'activity.settings': SettingsIcon,
  'activity.account': AccountIcon,
};

/** The rail: the four activities, then Settings (and Account, with a session) at the foot. */
export function Rail(): JSX.Element {
  const store = useShellStore();
  const activity = useShell((state) => state.side.activity);
  const open = useShell((state) => state.side.open);
  const session = useShell((state) => state.session);
  const shown = ACTIVITIES.filter((one) => one.needsSession !== true || session !== null);

  return (
    <nav className="rail" aria-label={text('Activities')}>
      {shown.map((one) => {
        const Mark = MARKS[one.id];
        const showing = one.id === activity && open;
        return (
          <button
            key={one.id}
            type="button"
            aria-pressed={showing}
            aria-label={text(one.label)}
            title={text(one.label)}
            className={one.foot === true ? 'act foot' : 'act'}
            data-activity={one.id}
            onClick={() => {
              // Clicking the activity that is already showing closes the side bar, and clicking
              // it again brings it back: the VS Code gesture, and the one way to get the width
              // back without going to the View menu.
              if (showing) store.getState().toggleSide(false);
              else store.getState().setActivity(one.id);
            }}
          >
            <Mark />
          </button>
        );
      })}
    </nav>
  );
}

/** The side bar: the selected activity's panel, with the edge that resizes it. */
export function Side(): JSX.Element {
  const store = useShellStore();
  const activity = useShell((state) => state.side.activity);
  const width = useShell((state) => state.side.width);
  const one = ACTIVITIES.find((each) => each.id === activity) ?? ACTIVITIES[0];
  const label = one === undefined ? '' : text(one.label);

  return (
    <aside className="side" aria-label={label} style={{ width }} data-width={width}>
      <div className="side-head">{label}</div>
      <label className="search">
        <FilterIcon />
        <input type="search" placeholder={text('Filter…')} aria-label={text('Filter…')} />
      </label>
      <div className="side-body" tabIndex={0}>
        <p className="note-line">{text('Nothing to show here yet.')}</p>
      </div>
      {/* The edge belongs to the region it resizes: inside it, so that it is part of the same
          landmark and cannot disturb the row's own flow. */}
      <Splitter
        orientation="vertical"
        at="right"
        size={width}
        min={BOUNDS.side.min}
        max={BOUNDS.side.max}
        grows={1}
        label={label}
        onSize={(next) => {
          store.getState().setSideWidth(next);
        }}
      />
    </aside>
  );
}

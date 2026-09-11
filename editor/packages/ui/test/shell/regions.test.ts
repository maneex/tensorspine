import { describe, expect, it } from 'vitest';

import {
  ACTIVITIES,
  BOUNDS,
  clampSize,
  defaultRegions,
  defaultSide,
  movePanel,
  PANELS,
  panelById,
  RAIL_WIDTH,
  regionOf,
  showPanel,
} from '../../src/shell/regions.js';
import { readLayout } from '../../src/shell/store.js';

// §4.2's region table, and the two rules the component inventory adds to it: the bottom panel has
// **exactly three** tabs, and every panel is a tab that can be dragged to the other region.

describe('the regions of §4.2', () => {
  it('opens at the sizes the plan states', () => {
    expect(RAIL_WIDTH).toBe(48);
    expect(defaultSide().width).toBe(264);
    expect(BOUNDS['region.right'].initial).toBe(340);
    expect(BOUNDS['region.bottom'].initial).toBe(200);
    // §4.2: Properties is "resizable to 520".
    expect(BOUNDS['region.right'].max).toBe(520);
  });

  it('opens with the side bar, Properties and the bottom panel all showing', () => {
    expect(defaultSide().open).toBe(true);
    const regions = defaultRegions();
    expect(regions['region.bottom'].open).toBe(true);
    expect(regions['region.right'].open).toBe(true);
  });

  it('shows Model explorer first, and puts Settings and Account at the foot', () => {
    expect(defaultSide().activity).toBe('activity.explorer');
    expect(ACTIVITIES.filter((one) => one.foot !== true).map((one) => one.label)).toEqual([
      'Model explorer',
      'Library',
      'Weights',
      'Search',
    ]);
    expect(ACTIVITIES.filter((one) => one.foot === true).map((one) => one.label)).toEqual([
      'Settings',
      'Account',
    ]);
    // Q8 defers the SaaS, so Account is the one that waits for a session.
    expect(ACTIVITIES.filter((one) => one.needsSession === true).map((one) => one.id)).toEqual([
      'activity.account',
    ]);
  });

  it('gives the bottom panel exactly three tabs, and Problems is the one showing', () => {
    const bottom = defaultRegions()['region.bottom'];
    expect(bottom.panels.map((id) => panelById(id)?.label)).toEqual(['Problems', 'Derived', 'Log']);
    expect(bottom.active).toBe('panel.problems');
  });

  it('has four panels and no fifth: the design pass’s fourth bottom tab was rejected', () => {
    expect(PANELS.map((panel) => panel.label)).toEqual(['Problems', 'Derived', 'Log', 'Properties']);
    expect(defaultRegions()['region.right'].panels).toEqual(['panel.properties']);
  });

  it('keeps a size inside its own bounds', () => {
    expect(clampSize('side', 10)).toBe(BOUNDS.side.min);
    expect(clampSize('side', 10_000)).toBe(BOUNDS.side.max);
    expect(clampSize('region.right', 600)).toBe(520);
    expect(clampSize('region.bottom', 217.4)).toBe(217);
  });
});

describe('dragging a panel to the other region', () => {
  it('moves it there, shows it, and leaves the region it came from with the rest', () => {
    const moved = movePanel(defaultRegions(), 'panel.derived', 'region.right');
    expect(moved['region.bottom'].panels).toEqual(['panel.problems', 'panel.log']);
    expect(moved['region.right'].panels).toEqual(['panel.properties', 'panel.derived']);
    expect(moved['region.right'].active).toBe('panel.derived');
    expect(regionOf(moved, 'panel.derived')).toBe('region.right');
  });

  it('moves the tab that was showing and leaves another showing behind it', () => {
    const moved = movePanel(defaultRegions(), 'panel.problems', 'region.right');
    expect(moved['region.bottom'].active).toBe('panel.derived');
  });

  it('closes a region a move emptied, and opens the one it filled', () => {
    let regions = defaultRegions();
    regions = movePanel(regions, 'panel.properties', 'region.bottom');
    expect(regions['region.right'].panels).toEqual([]);
    expect(regions['region.right'].open).toBe(false);
    expect(regions['region.bottom'].panels).toHaveLength(4);
    expect(regions['region.bottom'].active).toBe('panel.properties');

    regions = { ...regions, 'region.bottom': { ...regions['region.bottom'], open: false } };
    regions = movePanel(regions, 'panel.log', 'region.right');
    expect(regions['region.right'].open).toBe(true);
  });

  it('moves a panel onto the region it is already in by showing it', () => {
    const regions = movePanel(defaultRegions(), 'panel.log', 'region.bottom');
    expect(regions['region.bottom'].panels).toEqual(['panel.problems', 'panel.derived', 'panel.log']);
    expect(regions['region.bottom'].active).toBe('panel.log');
  });

  it('shows a panel wherever it sits, and opens its region to do it', () => {
    const closed = { ...defaultRegions() };
    closed['region.bottom'] = { ...closed['region.bottom'], open: false };
    const shown = showPanel(closed, 'panel.log');
    expect(shown['region.bottom'].open).toBe(true);
    expect(shown['region.bottom'].active).toBe('panel.log');
  });
});

describe('the arrangement read back out of the settings', () => {
  it('answers the defaults for anything that is not an arrangement', () => {
    for (const stored of [undefined, null, 7, 'x', [], {}] as const) {
      expect(readLayout(stored), JSON.stringify(stored ?? null)).toEqual({
        side: defaultSide(),
        regions: defaultRegions(),
      });
    }
  });

  it('keeps what it can read and clamps a size that left its bounds', () => {
    const read = readLayout({
      side: { open: false, width: 9000, activity: 'activity.library' },
      regions: { 'region.bottom': { open: false, size: 1, panels: ['panel.log'], active: 'panel.log' } },
    });
    expect(read.side).toEqual({ open: false, width: BOUNDS.side.max, activity: 'activity.library' });
    expect(read.regions['region.bottom'].size).toBe(BOUNDS['region.bottom'].min);
    expect(read.regions['region.bottom'].active).toBe('panel.log');
  });

  it('drops an activity and a panel this editor does not have', () => {
    const read = readLayout({
      side: { activity: 'activity.nonesuch' },
      regions: { 'region.right': { panels: ['panel.nonesuch', 'panel.properties'], active: 'panel.nonesuch' } },
    });
    expect(read.side.activity).toBe('activity.explorer');
    expect(read.regions['region.right'].panels).toEqual(['panel.properties']);
    expect(read.regions['region.right'].active).toBe('panel.properties');
  });

  it('brings a panel the stored arrangement lost back to the region it calls home', () => {
    const read = readLayout({
      regions: {
        'region.bottom': { panels: ['panel.problems'], active: 'panel.problems' },
        'region.right': { panels: ['panel.properties'], active: 'panel.properties' },
      },
    });
    const all = Object.values(read.regions).flatMap((region) => region.panels);
    expect([...all].sort()).toEqual([...PANELS.map((panel) => panel.id)].sort());
    expect(read.regions['region.bottom'].panels).toContain('panel.derived');
  });

  it('keeps a panel the user moved where the user moved it', () => {
    const moved = movePanel(defaultRegions(), 'panel.derived', 'region.right');
    const read = readLayout({ side: defaultSide(), regions: moved });
    expect(read.regions).toEqual(moved);
  });
});

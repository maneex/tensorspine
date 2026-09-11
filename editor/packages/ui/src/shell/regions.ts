/**
 * The shell's regions — plan §4.2's table, as a model the chrome renders and the settings
 * remember.
 *
 * | Region | Default |
 * |---|---|
 * | Activity bar (left) | 48 px, Model explorer selected |
 * | Side bar (left) | 264 px, open |
 * | Editor area | tabs, the rest of the width |
 * | Properties (right) | 340 px, open, resizable to 520 |
 * | Bottom panel | 200 px, Problems |
 *
 * **A panel is a tab, and a tab can move.** §4.2: "Every panel is a tab that can be dragged to
 * another region (Properties to the bottom, Derived to the right) — the VS Code convention;
 * layouts are saved in settings." So a *panel* is one of four things the editor can show beside
 * the canvas, a *region* is one of two places a panel can sit, and the arrangement is a map from
 * the second to the first that the user changes by dragging. The bottom panel's own default is
 * the component inventory's "**exactly three tabs**: Problems, Derived, Log" — three, and never a
 * fourth invented one; Properties arrives there only because the user dragged it.
 *
 * **Sizes are the plan's, and the bounds are the shell's.** A region that can be dragged to
 * nothing is a region the user loses; each carries the smallest width or height at which it still
 * reads, and Properties carries the 520 px §4.2 states as its maximum.
 */

/** One activity of the rail (§4.2: "Model explorer, Library, Weights, Search; below: Settings"). */
export type ActivityId =
  | 'activity.explorer'
  | 'activity.library'
  | 'activity.weights'
  | 'activity.search'
  | 'activity.settings'
  | 'activity.account';

/** One panel that can sit in a region. */
export type PanelId = 'panel.problems' | 'panel.derived' | 'panel.log' | 'panel.properties';

/** One place a panel can sit. */
export type RegionId = 'region.bottom' | 'region.right';

/** An activity and what the rail says about it. */
export interface Activity {
  readonly id: ActivityId;
  /** The English label — the side bar's heading, the rail's tooltip, the i18n key. */
  readonly label: string;
  /** At the foot of the rail rather than in the list (§4.2's "below: Settings, Account"). */
  readonly foot?: true;
  /** Shown only where the deployment has a session (Q8 defers the SaaS; `NoAuth` has none). */
  readonly needsSession?: true;
}

/** The rail, in §4.2's order. */
export const ACTIVITIES: readonly Activity[] = [
  { id: 'activity.explorer', label: 'Model explorer' },
  { id: 'activity.library', label: 'Library' },
  { id: 'activity.weights', label: 'Weights' },
  { id: 'activity.search', label: 'Search' },
  { id: 'activity.settings', label: 'Settings', foot: true },
  { id: 'activity.account', label: 'Account', foot: true, needsSession: true },
];

/** A panel and what its tab says. */
export interface Panel {
  readonly id: PanelId;
  /** The English label — the tab, the View menu's command, the i18n key. */
  readonly label: string;
  /** Where it sits until the user moves it. */
  readonly home: RegionId;
}

/**
 * The four panels.
 *
 * Three of them are the bottom panel's, and the fourth is Properties — which §4.2 puts on the
 * right and names as the example of a panel that can be dragged to the bottom. There is no fifth:
 * the design pass proposed one for the alternation strip and it was **rejected** (component
 * inventory §6), the strip sitting below the canvas in the drill-in instead.
 */
export const PANELS: readonly Panel[] = [
  { id: 'panel.problems', label: 'Problems', home: 'region.bottom' },
  { id: 'panel.derived', label: 'Derived', home: 'region.bottom' },
  { id: 'panel.log', label: 'Log', home: 'region.bottom' },
  { id: 'panel.properties', label: 'Properties', home: 'region.right' },
];

/** The state of one region: which panels are in it, which is showing, whether it is open, and how big. */
export interface RegionState {
  readonly open: boolean;
  /** The height of the bottom region, the width of the right one, in CSS pixels. */
  readonly size: number;
  readonly panels: readonly PanelId[];
  /** The panel whose body is showing, or `null` when the region holds none. */
  readonly active: PanelId | null;
}

/** How far a region may be dragged, and where it starts. */
export interface RegionBounds {
  readonly initial: number;
  readonly min: number;
  readonly max: number;
}

/**
 * The sizes of §4.2, and the bounds the shell adds.
 *
 * The side bar's 264 px and Properties' 340 px are "the site's `--nav-w`" and "the site's
 * `--insp-w`"; the 520 px maximum is §4.2's own ("resizable to 520"), and the bottom panel's
 * 200 px likewise. The minima and the other maxima are this feature's: a region has to keep
 * enough width for a path or a figure to read, and has to leave the editor area something.
 */
export const BOUNDS: Readonly<Record<'side' | 'region.right' | 'region.bottom', RegionBounds>> = {
  side: { initial: 264, min: 180, max: 520 },
  'region.right': { initial: 340, min: 240, max: 520 },
  'region.bottom': { initial: 200, min: 80, max: 640 },
};

/** The width of the activity rail (§4.2: "Activity bar (left, 48 px)"). */
export const RAIL_WIDTH = 48;

/** The side bar: the selected activity's panel, open by default. */
export interface SideState {
  readonly open: boolean;
  readonly width: number;
  readonly activity: ActivityId;
}

/** Where every panel sits, and how each region is showing them. */
export type Regions = Readonly<Record<RegionId, RegionState>>;

/** The arrangement §4.2 opens with. */
export function defaultSide(): SideState {
  return { open: true, width: BOUNDS.side.initial, activity: 'activity.explorer' };
}

/** The arrangement §4.2 opens with: three tabs at the bottom, Properties on the right. */
export function defaultRegions(): Regions {
  const inRegion = (region: RegionId): PanelId[] =>
    PANELS.filter((panel) => panel.home === region).map((panel) => panel.id);
  return {
    'region.bottom': {
      open: true,
      size: BOUNDS['region.bottom'].initial,
      panels: inRegion('region.bottom'),
      active: 'panel.problems',
    },
    'region.right': {
      open: true,
      size: BOUNDS['region.right'].initial,
      panels: inRegion('region.right'),
      active: 'panel.properties',
    },
  };
}

/** Which region a panel is in, or `undefined` if this arrangement has lost it. */
export function regionOf(regions: Regions, panel: PanelId): RegionId | undefined {
  return (Object.keys(regions) as RegionId[]).find((region) =>
    regions[region].panels.includes(panel),
  );
}

/** The panel with that identity. */
export function panelById(id: PanelId): Panel | undefined {
  return PANELS.find((panel) => panel.id === id);
}

/** Keep a size inside its region's bounds. */
export function clampSize(region: RegionId | 'side', size: number): number {
  const bounds = BOUNDS[region];
  return Math.min(bounds.max, Math.max(bounds.min, Math.round(size)));
}

/**
 * Move a panel into a region, and answer the arrangement that results.
 *
 * The panel leaves wherever it was — a panel is in one place — and lands at the end of the
 * region's tabs, showing. A region left with nothing is closed rather than shown empty, and a
 * region that gains a panel is opened, because a drag that put a panel somewhere invisible would
 * read as a drag that lost it. A move onto the region a panel is already in changes nothing but
 * which tab is showing.
 */
export function movePanel(regions: Regions, panel: PanelId, to: RegionId): Regions {
  const next: Record<RegionId, RegionState> = { ...regions };
  for (const id of Object.keys(next) as RegionId[]) {
    const state = next[id];
    if (id === to) continue;
    if (!state.panels.includes(panel)) continue;
    const panels = state.panels.filter((one) => one !== panel);
    next[id] = {
      ...state,
      panels,
      active: state.active === panel ? (panels[0] ?? null) : state.active,
      open: panels.length > 0 && state.open,
    };
  }
  const target = next[to];
  const panels = target.panels.includes(panel) ? target.panels : [...target.panels, panel];
  next[to] = { ...target, panels, active: panel, open: true };
  return next;
}

/** Show a panel: select its tab, open its region, and answer where it was found. */
export function showPanel(regions: Regions, panel: PanelId): Regions {
  const region = regionOf(regions, panel);
  if (region === undefined) return regions;
  return { ...regions, [region]: { ...regions[region], active: panel, open: true } };
}

/**
 * The shell's own state — plan §5.1's "Zustand + Immer for the store", read the way feature 2.1
 * left it to be read.
 *
 * **Zustand, and exactly where.** Feature 2.1 built `DocumentStore` framework-free, with a plain
 * `subscribe`, and said so: "§5.1's stack names Zustand, which is the shell's binding to make
 * (2.5) over this, not a dependency the store needs to be edited and tested". The binding is
 * here, and it is a division rather than a wrapper:
 *
 *  - **the document is the document store's**, and D1 is why — "the editor's store holds the
 *    `tensorspine/2.0` JSON … there is no second graph model". A copy of it here would be that
 *    second model, so there is none: nothing below holds a quantity, an instance, a binding or a
 *    problem. `test/shell/store.test.ts` snapshots this state's own keys so that a field which
 *    started holding the document would be a reviewed diff and not a discovery;
 *  - **the chrome is this store's** — which activity, how wide the side bar is, which panel sits
 *    where, which tabs are open, which theme, whether the palette is up. None of it is in the
 *    document (D6's rule one level up: "the grammar closes every object … the language would
 *    refuse it, rightly"), and all of it is remembered in the platform's settings.
 *
 * Zustand and not React's own `useSyncExternalStore` written out, because a shell re-renders on
 * every pointer move of a splitter and selector-scoped subscription is the thing that keeps that
 * from redrawing a canvas; and because §5.1 names it, so using it is the decision that needs no
 * argument and replacing it is the one that would.
 *
 * **What runs a command.** §4.4's table is `commands.ts`; the doing is here. A command the shell
 * can perform itself — the View menu's regions, panels, zoom and theme, the Help menu's links and
 * its two screens — has a handler below; every other one is registered and, until a feature wires
 * it, says so in the Log. A command that vanished from the menu because nobody had built it yet
 * would make the palette's claim false and would hide the plan from the person reading the menu.
 */
import type {
  ColourScheme as PlatformScheme,
  Platform,
  Session,
  SettingValue,
  WorkspaceRef,
} from '@tensorspine/store/platform';
import { createStore, type StoreApi } from 'zustand/vanilla';

import type { ProblemSeverity, ProblemSource } from '@tensorspine/lang/api';

import { SPECIFICATION_ANCHORS } from './anchors.js';
import { COMMANDS, commandById, type Command } from './commands.js';
import { NO_FILTER, type Grouping } from '../problems/rows.js';
import {
  ACTIVITIES,
  clampSize,
  defaultRegions,
  defaultSide,
  movePanel,
  PANELS,
  showPanel,
  type ActivityId,
  type PanelId,
  type RegionId,
  type Regions,
  type RegionState,
  type SideState,
} from './regions.js';
import { text, textWith } from './strings.js';
import {
  readThemeChoice,
  resolveTheme,
  writeThemeChoice,
  type ColourScheme,
  type ThemeChoice,
} from './theme.js';

/** One line of the Log panel: what the shell did, and when. */
export interface LogLine {
  /** ISO 8601, as everything else the editor timestamps. */
  readonly at: string;
  readonly text: string;
}

/**
 * One tab of the editor area (§4.2: "one per open model, one per drill-in … one per unit").
 *
 * `kind` names the view that draws its body; the shell knows its own two (`view.shortcuts` and
 * `view.about`, the Help menu's in-app screens) and is given the rest by whoever opens them, so
 * that a canvas, a source view and a primitive editor are tabs of the same strip without this
 * module knowing what any of them is.
 */
export interface Tab {
  readonly id: string;
  readonly title: string;
  readonly kind: string;
  /** The `template` mark of §4.3, or whatever a later tab kind needs beside its name. */
  readonly badge?: string;
  /** The dirty mark of §4.3 (●). */
  readonly dirty?: boolean;
}

/** The shell's state: chrome, and nothing of the document. */
export interface ShellState {
  readonly side: SideState;
  readonly regions: Regions;
  readonly tabs: readonly Tab[];
  readonly activeTab: string | null;
  /** What the user chose (§4.4's `Theme: Light / Dark / System`). */
  readonly theme: ThemeChoice;
  /** What the machine prefers, which `system` resolves against. */
  readonly scheme: ColourScheme;
  /** Which menu of the bar is open, if any. */
  readonly menu: string | null;
  readonly palette: boolean;
  /**
   * The panel a drag is carrying, if one is.
   *
   * State and not a local: the tab being dragged is in one region and the drop target is the
   * other, so the only place both can read it is here — and it is what lets a region say it
   * would take the drop before the drop happens.
   */
  readonly draggingPanel: PanelId | null;
  /** The canvas's zoom, as a fraction: the status bar's last-but-one field. */
  readonly zoom: number;
  /**
   * The four View toggles §4.7 puts on the canvas: families, derived figures, edge types,
   * identities.
   *
   * Chrome, and nothing of a document: what is remembered is which of the four readings the
   * author wants, not anything the document says. S1 draws Derived figures on and Identities off,
   * which is what {@link DEFAULT_CANVAS} carries.
   */
  readonly canvas: CanvasToggles;
  /** Who the user is, where the deployment knows — `null` under `NoAuth`, so no avatar. */
  readonly session: Session | null;
  readonly workspace: WorkspaceRef;
  readonly log: readonly LogLine[];
  /**
   * How the Problems panel is showing its rows — §4.17's filter and grouping.
   *
   * Chrome, and here for the reason feature 2.6 gave the explorer's filter box: the panel is
   * unmounted whenever another of the bottom panel's three tabs is showing, and a filter that was
   * typed and then lost because the reader looked at the Derived panel is a filter they have to
   * type again. The state's own keys are snapshotted (a reviewed diff), which is what stops a
   * field from quietly starting to hold a document's facts.
   */
  readonly problems: ProblemView;
  /**
   * What the Derived panel is showing — §4.18's tab, its filter and the split it was held to.
   *
   * Chrome, for the reason {@link ProblemView} is: the panel is unmounted whenever another of the
   * bottom panel's three tabs is showing, and a reader who had D6 open and looked at Problems
   * should find D6 open when they come back. Nothing of a document is kept here — the split is a
   * *name the reader chose*, carried with the tab it was chosen in so that another document does
   * not inherit it.
   */
  readonly derived: DerivedView;
  /**
   * Where the documentation site is, relative to this page.
   *
   * The Help menu resolves its links against it and so does a problem's code (§4.17): the editor
   * is deployed *beside* the site (D11), the page reads its own address, and no host is written
   * into any component.
   */
  readonly docsBase: string;
}

/**
 * What the Derived panel opens with: the first product, the filter armed, no split chosen.
 *
 * The filter is **on** by default because §4.18 describes it as the panel's own behaviour ("a
 * selection filter restricts every tab to the selected node, identity or split") and because with
 * nothing selected it holds nothing back: the chip appears when there is a subject, and the `×`
 * on it is how a reader turns it off.
 */
export const NO_PRODUCT: DerivedView = { product: null, filter: true, split: null };

/** What the Derived panel is showing (§4.18). */
export interface DerivedView {
  /** The product whose tab is showing, by the member the derived document writes it under. */
  readonly product: string | null;
  /** Whether every tab is held to the document's own selection. */
  readonly filter: boolean;
  /** The graph split a reader chose in D6, with the tab it was chosen in. */
  readonly split: { readonly tab: string; readonly name: string } | null;
}

/** What the Problems panel is showing (§4.17). */
export interface ProblemView {
  /** The sources kept; every one of them when it is empty. */
  readonly sources: readonly ProblemSource[];
  /** The severities kept; every one of them when it is empty. */
  readonly severities: readonly ProblemSeverity[];
  /** What the filter box holds. */
  readonly text: string;
  readonly group: Grouping;
}

/** What a command does. */
export type CommandHandler = () => void | Promise<void>;

/** The shell's state and the gestures that change it. */
export interface Shell extends ShellState {
  setActivity(activity: ActivityId): void;
  toggleSide(open?: boolean): void;
  setSideWidth(width: number): void;

  toggleRegion(region: RegionId, open?: boolean): void;
  setRegionSize(region: RegionId, size: number): void;
  selectPanel(region: RegionId, panel: PanelId): void;
  /** Move a panel to another region — §4.2's drag. */
  dockPanel(panel: PanelId, region: RegionId): void;
  /** Select a panel's tab and open its region, wherever it sits — `View ▸ Problems` and its kin. */
  revealPanel(panel: PanelId): void;
  /** Change how the Problems panel is showing its rows (§4.17). */
  setProblemView(view: Partial<ProblemView>): void;
  /** Change what the Derived panel is showing (§4.18). */
  setDerivedView(view: Partial<DerivedView>): void;
  /**
   * `View ▸ Toggle Properties`.
   *
   * Not a panel command like the other three: Properties is a panel that can have been dragged to
   * the bottom (§4.2 names that very move), so what the command shows or hides is the region it
   * is in at the time, and it hides only when Properties is the panel that region is showing.
   */
  toggleProperties(): void;

  openTab(tab: Tab): void;
  selectTab(id: string): void;
  /**
   * Change what a tab says about itself — its name, its badge, its dirty dot (§4.3).
   *
   * The strip is the shell's and the document behind it is not (D1), so the feature that owns the
   * document is what tells the strip a document has become dirty or turned out to be a template.
   */
  updateTab(id: string, patch: { title?: string; badge?: string; dirty?: boolean }): void;
  closeTab(id: string): void;
  closeAllTabs(): void;
  /**
   * What is asked before a tab closes — §4.3's "Close with unsaved changes asks".
   *
   * A guard and not a state member: it is the *document's* answer, and a copy of the answer here
   * would be a copy of the document's dirtiness, which D1 forbids. The default takes every close,
   * which is what the shell did before anything could be dirty.
   */
  guardClose(guard: (tab: Tab) => boolean | Promise<boolean>): void;

  setTheme(choice: ThemeChoice): void;
  openMenu(menu: string | null): void;
  setPalette(open: boolean): void;
  setDraggingPanel(panel: PanelId | null): void;
  setZoom(zoom: number): void;
  /** Turn one of §4.7's four canvas readings on or off. */
  setCanvas(canvas: Partial<CanvasToggles>): void;

  /** Write a line into the Log panel. */
  note(line: string): void;
  /** Run a command of §4.4 by its identity. */
  run(id: string): void;
  /** Bind handlers for commands a later feature performs. */
  bind(handlers: Readonly<Record<string, CommandHandler>>): void;
}

/** The store, as the React binding and the suites hold it. */
export type ShellStore = StoreApi<Shell>;

/** What {@link createShell} is given. */
export interface ShellOptions {
  readonly platform: Platform;
  /**
   * Where the documentation site is, relative to the page or absolute.
   *
   * The editor is deployed **beside** the site, not inside it (D11), so the Help menu's links are
   * resolved against this and never against a host written into a component. The application
   * passes what its own build knows.
   */
  readonly docsBase?: string;
  /** Handlers a feature that owns a command supplies. */
  readonly handlers?: Readonly<Record<string, CommandHandler>>;
}

/** The setting the arrangement is remembered under (§4.2: "layouts are saved in settings"). */
export const LAYOUT_SETTING = 'shell.layout';

/** The four readings §4.7's View menu turns on and off over the canvas. */
export interface CanvasToggles {
  /** View ▸ Show Families: the chips on a card. */
  readonly families: boolean;
  /** View ▸ Show Derived Figures on Diagram: the line under a card. */
  readonly derivedFigures: boolean;
  /** View ▸ Show Edge Types: D2's value type beside a wire. */
  readonly edgeTypes: boolean;
  /** View ▸ Show Identities: the dashed links between the chips of one identity. */
  readonly identities: boolean;
}

/** The readings S1 draws: the derived figures on, the identities off. */
export const DEFAULT_CANVAS: CanvasToggles = {
  families: true,
  derivedFigures: true,
  edgeTypes: false,
  identities: false,
};

/** Which command turns which reading on and off (§4.4's View menu). */
const CANVAS_COMMANDS: Readonly<Record<string, keyof CanvasToggles>> = {
  'view.show-families': 'families',
  'view.show-derived-figures': 'derivedFigures',
  'view.show-edge-types': 'edgeTypes',
  'view.show-identities': 'identities',
};

/** How the zoom steps, and how far it goes (§4.4's `Zoom In/Out/Fit`). */
const ZOOM = { step: 1.2, min: 0.2, max: 4, fit: 1 } as const;

/** How many lines the Log keeps. */
const LOG_LIMIT = 500;

/** The specification's own page, as `tools/site.sh` writes it. */
export const SPECIFICATION_PAGE = 'spec/specification.html';

/**
 * Where the specification states an identifier, as a link — §4.17's "a problem's code links to
 * its anchor in the specification (generated map, §1)".
 *
 * The map is generated from `docs/SPECIFICATION.md` at build time (`pnpm anchors`) and held to it
 * by `tests/audit/anchors.test.ts`, so a rule renumbered or a section re-anchored fails the audit
 * rather than shipping as a link that lands nowhere. `null` for a code the specification does not
 * state — the empty code every lint finding and most of the loader's refusals carry, and the
 * `schema` and `registry` codes, which are the editor's own words for a stage.
 */
export function specificationLink(
  id: string,
  docsBase: string,
): { readonly href: string; readonly title: string } | null {
  const found = SPECIFICATION_ANCHORS[id];
  if (found === undefined) return null;
  return { href: `${docsBase}${SPECIFICATION_PAGE}#${found.anchor}`, title: found.section };
}

/** The pages of the documentation site the Help menu names, as `tools/site.sh` writes them. */
export const HELP_PAGES: Readonly<Record<string, string>> = {
  'help.specification': SPECIFICATION_PAGE,
  'help.model-guide': 'spec/tensorspine-model_json.html',
  'help.unit-guide': 'spec/tensorspine-primitive-library-unit.html',
  'help.library-reference': 'primitive-library/index.html',
  'help.glossary': 'spec/glossary.html',
  'help.derived-products': 'spec/derived-products.html',
};

/**
 * Where the documentation site is, relative to wherever the editor is served from.
 *
 * The editor is deployed **beside** the site and not inside it (D11: "on GitHub Pages beside the
 * documentation site"), so `…/tensorspine/editor/` finds it at `…/tensorspine/`. One directory
 * up, read from the page's own address: no host is written into any source, and a build served
 * from somewhere else finds its own neighbour. Feature 2.18 settles the published base path.
 */
export function documentationBase(here: string): string {
  return new URL('../', here).href;
}

/** The Help menu's two in-app screens, as tabs of the editor area. */
export const HELP_TABS: Readonly<Record<string, Tab>> = {
  'help.keyboard-shortcuts': {
    id: 'tab.shortcuts',
    title: 'Keyboard Shortcuts',
    kind: 'view.shortcuts',
  },
  'help.about': { id: 'tab.about', title: 'About', kind: 'view.about' },
};

/** Which panel each of `View ▸ Problems`, `Derived` and `Log` reveals. */
const PANEL_COMMANDS: Readonly<Record<string, PanelId>> = {
  'view.problems': 'panel.problems',
  'view.derived': 'panel.derived',
  'view.log': 'panel.log',
};

/** Which theme each of `View ▸ Theme: …` pins. */
const THEME_COMMANDS: Readonly<Record<string, ThemeChoice>> = {
  'view.theme-light': 'light',
  'view.theme-dark': 'dark',
  'view.theme-system': 'system',
};

/** The arrangement as it is written into the settings. */
interface StoredLayout {
  readonly side: SideState;
  readonly regions: Regions;
}

/**
 * The arrangement as a settings value.
 *
 * Written out member by member rather than handed over whole: `SettingValue` is "anything that
 * survives `JSON.stringify` unchanged" and a TypeScript interface does not satisfy an index
 * signature, so the alternative was a cast — and a cast is exactly what would let a field that
 * does not survive the round trip reach the settings unnoticed.
 */
function storedLayout(side: SideState, regions: Regions): SettingValue {
  const region = (id: RegionId): SettingValue => ({
    open: regions[id].open,
    size: regions[id].size,
    panels: [...regions[id].panels],
    active: regions[id].active,
  });
  return {
    side: { open: side.open, width: side.width, activity: side.activity },
    regions: { 'region.bottom': region('region.bottom'), 'region.right': region('region.right') },
  };
}

/**
 * Whether a stored value is a truth.
 *
 * `value === true || value === false` and not `typeof value === 'boolean'`: the word is a value
 * of the unit schema's `argument_type.kind`, and catching rule (b) of §1 is a whole-literal scan
 * that cannot tell a JavaScript type test from a vocabulary item. Feature 2.1 met it in the same
 * place — the sidecar's own reader — and wrote the same line.
 */
function isTruth(value: unknown): value is boolean {
  return value === true || value === false;
}

/** Whether a stored string is an activity this editor has. */
function isActivity(value: unknown): value is ActivityId {
  return ACTIVITIES.some((one) => one.id === value);
}

/** Whether a stored string is a panel this editor has. */
function isPanel(value: unknown): value is PanelId {
  return PANELS.some((one) => one.id === value);
}

/**
 * Read the arrangement back, keeping only what this editor can still make sense of.
 *
 * Every value is checked against what this build has rather than trusted: the settings hold
 * whatever was written into them, by an older editor or by hand (§5.2), and an activity or a
 * panel this build does not have would otherwise reach the chrome as a name nothing answers. A
 * panel the stored arrangement lost comes back to the region it calls home, so a settings file
 * that dropped one does not take the panel with it.
 */
export function readLayout(stored: unknown): StoredLayout {
  const side = defaultSide();
  const regions = defaultRegions();
  if (typeof stored !== 'object' || stored === null) return { side, regions };
  const held = stored as { side?: unknown; regions?: unknown };

  const readSide = (value: unknown): SideState => {
    if (typeof value !== 'object' || value === null) return side;
    const one = value as { open?: unknown; width?: unknown; activity?: unknown };
    return {
      open: isTruth(one.open) ? one.open : side.open,
      width: typeof one.width === 'number' ? clampSize('side', one.width) : side.width,
      activity: isActivity(one.activity) ? one.activity : side.activity,
    };
  };

  const readRegions = (value: unknown): Regions => {
    if (typeof value !== 'object' || value === null) return regions;
    const all = value as Record<string, unknown>;
    const next: Record<RegionId, RegionState> = { ...regions };
    for (const id of Object.keys(regions) as RegionId[]) {
      const one = all[id];
      if (typeof one !== 'object' || one === null) continue;
      const state = one as { open?: unknown; size?: unknown; panels?: unknown; active?: unknown };
      const panels = Array.isArray(state.panels)
        ? (state.panels as unknown[]).filter(isPanel)
        : next[id].panels;
      next[id] = {
        open: isTruth(state.open) ? state.open : next[id].open,
        size: typeof state.size === 'number' ? clampSize(id, state.size) : next[id].size,
        panels,
        active: isPanel(state.active) && panels.includes(state.active) ? state.active : (panels[0] ?? null),
      };
    }
    const seen = new Set(Object.values(next).flatMap((state) => state.panels));
    for (const id of Object.keys(regions) as RegionId[]) {
      const missing = regions[id].panels.filter((panel) => !seen.has(panel));
      if (missing.length === 0) continue;
      next[id] = { ...next[id], panels: [...next[id].panels, ...missing] };
      if (next[id].active === null) next[id] = { ...next[id], active: missing[0] ?? null };
    }
    return next;
  };

  return { side: readSide(held.side), regions: readRegions(held.regions) };
}

/**
 * Build the shell's store over a platform.
 *
 * The three things the platform knows and the shell shows — who the user is, which workspace is
 * open, what the machine prefers — are subscribed to here and answered from state, so that a
 * component asks the store and never the platform. `dispose` stops all three; a page that lives
 * for the session never calls it, and a suite always does.
 */
export function createShell(options: ShellOptions): { store: ShellStore; dispose: () => void } {
  const { platform } = options;
  const settings = platform.settings;
  const stored = readLayout(settings.peek(LAYOUT_SETTING));
  const docsBase = options.docsBase ?? '../';
  let handlers: Record<string, CommandHandler> = { ...options.handlers };
  /** What is asked before a tab closes; every close is taken until a feature says otherwise. */
  let closing: (tab: Tab) => boolean | Promise<boolean> = () => true;

  const store: ShellStore = createStore<Shell>()((set, get) => {
    const remember = (): void => {
      const { side, regions } = get();
      settings.set(LAYOUT_SETTING, storedLayout(side, regions));
    };
    const changeRegions = (next: Regions): void => {
      set({ regions: next });
      remember();
    };

    const note = (line: string): void => {
      set((state) => ({
        log: [...state.log, { at: new Date().toISOString(), text: line }].slice(-LOG_LIMIT),
      }));
    };

    /** Take a tab out of the strip, the current one moving to its neighbour. */
    const remove = (id: string): void => {
      set((state) => {
        const index = state.tabs.findIndex((tab) => tab.id === id);
        if (index < 0) return {};
        const tabs = state.tabs.filter((tab) => tab.id !== id);
        if (state.activeTab !== id) return { tabs };
        const next = tabs[Math.min(index, tabs.length - 1)];
        return { tabs, activeTab: next?.id ?? null };
      });
    };

    const openTab = (tab: Tab): void => {
      set((state) =>
        state.tabs.some((one) => one.id === tab.id)
          ? { activeTab: tab.id }
          : { tabs: [...state.tabs, tab], activeTab: tab.id },
      );
    };

    const builtin: Record<string, CommandHandler> = {
      'view.toggle-side-bar': () => {
        get().toggleSide();
      },
      'view.toggle-properties': () => {
        get().toggleProperties();
      },
      'view.toggle-bottom-panel': () => {
        get().toggleRegion('region.bottom');
      },
      'view.zoom-in': () => {
        get().setZoom(get().zoom * ZOOM.step);
      },
      'view.zoom-out': () => {
        get().setZoom(get().zoom / ZOOM.step);
      },
      'view.zoom-fit': () => {
        get().setZoom(ZOOM.fit);
      },
      'edit.preferences': () => {
        get().setActivity('activity.settings');
      },
    };
    for (const [id, reading] of Object.entries(CANVAS_COMMANDS)) {
      builtin[id] = () => {
        get().setCanvas({ [reading]: !get().canvas[reading] });
      };
    }
    for (const [id, panel] of Object.entries(PANEL_COMMANDS)) {
      builtin[id] = () => {
        get().revealPanel(panel);
      };
    }
    for (const [id, choice] of Object.entries(THEME_COMMANDS)) {
      builtin[id] = () => {
        get().setTheme(choice);
      };
    }
    for (const [id, page] of Object.entries(HELP_PAGES)) {
      builtin[id] = () => platform.shell.openExternal(`${docsBase}${page}`);
    }
    for (const [id, tab] of Object.entries(HELP_TABS)) {
      builtin[id] = () => {
        openTab(tab);
      };
    }

    return {
      side: stored.side,
      regions: stored.regions,
      tabs: [],
      activeTab: null,
      theme: readThemeChoice(settings),
      scheme: platform.shell.colourScheme(),
      menu: null,
      palette: false,
      draggingPanel: null,
      zoom: ZOOM.fit,
      canvas: DEFAULT_CANVAS,
      session: platform.auth.current(),
      workspace: platform.workspace.root(),
      log: [],
      problems: { ...NO_FILTER, group: 'node' },
      derived: NO_PRODUCT,
      docsBase,

      setDerivedView: (view) => {
        set((state) => ({ derived: { ...state.derived, ...view } }));
      },
      setProblemView: (view) => {
        set((state) => ({ problems: { ...state.problems, ...view } }));
      },

      setActivity: (activity) => {
        set((state) => ({ side: { ...state.side, activity, open: true } }));
        remember();
      },
      toggleSide: (open) => {
        set((state) => ({ side: { ...state.side, open: open ?? !state.side.open } }));
        remember();
      },
      setSideWidth: (width) => {
        set((state) => ({ side: { ...state.side, width: clampSize('side', width) } }));
        remember();
      },

      toggleRegion: (region, open) => {
        set((state) => ({
          regions: {
            ...state.regions,
            [region]: { ...state.regions[region], open: open ?? !state.regions[region].open },
          },
        }));
        remember();
      },
      setRegionSize: (region, size) => {
        set((state) => ({
          regions: {
            ...state.regions,
            [region]: { ...state.regions[region], size: clampSize(region, size) },
          },
        }));
        remember();
      },
      selectPanel: (region, panel) => {
        set((state) => ({
          regions: {
            ...state.regions,
            [region]: { ...state.regions[region], active: panel, open: true },
          },
        }));
        remember();
      },
      dockPanel: (panel, region) => {
        changeRegions(movePanel(get().regions, panel, region));
      },
      revealPanel: (panel) => {
        changeRegions(showPanel(get().regions, panel));
      },
      toggleProperties: () => {
        const regions = get().regions;
        const id = (Object.keys(regions) as RegionId[]).find((region) =>
          regions[region].panels.includes('panel.properties'),
        );
        if (id === undefined) return;
        const region = regions[id];
        const showing = region.open && region.active === 'panel.properties';
        changeRegions(
          showing
            ? { ...regions, [id]: { ...region, open: false } }
            : showPanel(regions, 'panel.properties'),
        );
      },

      openTab,
      selectTab: (id) => {
        set((state) => (state.tabs.some((tab) => tab.id === id) ? { activeTab: id } : {}));
      },
      updateTab: (id, change) => {
        set((state) => ({
          tabs: state.tabs.map((tab) => (tab.id === id ? { ...tab, ...change } : tab)),
        }));
      },
      guardClose: (guard) => {
        closing = guard;
      },
      closeTab: (id) => {
        const tab = get().tabs.find((one) => one.id === id);
        if (tab === undefined) return;
        const answer = closing(tab);
        if (answer === true) {
          remove(id);
          return;
        }
        if (answer === false) return;
        void answer.then((may) => {
          if (may) remove(id);
        });
      },
      closeAllTabs: () => {
        for (const tab of [...get().tabs]) get().closeTab(tab.id);
      },

      setTheme: (choice) => {
        set({ theme: choice });
        writeThemeChoice(settings, choice);
      },
      openMenu: (menu) => {
        set({ menu });
      },
      setPalette: (open) => {
        set({ palette: open, menu: null });
      },
      setDraggingPanel: (panel) => {
        set({ draggingPanel: panel });
      },
      setZoom: (zoom) => {
        set({ zoom: Math.min(ZOOM.max, Math.max(ZOOM.min, zoom)) });
      },

      setCanvas: (canvas) => {
        set((state) => ({ canvas: { ...state.canvas, ...canvas } }));
      },

      note,
      bind: (more) => {
        handlers = { ...handlers, ...more };
      },
      run: (id) => {
        set({ menu: null, palette: false });
        const command = commandById(id);
        if (command === undefined) {
          note(textWith('There is no command {}.', id));
          return;
        }
        const handler = handlers[id] ?? builtin[id];
        if (handler === undefined) {
          note(textWith('{} is not built yet.', `“${text(command.label)}”`));
          return;
        }
        try {
          const answer = handler();
          if (answer instanceof Promise) {
            answer.catch((error: unknown) => {
              note(`${text(command.label)} — ${error instanceof Error ? error.message : String(error)}`);
            });
          }
        } catch (error) {
          note(`${text(command.label)} — ${error instanceof Error ? error.message : String(error)}`);
        }
      },
    };
  });

  // The Log's first lines: what this build is looking at. Written here rather than in a
  // component, because a component that wrote on mount would write again on every remount.
  const started = store.getState();
  started.note(textWith('Platform: {}.', platform.describe()));
  const workspace = platform.workspace.root();
  started.note(
    textWith(
      workspace.kind === 'empty' ? 'No workspace is open ({}).' : 'Workspace: {}.',
      workspace.kind === 'empty' ? workspace.kind : `${workspace.name} · ${workspace.kind}`,
    ),
  );
  if (!settings.persistent) started.note(text('Settings cannot be stored: they last this session.'));
  if (!platform.drafts.persistent) started.note(text('Drafts cannot be stored: they last this session.'));

  const unsubscribe = [
    platform.auth.onChange((session) => {
      store.setState({ session });
    }),
    platform.workspaces.onChange((workspace) => {
      store.setState({ workspace: workspace.root() });
    }),
    platform.shell.onColourSchemeChange((scheme: PlatformScheme) => {
      store.setState({ scheme });
    }),
  ];

  return { store, dispose: () => unsubscribe.forEach((stop) => stop()) };
}

/** Which theme is in force for a state — the choice, resolved against the machine's preference. */
export function themeOf(state: ShellState): ColourScheme {
  return resolveTheme(state.theme, state.scheme);
}

/** Every command, for the palette and for the native menu the platform may build (§4.4). */
export function allCommands(): readonly Command[] {
  return COMMANDS;
}

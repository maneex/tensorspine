/**
 * The shell — plan §4.2, §4.4 and §4.21, as drawn on S1 and S17.
 *
 * The frame the rest of the editor is built inside: the bar with §4.4's seven menus, the activity
 * rail, the side bar, the editor tabs, the bottom panel's three tabs, the Properties region, the
 * status bar, the command palette, the drag that moves a panel between regions, the arrangement
 * remembered in the settings, and the design's two themes with the machine's preference between
 * them.
 *
 * **What a later feature adds, and how.** Not by changing this: by opening a tab of a `kind` it
 * gives the shell a view for, by binding a handler to a command of §4.4's table, and by putting
 * its own body in a panel. `createShell` builds the store, `Shell` draws it, and both are the
 * whole seam.
 *
 * The stylesheet is `@tensorspine/ui/style.css` — a separate entry point, so that a suite that
 * imports the model can be a Node one, and the application that draws it asks for the tokens.
 */
export { Bar, initialsOf, Wordmark } from './Bar.js';
export {
  acceleratorText,
  COMMANDS,
  commandById,
  commandFor,
  commandsOf,
  matches,
  matching,
  MENUS,
  PALETTE_CHORD,
  scoreOf,
  type AcceleratorModifier,
  type Chord,
  type Command,
  type KeyStroke,
  type Menu,
  type MenuId,
} from './commands.js';
export { ShellProvider, useShell, useShellStore, usePlatform } from './context.js';
export { EditorArea, type TabViews } from './EditorArea.js';
export { MONOGRAM } from './logo.js';
export { Palette } from './Palette.js';
export { PANEL_TRANSFER, PanelRegion } from './Panels.js';
export { ActivityProvider, Rail, Side, type ActivityViews } from './Rail.js';
export {
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
  type Activity,
  type ActivityId,
  type Panel,
  type PanelId,
  type RegionBounds,
  type RegionId,
  type Regions,
  type RegionState,
  type SideState,
} from './regions.js';
export { isTyping, Shell, type ShellProps } from './Shell.js';
export { Splitter, type SplitterProps } from './Splitter.js';
export { StatusBar, themeLabel } from './StatusBar.js';
export {
  allCommands,
  createShell,
  documentationBase,
  HELP_PAGES,
  HELP_TABS,
  LAYOUT_SETTING,
  NO_PRODUCT,
  readLayout,
  themeOf,
  type CommandHandler,
  type DerivedView,
  type LogLine,
  type Shell as ShellModel,
  type ShellOptions,
  type ShellState,
  type ShellStore,
  type Tab,
} from './store.js';
export {
  currentLocale,
  forgetKeysUsed,
  keysUsed,
  setLocale,
  text,
  textWith,
  type Dictionary,
} from './strings.js';
export {
  DEFAULT_THEME,
  isThemeChoice,
  LIGHT_CLASS,
  readThemeChoice,
  resolveTheme,
  THEME_SETTING,
  writeThemeChoice,
  type ColourScheme,
  type ThemeChoice,
} from './theme.js';
export { About, KeyboardShortcuts, SHELL_VIEWS } from './views.js';

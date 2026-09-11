/**
 * The theme — plan §4.21, and the decision §9 Q3 left to implementation.
 *
 * > **Theme**: the token set the design pass chose, `plans/graph-editor-design/_ts.css` … with a
 * > light theme of the same stance; the default is decided before implementation (Q3); system
 * > preference honoured.
 *
 * **Q3, decided here: the default is `system`.** Q3 asked whether the default should be "the
 * site's light theme, with dark available", and deferred the answer to implementation; §4.21 asks
 * in the same breath that the system preference be honoured. `system` is the only choice that
 * keeps both: a machine set to light gets the light theme Q3's recommendation named, a machine
 * set to dark gets the ground every artboard is drawn on, and neither is imposed on the other.
 * `Theme: Light` and `Theme: Dark` (§4.4's View menu) pin it, and the pin is remembered in the
 * settings, so a choice survives a reload and an absence of choice keeps following the machine.
 *
 * There is no third answer to ask the machine for: `prefers-color-scheme` has `light` and `dark`
 * and nothing else, so a machine that expresses no preference answers light. That is why the
 * default reads as light on most machines rather than as the artboards' dark, and it is the
 * honest reading of "system preference honoured" rather than a preference invented for it.
 *
 * The tokens themselves are `tokens.css`, vendored from the design and held to it; this module
 * decides only which of the two blocks is in force.
 */
import type { SettingsStore } from '@tensorspine/store/platform';

/** What the machine prefers: the two values `prefers-color-scheme` has. */
export type ColourScheme = 'light' | 'dark';

/** What the user chose: either theme, or the machine's own preference. */
export type ThemeChoice = ColourScheme | 'system';

/** The setting the choice is remembered under (`SettingsStore`, §5.2's "user settings"). */
export const THEME_SETTING = 'theme';

/** The choice a session starts from when nothing was ever chosen. */
export const DEFAULT_THEME: ThemeChoice = 'system';

/** The class `tokens.css` redefines its names under; the dark block is `:root` and needs none. */
export const LIGHT_CLASS = 'theme-light';

const CHOICES: readonly ThemeChoice[] = ['light', 'dark', 'system'];

/** Whether a stored value is a choice this editor understands. */
export function isThemeChoice(value: unknown): value is ThemeChoice {
  return typeof value === 'string' && (CHOICES as readonly string[]).includes(value);
}

/** Which theme is in force: the choice, or the machine's preference under `system`. */
export function resolveTheme(choice: ThemeChoice, scheme: ColourScheme): ColourScheme {
  return choice === 'system' ? scheme : choice;
}

/**
 * The remembered choice.
 *
 * A settings store holds whatever was written into it, by an older editor or by hand (§5.2: "a
 * caller that cares looks before it reads"), so a value that is not one of the three is read as
 * no choice at all rather than as a theme nobody can name.
 */
export function readThemeChoice(settings: SettingsStore): ThemeChoice {
  const stored = settings.peek(THEME_SETTING);
  return isThemeChoice(stored) ? stored : DEFAULT_THEME;
}

/** Remember a choice. */
export function writeThemeChoice(settings: SettingsStore, choice: ThemeChoice): void {
  settings.set(THEME_SETTING, choice);
}

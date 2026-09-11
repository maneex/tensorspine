/**
 * The commands of plan §4.4, registered once — the menu bar, the palette and (later) every
 * context menu read this table and nothing else.
 *
 * > Every entry is a command available in the palette; shortcuts are the platform's conventions
 * > (⌘ on macOS, Ctrl elsewhere).
 *
 * And the rule that goes with it, from the same section:
 *
 * > Every edit has a visible affordance. … a keyboard shortcut is an accelerator for people who
 * > know it and never the only way, nor the main way, to edit anything. Every context-menu
 * > command is also a command of the menu bar or a button of the sheet.
 *
 * Which is why the table is the *source* and the three surfaces are readings of it: a command
 * that exists has a menu entry by construction, so none of them can become keyboard-only, and the
 * palette listing every one of them is a test rather than a promise
 * (`test/shell/commands.test.ts`).
 *
 * **What a command is here.** An identity, the menu it belongs to, the English label §4.4 gives
 * it (which is also its i18n key — `strings.ts`), and an optional accelerator. What it *does* is
 * not here: a handler is supplied by whoever can do it (`store.ts`), and a command nobody has
 * wired yet says so in the Log instead of vanishing from the menu. A feature that is not built is
 * a fact about the editor, not a reason to hide the command — and hiding it would make the
 * palette's claim false.
 *
 * **Identities are dotted and the labels are English sentences.** Neither can collide with the
 * language: catching rule (b) of §1 compares whole string literals against the schemas'
 * enumerations, and `derived`, `none`, `value` and `fixed` are among them — `panel.derived` and
 * `Derived` are not.
 */

/** A menu of the bar, in §4.4's order. */
export type MenuId =
  | 'menu.file'
  | 'menu.edit'
  | 'menu.view'
  | 'menu.model'
  | 'menu.library'
  | 'menu.weights'
  | 'menu.help';

/** One menu of the bar. */
export interface Menu {
  readonly id: MenuId;
  /** The English label, which is also the key of `strings.ts`. */
  readonly label: string;
}

/** The seven menus of §4.2's region table, in its order. */
export const MENUS: readonly Menu[] = [
  { id: 'menu.file', label: 'File' },
  { id: 'menu.edit', label: 'Edit' },
  { id: 'menu.view', label: 'View' },
  { id: 'menu.model', label: 'Model' },
  { id: 'menu.library', label: 'Library' },
  { id: 'menu.weights', label: 'Weights' },
  { id: 'menu.help', label: 'Help' },
];

/**
 * A key chord, stated in the platform-independent form §4.4 asks for.
 *
 * `mod` is "the platform's convention": ⌘ on Apple, Ctrl everywhere else — which of the two this
 * deployment uses is the platform's answer (`Shell.modifier`), never a guess made here.
 */
export interface Chord {
  readonly mod?: true;
  readonly shift?: true;
  readonly alt?: true;
  /** `KeyboardEvent.key`, with a letter written in upper case and compared without regard to case. */
  readonly key: string;
}

/** One command of §4.4. */
export interface Command {
  readonly id: string;
  readonly menu: MenuId;
  /** §4.4's own wording, and the i18n key. */
  readonly label: string;
  readonly accelerator?: Chord;
  /**
   * Whether the accelerator fires while the focus is in a text field.
   *
   * The editing chords — Ctrl+X/C/V/Z/A/D/F, Delete, F2 — belong to whatever is being typed in
   * while it is being typed in, and taking them from it would break the one thing a text field
   * is for. Save, the palette, the panel toggles and the function keys are wanted everywhere, so
   * they say so. Nothing is derived from the chord's shape: Ctrl+S and Ctrl+D have the same
   * shape and opposite answers.
   */
  readonly whileTyping?: true;
}

const F = 'menu.file';
const E = 'menu.edit';
const V = 'menu.view';
const M = 'menu.model';
const L = 'menu.library';
const W = 'menu.weights';
const H = 'menu.help';

/**
 * Every command of §4.4, in the order that section writes them.
 *
 * Transcribed, not designed. Where §4.4 writes a group on one line — "Cut/Copy/Paste
 * (Ctrl+X/C/V)", "Zoom In/Out/Fit (Ctrl+= / − / 0)", "Theme: Light / Dark / System" — the group
 * is the several commands it names, because the palette must offer each of them by name.
 *
 * Two accelerators are not written in §4.4 and are here because that section defers to "the
 * platform's conventions": **Redo** (⇧⌘Z / Ctrl+Shift+Z), which every editor on every platform
 * binds and whose absence beside a bound Undo would be a gap rather than a decision, and the
 * **command palette** itself, whose chord §4.2 states in its region table rather than in §4.4.
 */
export const COMMANDS: readonly Command[] = [
  // File
  { id: 'file.new-model', menu: F, label: 'New Model' },
  { id: 'file.new-template', menu: F, label: 'New Template' },
  { id: 'file.new-base', menu: F, label: 'New Base…' },
  { id: 'file.open-folder', menu: F, label: 'Open Folder…' },
  { id: 'file.open-model', menu: F, label: 'Open Model…' },
  { id: 'file.open-recent', menu: F, label: 'Open Recent' },
  { id: 'file.save', menu: F, label: 'Save', accelerator: { mod: true, key: 'S' }, whileTyping: true },
  { id: 'file.save-as', menu: F, label: 'Save As…' },
  { id: 'file.save-all', menu: F, label: 'Save All' },
  { id: 'file.download-zip', menu: F, label: 'Download Workspace as Zip' },
  { id: 'file.revert', menu: F, label: 'Revert' },
  { id: 'file.export-derived', menu: F, label: 'Export Derived Document…' },
  { id: 'file.export-diagram', menu: F, label: 'Export Diagram' },
  { id: 'file.close-tab', menu: F, label: 'Close Tab', accelerator: { mod: true, key: 'W' }, whileTyping: true },
  { id: 'file.close-all', menu: F, label: 'Close All' },

  // Edit
  { id: 'edit.undo', menu: E, label: 'Undo', accelerator: { mod: true, key: 'Z' } },
  { id: 'edit.redo', menu: E, label: 'Redo', accelerator: { mod: true, shift: true, key: 'Z' } },
  { id: 'edit.cut', menu: E, label: 'Cut', accelerator: { mod: true, key: 'X' } },
  { id: 'edit.copy', menu: E, label: 'Copy', accelerator: { mod: true, key: 'C' } },
  { id: 'edit.paste', menu: E, label: 'Paste', accelerator: { mod: true, key: 'V' } },
  { id: 'edit.duplicate', menu: E, label: 'Duplicate', accelerator: { mod: true, key: 'D' } },
  { id: 'edit.delete', menu: E, label: 'Delete', accelerator: { key: 'Delete' } },
  { id: 'edit.rename', menu: E, label: 'Rename', accelerator: { key: 'F2' } },
  { id: 'edit.select-all', menu: E, label: 'Select All', accelerator: { mod: true, key: 'A' } },
  { id: 'edit.find', menu: E, label: 'Find', accelerator: { mod: true, key: 'F' } },
  { id: 'edit.preferences', menu: E, label: 'Preferences' },

  // View
  { id: 'view.toggle-side-bar', menu: V, label: 'Toggle Side Bar', accelerator: { mod: true, key: 'B' }, whileTyping: true },
  { id: 'view.toggle-properties', menu: V, label: 'Toggle Properties' },
  { id: 'view.toggle-bottom-panel', menu: V, label: 'Toggle Bottom Panel', accelerator: { mod: true, key: 'J' }, whileTyping: true },
  { id: 'view.problems', menu: V, label: 'Problems', accelerator: { mod: true, shift: true, key: 'M' }, whileTyping: true },
  { id: 'view.derived', menu: V, label: 'Derived' },
  { id: 'view.log', menu: V, label: 'Log' },
  { id: 'view.zoom-in', menu: V, label: 'Zoom In', accelerator: { mod: true, key: '=' } },
  { id: 'view.zoom-out', menu: V, label: 'Zoom Out', accelerator: { mod: true, key: '-' } },
  { id: 'view.zoom-fit', menu: V, label: 'Zoom to Fit', accelerator: { mod: true, key: '0' } },
  { id: 'view.auto-layout', menu: V, label: 'Auto-layout', accelerator: { shift: true, alt: true, key: 'F' } },
  { id: 'view.reset-layout', menu: V, label: 'Reset Layout' },
  { id: 'view.show-identities', menu: V, label: 'Show Identities' },
  { id: 'view.show-derived-figures', menu: V, label: 'Show Derived Figures on Diagram' },
  { id: 'view.show-edge-types', menu: V, label: 'Show Edge Types' },
  { id: 'view.show-families', menu: V, label: 'Show Families' },
  { id: 'view.expanded-graph', menu: V, label: 'Expanded Graph' },
  { id: 'view.layer-preview', menu: V, label: 'Layer Preview' },
  { id: 'view.json-source', menu: V, label: 'JSON Source', accelerator: { mod: true, shift: true, key: 'J' }, whileTyping: true },
  { id: 'view.theme-light', menu: V, label: 'Theme: Light' },
  { id: 'view.theme-dark', menu: V, label: 'Theme: Dark' },
  { id: 'view.theme-system', menu: V, label: 'Theme: System' },

  // Model
  { id: 'model.validate-now', menu: M, label: 'Validate Now', accelerator: { key: 'F7' }, whileTyping: true },
  { id: 'model.derive-now', menu: M, label: 'Derive Now', accelerator: { shift: true, key: 'F7' }, whileTyping: true },
  { id: 'model.lint', menu: M, label: 'Lint' },
  { id: 'model.check-checkpoint', menu: M, label: 'Check Against Checkpoint…' },
  { id: 'model.set-preview-assignment', menu: M, label: 'Set Preview Assignment…' },
  { id: 'model.add-quantity', menu: M, label: 'Add Quantity' },
  { id: 'model.add-constant', menu: M, label: 'Add Constant' },
  { id: 'model.add-instance', menu: M, label: 'Add Instance…' },
  { id: 'model.add-composition', menu: M, label: 'Add Composition' },
  { id: 'model.add-input', menu: M, label: 'Add Input' },
  { id: 'model.add-output', menu: M, label: 'Add Output' },
  { id: 'model.extract-to-template', menu: M, label: 'Extract to Template…' },
  { id: 'model.inline-template', menu: M, label: 'Inline Template' },
  { id: 'model.upgrade-pins', menu: M, label: 'Upgrade Pins…' },
  { id: 'model.document-properties', menu: M, label: 'Document Properties' },

  // Library
  { id: 'library.open-base', menu: L, label: 'Open Base…' },
  { id: 'library.new-base', menu: L, label: 'New Base…' },
  { id: 'library.reload-bases', menu: L, label: 'Reload Bases' },
  { id: 'library.show-unit', menu: L, label: 'Show Unit' },
  { id: 'library.edit-unit', menu: L, label: 'Edit Unit' },
  { id: 'library.new-primitive', menu: L, label: 'New Primitive…' },
  { id: 'library.new-version', menu: L, label: 'New Version…' },
  { id: 'library.new-axis', menu: L, label: 'New Axis…' },
  { id: 'library.new-precision-role', menu: L, label: 'New Precision Role…' },
  { id: 'library.start-from', menu: L, label: 'Start From…' },
  { id: 'library.argument-schema', menu: L, label: 'Argument Schema' },
  { id: 'library.preview-instance', menu: L, label: 'Preview Instance' },
  { id: 'library.publish-template', menu: L, label: 'Publish Template…' },
  { id: 'library.unlock-base', menu: L, label: 'Unlock Base…' },

  // Weights
  { id: 'weights.open-checkpoint-folder', menu: W, label: 'Open Checkpoint Folder…' },
  { id: 'weights.open-hub-repository', menu: W, label: 'Open Hub Repository…' },
  { id: 'weights.adopt-config-values', menu: W, label: 'Adopt config.json Values as Quantities…' },
  { id: 'weights.locate-selected-slot', menu: W, label: 'Locate Selected Slot…' },
  { id: 'weights.check-locations', menu: W, label: 'Check Locations' },
  { id: 'weights.clear-checkpoint', menu: W, label: 'Clear Checkpoint' },

  // Help
  { id: 'help.specification', menu: H, label: 'Specification' },
  { id: 'help.model-guide', menu: H, label: 'Model JSON Guide' },
  { id: 'help.unit-guide', menu: H, label: 'Library-unit Guide' },
  { id: 'help.library-reference', menu: H, label: 'Library Reference' },
  { id: 'help.glossary', menu: H, label: 'Glossary' },
  { id: 'help.derived-products', menu: H, label: 'Derived Products' },
  { id: 'help.keyboard-shortcuts', menu: H, label: 'Keyboard Shortcuts' },
  { id: 'help.about', menu: H, label: 'About' },
];

/**
 * The command palette's own chord — §4.2's region table, "Ctrl/⌘ + Shift + P".
 *
 * It is not a command of §4.4 and so is not in the table above: the palette lists the commands,
 * and a list that listed its own opening would be listing something no menu carries. It is bound
 * beside them and shown on the bar's own palette button, so it is never the only way in (§4.4).
 */
export const PALETTE_CHORD: Chord = { mod: true, shift: true, key: 'P' };

/** The commands of one menu, in §4.4's order. */
export function commandsOf(menu: MenuId): Command[] {
  return COMMANDS.filter((command) => command.menu === menu);
}

/** The command with that identity, or `undefined`. */
export function commandById(id: string): Command | undefined {
  return COMMANDS.find((command) => command.id === id);
}

/**
 * How well a command answers a query — larger is better, 0 is no answer at all.
 *
 * The palette's ranking, kept here beside the table rather than in the component, so that what
 * "every command of §4.4 by name" means can be asked of it without a browser.
 */
export function scoreOf(command: Command, menu: string, query: string): number {
  if (query === '') return 1;
  const needle = query.toLowerCase();
  const label = command.label.toLowerCase();
  const within = menu.toLowerCase();
  const both = `${within} ${label}`;
  if (label.startsWith(needle)) return 100;
  if (label.split(/[\s:]+/).some((word) => word.startsWith(needle))) return 80;
  if (label.includes(needle)) return 60;
  if (within.startsWith(needle)) return 40;
  if (both.includes(needle)) return 20;
  return 0;
}

/** The commands a query finds, best first and otherwise in §4.4's own order. */
export function matching(query: string): { command: Command; menu: string }[] {
  const menuOf = (command: Command): string =>
    MENUS.find((one) => one.id === command.menu)?.label ?? '';
  return COMMANDS.map((command, at) => ({ command, menu: menuOf(command), at }))
    .map((one) => ({ ...one, score: scoreOf(one.command, one.menu, query) }))
    .filter((one) => one.score > 0)
    .sort((a, b) => b.score - a.score || a.at - b.at)
    .map(({ command, menu }) => ({ command, menu }));
}

/** Which modifier this platform's conventions use for an accelerator (§4.4). */
export type AcceleratorModifier = 'command' | 'control';

/** Whether a chord is written in Apple's symbols or in the spelled-out convention. */
function apple(modifier: AcceleratorModifier): boolean {
  return modifier === 'command';
}

/** How a key reads on a menu: `Del` rather than `Delete`, `+` rather than `=`. */
function keyText(key: string): string {
  if (key === 'Delete') return 'Del';
  if (key === '=') return '+';
  if (key === '-') return '−';
  return key;
}

/**
 * A chord as this platform writes it — `⇧⌘Z` on Apple, `Ctrl+Shift+Z` elsewhere.
 *
 * Apple's order is the one its own menus use (⌃⌥⇧⌘), and the parts are juxtaposed; everywhere
 * else the names are joined with `+` in the order the conventions write them.
 */
export function acceleratorText(chord: Chord, modifier: AcceleratorModifier): string {
  const key = keyText(chord.key);
  if (apple(modifier)) {
    return `${chord.alt === true ? '⌥' : ''}${chord.shift === true ? '⇧' : ''}${chord.mod === true ? '⌘' : ''}${key}`;
  }
  const parts: string[] = [];
  if (chord.mod === true) parts.push('Ctrl');
  if (chord.shift === true) parts.push('Shift');
  if (chord.alt === true) parts.push('Alt');
  parts.push(key);
  return parts.join('+');
}

/** What a key event carries, as much of it as a chord is decided on. */
export interface KeyStroke {
  readonly key: string;
  readonly ctrlKey: boolean;
  readonly metaKey: boolean;
  readonly shiftKey: boolean;
  readonly altKey: boolean;
}

/**
 * Whether a key event is this chord, under this platform's modifier.
 *
 * A letter is compared without regard to case, because the browser reports `z` and `Z` for the
 * same key depending on Shift; every modifier is compared exactly, so `⌘Z` is not `⇧⌘Z`.
 */
export function matches(chord: Chord, stroke: KeyStroke, modifier: AcceleratorModifier): boolean {
  const mod = apple(modifier) ? stroke.metaKey : stroke.ctrlKey;
  const other = apple(modifier) ? stroke.ctrlKey : stroke.metaKey;
  if (mod !== (chord.mod === true)) return false;
  if (other) return false;
  if (stroke.shiftKey !== (chord.shift === true)) return false;
  if (stroke.altKey !== (chord.alt === true)) return false;
  return stroke.key.toLowerCase() === chord.key.toLowerCase();
}

/** The command this key event is the accelerator of, or `undefined`. */
export function commandFor(
  stroke: KeyStroke,
  modifier: AcceleratorModifier,
  typing: boolean,
): Command | undefined {
  return COMMANDS.find(
    (command) =>
      command.accelerator !== undefined &&
      (!typing || command.whileTyping === true) &&
      matches(command.accelerator, stroke, modifier),
  );
}

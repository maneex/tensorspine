/**
 * The context menu of §4.7, and the rule that it is not a menu of its own.
 *
 * > Right-click | context menu: Edit, Rename, Duplicate, Delete, Drill In, Add to Composition…,
 * > Extract to Composition, Extract to Template, Edit Primitive, Upgrade Pin…, Show in Explorer,
 * > Show in JSON, Copy D1 identifier
 *
 * §4.4 states the rule the menu is held to: **"Every context-menu command is also a command of the
 * menu bar or a button of the sheet."** Seven of the thirteen are commands of §4.4's own table and
 * are run through it, so they cannot become canvas-only; the other six are named below with what
 * they are instead, because §4.4's table does not carry them and a feature whose standing rule is
 * that the table is *transcribed, not designed* does not add to it (the ledger records the
 * mismatch as a finding for the plan).
 *
 * | Entry | What backs it |
 * |---|---|
 * | Edit | the selection's own sheet (§4.4's canvas shortcuts: "Enter opens the sheet of the selection") |
 * | Rename · Duplicate · Delete | `edit.rename`, `edit.duplicate`, `edit.delete` |
 * | Drill In | §4.8's drill-in tab — feature 2.14 |
 * | Add to Composition… · Extract to Composition | §4.20's composition flows — feature 2.14 |
 * | Extract to Template | `model.extract-to-template` |
 * | Edit Primitive | `library.edit-unit` |
 * | Upgrade Pin… | `model.upgrade-pins` |
 * | Show in Explorer | a navigation, not an edit: it selects the row in the Model explorer |
 * | Show in JSON | `view.json-source` |
 * | Copy D1 identifier | the clipboard, not the document |
 */

/** One entry of the menu. */
export interface MenuEntry {
  /** The entry's own identity, which is a §4.4 command id wherever one exists. */
  readonly id: string;
  /** §4.7's own wording, which is also the key of the interface's dictionary. */
  readonly label: string;
  /** Whether the entry is run through the command table of §4.4 (and so is in a menu). */
  readonly command: boolean;
  /** A rule drawn above this entry. */
  readonly separated?: true;
}

/** §4.7's context menu, in §4.7's own order. */
export const CONTEXT_MENU: readonly MenuEntry[] = [
  { id: 'canvas.open-sheet', label: 'Edit', command: false },
  { id: 'edit.rename', label: 'Rename', command: true },
  { id: 'edit.duplicate', label: 'Duplicate', command: true },
  { id: 'edit.delete', label: 'Delete', command: true },
  { id: 'canvas.drill-in', label: 'Drill In', command: false, separated: true },
  { id: 'canvas.add-to-composition', label: 'Add to Composition…', command: false },
  { id: 'canvas.extract-to-composition', label: 'Extract to Composition', command: false },
  { id: 'model.extract-to-template', label: 'Extract to Template', command: true },
  { id: 'library.edit-unit', label: 'Edit Primitive', command: true, separated: true },
  { id: 'model.upgrade-pins', label: 'Upgrade Pin…', command: true },
  { id: 'canvas.show-in-explorer', label: 'Show in Explorer', command: false, separated: true },
  { id: 'view.json-source', label: 'Show in JSON', command: true },
  { id: 'canvas.copy-identifier', label: 'Copy D1 identifier', command: false },
];

/**
 * Which entries apply to a box of one role.
 *
 * Nothing is hidden for a semantic reason (Q5): an entry that does not apply is *absent*, which
 * is a fact about what the thing is — a terminal has no primitive to edit and no iteration to
 * drill into — and never a refusal of a gesture the author could have made.
 */
export function entriesFor(role: string, roles: { node: string; group: string; terminal: string }): MenuEntry[] {
  return CONTEXT_MENU.filter((entry) => {
    if (role === roles.terminal) return TERMINAL_ENTRIES.includes(entry.id);
    if (role === roles.group) return !NODE_ONLY.includes(entry.id);
    return entry.id !== 'canvas.extract-to-composition';
  });
}

/** What a terminal offers: it is a name, an endpoint and a kind, and nothing else. */
const TERMINAL_ENTRIES: readonly string[] = [
  'canvas.open-sheet',
  'edit.rename',
  'edit.delete',
  'canvas.show-in-explorer',
  'view.json-source',
];

/** What only an instance offers: a primitive, a pin, a template. */
const NODE_ONLY: readonly string[] = [
  'library.edit-unit',
  'model.upgrade-pins',
  'canvas.add-to-composition',
];

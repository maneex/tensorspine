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
    return true;
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

/**
 * What only an instance offers: a primitive, a pin, and §4.20's two moves.
 *
 * `Extract to Composition` "turns selected **roots** into a new composition with a fresh index"
 * and `Add to Composition…` "moves a root **instance** into a composition as a site" (§4.7,
 * §4.20), so both are gestures on an instance and neither is one on a composition — a composition
 * is what they make. Feature 2.9 had the first the other way round, with no test either way;
 * feature 2.14, which owns the three composition flows, is where it is decided.
 */
const NODE_ONLY: readonly string[] = [
  'library.edit-unit',
  'model.upgrade-pins',
  'canvas.add-to-composition',
  'canvas.extract-to-composition',
];

/**
 * The menu of a **port** — §4.15's two gestures.
 *
 * > Creating an input from an unfed port: right-click the port ▸ "Expose as input…" (name proposed
 * > from the port). Creating an output: right-click an output port ▸ "Expose as output…" with the
 * > generative toggle.
 *
 * Both are commands of §4.4's Model menu (`Add Input`, `Add Output`), so the rule that every
 * context-menu command is also a command of the bar holds here as it holds for the box's menu; the
 * labels are §4.15's own, because that is what a port's menu says. Which of the two a port offers
 * is the side it is on, and nothing is hidden for a semantic reason (Q5): a **fed** input port
 * still offers the gesture, and V7 is what says the port is then fed twice.
 */
export const PORT_MENU: readonly (MenuEntry & { readonly side: string })[] = [
  { id: 'model.add-input', label: 'Expose as input…', command: true, side: 'inputs' },
  { id: 'model.add-output', label: 'Expose as output…', command: true, side: 'outputs' },
];

/**
 * What a port of one side offers, with §4.8's carry where the canvas is a drill-in.
 *
 * > Dragging a handle onto a ghost is offered through the context menu "Connect from previous
 * > iteration…" which writes the override … and proposes the guard.
 *
 * A ghost column is generated from a rule and is not a drop target, so the gesture is offered
 * where the *rule* would start: on the producing port. Choosing it arms the same connection the
 * drag makes, with the override on its source — and there is one entry **per index**, because a
 * composition over two indices has two previous iterations and the plan's own sentence names one
 * (`layer`). With a single index it reads as §4.8 writes it.
 */
export function portEntries(side: string, indices: readonly string[] = []): MenuEntry[] {
  const own = PORT_MENU.filter((entry) => entry.side === side);
  if (side !== 'outputs' || indices.length === 0) return own;
  const carry = indices.map((index) => ({
    id: `${CARRY}:${index}`,
    label: indices.length === 1 ? 'Connect from previous iteration…' : `Connect from previous ${index}…`,
    command: false,
    separated: true as const,
  }));
  return [...own, ...carry];
}

/** The entry "Connect from previous iteration…" carries, with the index it is about after it. */
export const CARRY = 'canvas.connect-previous';

/**
 * The entries a **drill-in** offers on a site — §4.8's canvas, one level down.
 *
 * Three of §4.7's thirteen have no meaning inside a composition, and their absence is a fact
 * about where the reader is rather than a refusal (the rule {@link entriesFor} already states):
 * `Drill In` and `Add to Composition…` are the way *into* one, and `Extract to Composition` turns
 * root instances into a composition, which a site already is. What is offered instead is §4.20's
 * own: **Duplicate with complementary guard**, the gesture that writes the second half of a
 * periodic pattern (`attn` / `attn_full`, `ffn_sparse` / `ffn`).
 */
export const DRILL_MENU: readonly MenuEntry[] = [
  {
    id: 'canvas.duplicate-complementary',
    label: 'Duplicate with complementary guard',
    command: false,
  },
];

/** Which entries a box of one role offers inside a drill-in. */
export function drillEntriesFor(
  role: string,
  roles: { node: string; group: string; terminal: string },
): MenuEntry[] {
  const entries = entriesFor(role, roles).filter((entry) => !INSIDE_A_COMPOSITION.includes(entry.id));
  if (role !== roles.node) return entries;
  const at = entries.findIndex((entry) => entry.id === 'edit.duplicate');
  const before = entries.slice(0, at + 1);
  return [...before, ...DRILL_MENU, ...entries.slice(at + 1)];
}

/** The three entries that are the way into a composition, and so are not offered inside one. */
const INSIDE_A_COMPOSITION: readonly string[] = [
  'canvas.drill-in',
  'canvas.add-to-composition',
  'canvas.extract-to-composition',
];

/**
 * The compositions "Add to Composition…" offers — one entry per composition of the document.
 *
 * §4.7 writes the entry with an ellipsis, which is what a gesture that asks something looks like;
 * what it asks is *which* composition, and the answer is a list of the document's own. A document
 * with no composition is offered nothing, which is the same rule again: an entry that does not
 * apply is absent, never a refusal.
 */
export function compositionEntries(names: readonly string[]): MenuEntry[] {
  return names.map((name) => ({
    id: `canvas.add-to-composition:${name}`,
    label: name,
    command: false,
  }));
}

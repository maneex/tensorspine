import { describe, expect, it } from 'vitest';

import {
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
  type Chord,
  type KeyStroke,
} from '../../src/shell/commands.js';

// §4.4's table, held to the plan word for word — and the three claims that table makes about
// itself: every entry is a command, every command is in a menu, and the palette lists all of
// them. The first is what this file transcribes; the second and third are what it proves.

/**
 * The menus and their commands, as `plans/graph-editor-plan.md` §4.4 writes them.
 *
 * Written out here rather than derived from the table under test, because a test that read the
 * implementation would agree with whatever the implementation says. Where §4.4 puts several
 * commands on one line — `Cut/Copy/Paste`, `Zoom In/Out/Fit`, `Theme: Light / Dark / System` —
 * they are the several commands it names, because the palette offers each of them by name.
 */
const SECTION_4_4: Readonly<Record<string, readonly string[]>> = {
  File: [
    'New Model',
    'New Template',
    'New Base…',
    'Open Folder…',
    'Open Model…',
    'Open Recent',
    'Save',
    'Save As…',
    'Save All',
    'Download Workspace as Zip',
    'Revert',
    'Export Derived Document…',
    'Export Diagram',
    'Close Tab',
    'Close All',
  ],
  Edit: [
    'Undo',
    'Redo',
    'Cut',
    'Copy',
    'Paste',
    'Duplicate',
    'Delete',
    'Rename',
    'Select All',
    'Find',
    'Preferences',
  ],
  View: [
    'Toggle Side Bar',
    'Toggle Properties',
    'Toggle Bottom Panel',
    'Problems',
    'Derived',
    'Log',
    'Zoom In',
    'Zoom Out',
    'Zoom to Fit',
    'Auto-layout',
    'Reset Layout',
    'Show Identities',
    'Show Derived Figures on Diagram',
    'Show Edge Types',
    'Show Families',
    'Expanded Graph',
    'Layer Preview',
    'JSON Source',
    'Theme: Light',
    'Theme: Dark',
    'Theme: System',
  ],
  Model: [
    'Validate Now',
    'Derive Now',
    'Lint',
    'Check Against Checkpoint…',
    'Set Preview Assignment…',
    'Add Quantity',
    'Add Constant',
    'Add Instance…',
    'Add Composition',
    'Add Input',
    'Add Output',
    'Extract to Template…',
    'Inline Template',
    'Upgrade Pins…',
    'Document Properties',
  ],
  Library: [
    'Open Base…',
    'New Base…',
    'Reload Bases',
    'Show Unit',
    'Edit Unit',
    'New Primitive…',
    'New Version…',
    'New Axis…',
    'New Precision Role…',
    'Start From…',
    'Argument Schema',
    'Preview Instance',
    'Publish Template…',
    'Unlock Base…',
  ],
  Weights: [
    'Open Checkpoint Folder…',
    'Open Hub Repository…',
    'Adopt config.json Values as Quantities…',
    'Locate Selected Slot…',
    'Check Locations',
    'Clear Checkpoint',
  ],
  Help: [
    'Specification',
    'Model JSON Guide',
    'Library-unit Guide',
    'Library Reference',
    'Glossary',
    'Derived Products',
    'Keyboard Shortcuts',
    'About',
  ],
};

/** The accelerators §4.4 writes, in its own notation, for a machine that is not an Apple one. */
const SECTION_4_4_SHORTCUTS: Readonly<Record<string, string>> = {
  Save: 'Ctrl+S',
  'Close Tab': 'Ctrl+W',
  Undo: 'Ctrl+Z',
  Cut: 'Ctrl+X',
  Copy: 'Ctrl+C',
  Paste: 'Ctrl+V',
  Duplicate: 'Ctrl+D',
  Delete: 'Del',
  Rename: 'F2',
  'Select All': 'Ctrl+A',
  Find: 'Ctrl+F',
  'Toggle Side Bar': 'Ctrl+B',
  'Toggle Bottom Panel': 'Ctrl+J',
  Problems: 'Ctrl+Shift+M',
  'Zoom In': 'Ctrl++',
  'Zoom Out': 'Ctrl+−',
  'Zoom to Fit': 'Ctrl+0',
  'Auto-layout': 'Shift+Alt+F',
  'JSON Source': 'Ctrl+Shift+J',
  'Validate Now': 'F7',
  'Derive Now': 'Shift+F7',
};

/** A key event, as much of one as a chord is decided on. */
function stroke(key: string, held: Partial<KeyStroke> = {}): KeyStroke {
  return { key, ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, ...held };
}

describe('the commands of §4.4', () => {
  it('is the seven menus of §4.2, in its order', () => {
    expect(MENUS.map((menu) => menu.label)).toEqual(Object.keys(SECTION_4_4));
  });

  it('carries every command the plan names, in the plan’s order, and no other', () => {
    for (const menu of MENUS) {
      expect(
        commandsOf(menu.id).map((command) => command.label),
        menu.label,
      ).toEqual(SECTION_4_4[menu.label]);
    }
    expect(COMMANDS).toHaveLength(Object.values(SECTION_4_4).flat().length);
  });

  it('registers each of them exactly once, under an identity nothing else carries', () => {
    const ids = COMMANDS.map((command) => command.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) expect(commandById(id)?.id).toBe(id);
    expect(commandById('file.no-such-command')).toBeUndefined();
  });

  it('names no vocabulary of the language in an identity or a label', () => {
    // Catching rule (b) of §1 is a scan of the sources; this is the same rule stated where the
    // strings are written, so a label chosen carelessly fails here first and with a reason.
    // `derived`, `none`, `value` and `fixed` are the words that come closest.
    const vocabulary = new Set(['derived', 'none', 'value', 'fixed', 'add', 'read', 'write']);
    for (const command of COMMANDS) {
      expect(vocabulary.has(command.id), command.id).toBe(false);
      expect(vocabulary.has(command.label), command.label).toBe(false);
    }
  });

  it('writes the accelerators §4.4 writes, and gives two more only where the plan defers', () => {
    const written = new Map(
      COMMANDS.filter((command) => command.accelerator !== undefined).map((command) => [
        command.label,
        acceleratorText(command.accelerator as Chord, 'control'),
      ]),
    );
    for (const [label, chord] of Object.entries(SECTION_4_4_SHORTCUTS)) {
      expect(written.get(label), label).toBe(chord);
    }
    // §4.4 defers to "the platform's conventions" and writes no chord for Redo; §4.2 states the
    // palette's in its region table rather than in §4.4. Those are the only two additions.
    expect([...written.keys()].filter((label) => SECTION_4_4_SHORTCUTS[label] === undefined)).toEqual(
      ['Redo'],
    );
    expect(acceleratorText(PALETTE_CHORD, 'control')).toBe('Ctrl+Shift+P');
  });

  it('writes a chord in Apple’s symbols where the platform says so', () => {
    expect(acceleratorText({ mod: true, key: 'S' }, 'command')).toBe('⌘S');
    expect(acceleratorText({ mod: true, shift: true, key: 'Z' }, 'command')).toBe('⇧⌘Z');
    expect(acceleratorText({ shift: true, alt: true, key: 'F' }, 'command')).toBe('⌥⇧F');
    expect(acceleratorText({ key: 'F7' }, 'command')).toBe('F7');
    expect(acceleratorText({ key: 'Delete' }, 'command')).toBe('Del');
  });
});

describe('a key event and the command it is', () => {
  it('reads `mod` as this platform’s modifier and refuses the other one', () => {
    const chord: Chord = { mod: true, key: 'S' };
    expect(matches(chord, stroke('s', { ctrlKey: true }), 'control')).toBe(true);
    expect(matches(chord, stroke('s', { metaKey: true }), 'control')).toBe(false);
    expect(matches(chord, stroke('s', { metaKey: true }), 'command')).toBe(true);
    expect(matches(chord, stroke('s', { ctrlKey: true }), 'command')).toBe(false);
  });

  it('compares a letter without regard to case and every modifier exactly', () => {
    expect(matches({ mod: true, key: 'Z' }, stroke('Z', { ctrlKey: true }), 'control')).toBe(true);
    expect(
      matches({ mod: true, key: 'Z' }, stroke('Z', { ctrlKey: true, shiftKey: true }), 'control'),
    ).toBe(false);
    expect(
      matches(
        { mod: true, shift: true, key: 'Z' },
        stroke('Z', { ctrlKey: true, shiftKey: true }),
        'control',
      ),
    ).toBe(true);
  });

  it('leaves the editing chords to whatever is being typed in', () => {
    const typing = true;
    expect(commandFor(stroke('x', { ctrlKey: true }), 'control', !typing)?.id).toBe('edit.cut');
    expect(commandFor(stroke('x', { ctrlKey: true }), 'control', typing)).toBeUndefined();
    expect(commandFor(stroke('Delete'), 'control', typing)).toBeUndefined();
    expect(commandFor(stroke('F2'), 'control', typing)).toBeUndefined();
    // Save, the panel toggles and the function keys are wanted wherever the focus is.
    expect(commandFor(stroke('s', { ctrlKey: true }), 'control', typing)?.id).toBe('file.save');
    expect(commandFor(stroke('b', { ctrlKey: true }), 'control', typing)?.id).toBe(
      'view.toggle-side-bar',
    );
    expect(commandFor(stroke('F7'), 'control', typing)?.id).toBe('model.validate-now');
    expect(commandFor(stroke('F7', { shiftKey: true }), 'control', typing)?.id).toBe(
      'model.derive-now',
    );
  });

  it('answers nothing for a stroke no command claims', () => {
    expect(commandFor(stroke('q', { ctrlKey: true }), 'control', false)).toBeUndefined();
    expect(commandFor(stroke('a'), 'control', false)).toBeUndefined();
  });

  it('gives no two commands the same chord', () => {
    const seen = new Map<string, string>();
    for (const command of COMMANDS) {
      if (command.accelerator === undefined) continue;
      const written = `${acceleratorText(command.accelerator, 'control')}${command.whileTyping === true ? ' typing' : ''}`;
      expect(seen.get(written), `${written} is ${seen.get(written) ?? ''} and ${command.label}`).toBeUndefined();
      seen.set(written, command.label);
    }
  });
});

describe('the command palette', () => {
  it('lists every command of §4.4 when nothing is typed', () => {
    expect(matching('').map((one) => one.command.id)).toEqual(COMMANDS.map((command) => command.id));
  });

  it('finds a command by its own name before its menu’s', () => {
    const found = matching('model').map((one) => one.command.id);
    // `Model JSON Guide` begins with the word; `New Model` and `Open Model…` carry it; the
    // fifteen commands of the Model menu are found by their menu and come after all three.
    expect(found[0]).toBe('help.model-guide');
    expect(found.slice(1, 3)).toEqual(['file.new-model', 'file.open-model']);
    expect(found.filter((id) => id.startsWith('model.'))).toHaveLength(15);
    expect(found.indexOf('model.lint')).toBeGreaterThan(found.indexOf('file.open-model'));
  });

  it('ranks two commands that both begin with the query in §4.4’s own order', () => {
    const found = matching('derive').map((one) => one.command.id);
    expect(found.slice(0, 2)).toEqual(['view.derived', 'model.derive-now']);
    expect(found).toContain('file.export-derived');
  });

  it('finds a whole menu by its name', () => {
    const found = matching('weights').map((one) => one.command.menu);
    expect(new Set(found.slice(0, 6))).toEqual(new Set(['menu.weights']));
  });

  it('ranks a label that starts with the query above one that merely contains it', () => {
    const menu = 'View';
    expect(scoreOf({ id: 'a', menu: 'menu.view', label: 'Zoom In' }, menu, 'zoom')).toBeGreaterThan(
      scoreOf({ id: 'b', menu: 'menu.view', label: 'Reset Zoom' }, menu, 'zoom'),
    );
  });

  it('answers nothing for a query nothing matches, and says so rather than listing all', () => {
    expect(matching('zzzz')).toEqual([]);
  });

  it('keeps §4.4’s own order among equals', () => {
    const found = matching('new').map((one) => one.command.id);
    expect(found.slice(0, 3)).toEqual(['file.new-model', 'file.new-template', 'file.new-base']);
  });
});

import { describe, expect, it } from 'vitest';

import { EditLog } from '../src/log.js';

// The command log of plan D13: "every gesture is a named command … with forward and inverse
// patches; the Edit menu names them", and the revision counter of §5.4, which is what every
// product's freshness is measured against.

interface Counter {
  readonly value: number;
  readonly names: readonly string[];
}

interface MutableCounter {
  value: number;
  names: string[];
}

const start: Counter = { value: 0, names: ['a'] };

function log(): EditLog<Counter, MutableCounter> {
  return new EditLog<Counter, MutableCounter>(start);
}

describe('an edit log', () => {
  it('answers the patches and the inverse of every edit', () => {
    const one = log();
    const edit = one.apply('Set value', (draft) => {
      draft.value = 3;
    });
    expect(edit.changed).toBe(true);
    expect(edit.patches).toEqual([{ op: 'replace', path: ['value'], value: 3 }]);
    expect(edit.inverse).toEqual([{ op: 'replace', path: ['value'], value: 0 }]);
    expect(edit.state).toEqual({ value: 3, names: ['a'] });
  });

  it('leaves the state it was given untouched', () => {
    const one = log();
    one.apply('Add a name', (draft) => {
      draft.names.push('b');
    });
    expect(start).toEqual({ value: 0, names: ['a'] });
    expect(one.state.names).toEqual(['a', 'b']);
  });

  it('counts a revision per state, an undo among them', () => {
    // §5.4 dims a product computed for an older revision. An undo is a new state, so it is a new
    // revision: the products of the state it restores were computed against a revision that will
    // never come back.
    const one = log();
    expect(one.revision).toBe(0);
    one.apply('Set value', (draft) => {
      draft.value = 1;
    });
    expect(one.revision).toBe(1);
    one.undo();
    expect(one.revision).toBe(2);
    one.redo();
    expect(one.revision).toBe(3);
    expect(one.state).toEqual({ value: 1, names: ['a'] });
  });

  it('records nothing for an edit that changes nothing', () => {
    const one = log();
    const edit = one.apply('Set value', (draft) => {
      draft.value = 0;
    });
    expect(edit.changed).toBe(false);
    expect(one.revision).toBe(0);
    expect(one.canUndo).toBe(false);
  });

  it('names what an undo and a redo would do — the Edit menu’s two lines', () => {
    const one = log();
    expect(one.undoLabel).toBeNull();
    expect(one.redoLabel).toBeNull();
    one.apply('Add instance attn', (draft) => {
      draft.names.push('attn');
    });
    expect(one.undoLabel).toBe('Add instance attn');
    expect(one.redoLabel).toBeNull();
    one.undo();
    expect(one.undoLabel).toBeNull();
    expect(one.redoLabel).toBe('Add instance attn');
  });

  it('restores exactly what was there, through several edits', () => {
    const one = log();
    one.apply('first', (draft) => {
      draft.names.splice(0, 0, 'z');
    });
    one.apply('second', (draft) => {
      draft.value = 7;
      draft.names.push('y');
    });
    expect(one.state).toEqual({ value: 7, names: ['z', 'a', 'y'] });
    one.undo();
    expect(one.state).toEqual({ value: 0, names: ['z', 'a'] });
    one.undo();
    expect(one.state).toEqual(start);
    expect(one.undo()).toBeNull();
  });

  it('drops the redo stack when a new edit is made', () => {
    const one = log();
    one.apply('first', (draft) => {
      draft.value = 1;
    });
    one.undo();
    expect(one.canRedo).toBe(true);
    one.apply('another', (draft) => {
      draft.value = 2;
    });
    expect(one.canRedo).toBe(false);
    expect(one.state.value).toBe(2);
  });

  it('keeps a history as deep as it was told to', () => {
    const one = new EditLog<Counter, MutableCounter>(start, { depth: 2 });
    for (const value of [1, 2, 3]) {
      one.apply(`set ${String(value)}`, (draft) => {
        draft.value = value;
      });
    }
    expect(one.history).toEqual(['set 2', 'set 3']);
    one.undo();
    one.undo();
    expect(one.canUndo).toBe(false);
    expect(one.state.value).toBe(1);
  });

  it('tells its listeners after every state change, until they stop listening', () => {
    const one = log();
    const seen: number[] = [];
    const stop = one.subscribe((_, revision) => seen.push(revision));
    one.apply('first', (draft) => {
      draft.value = 1;
    });
    one.undo();
    stop();
    one.redo();
    expect(seen).toEqual([1, 2]);
  });
});

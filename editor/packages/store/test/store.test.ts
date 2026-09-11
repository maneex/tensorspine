import { JsonParseError, serialize } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import { rename, setValue } from '../src/commands.js';
import { DocumentStore, MODEL_ROLE, packageName } from '../src/index.js';
import { EditError } from '../src/path.js';
import { corpusText, shapes } from './source.js';
import { QUANTITY } from './selectors.js';

describe('@tensorspine/store', () => {
  it('names itself as the workspace declares it', () => {
    expect(packageName).toBe('@tensorspine/store');
  });
});

// The store of plan D1: the document *is* the model, and everything else is a projection of it.
// What the store adds to the tree is the history, the revision counter §5.4 measures freshness
// against, the reference index, and the text the file is saved as.

describe('the document store', () => {
  function open(name = 'llama3-8b'): DocumentStore {
    return DocumentStore.open(corpusText(name), shapes());
  }

  it('holds the tree the core parsed, and writes the bytes the file holds (D12)', () => {
    const text = corpusText('llama3-8b');
    const store = open();
    expect(store.text).toBe(text);
    expect(serialize(store.tree)).toBe(text);
    expect(store.role).toBe(MODEL_ROLE);
  });

  it('refuses a text that is not a document, in the core’s own words', () => {
    expect(() => DocumentStore.open('[]', shapes())).toThrow(EditError);
    expect(() => DocumentStore.open('{', shapes())).toThrow(JsonParseError);
  });

  it('counts a revision per state and re-renders only when the state moved', () => {
    const store = open();
    expect(store.revision).toBe(0);
    const first = store.text;
    expect(store.text).toBe(first);

    store.run((context) =>
      setValue(context, { path: ['model'], value: 'llama3-8b-probe', label: 'Set the model id' }),
    );
    expect(store.revision).toBe(1);
    expect(store.text).not.toBe(first);
    expect(store.text).toContain('"model": "llama3-8b-probe"');

    store.undo();
    expect(store.revision).toBe(2);
    expect(store.text).toBe(first);
  });

  it('indexes the document it holds, and again after an edit', () => {
    const store = open();
    const before = store.index;
    expect(store.index).toBe(before);
    expect(before.of('quantity', 'd')).toHaveLength(10);

    store.run((context) =>
      rename(context, { path: ['quantities', 'd'], to: 'width', references: QUANTITY }),
    );
    expect(store.index).not.toBe(before);
    expect(store.index.of('quantity', 'd')).toEqual([]);
    expect(store.index.of('quantity', 'width')).toHaveLength(10);
  });

  it('reads what the schemas say about names once, whatever the document', () => {
    const first = open();
    const second = open('qwen3.5-4b-text');
    expect(second.referenceTags).toBe(first.referenceTags);
  });

  it('names the edits for the Edit menu, and takes them back one at a time', () => {
    const store = open();
    expect(store.canUndo).toBe(false);
    expect(store.undo()).toBeNull();
    expect(store.redo()).toBeNull();

    store.run((context) =>
      rename(context, { path: ['quantities', 'd'], to: 'width', references: QUANTITY }),
    );
    expect(store.history).toEqual(['Rename d to width']);
    expect(store.undoLabel).toBe('Rename d to width');
    expect(store.redoLabel).toBeNull();

    const undone = store.undo();
    expect(undone?.label).toBe('Rename d to width');
    expect(undone?.moves).toEqual([]);
    expect(store.redoLabel).toBe('Rename d to width');
    expect(store.redo()?.changed).toBe(true);
    expect(store.text).toContain('"quantity": "width"');
  });

  it('tells its listeners, which is what the pipeline of §5.4 hangs off', () => {
    const store = open();
    const seen: number[] = [];
    const stop = store.subscribe((_, revision) => seen.push(revision));
    store.run((context) => setValue(context, { path: ['model'], value: 'one' }));
    store.run((context) => setValue(context, { path: ['model'], value: 'two' }));
    stop();
    store.run((context) => setValue(context, { path: ['model'], value: 'three' }));
    expect(seen).toEqual([1, 2]);
  });

  it('is untouched by a command that raises half-way through', () => {
    // A recipe that throws discards its draft: the store is what it was, and the revision has not
    // moved, so a caller that catches the refusal is holding the document it was holding.
    const store = open();
    const text = store.text;
    expect(() =>
      store.apply({
        label: 'Half an edit',
        moves: [],
        edit(draft) {
          draft.members.push({ name: 'stray', value: null });
          throw new EditError('no further');
        },
      }),
    ).toThrow(EditError);
    expect(store.text).toBe(text);
    expect(store.revision).toBe(0);
    expect(store.canUndo).toBe(false);
  });

  it('applies a command built against a context the caller already holds', () => {
    // The confirmation of a delete is shown from the command and applied afterwards (§4.7), so
    // the two steps are separate on purpose.
    const store = open();
    const command = setValue(store.context, { path: ['model'], value: 'shown-first' });
    expect(store.text).toContain('"model": "llama3-8b"');
    const applied = store.apply(command);
    expect(applied.changed).toBe(true);
    expect(store.text).toContain('"model": "shown-first"');
  });
});

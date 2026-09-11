import { describe, expect, it } from 'vitest';

import { literalIndex, serialize, type FoldedHandle, type JsonObject } from '@tensorspine/lang';
import { DocumentStore, pointerOf, type Path } from '@tensorspine/store';

import {
  addInstance,
  connectHandles,
  declaresAt,
  duplicateAt,
  referencesTo,
  removeAt,
  renameAt,
  type GestureContext,
} from '../../src/canvas/gestures.js';
import { bindings, corpusTree, reading, registry, shapes } from './source.js';

/**
 * Every gesture of §4.7's interactions table, as a command of the store.
 *
 * The rule each of them is held to is Q5's: **no gesture is refused for a semantic reason**. A
 * connection onto an input that is already fed replaces the older edge rather than refusing; a
 * delete that cannot empty a container the grammar requires says so instead of stopping; a
 * dropped primitive writes D5's skeleton, which V2 will refuse and Problems will list.
 */

const context: GestureContext = { shapes, bindings };

/** One corpus document in a store, so a gesture can be made on it and read back. */
function storeOf(name: string): DocumentStore {
  return new DocumentStore(corpusTree(name), shapes);
}

/** The handle of one box's port, as the canvas builds it (`Canvas.handleFor`'s own reading). */
function handle(name: string, pointer: string, port: string): FoldedHandle {
  const node = reading(name).folded.byPointer.get(pointer);
  if (node === undefined) throw new Error(`no box at ${pointer}`);
  return {
    box: node.parent ?? node.pointer,
    site: node.pointer,
    name: node.name,
    port,
    indices: (node.site?.indices ?? []).map((one) => ({
      name: one.name,
      written: literalIndex(one.value as bigint),
      value: one.value,
    })),
    where: node.where,
    value: node.where === null ? null : `${node.where}.${port}`,
    boundary: node.parent !== null,
  };
}

describe('dropping a primitive from the palette', () => {
  it('writes D5’s skeleton with the proposed name and family (§9 Q4)', () => {
    const store = storeOf('llama3-8b');
    const applied = store.apply(
      addInstance(store.context, { primitive: 'norm.rms', version: '1.0.0' }),
    );
    expect(applied.changed).toBe(true);
    expect(applied.label).toBe('Add instance rms');
    const written = store.text;
    expect(written).toContain('"rms": {');
    expect(written).toContain('"name": "norm.rms"');
    expect(written).toContain('"arguments": {}');
    expect(written).toContain('"norm"');
  });

  it('uniquifies the proposal against what the document already has', () => {
    const store = storeOf('llama3-8b');
    store.apply(addInstance(store.context, { primitive: 'norm.rms', version: '1.0.0' }));
    const second = store.apply(
      addInstance(store.context, { primitive: 'norm.rms', version: '1.0.0' }),
    );
    expect(second.label).toBe('Add instance rms_2');
  });

  it('leaves the document on the grammar, which is where V2 refuses it (D5)', () => {
    const store = storeOf('llama3-8b');
    store.apply(addInstance(store.context, { primitive: 'attention.dense', version: '1.0.0' }));
    expect(registry.conforms(store.tree, 'model')).toBe(true);
  });
});

describe('connecting two handles', () => {
  it('writes the endpoints the schema has, byte for byte as the corpus writes them', () => {
    const store = storeOf('llama3-8b');
    const from = handle('llama3-8b', '/instances/final_n', 'output');
    const to = handle('llama3-8b', '/instances/lm_head', 'input');
    const before = bindingAt(corpusTree('llama3-8b'), 'lm_head.in');
    const applied = store.apply(connectHandles(store.context, from, to));
    // The corpus writes that very edge under the author's own name `lm_head.in`; §4.7 names a new
    // binding `<to>.<port>`, so the connection *replaces* it and the endpoints are the same bytes
    // under the proposed name. What a rule is called is the author's, and the canvas proposes.
    expect(pointerOf(applied.replaced as Path)).toBe('/bindings/values/lm_head.in');
    expect(serialize(bindingAt(store.tree, 'lm_head.input'))).toBe(serialize(before));
    expect(store.text).not.toContain('"lm_head.in"');
  });

  it('replaces the edge an already-fed input had, and says which (V7, Q5)', () => {
    const store = storeOf('llama3-8b');
    const from = handle('llama3-8b', '/instances/embed', 'output');
    const to = handle('llama3-8b', '/instances/lm_head', 'input');
    const applied = store.apply(connectHandles(store.context, from, to));
    expect(applied.replaced).toBeDefined();
    expect(pointerOf(applied.replaced as Path)).toBe('/bindings/values/lm_head.in');
    expect(store.text).toContain('"instance": "embed"');
    // One command, so one undo takes the replacement back with it (D13).
    store.undo();
    expect(store.text).toBe(serialize(corpusTree('llama3-8b')));
  });

  it('never refuses a connection the core would refuse (Q5)', () => {
    const store = storeOf('llama3-8b');
    // `embed.tokens` is an *input*, and `lm_head.logits` is a generative output: a wire from one
    // to the other is nonsense, and the gesture makes it anyway. The refusal is Problems'.
    const from = handle('llama3-8b', '/instances/lm_head', 'logits');
    const to = handle('llama3-8b', '/instances/embed', 'tokens');
    const applied = store.apply(connectHandles(store.context, from, to));
    expect(applied.changed).toBe(true);
    expect(store.text).toContain('"embed.tokens"');
    expect(registry.conforms(store.tree, 'model')).toBe(true);
  });

  it('names a site of a composition at the iteration its card stands for', () => {
    const store = storeOf('llama3-8b');
    const from = handle('llama3-8b', '/instances/embed', 'output');
    const to = handle('llama3-8b', '/compositions/decoder/instances/attn_n', 'input');
    const before = bindingAt(corpusTree('llama3-8b'), 'decoder.entry');
    const applied = store.apply(connectHandles(store.context, from, to));
    // The card stands for `[layer=0]`, which is exactly what `decoder.entry` writes: the same
    // selector, the same literal index, the same bytes — so the connection replaces that edge.
    expect(pointerOf(applied.replaced as Path)).toBe('/bindings/values/decoder.entry');
    expect(serialize(bindingAt(store.tree, 'attn_n.input'))).toBe(serialize(before));
  });
});

describe('deleting a box', () => {
  it('cascades over everything that names it, and lists what it kept (§4.7)', () => {
    const store = storeOf('llama3-8b');
    const command = removeAt(context, store.context, ['instances', 'final_n']);
    expect(command.label).toBe('Delete instance final_n');
    const cascade = command.cascade;
    expect(cascade).toBeDefined();
    expect((cascade?.removed ?? []).map((place) => pointerOf(place))).toEqual([
      '/bindings/values/lm_head.in',
      '/bindings/values/final_n.in',
      '/bindings/parameters/final_n.weight',
      '/instances/final_n',
    ]);
    const applied = store.apply(command);
    expect(applied.changed).toBe(true);
    expect(store.text).not.toContain('"final_n"');
    expect(registry.conforms(store.tree, 'model')).toBe(true);
  });

  it('keeps a reference the grammar will not let go, and reports it (wire first, fix later)', () => {
    const store = storeOf('llama3-8b');
    const command = removeAt(context, store.context, ['instances', 'embed']);
    expect((command.cascade?.kept ?? []).map((place) => pointerOf(place))).toEqual([
      '/interfaces/inputs/tokens/to/0',
    ]);
  });
});

describe('renaming a box', () => {
  it('rewrites every occurrence the bindings say names it', () => {
    const store = storeOf('llama3-8b');
    const applied = store.apply(renameAt(context, store.context, ['instances', 'final_n'], 'tail_n'));
    expect(applied.label).toBe('Rename instance final_n to tail_n');
    expect(store.text).not.toContain('"final_n"');
    expect(store.text).toContain('"instance": "tail_n"');
    expect(registry.conforms(store.tree, 'model')).toBe(true);
  });

  it('renames a site of a composition against its own scope', () => {
    const store = storeOf('llama3-8b');
    store.apply(
      renameAt(context, store.context, ['compositions', 'decoder', 'instances', 'attn'], 'sa'),
    );
    expect(store.text).toContain('"sa": {');
    expect(store.text).toContain('"site": "sa"');
    expect(registry.conforms(store.tree, 'model')).toBe(true);
  });
});

describe('duplicating a box', () => {
  it('copies the declaration under a name nothing else has, and connects nothing', () => {
    const store = storeOf('llama3-8b');
    const command = duplicateAt(context, store.context, ['instances', 'final_n']);
    expect(command.name).toBe('final_n_copy');
    store.apply(command);
    expect(store.text).toContain('"final_n_copy"');
    expect(registry.conforms(store.tree, 'model')).toBe(true);
    // Its ports are fed by nothing, which V7 will say and Problems will list.
    expect(store.text.match(/"instance": "final_n_copy"/g)).toBeNull();
  });
});

describe('what a gesture reads from presentation.json', () => {
  it('names what each map declares', () => {
    expect(declaresAt(context, ['instances', 'embed'])).toBe('instance');
    expect(declaresAt(context, ['compositions', 'decoder'])).toBe('composition');
    expect(declaresAt(context, ['compositions', 'decoder', 'instances', 'attn'])).toBe('site');
    expect(declaresAt(context, ['interfaces', 'inputs', 'tokens'])).toBe('input');
  });

  it('reads a scoped rule against the composition the declaration sits in', () => {
    const selectors = referencesTo(context, ['compositions', 'decoder', 'instances', 'attn']);
    expect(selectors.length).toBeGreaterThan(0);
    expect(selectors.some((one) => one.qualifiers?.['composition'] === 'decoder')).toBe(true);
  });
});

describe('every gesture, on every corpus document', () => {
  const MODELS = ['llama3-8b', 'qwen3.5-4b-text', 'gemma3n-kvshare', 'colbert-v2'];

  it.each(MODELS)('%s stays on the grammar whatever is done to it (D5)', (name) => {
    for (const gesture of ['add', 'rename', 'delete', 'duplicate'] as const) {
      const store = storeOf(name);
      const first = firstInstance(store.tree);
      if (first === null) continue;
      if (gesture === 'add') {
        store.apply(addInstance(store.context, { primitive: 'norm.rms', version: '1.0.0' }));
      }
      if (gesture === 'rename') {
        store.apply(renameAt(context, store.context, first, `${String(first[1])}_x`));
      }
      if (gesture === 'delete') store.apply(removeAt(context, store.context, first));
      if (gesture === 'duplicate') store.apply(duplicateAt(context, store.context, first));
      expect(registry.conforms(store.tree, 'model'), `${name} ${gesture}`).toBe(true);
    }
  });

  it.each(MODELS)('%s comes back byte for byte when a gesture is undone', (name) => {
    const store = storeOf(name);
    const before = store.text;
    const first = firstInstance(store.tree);
    if (first === null) return;
    store.apply(renameAt(context, store.context, first, 'renamed_for_the_suite'));
    expect(store.text).not.toBe(before);
    store.undo();
    expect(store.text).toBe(before);
  });
});

/** One value binding of a document, by its rule name. */
function bindingAt(tree: JsonObject, rule: string): JsonObject {
  const bindings = member(tree, 'bindings');
  const values = bindings === null ? null : member(bindings, 'values');
  const found = values === null ? null : member(values, rule);
  if (found === null) throw new Error(`no binding ${rule}`);
  return found;
}

/** One member of an object node, or `null`. */
function member(node: JsonObject, name: string): JsonObject | null {
  const found = node.members.find((one) => one.name === name)?.value;
  return found === undefined || found === null || typeof found !== 'object' || Array.isArray(found)
    ? null
    : (found as JsonObject);
}

/** The first root instance of a document, as a path. */
function firstInstance(tree: JsonObject): Path | null {
  const instances = tree.members.find((member) => member.name === 'instances')?.value;
  if (instances === null || instances === undefined || typeof instances !== 'object') return null;
  const node = instances as JsonObject;
  const first = node.members[0];
  return first === undefined ? null : ['instances', first.name];
}

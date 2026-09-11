import { isJsonObject, type JsonObject, type JsonValue } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import {
  addToMap,
  connect,
  insertItem,
  remove,
  rename,
  setMemberAt,
  unique,
  type Command,
  type EditContext,
} from '../src/commands.js';
import { DocumentStore } from '../src/document.js';
import { nodeAt, type Path } from '../src/path.js';
import { corpusNames, corpusText, MODEL, registry, shapes } from './source.js';
import { COMPOSITION, INDEX, QUANTITY, ROOT_INSTANCE, site } from './selectors.js';

// The gesture test of plan §2 D5: **every command applied to every corpus document leaves it on
// the grammar**, and — D13's other half — **an undo restores a byte-identical serialisation**.
//
// The two are one suite because they are one claim about a gesture: it produces a document the
// editor can go on reading, and it can be taken back exactly. The grammar is Ajv over the schema
// files themselves (D4, feature 1.1's `conforms`, 1–6 ms on the corpus), and the bytes are the
// core's serializer (D12), which feature 0.3 measured to reproduce every file of the corpus.
//
// The gestures are derived from each document rather than written per document: whatever the
// first quantity, the first composition, the first site or the first binding of a document is,
// the same nine kinds of gesture are made on it, and the sites and root instances are each
// deleted in turn. That is what makes the suite say something about the corpus and not about
// `llama3-8b`.

/**
 * One gesture, with the name the failure will carry.
 *
 * A gesture resolves its own target against the tree it is made on and answers `null` when the
 * document no longer has it: the suite makes each one on the document as it stands, first one at
 * a time and then one after another, and after a rename or a delete some of them have nothing
 * left to act on. That is the honest reading of "a gesture" — the canvas offers what the document
 * has — and it is what lets the same list be run twice.
 */
interface Gesture {
  readonly what: string;
  readonly make: (context: EditContext) => Command | null;
}

function members(tree: JsonObject, path: Path): readonly string[] {
  const node = nodeAt(tree, path);
  return node !== undefined && isJsonObject(node) ? node.members.map((member) => member.name) : [];
}

function valueAt(tree: JsonObject, path: Path): JsonValue | null {
  return nodeAt(tree, path) ?? null;
}

/** An empty argument map, which is what D5's placeholder writes. */
const NO_ARGUMENTS: JsonValue = { kind: 'object', members: [] };

/**
 * The gestures of §4.7's interactions table, read off the document they are made on.
 *
 * A document without a top-level value binding (the template, whose every edge is scoped) simply
 * has no connection gesture; nothing here invents a shape a document has not.
 */
function gesturesFor(tree: JsonObject): Gesture[] {
  const found: Gesture[] = [];
  const instances = members(tree, ['instances']);
  const quantities = members(tree, ['quantities']);
  const compositions = members(tree, ['compositions']);
  const edges = members(tree, ['bindings', 'values']);
  const first = instances[0];
  const composition = compositions[0];

  if (first !== undefined) {
    found.push({
      what: 'drop a primitive from the palette (D5’s placeholder)',
      make: (context) => {
        const primitive = valueAt(context.tree, ['instances', first, 'primitive']);
        if (primitive === null) return null;
        return addToMap(context, {
          path: ['instances'],
          name: unique('probe', members(context.tree, ['instances'])),
          values: { primitive, arguments: NO_ARGUMENTS, families: ['probe_family'] },
        });
      },
    });
    found.push({
      what: 'add a family chip',
      make: (context) => {
        const families = valueAt(context.tree, ['instances', first, 'families']);
        if (families === null || !Array.isArray(families) || families.includes('probe_family')) {
          return null;
        }
        return insertItem(context, {
          path: ['instances', first, 'families'],
          value: 'probe_family',
        });
      },
    });
    found.push({
      what: 'add a guard',
      make: (context) => {
        if (valueAt(context.tree, ['instances', first]) === null) return null;
        return setMemberAt(context, {
          path: ['instances', first],
          name: 'when',
          value: { kind: 'object', members: [{ name: 'boolean', value: true }] },
        });
      },
    });
    found.push({
      what: 'rename a root instance',
      make: (context) => {
        if (valueAt(context.tree, ['instances', first]) === null) return null;
        return rename(context, {
          path: ['instances', first],
          to: unique('probe_name', members(context.tree, ['instances'])),
          references: ROOT_INSTANCE,
        });
      },
    });
    for (const name of instances) {
      found.push({
        what: `delete the root instance ${name}`,
        make: (context) =>
          valueAt(context.tree, ['instances', name]) === null
            ? null
            : remove(context, { path: ['instances', name], references: ROOT_INSTANCE }),
      });
    }
  }

  const quantity = quantities[0];
  if (quantity !== undefined) {
    found.push({
      what: 'rename a quantity',
      make: (context) =>
        valueAt(context.tree, ['quantities', quantity]) === null
          ? null
          : rename(context, {
              path: ['quantities', quantity],
              to: unique('probe_quantity', members(context.tree, ['quantities'])),
              references: QUANTITY,
            }),
    });
  }

  if (composition !== undefined) {
    const sites = members(tree, ['compositions', composition, 'instances']);
    const indices = members(tree, ['compositions', composition, 'indices']);
    const site0 = sites[0];
    const index0 = indices[0];
    found.push({
      what: 'drop a primitive into a composition (the drill-in)',
      make: (context) => {
        if (site0 === undefined) return null;
        const primitive = valueAt(context.tree, [
          'compositions',
          composition,
          'instances',
          site0,
          'primitive',
        ]);
        if (primitive === null) return null;
        return addToMap(context, {
          path: ['compositions', composition, 'instances'],
          name: unique('probe', members(context.tree, ['compositions', composition, 'instances'])),
          values: { primitive, arguments: NO_ARGUMENTS, families: ['probe_family'] },
        });
      },
    });
    if (site0 !== undefined) {
      found.push({
        what: 'rename a site',
        make: (context) =>
          valueAt(context.tree, ['compositions', composition, 'instances', site0]) === null
            ? null
            : rename(context, {
                path: ['compositions', composition, 'instances', site0],
                to: unique(
                  'probe_site',
                  members(context.tree, ['compositions', composition, 'instances']),
                ),
                references: site(composition),
              }),
      });
      for (const name of sites) {
        found.push({
          what: `delete the site ${name}`,
          make: (context) =>
            valueAt(context.tree, ['compositions', composition, 'instances', name]) === null
              ? null
              : remove(context, {
                  path: ['compositions', composition, 'instances', name],
                  references: site(composition),
                }),
        });
      }
    }
    if (index0 !== undefined) {
      found.push({
        what: 'rename an index',
        make: (context) =>
          valueAt(context.tree, ['compositions', composition, 'indices', index0]) === null
            ? null
            : rename(context, {
                path: ['compositions', composition, 'indices', index0],
                to: unique('probe_index', members(context.tree, ['compositions', composition, 'indices'])),
                references: INDEX(composition),
              }),
      });
    }
    found.push({
      what: 'rename a composition',
      make: (context) =>
        valueAt(context.tree, ['compositions', composition]) === null
          ? null
          : rename(context, {
              path: ['compositions', composition],
              to: unique('probe_composition', members(context.tree, ['compositions'])),
              references: COMPOSITION,
            }),
    });
  }

  const edge = edges[0];
  const other = edges[1];
  if (edge !== undefined) {
    found.push({
      what: 'connect an output to an input already fed (the older edge is replaced, V7)',
      make: (context) => {
        const from = valueAt(context.tree, ['bindings', 'values', edge, 'from']);
        const to = valueAt(context.tree, ['bindings', 'values', edge, 'to']);
        if (from === null || to === null) return null;
        return connect(context, {
          path: ['bindings', 'values'],
          name: unique('probe.edge', members(context.tree, ['bindings', 'values'])),
          values: { from, to },
          into: 'to',
        });
      },
    });
    if (other !== undefined) {
      found.push({
        what: 'connect an output to another input',
        make: (context) => {
          const from = valueAt(context.tree, ['bindings', 'values', edge, 'from']);
          const to = valueAt(context.tree, ['bindings', 'values', other, 'to']);
          if (from === null || to === null) return null;
          return connect(context, {
            path: ['bindings', 'values'],
            name: unique('probe.edge', members(context.tree, ['bindings', 'values'])),
            values: { from, to },
            into: 'to',
          });
        },
      });
    }
  }

  return found;
}

describe('every gesture, on every corpus document', () => {
  const names = corpusNames();

  it('reads the fourteen models and the template', () => {
    expect(names).toHaveLength(15);
    expect(names).toContain('llama3-8b');
    expect(names).toContain('decoder-causal-yarn/1.0.0');
  });

  it.each(names)(
    '%s stays on the grammar, and every undo restores its bytes',
    (name) => {
      const text = corpusText(name);
      const store = DocumentStore.open(text, shapes());
      expect(store.text, `${name} is not written as the core writes it`).toBe(text);
      expect(registry().conforms(store.tree, MODEL), `${name} is off the grammar as it stands`).toBe(
        true,
      );

      const gestures = gesturesFor(store.tree);
      expect(gestures.length, name).toBeGreaterThan(4);

      for (const gesture of gestures) {
        const command = gesture.make(store.context);
        expect(command, `${name}: ${gesture.what} has nothing to act on`).not.toBeNull();
        const applied = store.apply(command!);
        expect(
          registry().conforms(store.tree, MODEL),
          `${name}: ${gesture.what} left the grammar`,
        ).toBe(true);
        if (!applied.changed) {
          // A gesture with no answer on the grammar writes nothing and says why (the delete
          // cascade's `kept`). There is nothing to undo, and the document is untouched.
          expect(applied.cascade?.kept.length ?? 0, `${name}: ${gesture.what}`).toBeGreaterThan(0);
          expect(store.text, `${name}: ${gesture.what}`).toBe(text);
          continue;
        }
        expect(store.undo()?.label, `${name}: ${gesture.what}`).toBe(applied.label);
        expect(store.text, `${name}: ${gesture.what} did not come back`).toBe(text);
      }

      // And once more, the whole battery one after another: a history several edits deep comes
      // back to the same bytes, one undo at a time.
      const made: string[] = [];
      for (const gesture of gestures) {
        const command = gesture.make(store.context);
        if (command === null) continue;
        const applied = store.apply(command);
        if (applied.changed) made.push(applied.label);
        expect(registry().conforms(store.tree, MODEL), `${name}: ${gesture.what} in sequence`).toBe(
          true,
        );
      }
      expect(store.history).toEqual(made);
      expect(made.length, name).toBeGreaterThan(4);
      while (store.canUndo) store.undo();
      expect(store.text, `${name} did not come back after ${String(made.length)} edits`).toBe(text);
    },
    120_000,
  );
});

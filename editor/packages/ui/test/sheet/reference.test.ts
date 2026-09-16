import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  identityKey,
  isJsonObject,
  parse,
  primitiveCatalog,
  serialize,
  type JsonObject,
  type JsonValue,
} from '@tensorspine/lang';
import {
  DocumentStore,
  pathOfPointer,
  pointerOf,
  SchemaShapes,
  type Path,
} from '@tensorspine/store';

import { shapeAt } from '../../src/sheet/places.js';
import { referenceMembers, referenceOf, writeReference } from '../../src/sheet/reference.js';
import { identityOf } from '../../src/library/primitives.js';
import { formContext } from '../../src/forms/index.js';
import { readPresentation } from '../../src/presentation/index.js';
import { context, library, registry } from './source.js';
import { repositoryRoot } from '../presentation/source.js';

// One field over `primitive_reference`'s two members — feature 2.21, plan §4.11.
//
// Two claims, and they are the ones the corpus can settle: **the field reads** `name@version` on
// every pinned primitive of every corpus document, and **what it writes is byte-identical** to
// what the file already had. The second is the round trip of feature 0.3 with an edit in the
// middle of it: every reference of every document is rewritten with the identity the field would
// commit, and the bytes must not move — which is the only way to know that one field over two
// members has not quietly reordered, retyped or reformatted either half.

const shapes = new SchemaShapes(registry);

/** Every corpus document, by name. */
function corpus(): string[] {
  return readdirSync(join(repositoryRoot, 'data', 'models'))
    .filter((name) => name.endsWith('.json'))
    .sort();
}

function textOf(file: string): string {
  return readFileSync(join(repositoryRoot, 'data', 'models', file), 'utf8');
}

/** Every place a document writes a pinned primitive, found by walking for the widget's anchor. */
function referencePlaces(tree: JsonObject): Path[] {
  const found: Path[] = [];
  const walk = (value: JsonValue | undefined, at: Path): void => {
    if (Array.isArray(value)) {
      value.forEach((item: JsonValue, index) => {
        walk(item, [...at, index] as Path);
      });
      return;
    }
    if (value === undefined || !isJsonObject(value)) return;
    // The place is read from the **schema**, not from a member name: a place whose shape is the
    // definition the editor is bound at is a pinned primitive, wherever the grammar writes one.
    if (referenceMembers(context, shapeAt(shapes, at, 'model')) !== null && at.length > 0) {
      found.push(at);
      return;
    }
    for (const member of value.members) walk(member.value, [...at, member.name] as Path);
  };
  walk(tree, [] as Path);
  return found;
}

describe('which member is which', () => {
  const at = pathOfPointer('/instances/embed/primitive');

  it('pairs them so that the document reads as an identity the catalog carries', () => {
    const tree = parse(textOf('llama3-8b.json')) as JsonObject;
    const members = referenceMembers(context, shapeAt(shapes, at, 'model'));
    expect(members).not.toBeNull();
    const held = referenceOf(nodeOf(tree, at), members as NonNullable<typeof members>);
    expect(primitiveCatalog(library()).some((one) => one.id === identityOf(held))).toBe(true);
  });

  it('is told apart by the picker the binding gives the version, never by its name', () => {
    // The claim, by mutation: with the version's own binding gone there is nothing to tell the
    // two members apart by, and the editor answers `null` — the row then renders generically,
    // which is §1's own answer for a construct nobody has bound, rather than writing the halves
    // the wrong way round.
    const file = JSON.parse(
      readFileSync(join(repositoryRoot, 'editor/packages/ui/src/presentation.json'), 'utf8'),
    ) as Record<string, unknown>;
    delete file[
      'https://tensorspine.dev/schema/2.0/model.json#/$defs/primitive_reference/properties/version'
    ];
    const without = formContext(registry, readPresentation(file));
    expect(referenceMembers(without, shapeAt(shapes, at, 'model'))).toBeNull();
  });

  it('answers nothing at a place that is no such pair, so the row renders generically', () => {
    expect(
      referenceMembers(context, shapeAt(shapes, pathOfPointer('/interfaces/inputs/tokens'), 'model')),
    ).toBeNull();
  });
});

describe('the field reads every pinned primitive of the corpus', () => {
  it('shows `name@version`, the library’s own form, at every one of them', () => {
    let seen = 0;
    for (const file of corpus()) {
      const tree = parse(textOf(file)) as JsonObject;
      for (const at of referencePlaces(tree)) {
        const members = referenceMembers(context, shapeAt(shapes, at, 'model'));
        expect(members, `${file} ${pointerOf(at)}`).not.toBeNull();
        const held = referenceOf(nodeOf(tree, at), members as NonNullable<typeof members>);
        expect(identityOf(held), `${file} ${pointerOf(at)}`).toBe(
          identityKey(held.name, held.version),
        );
        expect(identityOf(held)).toMatch(/^[^@]+@\d+\.\d+\.\d+$/);
        seen += 1;
      }
    }
    // Every corpus document pins primitives, and there are a hundred and more of them.
    expect(seen).toBeGreaterThan(100);
  });

  it('reads the identity the catalog carries, wherever the document pins one of the base’s', () => {
    const catalog = new Set(primitiveCatalog(library()).map((one) => one.id));
    const tree = parse(textOf('llama3-8b.json')) as JsonObject;
    for (const at of referencePlaces(tree)) {
      const members = referenceMembers(context, shapeAt(shapes, at, 'model'));
      const held = referenceOf(nodeOf(tree, at), members as NonNullable<typeof members>);
      expect(catalog, pointerOf(at)).toContain(identityOf(held));
    }
  });
});

describe('what the field writes', () => {
  it('is byte-identical to the corpus’s own spelling, document by document', () => {
    for (const file of corpus()) {
      const text = textOf(file);
      const store = DocumentStore.open(text, shapes);
      for (const at of referencePlaces(store.tree)) {
        const members = referenceMembers(context, shapeAt(shapes, at, 'model'));
        const held = referenceOf(nodeOf(store.tree, at), members as NonNullable<typeof members>);
        store.apply(
          writeReference(store.context, at, members as NonNullable<typeof members>, held, 'Set'),
        );
      }
      expect(serialize(store.tree), file).toBe(text);
    }
  });

  it('is one command, so one gesture is one undo (D13)', () => {
    const store = DocumentStore.open(textOf('llama3-8b.json'), shapes);
    const at = pathOfPointer('/instances/embed/primitive');
    const members = referenceMembers(context, shapeAt(shapes, at, 'model'));
    const before = serialize(store.tree);
    const wanted = primitiveCatalog(library()).find(
      (one) => one.id !== identityOf(referenceOf(nodeOf(store.tree, at), members as NonNullable<typeof members>)),
    );
    store.apply(
      writeReference(
        store.context,
        at,
        members as NonNullable<typeof members>,
        { name: wanted?.name ?? '', version: wanted?.version ?? '' },
        'Set primitive',
      ),
    );
    expect(serialize(store.tree)).not.toBe(before);
    store.undo();
    expect(serialize(store.tree)).toBe(before);
  });

  it('writes both halves, and writes them where the file already wrote them', () => {
    const store = DocumentStore.open(textOf('llama3-8b.json'), shapes);
    const at = pathOfPointer('/instances/embed/primitive');
    const members = referenceMembers(context, shapeAt(shapes, at, 'model'));
    const held = nodeOf(store.tree, at) as JsonObject;
    const order = held.members.map((one) => one.name);
    const wanted = primitiveCatalog(library())[0];
    store.apply(
      writeReference(
        store.context,
        at,
        members as NonNullable<typeof members>,
        { name: wanted?.name ?? '', version: wanted?.version ?? '' },
        'Set primitive',
      ),
    );
    const after = nodeOf(store.tree, at) as JsonObject;
    expect(after.members.map((one) => one.name)).toEqual(order);
    expect(
      referenceOf(after, members as NonNullable<typeof members>),
    ).toEqual({ name: wanted?.name, version: wanted?.version });
  });
});

/** The value at a place of a tree — the suites' own step-down, since the store keeps no reader. */
function nodeOf(tree: JsonObject, at: Path): JsonValue | undefined {
  let held: JsonValue | undefined = tree;
  for (const step of at) {
    if (held === undefined) return undefined;
    if (typeof step === 'number') {
      held = Array.isArray(held) ? (held[step] as JsonValue | undefined) : undefined;
      continue;
    }
    held = isJsonObject(held)
      ? held.members.find((one) => one.name === step)?.value
      : undefined;
  }
  return held;
}

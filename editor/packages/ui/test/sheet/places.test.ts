import { describe, expect, it } from 'vitest';

import {
  basesOf,
  derive,
  foldedGraph,
  identityReadings,
  isJsonObject,
  loadLibrary,
  quantityReadings,
  serialize,
  streamRows,
  toPython,
  valueRows,
  type JsonObject,
  type Library,
  type PyRecord,
  type PyValue,
} from '@tensorspine/lang';
import { DocumentStore, type Path } from '@tensorspine/store';

import { outlineOf, type OutlineRow } from '../../src/explorer/outline.js';
import { presentation } from '../../src/presentation/index.js';
import {
  addDeclaration,
  declaringMaps,
  exposeAt,
  interfaceMap,
  mapDeclaring,
} from '../../src/sheet/add.js';
import { mapTable, placeSheet, type PlaceRequest } from '../../src/sheet/places.js';
import { blankValue } from '../../src/sheet/skeleton.js';
import { context, fileSource, MODEL, registry, tree } from './source.js';

/**
 * The other sheets of §4.11 and the tables of §4.16, as data — feature 2.12.
 *
 * What these hold to account is the *claim* the sheets rest on: **every row is the generated form
 * of the place's own `$def`, and every section beside it is a fact the core or the products
 * answered**. So the suite walks every declaration of every corpus document through `placeSheet`
 * and requires a form for each, then pins the facts on the places the artboard draws — the
 * quantity's resolved value and its used-by count, the identity's D3 rows, the interface's D2
 * value, the composition's counts — against the core's own answers.
 */

const LLAMA = 'llama3-8b';

/** The reference base, gathered once through the loader's own file interface. */
let gathered: Library | null = null;

function library(document: JsonObject, name: string): Library {
  if (gathered !== null) return gathered;
  const { bases } = basesOf(`data/models/${name}.json`, toPython(document));
  gathered = loadLibrary(bases, { schemas: registry, source: fileSource() });
  return gathered;
}

/** The derived document of a corpus model, derived once per suite run. */
const derivations = new Map<string, PyValue>();

function derivedOf(name: string): PyValue {
  const held = derivations.get(name);
  if (held !== undefined) return held;
  const document = tree(name);
  const made = derive(document, { schemas: registry, library: library(document, name) });
  derivations.set(name, made);
  return made;
}

/** Everything a sheet of one document is built from, as the interface gathers it per revision. */
function requestFor(name: string, path: Path, options: { derived?: boolean } = {}): PlaceRequest {
  const document = tree(name);
  const python = toPython(document);
  const derived = options.derived === false ? null : derivedOf(name);
  const rows = outlineOf({
    tree: document,
    shapes: context.shapes,
    bindings: presentation(),
    role: MODEL,
    openAll: true,
  });
  const at = pointerOf(path);
  return {
    tree: document,
    role: MODEL,
    context,
    path,
    row: rows.find((row) => row.pointer === at) ?? null,
    rows: new Map(rows.map((row) => [row.pointer, row])),
    index: DocumentStore.open(serialize(document), context.shapes).context.index,
    quantities: quantityReadings(python as PyRecord),
    identities: identityReadings(python),
    derived,
    values: derived === null ? [] : valueRows(derived),
    streams: derived === null ? [] : streamRows(derived),
    folded: foldedGraph(document),
    problems: [],
  };
}

/** RFC 6901, as the store writes it. */
function pointerOf(path: Path): string {
  return path.map((step) => `/${String(step).replace(/~/g, '~0').replace(/\//g, '~1')}`).join('');
}

/** Every row of the outline of one document, whatever is open. */
function outline(name: string): readonly OutlineRow[] {
  return outlineOf({
    tree: tree(name),
    shapes: context.shapes,
    bindings: presentation(),
    role: MODEL,
    openAll: true,
  });
}

describe('every declaration of the corpus gets a sheet', () => {
  for (const name of [LLAMA, 'qwen3.5-4b-text', 'gemma3n-kvshare']) {
    it(`generates one for every place ${name} declares`, () => {
      const rows = outline(name).filter((row) => row.kind === 'entry' || row.kind === 'document');
      expect(rows.length).toBeGreaterThan(10);
      for (const row of rows) {
        const sheet = placeSheet({ ...requestFor(name, row.path, { derived: false }), row });
        // A form with rows, and every widget one the walker decides or a binding names.
        expect(sheet.form.rows.length, row.pointer).toBeGreaterThan(0);
        for (const each of sheet.form.rows) {
          expect(typeof each.widget, `${row.pointer} ${each.path}`).toBe('string');
        }
        expect(sheet.pointer).toBe(row.pointer);
      }
    });
  }

  it('renders nothing generically that the walker has no reading for', () => {
    // §1's "unknown constructs get the generic widget … and is listed in the log": what the
    // corpus writes is rendered by a widget the walker decides, never by the fallback.
    const generic: string[] = [];
    for (const row of outline(LLAMA)) {
      const sheet = placeSheet({ ...requestFor(LLAMA, row.path, { derived: false }), row });
      for (const each of sheet.form.rows) {
        if (each.generic !== undefined) generic.push(`${row.pointer}${each.path}: ${each.generic}`);
      }
    }
    expect(generic).toEqual([]);
  });
});

describe('the quantity sheet', () => {
  it('shows what the core resolves it to, and what a rename would rewrite', () => {
    const sheet = placeSheet(requestFor(LLAMA, ['quantities', 'd'] as Path));
    expect(sheet.declares).toBe('quantity');
    expect(sheet.quantity?.value).toBe(4096n);
    expect(sheet.quantity?.computed).toBe(false);
    // Erratum E13's own figure for this document: ten, not the corpus's 604.
    expect(sheet.usedBy).toHaveLength(10);
    expect(sheet.referenced).toBe(true);
    expect(sheet.usedBy.every((used) => used.tag === 'quantity')).toBe(true);
  });

  it('says a literal declares its derivation, which is where §4.11 puts the editor', () => {
    const sheet = placeSheet(requestFor(LLAMA, ['quantities', 'head_dim'] as Path));
    expect(sheet.quantity?.computed).toBe(true);
    expect(sheet.quantity?.value).toBe(128n);
    // The row is under `source`, which is where the schema writes it: the sheet does not lift it.
    const row = sheet.form.rows.find((each) => each.path === '/source/derivation');
    expect(row?.present).toBe(true);
    expect(row?.widget).toBe('expression');
  });

  it('does not count a declaration’s own place among the uses of it', () => {
    // A composition's indices are referred to under the map's own member as well as by `{"index":
    // …}`, so the declaration would count itself if the place were not taken out.
    const sheet = placeSheet(
      requestFor(LLAMA, ['compositions', 'decoder', 'indices', 'layer'] as Path),
    );
    expect(sheet.referenced).toBe(true);
    expect(sheet.usedBy.map((used) => used.pointer)).not.toContain(sheet.pointer);
    expect(sheet.usedBy.length).toBeGreaterThan(0);
  });

  it('finds the one quantity of the corpus nothing names — deepseek’s `kv_heads`', () => {
    const sheet = placeSheet(requestFor('deepseek-v4-pro', ['quantities', 'kv_heads'] as Path, {
      derived: false,
    }));
    expect(sheet.usedBy).toEqual([]);
    expect(sheet.referenced).toBe(true);
  });
});

describe('§4.16’s table', () => {
  const table = mapTable(requestFor(LLAMA, ['quantities'] as Path));

  it('has the entry definition’s own members as columns, and the core’s two after them', () => {
    expect(table?.declares).toBe('quantity');
    expect(table?.columns).toEqual(['name', 'type', 'source', 'domain', 'used by', 'resolved']);
    expect(table?.rows).toHaveLength(9);
  });

  it('marks the row the editor has a notice about — §4.16’s own hint, feature 2.8’s row', () => {
    // The mark is the *notice*, not a rule of the table's: the panel and the table say the same
    // thing about the same declaration. deepseek-v4-pro's `kv_heads` is the corpus's one unread
    // quantity, and llama3-8b has none.
    const request = {
      ...requestFor('deepseek-v4-pro', ['quantities'] as Path, { derived: false }),
      notices: [{ path: '/quantities/kv_heads' }],
    };
    const marked = mapTable(request)?.rows.filter((row) => row.unused) ?? [];
    expect(marked.map((row) => row.name)).toEqual(['kv_heads']);
    expect(mapTable(requestFor(LLAMA, ['quantities'] as Path))?.rows.some((row) => row.unused)).toBe(
      false,
    );
  });

  it('reads each cell from the row of the entry’s own form', () => {
    const row = table?.rows.find((each) => each.name === 'd');
    expect(row?.cells.map((cell) => cell?.text ?? null)).toEqual(['cardinality', 'literal 4096', null]);
    expect(row?.usedBy).toBe(10);
    expect(row?.resolved).toBe('4096');
    expect(row?.unused).toBe(false);
  });

  it('draws the derivation under the row S15 draws it under', () => {
    const row = table?.rows.find((each) => each.name === 'head_dim');
    expect(row?.note).toBe('derivation d div heads');
    expect(row?.cells[1]?.text).toBe('literal 128');
  });

  it('is a table of every map of declarations, not of three', () => {
    // The rule is the map's own binding, so the interfaces and the constants have one too — and
    // so would a map the grammar grew.
    const inputs = mapTable(requestFor(LLAMA, ['interfaces', 'inputs'] as Path));
    expect(inputs?.declares).toBe('input');
    expect(inputs?.columns).toEqual(['name', 'to', 'kind', 'stream', 'fragmented', 'used by']);
    expect(inputs?.rows[0]?.name).toBe('tokens');
    const constants = mapTable(requestFor(LLAMA, ['constants'] as Path));
    expect(constants?.declares).toBe('constant');
    // No corpus document declares one (finding F8): the table exists and is empty.
    expect(constants?.rows).toEqual([]);
  });

  it('is `null` where the place is not a map of declarations', () => {
    expect(mapTable(requestFor(LLAMA, ['quantities', 'd'] as Path))).toBeNull();
    expect(mapTable(requestFor(LLAMA, []))).toBeNull();
  });
});

describe('the identity, edge and interface sheets', () => {
  it('gives a scoped parameter rule its identity and the products’ rows', () => {
    const sheet = placeSheet(
      requestFor(LLAMA, ['compositions', 'decoder', 'bindings', 'parameters', 'attn.q'] as Path),
    );
    expect(sheet.identity?.reading.identity).toBe('decoder.attn.q');
    expect(sheet.identity?.reading.kind).toBe('parameters');
    expect(sheet.identity?.derived.instances).toBe(32);
    expect(sheet.identity?.derived.bytes).toBe(32n * 33554432n);
    // Nothing refers to a rule name, so the sheet says so rather than showing an empty list.
    expect(sheet.referenced).toBe(false);
  });

  it('gives an edge the D2 value its producing end carries', () => {
    const sheet = placeSheet(requestFor(LLAMA, ['bindings', 'values', 'decoder.entry'] as Path));
    expect(sheet.edge?.rule).toBe('decoder.entry');
    expect(sheet.value?.value).toBe('embed.output');
    expect(sheet.value?.geometry).toBe('bf16[tokens, model.width=4096]');
    expect(sheet.identity).toBeNull();
  });

  it('gives a public input its value, what it is required for, and its stream', () => {
    const sheet = placeSheet(requestFor(LLAMA, ['interfaces', 'inputs', 'tokens'] as Path));
    expect(sheet.value?.input).toBe('tokens');
    expect(sheet.value?.requiredFor).toEqual(['logits']);
    expect(sheet.stream?.name).toBe('tokens');
    expect(sheet.stream?.kind).toBe('token');
  });

  it('gives a public output the value it exposes', () => {
    const sheet = placeSheet(requestFor(LLAMA, ['interfaces', 'outputs', 'logits'] as Path));
    expect(sheet.value?.value).toBe('lm_head.logits');
    expect(sheet.value?.bytesPerElement).toBe(513024);
    // Nothing in a document names an output (feature 2.2's `"refers": []`).
    expect(sheet.referenced).toBe(false);
  });

  it('gives a composition what it holds — S3’s own summary line', () => {
    const sheet = placeSheet(requestFor(LLAMA, ['compositions', 'decoder'] as Path));
    // The core's own summary of a composition (feature 2.9's `FoldedHeld`), which is what S3
    // writes on the box: the sites and the scoped bindings, container by container.
    expect(sheet.held.map((held) => `${held.name} ${String(held.count)}`)).toEqual([
      'instances 6',
      'values 8',
      'parameters 9',
      'states 1',
    ]);
    // Its indices are rows of the sheet, which is what §4.11 asks of a composition.
    const stop = sheet.form.rows.find((row) => row.path === '/indices/layer/stop');
    expect(stop?.widget).toBe('expression');
  });

  it('gives the document’s root the sheet §4.11 calls the Document sheet', () => {
    const sheet = placeSheet(requestFor(LLAMA, []));
    expect(sheet.pointer).toBe('');
    expect(sheet.table).toBeNull();
    const members = sheet.form.rows.filter((row) => row.depth === 1).map((row) => row.label);
    // The root's own members, in the schema's order — the counts S15 draws are those maps'.
    // The walker's own order: the schema's `properties`, required first (`version` is optional
    // and is declared between `model` and `primitive_libraries`, which is where it is shown).
    expect(members).toEqual([
      'schema',
      'model',
      'version',
      'primitive_libraries',
      'quantities',
      'constants',
      'instances',
      'compositions',
      'bindings',
      'interfaces',
    ]);
  });
});

describe('what a gesture writes', () => {
  it('finds every map of declarations the document has, and the word each declares', () => {
    const maps = declaringMaps(context, tree(LLAMA), MODEL);
    expect(maps.map((one) => `${pointerOf(one.path)} ${one.declares}`)).toEqual([
      '/quantities quantity',
      '/constants constant',
      '/instances instance',
      '/compositions composition',
      '/interfaces/inputs input',
      '/interfaces/outputs output',
    ]);
    expect(mapDeclaring(context, tree(LLAMA), MODEL, 'quantity')?.path).toEqual(['quantities']);
    // §4.15's two, told apart by the side `presentation.json` draws their terminals on.
    expect(interfaceMap(context, tree(LLAMA), MODEL, 'left')).toEqual(['interfaces', 'inputs']);
    expect(interfaceMap(context, tree(LLAMA), MODEL, 'right')).toEqual(['interfaces', 'outputs']);
  });

  it('adds a quantity on the grammar, from the schema alone (D5)', () => {
    const store = opened(LLAMA);
    let added = '';
    store.run((edit) => {
      const command = addDeclaration(edit, { context, role: MODEL, path: ['quantities'] as Path });
      added = command.name;
      return command;
    });
    expect(added).toBe('quantity');
    const made = store.tree;
    const written = memberAt(made, ['quantities', 'quantity']);
    expect(written).toEqual({
      type: { kind: 'cardinality' },
      source: { kind: 'literal', value: 0 },
    });
    // On the grammar: Ajv on the schema itself says so, which is what D5 requires of a gesture.
    expect(registry.structural(made, MODEL)).toEqual([]);
  });

  it('uniquifies the name it proposes, which is the word the map declares', () => {
    const store = opened(LLAMA);
    for (const expected of ['quantity', 'quantity_2', 'quantity_3']) {
      let added = '';
      store.run((edit) => {
        const command = addDeclaration(edit, { context, role: MODEL, path: ['quantities'] as Path });
        added = command.name;
        return command;
      });
      expect(added).toBe(expected);
    }
  });

  it('exposes a port as an output, with the endpoint the core writes', () => {
    const store = opened(LLAMA);
    const graph = foldedGraph(store.tree);
    const box = graph.byPointer.get('/instances/final_n');
    const handle = {
      box: '/instances/final_n',
      site: '/instances/final_n',
      name: 'final_n',
      port: 'output',
      indices: [],
      where: box?.where ?? null,
      value: 'final_n.output',
      boundary: false,
    };
    let added = '';
    store.run((edit) => {
      const command = exposeAt(edit, {
        context,
        role: MODEL,
        path: ['interfaces', 'outputs'] as Path,
        handle,
      });
      added = command.name;
      return command;
    });
    expect(added).toBe('output');
    const made = store.tree;
    expect(memberAt(made, ['interfaces', 'outputs', 'output'])).toEqual({
      from: { instance: { kind: 'root', instance: 'final_n' }, port: 'output' },
      generative: false,
    });
    // The document it leaves is on the grammar, which the gesture from a port always can be.
    expect(registry.structural(made, MODEL)).toEqual([]);
  });

  it('exposes a port as an input, whose endpoint goes in the list its schema declares', () => {
    const store = opened(LLAMA);
    const handle = {
      box: '/instances/lm_head',
      site: '/instances/lm_head',
      name: 'lm_head',
      port: 'input',
      indices: [],
      where: 'lm_head',
      value: 'lm_head.input',
      boundary: false,
    };
    store.run((edit) =>
      exposeAt(edit, { context, role: MODEL, path: ['interfaces', 'inputs'] as Path, handle }),
    );
    const made = store.tree;
    expect(memberAt(made, ['interfaces', 'inputs', 'input'])).toEqual({
      to: [{ instance: { kind: 'root', instance: 'lm_head' }, port: 'input' }],
      kind: 'sequence',
    });
  });
});

describe('the blank of a place', () => {
  it('is the first value the grammar admits, alternative by alternative', () => {
    const shape = context.shapes.at(
      'https://tensorspine.dev/schema/2.0/model.json#/$defs/quantity_domain',
    );
    // The first alternative of the union, built from its own required members: an interval whose
    // bounds are optional is its `kind` alone.
    expect(plain(blankValue(context, shape))).toEqual({ kind: 'interval' });
  });

  it('is off the grammar where the grammar needs a name nobody can guess', () => {
    const shape = context.shapes.at(
      'https://tensorspine.dev/schema/2.0/model.json#/$defs/value_endpoint',
    );
    // An endpoint needs an instance and a port, and the blank writes the empty text for both:
    // D5's own arrangement — Ajv says so on the row at once — and the reason `Expose as …` fills
    // them in from the port rather than starting from this.
    expect(plain(blankValue(context, shape))).toEqual({
      instance: { kind: 'root', instance: '' },
      port: '',
    });
  });
});

// --- what a suite needs of the store: the store itself, which is what the editor runs ---------

/** One corpus document in a store of its own, as the editor holds an open document (D1). */
function opened(name: string): DocumentStore {
  return DocumentStore.open(serialize(tree(name)), context.shapes);
}

/** A member of a tree, as plain JSON — what an assertion reads. */
function memberAt(document: JsonObject, path: readonly string[]): unknown {
  let held: unknown = document;
  for (const step of path) {
    if (!isJsonObject(held as never)) return undefined;
    held = (held as JsonObject).members.find((member) => member.name === step)?.value;
  }
  return plain(held);
}

/** A node of the tree as plain JSON: the members in order, the numbers as numbers. */
function plain(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((one) => plain(one));
  if (value !== null && typeof value === 'object') {
    const node = value as { kind?: string; value?: number; members?: { name: string; value: unknown }[] };
    if (node.kind === 'number') return node.value;
    if (node.members !== undefined) {
      return Object.fromEntries(node.members.map((member) => [member.name, plain(member.value)]));
    }
  }
  return value;
}

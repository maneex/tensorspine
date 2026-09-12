import { describe, expect, it } from 'vitest';

import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

import {
  languageAnchors,
  parse,
  serialize,
  type JsonValue,
  type SchemaRegistry,
} from '@tensorspine/lang';
import { SchemaShapes, type Shape } from '@tensorspine/store';

import { parseAt, printAt, type PrintContext } from '../../src/expressions/index.js';
import { presentation } from '../../src/presentation/index.js';
import {
  readRepositoryFile,
  registry as repositorySchemas,
  repositoryRoot,
} from '../presentation/source.js';

/**
 * The round trip of plan §6 and of §4.13's own sentence:
 *
 * > Round-trip is exact: text → JSON → text is the identity on the corpus's expressions and on
 * > every expression of the reference base (a test).
 *
 * Both directions are asserted, because only the pair says what the feature claims. **Text →
 * JSON → text** is what §4.13 writes and what a reader typing over a printed expression depends
 * on. **JSON → text → JSON** is the stronger one and the one D1 needs: the document is the model,
 * so an expression opened in the text view and saved back untouched has to be the same bytes —
 * the same tags, the same nesting, the same number lexemes (`1e-05` is not `0.00001`, `1.0` is not
 * `1`).
 *
 * The places are found **by the grammar**, never by a list of tags: a value is an expression where
 * the schema at its place names one of the four language anchors *and the grammar accepts the
 * value there* — which is what separates `{"op": …}` from the `{"record": …}` written at the very
 * same place, `argument_value` being either. So the walk covers exactly what the editor will open
 * an expression editor on, and a construct nobody thought of is covered by existing rather than by
 * being remembered.
 */

const registry: SchemaRegistry = repositorySchemas();
const shapes = new SchemaShapes(registry);
const context: PrintContext = { registry, shapes, bindings: presentation() };
const anchors = languageAnchors(registry);
const LANGUAGE = [anchors.expression, anchors.condition, anchors.unitExpression, anchors.unitCondition];

/** One expression of the repository, with the anchor it stands at and the file it is written in. */
interface Written {
  readonly anchor: string;
  readonly value: JsonValue;
  readonly file: string;
  /** Where in the document, as a JSON pointer — what a failure names. */
  readonly at: string;
}

/** Every `.json` under a directory of the repository, in the order the file system lists them. */
function filesUnder(directory: string, into: string[] = []): string[] {
  for (const name of readdirSync(join(repositoryRoot, directory)).sort()) {
    const path = `${directory}/${name}`;
    if (statSync(join(repositoryRoot, path)).isDirectory()) filesUnder(path, into);
    else if (name.endsWith('.json')) into.push(path);
  }
  return into;
}

/** The walk: every place of a document whose schema and value make it an expression. */
function collect(shape: Shape, value: JsonValue, where: Written, into: Written[]): void {
  const named = new Set(shape.all.map((place) => place.anchor));
  for (const anchor of LANGUAGE) {
    if (!named.has(anchor)) continue;
    if (!registry.accepts(value, anchor)) continue;
    into.push({ ...where, anchor, value });
    return;
  }
  if (Array.isArray(value)) {
    (value as readonly JsonValue[]).forEach((item, index) => {
      collect(shapes.item(shape, index), item, { ...where, at: `${where.at}/${String(index)}` }, into);
    });
    return;
  }
  if (typeof value !== 'object' || value === null || !('members' in value)) return;
  for (const member of (value as { members: readonly { name: string; value: JsonValue }[] }).members) {
    collect(
      shapes.member(shape, member.name),
      member.value,
      { ...where, at: `${where.at}/${member.name}` },
      into,
    );
  }
}

/** Every expression and condition the repository writes, corpus and reference base alike. */
function repositoryExpressions(): readonly Written[] {
  const found: Written[] = [];
  for (const file of [...filesUnder('data/models'), ...filesUnder('data/primitive-library')]) {
    const role = file.startsWith('data/models') ? 'model' : 'primitive-library-unit';
    const tree = parse(readRepositoryFile(file));
    collect(shapes.root(role), tree, { anchor: '', value: tree, file, at: '' }, found);
  }
  return found;
}

const written = repositoryExpressions();

describe('every expression the repository writes', () => {
  it('is found by the grammar, on both sides of the language', () => {
    const byAnchor = new Map<string, number>();
    for (const one of written) byAnchor.set(one.anchor, (byAnchor.get(one.anchor) ?? 0) + 1);
    // Four anchors, all of them exercised: the model's expressions and conditions, the unit's.
    expect([...byAnchor.keys()].sort()).toEqual([...LANGUAGE].sort());
    for (const anchor of LANGUAGE) expect(byAnchor.get(anchor) ?? 0, anchor).toBeGreaterThan(50);
    // 1 758 on the fifteen corpus documents and the 131 files of the reference base, measured:
    // 1 166 model expressions, 79 model conditions, 411 unit expressions, 102 unit conditions.
    expect(written.length).toBeGreaterThan(1700);
  });

  it('prints, parses and prints again to the same text — §4.13’s own identity', () => {
    const moved: string[] = [];
    for (const one of written) {
      const text = printAt(context, one.anchor, one.value);
      const read = parseAt(context, one.anchor, text);
      if (read.value === undefined) {
        moved.push(`${one.file}${one.at}: ${text} — ${read.problems.map((p) => p.message).join('; ')}`);
        continue;
      }
      const again = printAt(context, one.anchor, read.value);
      if (again !== text) moved.push(`${one.file}${one.at}: ${text} → ${again}`);
    }
    expect(moved).toEqual([]);
  });

  it('parses back to the very JSON it was printed from, lexemes and nesting included', () => {
    const moved: string[] = [];
    for (const one of written) {
      const text = printAt(context, one.anchor, one.value);
      const read = parseAt(context, one.anchor, text);
      if (read.value === undefined) continue;
      // Compared as bytes, because that is what D12 makes the editor answerable for: a number
      // written `1.0` and one written `1` are the same double and only the text tells them apart.
      const before = serialize(one.value);
      const after = serialize(read.value);
      if (before !== after) moved.push(`${one.file}${one.at}: ${text}\n  ${before}\n  ${after}`);
    }
    expect(moved).toEqual([]);
  });

  it('stays on the grammar: every parsed value is accepted where it stands', () => {
    for (const one of written) {
      const text = printAt(context, one.anchor, one.value);
      const read = parseAt(context, one.anchor, text);
      expect(read.value !== undefined && registry.accepts(read.value, one.anchor), text).toBe(true);
    }
  });
});

describe('what the repository’s own expressions make the text form answer', () => {
  const textsOf = (anchor: string): Set<string> =>
    new Set(written.filter((one) => one.anchor === anchor).map((one) => printAt(context, anchor, one.value)));

  it('writes a string literal quoted, so a name and a literal are two texts', () => {
    // `mask = causal` and `causal = true` are both written in the reference base: the first
    // compares an argument with the literal text `causal`, the second an argument *named* causal
    // with `true`. A bare `causal` would read as the second in both places.
    const unit = textsOf(anchors.unitCondition);
    expect(unit.has('mask = "causal"')).toBe(true);
    expect(unit.has('causal = true')).toBe(true);
  });

  it('writes a presence test as present(path), which §4.13’s symbol list ends with', () => {
    expect(textsOf(anchors.unitCondition).has('streaming and streaming = true')).toBe(false);
    expect(textsOf(anchors.unitCondition).has('present(streaming) and streaming = true')).toBe(true);
  });

  it('writes the one conditional expression the corpus holds with its keywords', () => {
    const model = textsOf(anchors.expression);
    expect([...model].filter((one) => one.startsWith('if '))).toEqual([
      'if $layer >= 20 then "shared" else "own"',
    ]);
  });

  it('keeps the two readings of a * b * c apart, which the repository writes both of', () => {
    const unit = textsOf(anchors.unitExpression);
    // `attention.latent_compressed` writes `multiply(2, multiply(overlap, head_dim))` and the
    // corpus's `gdn_conv` the flat `multiply(2, gdn_k, gdn_hd)`: the parentheses are the only
    // thing that tells the parser which of the two a text means.
    expect(unit.has('2 * (compress.overlap * head_dim)')).toBe(true);
    expect(textsOf(anchors.expression).has('2 * gdn_k * gdn_hd + gdn_v * gdn_hd')).toBe(true);
    // And an equal precedence read left to right: `floor_divide(multiply(h, hd), o_groups)`.
    expect(unit.has('heads * head_dim div o_groups')).toBe(true);
  });

  it('prints artboard S7’s three lines', () => {
    expect(textsOf(anchors.expression).has('d div heads')).toBe(true);
    expect(textsOf(anchors.condition).has('$layer mod 5 = 4 and $layer >= 4')).toBe(true);
    expect(textsOf(anchors.unitCondition).has('heads mod kv_heads = 0')).toBe(true);
  });
});

/**
 * The blank of a place: the smallest value its schema admits.
 *
 * Plan D5 is the whole of it — "the document stays on-schema while editing; incomplete meaning is
 * the semantic stage's report". Every gesture that *creates* something writes one of these: `Add
 * quantity` writes a `quantity_definition`, a chooser set to another alternative writes that
 * alternative, a list's `Add` appends its item, a map's takes a name and writes its value. What
 * each of them writes is read from the schema at that place and from nothing else:
 *
 * - a member the schema **fixes** takes its constant (`{"kind": "literal"}`, `tensorspine/2.0`);
 * - a member it **enumerates** takes the first value of the enumeration, in the schema's order;
 * - a **union** takes its first alternative, built the same way;
 * - an **object** takes its required members, each built the same way, in the schema's order;
 * - a **map** is empty and a **list** is empty, whatever their minimum — a map with no name to
 *   give an entry and a list with no item to put in it are what the author is about to fill, and
 *   inventing a name would be inventing a declaration;
 * - a **scalar** takes `blankOf`'s answer (feature 2.3): the first option, zero, false, or the
 *   empty text.
 *
 * **The blank can be off the grammar, and that is the honest answer.** A `value_endpoint` needs an
 * instance name and a port, and neither can be guessed: the blank writes the empty text, Ajv says
 * so on the row at once, and the author fills it in. Feature 2.6 met this with New Model's own
 * skeleton and recorded the same thing rather than inventing a site nobody asked for; the gestures
 * that *can* supply the names — `Expose as input…`, `Expose as output…` — do supply them.
 */
import { jsonObject, type JsonValue } from '@tensorspine/lang';
import type { Shape } from '@tensorspine/store';

import {
  alternationAt,
  blankOf,
  factsOfShape,
  type Alternation,
  type FormContext,
  type FormMode,
} from '../forms/index.js';

/** How deep a blank is built before it stops — a recursive `$def` has no smallest value. */
const DEPTH = 8;

/** The smallest value the schema at a place admits. */
export function blankValue(context: FormContext, shape: Shape, depth = 0): JsonValue {
  const facts = factsOfShape(shape);
  if (facts.fixed && facts.constant !== null && facts.constant !== undefined) {
    return facts.constant as JsonValue;
  }
  if (depth >= DEPTH) return null;
  const union = unionOf(context, shape);
  if (union !== undefined) {
    const first = union.alternatives[0];
    if (first !== undefined) return blankValue(context, context.shapes.at(first.anchor), depth + 1);
  }
  // The first value of the enumeration, in the schema's own order: `blankOf` takes the *options*
  // a row carries, and a place built from the schema alone has the facts instead.
  if (facts.choices !== null) return (facts.choices[0] ?? null) as JsonValue;
  if (facts.holdsArray || facts.listed) return [];
  if (facts.keyed) return jsonObject([]);
  if (facts.holdsObject || facts.closed || facts.members.length > 0) {
    const required = context.shapes.constraintsOf(shape).required;
    return jsonObject(
      context.shapes
        .propertyOrder(shape)
        .filter((name) => required.has(name))
        .map((name) => ({
          name,
          value: blankValue(context, context.shapes.member(shape, name), depth + 1),
        })),
    );
  }
  return blankOf(facts, undefined) as JsonValue;
}

/** The shape of one alternative of a union — what a chooser set to that mode writes. */
export function memberShape(context: FormContext, shape: Shape, mode: FormMode): Shape {
  const anchor = mode.anchors[0];
  return anchor === undefined ? shape : context.shapes.at(anchor);
}

/** The shape of one item of a list. */
export function itemShape(context: FormContext, shape: Shape, index: number): Shape {
  return context.shapes.item(shape, index);
}

/** The shape of one entry of a map, by the name it is written under. */
export function valueShape(context: FormContext, shape: Shape, name: string): Shape {
  return context.shapes.step(shape, name);
}

/** The union at a place, from the first anchor of its chain that names one (the walker's rule). */
function unionOf(context: FormContext, shape: Shape): Alternation | undefined {
  const vocabulary = context.registry.vocabulary();
  for (const place of shape.direct.length > 0 ? shape.direct : shape.all) {
    const known = context.unions.get(place.anchor);
    if (known !== undefined) {
      if (known !== null) return known;
      continue;
    }
    const found = alternationAt(vocabulary, context.shapes, place.anchor) ?? null;
    context.unions.set(place.anchor, found);
    if (found !== null) return found;
  }
  return undefined;
}

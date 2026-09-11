/**
 * The derived document's schema, as the figure walk reads it — feature 2.6.
 *
 * `SchemaShapes` (the store's) answers what subschemas describe a place and what they assert; the
 * figure walk needs three of those answers and nothing else. This is the adapter, kept apart so
 * that `figures.ts` — which renders — depends on an interface rather than on the store's reading
 * of a document, and so that a suite can walk a schema of its own.
 */
import { mergeFacts, type SchemaRegistry } from '@tensorspine/lang';
import { SchemaShapes, type Shape } from '@tensorspine/store';

import type { FigureShapes } from './figures.js';

/** The role the registry indexes the derived schema under — its file's own name. */
export const DERIVED_ROLE = 'derived';

/** The derived schema, as {@link FigureShapes}. */
export function shapesFor(registry: SchemaRegistry, role: string = DERIVED_ROLE): FigureShapes<Shape> {
  const shapes = new SchemaShapes(registry);
  return {
    root: () => shapes.root(role),
    member: (shape, name) => shapes.member(shape, name),
    anchors: (shape) => shape.all.map((place) => place.anchor),
    holdsNumber: (shape) => {
      const facts = mergeFacts(shape.all.map((place) => place.node));
      return facts.holdsNumber || facts.holdsWholeNumber;
    },
  };
}

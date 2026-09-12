/**
 * Derivation: the port of `tools/derive.py` (plan §5.3, `derive`).
 *
 * §7 — "A valid document and its referenced primitives make all products below computable without
 * inference code or human knowledge of a named mechanism" — and this is where the six are
 * computed. D1 is its own module (`d1/`, feature 1.7), because a consumer that never validated can
 * emit it; D2–D6 are here, each over the **expanded** graph of `expand.ts` and none over the
 * document.
 *
 * The order is the tools' own and it is not the products' numbering: D3 first, then D4, then D2,
 * then D5 — the costs read the three inventories — then D1 and the agreement between the two
 * expansions, then D6. `products.ts` holds the gate every one of them is behind (`--validate`'s
 * two stages, in their order) and that agreement; `figures.ts` holds the widths, counts and shapes
 * they all write.
 *
 * Features 1.8a–1.8e filled it in one product at a time, and it is complete: the expansion, the
 * figures, the agreement, the six products and `derive` itself — the whole document, validated
 * against the derived schema before it is returned. Beside them, the value-type label a diagram
 * prints over an edge, which is a reading of D2 and not a product (finding F6), and the
 * qualified-value algebra of §2.2, of which the tools implement the one row D5's totals need.
 */
export {
  BYTES,
  defaultDtype,
  elementsOf,
  numberOf,
  orZero,
  productShape,
  pySum,
  selectedDtype,
  sensitivityOf,
  sound,
  widthOf,
  type Figure,
} from './figures.js';
export {
  expandAnalysis,
  expandedKeyOf,
  expandedPortKeyOf,
  identOf,
  locatedValue,
  nodeAt,
  prefixedLocation,
  type ExpandedComposition,
  type ExpandedEdge,
  type ExpandedGraph,
  type ExpandedMember,
  type ExpandedMeta,
  type ExpandedNode,
  type ExpandedPort,
  type ExpandedPortDomain,
  type ExpandedSite,
  type ExpandedStateInstance,
  type ExpandedTensorInstance,
} from './expand.js';
export {
  consistent,
  derivationGraph,
  derive,
  DerivedSchemaError,
  type Derivation,
  type DerivationOptions,
} from './products.js';
export {
  ancestors,
  counts,
  d2,
  peakLive,
  pyRound,
  structuralGraphSplits,
  valueId,
  type Count,
  type StructuralSplit,
} from './d2.js';
export { d3 } from './d3.js';
export { d4 } from './d4.js';
export { d5, OPERATION_COUNTERS } from './d5.js';
export { d6 } from './d6.js';
export {
  conditionalStatus,
  flipped,
  propagate,
  propagationOf,
  PROPAGATIONS,
  roundQualified,
  STATUSES,
  sumStatus,
  type Propagation,
  type Rounding,
  type Status,
} from './qualified.js';
export { shapeAxes, shapeText, streamAxis, valueGeometry } from './labels.js';

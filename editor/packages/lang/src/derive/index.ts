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
 * Features 1.8a–1.8e fill it in one product at a time. What is here today: the expansion, the
 * figures, the agreement, D3 and D4.
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
  type Derivation,
  type DerivationOptions,
} from './products.js';
export { d3 } from './d3.js';
export { d4 } from './d4.js';

/**
 * D1, the Derived Computation Graph: the port of `tools/d1.py` (plan §5.3, `expand`).
 *
 * §5.1 — "A document denotes one finite instance graph. Families, stacks, trunks, repetitions and
 * patterns are syntactic sugar with the expansion defined here" — and §7's D1 is that graph
 * written down: "instances, edges, and families; per instance, whether its primitive reads across
 * positions". Everything the other five products say is said about *these* identifiers.
 *
 * What the module answers is the derived document `--d1` writes: the head of §2 of the derived
 * guide (`schema`, `model`, `primitive_libraries`, `assignment`) with the one product `d1` under
 * it. It is a value, not a class: it survives the structured clone into the worker (§5.3), the
 * serializer writes it with the corpus's own bytes (D12), and the derivation of D2–D6 reads it as
 * the tools read it.
 *
 * The four rules of §5.2 it is written against, and where each one lives:
 *
 * - **rule 2, identifiers** — a root instance by its name, a generated one as
 *   `<composition>/<site>[<index>=<value>,…]` with the indices in name order, an instance of a
 *   template prefixed by its instance. It is `whereOfSite` of `validate/graph.ts`: the validator's
 *   `where` and the emitter's `identity` are the same function, and a refusal about a site and the
 *   node D1 lists for it therefore name it identically.
 * - **rule 3, resolved references** — "a binding is emitted only where every instance it names is
 *   emitted; an instance absent by its guard is not a reference failure". A guarded-out site is
 *   remembered as absent and the bindings naming it are dropped, with no guard repeated on an edge.
 * - **rule 4, set semantics** — "any listing is conforming; the canonical listing orders instances
 *   by identifier, edges by (source, destination)". That is the listing here, and the topological
 *   order is taken with the successors sorted so that it, too, is a function of the graph alone.
 * - **rule 8, templates** — an instance expands its template under the assignment its arguments
 *   make, its nodes are prefixed by the instance, an edge into it fans out to the template's own
 *   destinations and an edge out of it starts at the template's source. "Two instances share
 *   nothing."
 */
export { acrossPositions, recordDefaults } from './arguments.js';
export { expand, expandText, MAX_D1_DEPTH, type ExpandOptions } from './expand.js';

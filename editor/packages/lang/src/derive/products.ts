/**
 * `derive.products`: what every derivation crosses before a product is computed.
 *
 * "D1–D6 of one document, or `ValueError('not valid, no products: …')` with the validator's first
 * line. **Both stages of `--validate` are crossed here, in their order** — the structural one (the
 * schema, V12) before the semantic one — because this is the one entry every derivation takes:
 * the command line, the status and artifact tools, the capabilities reader and every generator. A
 * document off the schema has no products, whatever asked for them."
 *
 * Two things live here and nowhere else. {@link derivationGraph} is that gate and the expansion
 * behind it — the analysis of a valid document with every template instance expanded in place
 * (§5.1) — which is what D2, D3, D4, D5 and D6 are each written over. {@link consistent} is the
 * agreement the tools require between the two expansions the repository carries:
 *
 * > "D1's arguments are the validator's, node by node. Two resolutions exist — the emitter's,
 * > which D1 carries, and the analysis's, which D4's carrying, V18 and the costs read — and their
 * > agreement is checked here rather than assumed: a node the two resolve differently, or one that
 * > only one of them knows, fails the derivation by name."
 *
 * Feature 1.7 measured where the two can differ: `d1.emit` reads `cat['primitives'][name]` by name
 * alone where `analyse` reads the pinned version first, and builds a template's sub-assignment
 * from the *written* arguments where `analyse` builds it from the resolved ones. Neither can
 * happen in a document `analyse` accepted, which is why the check is an exception and not a
 * problem row — but it is *checked*, so that a divergence is named rather than derived from.
 *
 * The assembly of the whole derived document is feature 1.8e's: it runs the six products in the
 * tools' own order (D3, D4, D2, D5, then D1 and this agreement, then D6) and validates the result
 * against the derived schema.
 */
import { PyValueError } from '../expr/errors.js';
import { toPython, type PyRecord, type PyValue } from '../expr/value.js';
import type { JsonValue } from '../json/index.js';
import { asRecord, demand, entries, optional } from '../library/access.js';
import type { Library } from '../library/load.js';
import { pyRepr } from '../library/repr.js';
import { pyEqual } from '../expr/arithmetic.js';
import { formatProblem, type SchemaRegistry } from '../schema/index.js';
import { comparePythonStrings } from '../schema/repr.js';
import { analyse, formatSemanticProblem, type Analysis } from '../validate/index.js';
import { expandAnalysis, identOf, type ExpandedGraph } from './expand.js';

/** What a derivation needs beside the document: the grammar, the library, and the assignment. */
export interface DerivationOptions {
  /** The schemas, for the structural stage `products` crosses first. */
  readonly schemas: SchemaRegistry;
  /** The gathered primitive library, as `loadLibrary` answers it. */
  readonly library: Library;
  /** The values of the external quantities (§4.6); absent for a concrete model. */
  readonly assignment?: PyRecord;
}

/** The state every product is computed from. */
export interface Derivation {
  /** The document as the evaluators read it — `toPython` of the tree, read once. */
  readonly document: PyValue;
  /** The whole semantic stage, whose `stats` D5 reads and whose graph is expanded below. */
  readonly analysis: Analysis;
  /** `_expand(result['graph'])`: the graph every product is written over. */
  readonly graph: ExpandedGraph;
}

/**
 * The two stages of `--validate`, then the expansion: everything `products` does before D3.
 *
 * The tree is the lexeme-preserving one the store holds (D1), because that is what the grammar
 * stage validates; the semantic stage and the expansion read it as a value, converted once.
 */
export function derivationGraph(tree: JsonValue, options: DerivationOptions): Derivation {
  const structural = options.schemas.structural(tree, 'model');
  const first = structural[0];
  if (first !== undefined) {
    throw new PyValueError(`not valid, no products: ${formatProblem(first)}`);
  }
  const document = toPython(tree);
  const analysis = analyse(document, options.library, {
    ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
  });
  const refusal = analysis.problems[0];
  if (refusal !== undefined) {
    throw new PyValueError(`not valid, no products: ${formatSemanticProblem(refusal)}`);
  }
  return { document, analysis, graph: expandAnalysis(analysis) };
}

/**
 * `_consistent(nodes, graph)`: D1's nodes and the validator's, name by name and argument by
 * argument.
 *
 * "Every derived fact that is a function of the arguments (D1's `across_positions`, D4's
 * `carried_across_fragments`) agrees when the arguments do." A disagreement raises, naming the
 * first node or the first argument in code-point order — the tools' `sorted`, so that the same
 * document names the same one twice running.
 */
export function consistent(nodes: PyValue, graph: ExpandedGraph): void {
  const resolved = new Map<string, PyRecord>();
  for (const [, node] of graph.resolved) resolved.set(identOf(node.site), node.args);
  const emitted = new Map(entries(nodes));

  const odd: string[] = [];
  for (const name of resolved.keys()) if (!emitted.has(name)) odd.push(name);
  for (const name of emitted.keys()) if (!resolved.has(name)) odd.push(name);
  if (odd.length > 0) {
    odd.sort(comparePythonStrings);
    const name = odd[0] as string;
    const said = emitted.has(name)
      ? 'emitted by D1, unknown to the validator'
      : 'resolved by the validator, absent from D1';
    throw new PyValueError(`${name}: ${said}`);
  }

  for (const [name, node] of emitted) {
    const mine = resolved.get(name) as PyRecord;
    const theirs = demand(node, 'arguments');
    if (pyEqual(mine, theirs)) continue;
    const differing = [...new Set([...Object.keys(mine), ...Object.keys(asRecord(theirs))])].filter(
      (argument) => !pyEqual(optional(mine, argument, null), optional(theirs, argument, null)),
    );
    differing.sort(comparePythonStrings);
    const argument = differing[0] as string;
    throw new PyValueError(
      `${name}: D1 and the validator resolve argument '${argument}' differently — ` +
        `${pyRepr(optional(theirs, argument, null))} against ${pyRepr(optional(mine, argument, null))}`,
    );
  }
}

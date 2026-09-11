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
 * {@link derive} is the third: the assembly of the whole document. It runs the six products in the
 * tools' own order — D3, D4, D2, D5, then D1 and this agreement, then D6 — and validates the
 * result against the derived schema before returning it, which is `d1.self_check`, "an emitter
 * validates its own output against the derived schema before writing it: a document it cannot
 * vouch for is not written".
 */
import { PyValueError } from '../expr/errors.js';
import { toJsonValue, toPython, type PyRecord, type PyValue } from '../expr/value.js';
import type { JsonValue } from '../json/index.js';
import { asRecord, demand, entries, listOf, optional } from '../library/access.js';
import type { Library } from '../library/load.js';
import { pyRepr } from '../library/repr.js';
import { pyEqual } from '../expr/arithmetic.js';
import { put } from '../json/tree.js';
import { expand } from '../d1/expand.js';
import {
  formatProblem,
  type SchemaRegistry,
  type StructuralProblem,
} from '../schema/index.js';
import { comparePythonStrings } from '../schema/repr.js';
import { analyse, formatSemanticProblem, type Analysis } from '../validate/index.js';
import { d2 } from './d2.js';
import { d3 } from './d3.js';
import { d4 } from './d4.js';
import { d5 } from './d5.js';
import { d6 } from './d6.js';
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

/**
 * Raised when the derived document the core built is off the derived schema.
 *
 * `d1.self_check` is what the tools run on their own output, and `derive.run` prints its lines and
 * **writes nothing**: "a document it cannot vouch for is not written". The core's counterpart is
 * to return nothing — the products are not an answer the caller may use — so the refusal is an
 * exception, worded like the derivation's other two (`not valid, no products: <first line>`) and
 * carrying every line rather than the five the tools print.
 *
 * The document that failed travels on the error, for a log that must explain *what* was wrong with
 * it. It is not a product: nothing but a report may read it.
 */
export class DerivedSchemaError extends Error {
  /** Every line `self_check` would have printed, in `--validate`'s own wording. */
  readonly problems: readonly StructuralProblem[];
  /** The document the check refused, for the log alone. */
  readonly document: PyRecord;

  constructor(problems: readonly StructuralProblem[], document: PyRecord) {
    const first = problems[0];
    super(
      'the emitted document is off the derived schema' +
        (first === undefined ? '' : `: ${formatProblem(first)}`),
    );
    this.name = 'DerivedSchemaError';
    this.problems = problems;
    this.document = document;
  }
}

/**
 * `products(model_path, cat, assignment)`: D1–D6 of one document, as one derived document.
 *
 * The order is the tools' own and it is not the products' numbering: **D3, D4, D2, D5, then D1 and
 * the agreement, then D6**. It is not arbitrary — D5 reads the three inventories before it, and D6
 * reads D2's splits, D4's states and D1's published order — and it is kept because the two
 * refusals that can interrupt it (`_sound`'s R11 line, `_consistent`'s) then come out in the order
 * the tools produce them.
 *
 * D1 is *re-emitted* rather than read off the analysis: `d1.emit` resolves an instance's arguments
 * as written where `analyse` resolves them typed (feature 1.7), so the repository carries two
 * expansions and {@link consistent} is what holds them to each other. The emitted document is also
 * the envelope — `schema`, `model`, `primitive_libraries`, `assignment`, `d1` — and the five
 * products are appended to it in numerical order, which is what `document.update` does.
 *
 * The last step is the self-check. In the tools it is not `products`' but its caller's — `run`
 * asks `d1.self_check` and writes nothing when it answers — and the two are folded here because
 * the core's "writing" is its returning: a document it cannot vouch for is not returned, and
 * {@link DerivedSchemaError} is what comes out instead.
 */
export function derive(tree: JsonValue, options: DerivationOptions): PyRecord {
  const derivation = derivationGraph(tree, options);
  const { analysis, graph } = derivation;
  const p3 = d3(graph, options.library);
  const p4 = d4(graph, options.library);
  const p2 = d2(graph, options.library);
  const p5 = d5(graph, p3, p4, p2, analysis.stats);

  const document = expand(derivation.document, options.library, {
    ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
  });
  const emitted = demand(document, 'd1');
  consistent(demand(emitted, 'nodes'), graph);
  const p6 = d6(graph, p2, p4, listOf(demand(emitted, 'topological_order')));

  put(document, 'd2', p2);
  put(document, 'd3', p3);
  put(document, 'd4', p4);
  put(document, 'd5', p5);
  put(document, 'd6', p6);

  // `d1.self_check(doc, schema_dir)`: the emitter's own reading of its own output, against the
  // schema of the `derived` role — and the registry's "no schema with $id ending in /derived.json"
  // line when the workspace carries none, which is `locate`'s own answer there too.
  const problems = options.schemas.structural(toJsonValue(document), 'derived');
  if (problems.length > 0) throw new DerivedSchemaError(problems, document);
  return document;
}

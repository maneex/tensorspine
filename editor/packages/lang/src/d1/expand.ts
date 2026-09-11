/**
 * `d1.emit`: the expanded graph of one document (§5.2, §7 D1).
 *
 * "The model definition is not the graph — it is a generator." A composition denotes one site per
 * point of its index grid, a guard removes some of them, a template instance stands for a whole
 * document expanded under the assignment its arguments make, and a binding is emitted once per
 * index environment it fires in. D1 is the form every consumer reads.
 *
 * **This is not a second reading of `validate.analyse`.** The tools write the expansion twice —
 * once inside `analyse`, which resolves arguments *typed* and refuses what does not conform, and
 * once in `d1.emit`, which resolves them *as written* and raises where the validator would have
 * produced a line — and `derive.products` runs both and checks that they agree node by node.
 * The port keeps them apart for the same reason: D1 must be computable by a consumer that never
 * validated, and the parity job compares each against its own tool. What the two *do* share is
 * shared here rather than written twice: the site key and its identifier (§5.2 rule 2), the
 * selector, the index grid, the physical name of a weights prefix, and Kahn's algorithm — all of
 * `validate/graph.ts`, read through the readings it names.
 *
 * **The answer is read, never written into.** The emitter builds fresh records for everything it
 * derives, but a node's `primitive` reference and the document's `primitive_libraries` are the
 * document's own objects, shared as the tools share them — the expansion does not copy the
 * document and does not modify it. A consumer that wants to edit one edits the document, which is
 * the store's single tree (plan D1), and expands again.
 *
 * **Where the tools raise, this raises.** `d1.run` catches `ValueError`, `KeyError`, `OSError` and
 * `ModelError` and prints `failed: …` for the document; `emit` itself raises them. An undecidable
 * guard (§5.2 rule 6, I7), a primitive the library does not hold, an argument the
 * `across_positions` condition needs, a cycle (V6), a weights prefix whose index does not resolve
 * — each is a refusal, and the editor's caller catches it where the tools' caller does.
 */
import { pyOrder } from '../expr/arithmetic.js';
import { PyKeyError, PyValueError } from '../expr/errors.js';
import {
  indexGrid,
  modelCondition,
  modelValue,
  resolveQuantities,
  staticArgument,
  type Env,
  type Quantities,
} from '../expr/model.js';
import { primitiveValue } from '../expr/primitive.js';
import {
  truthy,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../expr/value.js';
import { put } from '../json/tree.js';
import { demand, entries, has, listOf, optional } from '../library/access.js';
import { templateOf, type Library } from '../library/load.js';
import { pyRepr, pyStr } from '../library/repr.js';
import { loadModel, normalise } from '../model/normalise.js';
import { comparePythonStrings } from '../schema/repr.js';
import { acrossPositions, recordDefaults } from './arguments.js';
import {
  generatedSite,
  kahnOrder,
  keyOf,
  loopEnvsWith,
  physicalName,
  portKeyOf,
  reprEnv,
  rootSite,
  selectSite,
  valueToken,
  whereOfSite,
  type SiteKey,
} from '../validate/graph.js';

/** `MAX_DEPTH`: how deep a template primitive may invoke another one (§4.6). */
export const MAX_D1_DEPTH = 8;

/** What an expansion needs beside the document and the library. */
export interface ExpandOptions {
  /**
   * The external quantities, by name (§4.6).
   *
   * A template receives its own from the arguments of the instance that invokes it; a model read
   * from the command line receives `--assign`'s. Only the names the document declares `external`
   * reach the emitted document's `assignment`.
   */
  readonly assignment?: PyRecord;
  /** `_prefix`: the identifier of the template instance this reading stands inside, with its `/`. */
  readonly prefix?: string;
  /** `_depth`: how many template call sites this reading is under. */
  readonly depth?: number;
  /** `_stack`: the template primitives being expanded, innermost last — the cycle check of §4.6. */
  readonly stack?: readonly string[];
}

/**
 * `emit(model_path, cat, assignment)`: the D1 document of one model, read from its text.
 *
 * Reading the text is inside what the tools' caller catches: a duplicate member name and a
 * scoped-binding refusal are both `model.load`'s (feature 1.4's `ModelError`).
 */
export function expandText(text: string, library: Library, options: ExpandOptions = {}): PyRecord {
  // `model.load`'s reading: the parser's duplicate-member refusal (V12) and the hoisting of §5.2
  // rule 7. {@link expand} normalises again, which is idempotent and leaves its input untouched.
  return expand(loadModel(text), library, options);
}

/**
 * The same over a document already parsed — the editor's reading, where the tree exists before
 * anything validated it.
 *
 * The document is taken *as read*; the scoped bindings of §5.2 rule 7 are hoisted here, as
 * `model.load` hoists them for the tools.
 */
export function expand(
  document: PyValue,
  library: Library,
  options: ExpandOptions = {},
): PyRecord {
  const model = normalise(document);
  const assignment = options.assignment ?? {};
  const prefix = options.prefix ?? '';
  const depth = options.depth ?? 0;
  const stack = options.stack ?? [];
  const quantities = resolveQuantities(model, assignment);

  const value = (expression: PyValue, env?: Env): PyValue =>
    modelValue(expression, quantities, env);
  const staticOf = (one: PyValue, env: Env): PyValue => staticArgument(one, quantities, env);

  // --- unroll the instances; a guarded-out site is remembered as absent ---
  const { keys, absent } = unrollSites(model, quantities);

  const edges: PyValue[] = [];
  /** The template instances, in the order they were expanded — D1's `instances`. */
  const instances = new Map<string, PyRecord>();
  /** The nodes of the expanded templates, already prefixed by their instance. */
  const nodes = new Map<string, PyValue>();
  /** `inputs_of`: a template instance's input port, and the destinations it fans out to. */
  const inputsOf = new Map<string, [PyValue, PyValue][]>();
  /** `outputs_of`: a template instance's output port, and the source it starts at. */
  const outputsOf = new Map<string, [PyValue, PyValue]>();

  for (const id of [...keys.keys()]) {
    const site = keys.get(id) as DeclaredInstance;
    const name = pyStr(demand(demand(site.instance, 'primitive'), 'name'));
    const definition = primitiveByName(library, name);
    if (!has(definition, 'template')) continue; // a primitive definition
    if (stack.includes(name)) {
      throw new PyValueError(`primitive cycle: ${[...stack, name].join(' -> ')}`);
    }
    if (depth + 1 > MAX_D1_DEPTH) {
      throw new PyValueError(`primitive nesting deeper than ${MAX_D1_DEPTH}`);
    }
    keys.delete(id);
    const instance = prefix + whereOfSite(site.key);
    // A generated site's indices are in scope for its arguments and its prefix (§5.2).
    const env: Env = new Map(site.key.indices.map((one) => [one.name, one.value]));
    const subAssignment: PyRecord = {};
    for (const [argument, written] of entries(demand(site.instance, 'arguments'))) {
      const one = staticOf(written, env);
      if (one !== UNRESOLVED) put(subAssignment, argument, one);
    }
    const sub = expand(templateOf(library, definition).document, library, {
      assignment: subAssignment,
      prefix: `${instance}/`,
      depth: depth + 1,
      stack: [...stack, name],
    });
    const graph = demand(sub, 'd1');
    for (const edge of listOf(demand(graph, 'edges'))) edges.push(edge);
    const record: PyRecord = {};
    put(record, 'primitive', demand(site.instance, 'primitive'));
    put(record, 'arguments', { ...subAssignment });
    if (has(site.instance, 'weights_location_prefix')) {
      const written = physicalName(
        demand(site.instance, 'weights_location_prefix'),
        env,
        value,
        new Map(),
      );
      if (written.problem !== null) {
        throw new PyValueError(`${instance}: weights_location_prefix: ${written.problem}`);
      }
      put(record, 'weights_location_prefix', written.name as string);
    }
    instances.set(instance, record);
    for (const [nested, one] of entries(optional(graph, 'instances', {}))) {
      instances.set(nested, one as PyRecord);
    }
    for (const [node, one] of entries(demand(graph, 'nodes'))) nodes.set(node, one);
    const exposed = demand(graph, 'interfaces');
    for (const [port, declared] of entries(demand(exposed, 'inputs'))) {
      inputsOf.set(
        portKeyOf(site.key, port),
        listOf(demand(declared, 'to')).map((one) => [demand(one, 'node'), demand(one, 'port')]),
      );
    }
    for (const [port, declared] of entries(demand(exposed, 'outputs'))) {
      outputsOf.set(portKeyOf(site.key, port), [demand(declared, 'node'), demand(declared, 'port')]);
    }
  }

  // --- the instances that are not template instances ---------------------
  for (const [, site] of keys) {
    const name = pyStr(demand(demand(site.instance, 'primitive'), 'name'));
    const definition = primitiveByName(library, name);
    const env: Env = new Map(site.key.indices.map((one) => [one.name, one.value]));
    const args: PyRecord = {};
    for (const [argument, written] of entries(demand(site.instance, 'arguments'))) {
      put(args, argument, staticOf(written, env));
    }
    for (const [argument, declaration] of entries(demand(definition, 'arguments'))) {
      if (!Object.hasOwn(args, argument) && has(declaration, 'default')) {
        put(args, argument, primitiveValue(demand(declaration, 'default'), args));
      }
    }
    recordDefaults(demand(definition, 'arguments'), args, args);
    const identifier = prefix + whereOfSite(site.key);
    const node: PyRecord = {};
    put(node, 'primitive', demand(site.instance, 'primitive'));
    put(node, 'arguments', settledArguments(args));
    put(node, 'families', familiesOf(model, site));
    put(node, 'across_positions', acrossPositions(definition, args, identifier));
    nodes.set(identifier, node);
  }

  // --- unroll the edges --------------------------------------------------
  const loopEnvs = (binding: PyValue, label: string): Env[] =>
    loopEnvsWith(binding, label, quantities, (message) => {
      throw new PyValueError(message);
    });

  /** `sources(sel, port, env)`: where an edge out of that port starts, the template's if it is one. */
  const sources = (selector: PyValue, port: PyValue, env: Env): [SiteKey, [PyValue, PyValue][]] => {
    const key = selectSite(selector, quantities, env);
    const exposed = outputsOf.get(portKeyOf(key, port));
    if (exposed !== undefined) return [key, [exposed]];
    return [key, [[prefix + whereOfSite(key), port]]];
  };

  /** `destinations(sel, port, env)`: every port an edge into that port reaches. */
  const destinations = (
    selector: PyValue,
    port: PyValue,
    env: Env,
  ): [SiteKey, [PyValue, PyValue][]] => {
    const key = selectSite(selector, quantities, env);
    const exposed = inputsOf.get(portKeyOf(key, port));
    if (exposed !== undefined) return [key, [...exposed]];
    return [key, [[prefix + whereOfSite(key), port]]];
  };

  for (const [bid, binding] of entries(demand(demand(model, 'bindings'), 'values'))) {
    for (const env of loopEnvs(binding, bid)) {
      const from = demand(binding, 'from');
      const to = demand(binding, 'to');
      const [sourceKey, starts] = sources(demand(from, 'instance'), demand(from, 'port'), env);
      const [targetKey, ends] = destinations(demand(to, 'instance'), demand(to, 'port'), env);
      // §5.2 rule 3: a binding is emitted only where every instance it names is.
      if (absent.has(keyOf(sourceKey)) || absent.has(keyOf(targetKey))) continue;
      for (const [sourceNode, sourcePort] of starts) {
        for (const [targetNode, targetPort] of ends) {
          const edge: PyRecord = {};
          put(edge, 'rule', prefix + bid);
          put(edge, 'from', endpoint(sourceNode, sourcePort));
          put(edge, 'to', endpoint(targetNode, targetPort));
          edges.push(edge);
        }
      }
    }
  }

  // --- the public interfaces ---------------------------------------------
  const declaredInterfaces = demand(model, 'interfaces');
  const inputs: PyRecord = {};
  for (const [name, declared] of entries(demand(declaredInterfaces, 'inputs'))) {
    const to: PyValue[] = [];
    for (const one of listOf(demand(declared, 'to'))) {
      const [, ends] = destinations(demand(one, 'instance'), demand(one, 'port'), new Map());
      for (const [node, port] of ends) to.push(endpoint(node, port));
    }
    const entry: PyRecord = {};
    put(entry, 'to', to);
    put(entry, 'kind', demand(declared, 'kind'));
    if (has(declared, 'stream')) put(entry, 'stream', demand(declared, 'stream'));
    if (truthy(optional(declared, 'fragmented', null))) put(entry, 'fragmented', true);
    put(inputs, name, entry);
  }
  const outputs: PyRecord = {};
  for (const [name, declared] of entries(demand(declaredInterfaces, 'outputs'))) {
    const from = demand(declared, 'from');
    const [, starts] = sources(demand(from, 'instance'), demand(from, 'port'), new Map());
    const [node, port] = starts[0] as [PyValue, PyValue];
    const entry: PyRecord = {};
    put(entry, 'node', node);
    put(entry, 'port', port);
    put(entry, 'generative', demand(declared, 'generative'));
    put(outputs, name, entry);
  }

  // --- the canonical listing (§5.2 rule 4) and one topological order -----
  const listed = [...nodes.keys()].sort(comparePythonStrings);
  edges.sort(compareEdges);
  const order = kahnOrder(
    listed,
    edges.map((edge) => [nodeOf(edge, 'from'), nodeOf(edge, 'to')] as const),
    {
      identify: (one) => one,
      compare: comparePythonStrings,
      sortSuccessors: true,
      unknownDestination: (one) => {
        throw new PyKeyError(pyRepr(one));
      },
    },
  );
  if (order.length !== listed.length) {
    throw new PyValueError(
      `${pyStr(demand(model, 'model'))}: cyclic graph, ${listed.length - order.length} ` +
        `node(s) in a cycle — no D1 (V6 rejection)`,
    );
  }

  const listedNodes: PyRecord = {};
  for (const name of listed) put(listedNodes, name, nodes.get(name) as PyValue);
  const graph: PyRecord = {};
  put(graph, 'nodes', listedNodes);
  put(graph, 'edges', edges);
  put(graph, 'interfaces', { inputs, outputs });
  put(graph, 'topological_order', order);
  if (instances.size > 0) {
    const emitted: PyRecord = {};
    for (const [name, one] of instances) put(emitted, name, one);
    put(graph, 'instances', emitted);
  }

  // "The values supplied for the external quantities — empty for a concrete model, the call-site
  // or `--assign` values for a template" (derived guide §2): what the caller supplied beyond them
  // is not part of what this document denotes, and is dropped.
  const externals = new Set<string>();
  for (const [name, quantity] of entries(demand(model, 'quantities'))) {
    if (demand(demand(quantity, 'source'), 'kind') === 'external') externals.add(name);
  }
  const supplied: PyRecord = {};
  for (const [name, one] of Object.entries(assignment)) {
    if (externals.has(name)) put(supplied, name, one);
  }
  const document_: PyRecord = {};
  put(document_, 'schema', 'tensorspine-derived/2.1');
  put(document_, 'model', demand(model, 'model'));
  put(document_, 'primitive_libraries', demand(model, 'primitive_libraries'));
  put(document_, 'assignment', supplied);
  put(document_, 'd1', graph);
  return document_;
}

/** A site the document declares, at one point of its grid. */
interface DeclaredInstance {
  readonly key: SiteKey;
  readonly instance: PyValue;
}

/**
 * The sites the guards keep, and the sites they remove.
 *
 * "A guard that cannot be decided is a rejection (V10), never false: an undecidable guard would
 * otherwise drop instances silently (I7)" — and here, unlike in the validator, the rejection is an
 * exception, because a document with no decidable graph has no D1 at all.
 */
function unrollSites(
  model: PyRecord,
  quantities: Quantities,
): { keys: Map<string, DeclaredInstance>; absent: Set<string> } {
  const keys = new Map<string, DeclaredInstance>();
  const absent = new Set<string>();
  for (const [name, instance] of entries(demand(model, 'instances'))) {
    const key = rootSite(name);
    if (has(instance, 'when')) {
      const truth = modelCondition(demand(instance, 'when'), quantities, new Map());
      if (truth === UNRESOLVED) throw new PyValueError(`${name}: \`when\` does not resolve`);
      if (!truthy(truth)) {
        absent.add(keyOf(key));
        continue;
      }
    }
    keys.set(keyOf(key), { key, instance });
  }
  for (const [composition, declared] of entries(demand(model, 'compositions'))) {
    const { names, ranges } = indexGrid(demand(declared, 'indices'), quantities);
    for (const combination of product(ranges)) {
      const env: Env = new Map(names.map((name, at) => [name, combination[at] as bigint]));
      for (const [site, instance] of entries(demand(declared, 'instances'))) {
        const key = generatedSite(
          composition,
          site,
          [...env].map(([name, one]) => ({ name, value: one })),
        );
        if (has(instance, 'when')) {
          const truth = modelCondition(demand(instance, 'when'), quantities, env);
          if (truth === UNRESOLVED) {
            throw new PyValueError(
              `${composition}.${site}${reprEnv(env)}: \`when\` does not resolve`,
            );
          }
          if (!truthy(truth)) {
            absent.add(keyOf(key));
            continue;
          }
        }
        keys.set(keyOf(key), { key, instance });
      }
    }
  }
  return { keys, absent };
}

/**
 * `cat['primitives'][name]`: the primitive by *name alone*, which is how D1 reads one.
 *
 * `validate.analyse` reads `primitive_library.primitive(cat, ref)` — the pinned version first,
 * falling back to the name, so that a version mismatch is reported as one; the emitter indexes by
 * name, whose entry is the highest version the bases carry. A document pinning a version the base
 * does not hold therefore derives against the highest, and a name the base does not hold at all is
 * the `KeyError` Python raises. Both are reproduced; V1 is what reports them.
 */
function primitiveByName(library: Library, name: string): PyValue {
  const definition = library.primitives.get(name);
  if (definition === undefined) throw new PyKeyError(`'${name}'`);
  return definition;
}

/** `{k: v for k, v in args.items() if v is not UNRESOLVED}`: what the node carries. */
function settledArguments(args: PyRecord): PyRecord {
  const settled: PyRecord = {};
  for (const [name, value] of Object.entries(args)) {
    if (value !== UNRESOLVED) put(settled, name, value);
  }
  return settled;
}

/**
 * The families of one node: the site's own, and its composition's beside them.
 *
 * A root instance keeps the list as written; a generated one carries the union, sorted — a Python
 * set of names, so the order is the identifiers' own and not either list's.
 */
function familiesOf(model: PyRecord, site: DeclaredInstance): PyValue[] {
  const own = listOf(demand(site.instance, 'families'));
  if (site.key.kind === 'root') return own;
  const composition = demand(demand(model, 'compositions'), site.key.composition);
  const united = new Map<string, PyValue>();
  for (const family of [...listOf(demand(composition, 'families')), ...own]) {
    // A Python set: two families are one when they are `==` and hash alike, which is what
    // `valueToken` writes (feature 1.6b).
    united.set(valueToken(family), family);
  }
  return [...united.values()].sort((one, other) => pyOrder(one, other) ?? 0);
}

/** `{"node": …, "port": …}`: one end of an edge. */
function endpoint(node: PyValue, port: PyValue): PyRecord {
  const one: PyRecord = {};
  put(one, 'node', node);
  put(one, 'port', port);
  return one;
}

/** The node identifier at one end of an emitted edge. */
function nodeOf(edge: PyValue, side: 'from' | 'to'): string {
  return pyStr(demand(demand(edge, side), 'node'));
}

/**
 * `edges.sort(key=lambda e: (from.node, from.port, to.node, to.port))`: the canonical order of the
 * edges (§5.2 rule 4), a tuple compared member by member.
 */
function compareEdges(one: PyValue, other: PyValue): number {
  for (const [side, part] of [
    ['from', 'node'],
    ['from', 'port'],
    ['to', 'node'],
    ['to', 'port'],
  ] as const) {
    const order = pyOrder(demand(demand(one, side), part), demand(demand(other, side), part));
    if (order !== undefined && order !== 0) return order;
  }
  return 0;
}

/** `itertools.product(*ranges)`: one empty point when there is no range at all. */
function product(ranges: readonly (readonly bigint[])[]): bigint[][] {
  let points: bigint[][] = [[]];
  for (const range of ranges) {
    const grown: bigint[][] = [];
    for (const point of points) {
      for (const one of range) grown.push([...point, one]);
    }
    points = grown;
  }
  return points;
}

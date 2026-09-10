/**
 * The graph of one document: `analyse`'s sites, its edges and its public interfaces — the port of
 * `walk`, `where`, `loop_envs`, `select`, `port_shape` and the edge and interface blocks of
 * `tools/validate.py`.
 *
 * "The document is a generator, not the graph" (§5.1, §5.2). A composition denotes one site per
 * point of its index grid, a guard removes some of them, a template instance stands for a whole
 * document expanded under the assignment its arguments make — and what the rules of §6 are written
 * over is the graph that comes out, not the text. This module builds it once and refuses what does
 * not hold on it:
 *
 * - **V10** — a repetition, a guard or a derivation resolves. "A guard that cannot be decided is a
 *   rejection, never false: an undecidable guard would otherwise drop instances silently (I7)."
 *   A site whose guard is *false* is not an error: it is remembered as absent, and a binding
 *   naming it is not emitted there (§5.2 rule 3).
 * - **V1** — every primitive, instance, port and stream reference resolves; a template primitive's
 *   citation graph is acyclic and of bounded depth (§4.6).
 * - **V4** — shapes compose: "the same rank and, position by position, the same axis identity and
 *   equal extents; local names and natures are not compared". A public input is an edge like any
 *   other, so the ports one input feeds must agree with each other too.
 * - **V7** — every present input port of an emitted instance is fed exactly once (the slot half is
 *   feature 1.6c's); **V13** — every present output port is consumed by an edge or exposed.
 * - **V6** — the value graph is acyclic within an invocation.
 * - **V5** — indexing domains agree on every edge (§5.3): a domain is a pair (kind, stream), an
 *   instance's own domain is the common domain of its untransformed inputs, and an output's comes
 *   from its declared kind, its `inherit` source or a `merge` transform.
 * - **V19** — a public input that joins a stream joins it at a kind the stream carries
 *   *independently* of it (§2.3).
 *
 * **A template instance is a document expanded at the call site** (§4.6, §5.2 rule 8): the template
 * is analysed under the assignment its arguments make, its errors are repeated with the instance
 * named, its counts are merged into the caller's, and the interface ports the expansion resolved
 * become the instance's ports — so V4 and V5 apply across the boundary. "Two invocations share
 * nothing", and the memo is keyed by the file and the assignment for that reason.
 *
 * **What this module answers beside the refusals** is the state every later stage reads: the
 * resolved sites with their arguments and definitions, the edges, the domains, the topological
 * order, and the public interface ports with their kinds, streams and evaluated shapes — the
 * `ports` a caller of a template reads back. Features 1.6c (slots, identities, states) and 1.7
 * (D1) continue from {@link GraphAnalysis}; nothing here is computed twice there.
 *
 * **Where the tools raise, this raises.** `analyse` catches exactly one thing — the refusal
 * `model.load` raises, which it reports as V12 or V1 — and lets everything else through: an
 * index bound that does not resolve, an interface written in a form the grammar refuses, a
 * comparison Python cannot make. Meaning assumes grammar (feature 1.2's note, feature 1.5's), so
 * a caller that hands this an off-schema document gets the tools' exception, not a verdict.
 */
import { pyEqual, pyOrder } from '../expr/arithmetic.js';
import { PyIndexError, PyTypeError } from '../expr/errors.js';
import {
  indexGrid,
  modelCondition,
  modelValue,
  resolveQuantities,
  staticArgument,
  type Env,
  type Quantities,
} from '../expr/model.js';
import { primitiveCondition, primitiveValue } from '../expr/primitive.js';
import { truthy, UNRESOLVED, type PyRecord, type PyValue } from '../expr/value.js';
import { demand, entries, has, listOf, optional } from '../library/access.js';
import { primitiveOf, templatePinOf, type Library } from '../library/load.js';
import { PrimitiveLibraryError } from '../library/problems.js';
import { pyRepr, pyStr } from '../library/repr.js';
import { templateInterface } from '../library/template.js';
import { ModelError } from '../model/errors.js';
import { loadModel, normalise } from '../model/normalise.js';
import { comparePythonStrings } from '../schema/index.js';
import type { PathSegment } from '../schema/types.js';

import { describeArguments, type ArgumentDescription } from './arguments.js';
import { semanticProblem, withMessage, type SemanticProblem } from './problems.js';
import { checkQuantities } from './quantities.js';

/** `MAX_PRIMITIVE_DEPTH`: how deep a template primitive may cite another one (§4.6). */
export const MAX_PRIMITIVE_DEPTH = 8;

/** One index of a generated site, bound to its value at this point of the grid. */
export interface IndexBinding {
  readonly name: string;
  readonly value: PyValue;
}

/**
 * A site of the expanded graph: `('root', name)` or `('gen', composition, site, indices)`, the
 * tuple `analyse` keys its dictionaries by.
 *
 * The indices are sorted by name, as `tuple(sorted(env.items()))` sorts them, so two selectors
 * naming the same point are the same key however they were written (§5.2 rule 2).
 */
export interface SiteKey {
  readonly kind: 'root' | 'gen';
  /** The composition, `''` for a root instance. */
  readonly composition: string;
  /** The root instance's name, or the site's name inside its composition. */
  readonly name: string;
  /** The indices in scope, sorted by name; empty for a root instance. */
  readonly indices: readonly IndexBinding[];
}

/** `('root', name)`. */
export function rootSite(name: string): SiteKey {
  return { kind: 'root', composition: '', name, indices: [] };
}

/** `('gen', composition, site, tuple(sorted(indices)))`. */
export function generatedSite(
  composition: string,
  name: string,
  indices: readonly IndexBinding[],
): SiteKey {
  const sorted = [...indices].sort((one, other) => comparePythonStrings(one.name, other.name));
  return { kind: 'gen', composition, name, indices: sorted };
}

/**
 * The site key as a string, so that a `Map` keys what Python keys by a tuple.
 *
 * Python's dictionary reads two keys as one when they are `==` and hash alike, so an index
 * standing at `1`, `1.0` or `True` names one site — {@link valueToken} therefore writes a
 * number by its *value* and not by its float-ness, which is the one place in the core where the
 * distinction feature 1.2 keeps is deliberately dropped. `\u0000` separates the parts: no
 * identifier of the grammar carries one, and a string value is written escaped.
 */
export function keyOf(site: SiteKey): string {
  if (site.kind === 'root') return `root\u0000${site.name}`;
  const indices = site.indices.map((one) => `${one.name}\u0000${valueToken(one.value)}`);
  return ['gen', site.composition, site.name, ...indices].join('\u0000');
}

/** A `(site, port)` pair as one key, which is how `producers`, `domains` and `consumed` are keyed. */
export function portKeyOf(site: SiteKey, port: PyValue): string {
  return `${keyOf(site)}\u0000\u0000${valueToken(port)}`;
}

/**
 * A value as a dictionary key: what Python's `hash` and `==` make of it.
 *
 * `1`, `1.0` and `True` are one key there, so they are one token here; a string is written
 * escaped, so that a value carrying the separator cannot be read as one; and a value Python
 * cannot hash raises as Python raises.
 */
export function valueToken(value: PyValue): string {
  if (value === UNRESOLVED) return 'unresolved';
  if (value === null) return 'None';
  if (typeof value === 'string') return `s${JSON.stringify(value)}`;
  if (typeof value === 'boolean') return `#${value ? '1' : '0'}`;
  if (typeof value === 'bigint') return `#${value.toString()}`;
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return '#nan';
    if (value === Infinity) return '#inf';
    if (value === -Infinity) return '#-inf';
    // An exact integer hashes and compares as the integer does, `-0.0` as `0`.
    if (Number.isInteger(value)) return `#${BigInt(value).toString()}`;
    return `#${String(value)}`;
  }
  throw new PyTypeError(`unhashable type: '${Array.isArray(value) ? 'list' : 'dict'}'`);
}

/** `where(key)`: how a refusal names a site — `embed`, or `decoder/attn[layer=3]`. */
export function whereOfSite(site: SiteKey): string {
  if (site.kind === 'root') return site.name;
  const indices = site.indices.map((one) => `${one.name}=${pyStr(one.value)}`);
  return `${site.composition}/${site.name}[${indices.join(',')}]`;
}

/** Where a site is written in the document, for the pointer a problem carries beside its words. */
export function segmentsOfSite(site: SiteKey): PathSegment[] {
  return site.kind === 'root'
    ? ['instances', site.name]
    : ['compositions', site.composition, 'instances', site.name];
}

/**
 * The order Python puts two site keys in — `sorted(n for n in nodes if indegree[n] == 0)`, which
 * decides the topological order and hence the order the V5 block reports in.
 *
 * A tuple compares element by element: `'gen'` before `'root'`, then the composition, the site,
 * and the index bindings pair by pair. Two values Python cannot order raise there and here.
 */
export function compareSiteKeys(one: SiteKey, other: SiteKey): number {
  if (one.kind !== other.kind) return one.kind === 'gen' ? -1 : 1;
  if (one.kind === 'root') return comparePythonStrings(one.name, other.name);
  const composition = comparePythonStrings(one.composition, other.composition);
  if (composition !== 0) return composition;
  const name = comparePythonStrings(one.name, other.name);
  if (name !== 0) return name;
  const width = Math.min(one.indices.length, other.indices.length);
  for (let index = 0; index < width; index += 1) {
    const mine = one.indices[index] as IndexBinding;
    const theirs = other.indices[index] as IndexBinding;
    const byName = comparePythonStrings(mine.name, theirs.name);
    if (byName !== 0) return byName;
    const byValue = pyOrder(mine.value, theirs.value);
    if (byValue === undefined) return 0; // a `nan`: unordered, as Python leaves it
    if (byValue !== 0) return byValue;
  }
  return one.indices.length - other.indices.length;
}

/** A shape as V4 compares it: axis identity and extent, position by position. */
export type ShapeIdentity = readonly (readonly [PyValue, PyValue])[];

/** An indexing domain (§5.3): the pair (kind, stream) the tools carry as a tuple. */
export type Indexing = readonly [PyValue, PyValue];

/**
 * One entry of `domains`: the `(site, port)` the tools key it by, and the domain it carries.
 *
 * The tools key that dictionary by a tuple and read the site back out of the key (the V19 block
 * does, to ask whether the joining input reaches it); a string key cannot be read back, so the
 * pair travels with the value.
 */
export interface PortDomain {
  readonly site: SiteKey;
  readonly port: PyValue;
  readonly domain: Indexing;
}

/** One `(site, port)` the tools keep in a set or a dictionary, carried beside its string key. */
export interface PortReference {
  readonly site: SiteKey;
  readonly port: PyValue;
}

/** What feeds one input port: a value binding's name, or `input:<name>` for a public input. */
export interface PortProducer extends PortReference {
  readonly by: string;
}

/** `_shape_identity`: "a shape as V4 compares it: axis identity and extent, position by position". */
export function shapeIdentity(shape: PyValue, args: PyRecord): ShapeIdentity {
  return listOf(demand(shape, 'axes')).map(
    (axis) => [demand(axis, 'axis'), primitiveValue(demand(axis, 'extent'), args)] as const,
  );
}

/** `port_shape`: the shape a port declares, evaluated in the instance's arguments; `null` for none. */
export function portShape(port: PyValue, args: PyRecord): ShapeIdentity | null {
  return has(port, 'shape') ? shapeIdentity(demand(port, 'shape'), args) : null;
}

/** `_present`: whether a declared element exists under these arguments (§4.1's `present_when`). */
export function present(element: PyValue, args: PyRecord): boolean {
  return has(element, 'present_when')
    ? truthy(primitiveCondition(demand(element, 'present_when'), args))
    : true;
}

/** Two shape identities as Python's `==` reads them: `None` on either side is never equal to one. */
export function shapesAgree(one: ShapeIdentity | null, other: ShapeIdentity | null): boolean {
  if (one === null || other === null) return one === other;
  if (one.length !== other.length) return false;
  return one.every(
    (axis, index) =>
      pyEqual(axis[0], (other[index] as readonly [PyValue, PyValue])[0]) &&
      pyEqual(axis[1], (other[index] as readonly [PyValue, PyValue])[1]),
  );
}

/** `f"{list(shape)}"`: a list of 2-tuples, as Python writes one into a message. */
export function reprShape(shape: ShapeIdentity): string {
  return `[${shape.map((axis) => reprTuple(axis)).join(', ')}]`;
}

/** `repr` of a tuple: `('model.width', 4096)`, a lone member keeping Python's trailing comma. */
export function reprTuple(items: readonly PyValue[]): string {
  const written = items.map((one) => pyRepr(one)).join(', ');
  return items.length === 1 ? `(${written},)` : `(${written})`;
}

/**
 * `_physical_name`: a physical name evaluated in an index environment — literal strings, `{index}`
 * in decimal, `{coordinate}` inside a stack (§3.4).
 *
 * Feature 1.6b needs it for the one place a name is evaluated before the slots are: an instance's
 * `weights_location_prefix`, which `analyse` reads in the site loop, before the template interface
 * replaces the definition. Feature 1.6c reads it again for the locations themselves.
 */
export function physicalName(
  items: PyValue,
  env: Env,
  value: (expression: PyValue, env: Env) => PyValue,
  coordinates: ReadonlyMap<string, PyValue>,
): { name: string | null; problem: string | null } {
  const out: string[] = [];
  for (const item of listOf(items)) {
    if (typeof item === 'string') {
      out.push(item);
    } else if (has(item, 'index')) {
      const index = demand(item, 'index');
      const resolved = value({ index }, env);
      if (resolved === UNRESOLVED || typeof resolved === 'boolean' || typeof resolved !== 'bigint') {
        return {
          name: null,
          problem: `index '${pyStr(index)}' does not resolve in the physical name`,
        };
      }
      out.push(pyStr(resolved));
    } else {
      const coordinate = demand(item, 'coordinate');
      const at = typeof coordinate === 'string' ? coordinates.get(coordinate) : undefined;
      if (at === undefined) {
        return {
          name: null,
          problem: `\`coordinate\` '${pyStr(coordinate)}' outside a stack over that axis`,
        };
      }
      out.push(pyStr(at));
    }
  }
  return { name: out.join(''), problem: null };
}

/** One public interface port of a document, as the expansion resolved it. */
export interface InterfacePort {
  /** The kind the input declares, or the kind the expansion gave the output; `null` when undetermined. */
  readonly kind: PyValue | null;
  /** The stream the input introduces or joins, or the output's; `null` when undetermined. */
  readonly stream: PyValue | null;
  /** The shape of the instance port it names, evaluated; `null` when the port declares none. */
  readonly shape: ShapeIdentity | null;
}

/** "What this document exposes to a caller: its interface ports, resolved." */
export interface InterfacePorts {
  readonly inputs: ReadonlyMap<string, InterfacePort>;
  readonly outputs: ReadonlyMap<string, InterfacePort>;
}

/** One site of the expanded graph, with the primitive it resolved to and its arguments. */
export interface ResolvedSite {
  readonly key: SiteKey;
  /** The primitive's name, as the instance writes it. */
  readonly primitive: PyValue;
  /** The primitive definition — a template's *interface*, its ports replaced by the expansion's. */
  readonly definition: PyValue;
  /** "The complete, typed argument map of one instance": `resolve_arguments`' answer. */
  readonly args: PyRecord;
  /** The facts feature 1.6a records on the way, for the property sheet (§4.12). */
  readonly arguments: ArgumentDescription;
  /** The site as the document writes it. */
  readonly instance: PyValue;
}

/** One value edge of the expanded graph. */
export interface ValueEdge {
  readonly from: SiteKey;
  readonly fromPort: PyValue;
  readonly to: SiteKey;
  readonly toPort: PyValue;
  /** The binding that emitted it, `decoder.attn.norm_in`. */
  readonly binding: string;
}

/** A site the document declares, at one point of its grid, before its primitive is resolved. */
export interface DeclaredSite {
  readonly key: SiteKey;
  readonly instance: PyValue;
}

/**
 * What `analyse` answers about the graph, up to the bindings feature 1.6c reads.
 *
 * The refusals are the tools' lines; everything else is the state the later stages continue from
 * — `analyse`'s own `graph` dictionary, under names rather than tuple positions.
 */
export interface GraphAnalysis {
  /** Every refusal, in the order `fail` appends them. */
  readonly problems: readonly SemanticProblem[];
  /** `--lint`'s advisories carried up from an expanded template; this stage raises none of its own. */
  readonly advisories: readonly string[];
  /** The counters `--validate` prints, by the tools' own names. */
  readonly stats: ReadonlyMap<string, PyValue>;
  /** The public interface ports, resolved: what a caller of this document as a template reads. */
  readonly ports: InterfacePorts;
  /** The document normalised, or `null` when the reading refused it. */
  readonly model: PyRecord | null;
  /** The quantities under the assignment in force. */
  readonly quantities: Quantities;
  /** Every site whose guard fires, by {@link keyOf}. */
  readonly sites: ReadonlyMap<string, DeclaredSite>;
  /** The sites a guard removed: "not a reference failure" (§5.2 rule 3). */
  readonly absent: ReadonlyMap<string, SiteKey>;
  /** Every site whose primitive resolved, with its arguments. */
  readonly resolved: ReadonlyMap<string, ResolvedSite>;
  /** The value edges, in the order the bindings emitted them. */
  readonly edges: readonly ValueEdge[];
  /** The indexing domain of every `(site, port)` the walk decided, by {@link portKeyOf}. */
  readonly domains: ReadonlyMap<string, PortDomain>;
  /** Each instance's own domain — the common domain of its untransformed inputs — or `null`. */
  readonly own: ReadonlyMap<string, Indexing | null>;
  /** The topological order the V6 block computed; shorter than the graph when there is a cycle. */
  readonly order: readonly SiteKey[];
  /** The streams a fragmented public input delivers (§5.3). */
  readonly fragmented: ReadonlySet<string>;
  /** The `(site, port)` a public input seeds, with the domain it introduces. */
  readonly seeds: ReadonlyMap<string, PortDomain>;
  /** What feeds each `(site, port)`: a binding's name, or `input:<name>`. */
  readonly producers: ReadonlyMap<string, PortProducer>;
  /** The `(site, port)` an edge or a public output consumes. */
  readonly consumed: ReadonlyMap<string, PortReference>;
  /** The evaluated `weights_location_prefix` of every template instance that carries one. */
  readonly weightsPrefixes: ReadonlyMap<string, string>;
  /** The expansion of each template instance, by the key of its site. */
  readonly subResults: ReadonlyMap<string, GraphAnalysis>;
}

/** What a reading needs beside the document: the library, the assignment, and the expansion memo. */
export interface GraphOptions {
  /** The assignment the external quantities are read under (§4.6); absent for a closed document. */
  readonly assignment?: PyRecord;
  /** `_depth`: how many template call sites this reading is under. */
  readonly depth?: number;
  /** `_cache`: one analysis per (template, assignment), shared down the expansion. */
  readonly cache?: Map<string, GraphAnalysis>;
}

/**
 * `analyse(model_path, cat, assignment)`: the semantic stage over a document read from its text.
 *
 * The text is what the tools are given, and reading it is inside the refusal they catch: a
 * duplicate member name and a scoped-binding refusal are both `model.load`'s, reported as V12 or
 * V1 (feature 1.4's `ModelError`).
 */
export function analyseGraphText(
  text: string,
  library: Library,
  options: GraphOptions = {},
): GraphAnalysis {
  let model: PyRecord;
  try {
    model = loadModel(text);
  } catch (error) {
    if (error instanceof ModelError) return refusalOf(error);
    throw error;
  }
  return analyseNormalised(model, library, options);
}

/**
 * The same over a document already parsed — the editor's reading, where the tree exists before any
 * validation and the duplicate member name was refused when it was parsed (feature 0.3).
 *
 * `document` is the document *as read*; the scoped bindings of §5.2 rule 7 are hoisted here, as
 * `analyse` hoists them, and the refusal that hoisting can raise is reported as the tools report it.
 */
export function analyseGraph(
  document: PyValue,
  library: Library,
  options: GraphOptions = {},
): GraphAnalysis {
  let model: PyRecord;
  try {
    model = normalise(document);
  } catch (error) {
    if (error instanceof ModelError) return refusalOf(error);
    throw error;
  }
  return analyseNormalised(model, library, options);
}

/**
 * `'V12' if 'duplicate' in str(e) or 'declared both' in str(e) else 'V1'`: the mapping is
 * `analyse`'s line, and `ModelError.kind` says the same without a substring test (feature 1.4).
 */
function refusalOf(error: ModelError): GraphAnalysis {
  const code = error.kind === 'duplicate' || error.kind === 'collision' ? 'V12' : 'V1';
  return {
    ...empty(),
    problems: [semanticProblem(code, error.message)],
  };
}

/** `empty`: what `analyse` answers when the document cannot be read at all. */
function empty(): GraphAnalysis {
  return {
    problems: [],
    advisories: [],
    stats: new Map(),
    ports: { inputs: new Map(), outputs: new Map() },
    model: null,
    quantities: new Map(),
    sites: new Map(),
    absent: new Map(),
    resolved: new Map(),
    edges: [],
    domains: new Map(),
    own: new Map(),
    order: [],
    fragmented: new Set(),
    seeds: new Map(),
    producers: new Map(),
    consumed: new Map(),
    weightsPrefixes: new Map(),
    subResults: new Map(),
  };
}

/**
 * `select(sel, env)`: the site an instance selector designates, at the indices in scope.
 *
 * A root selector names an instance; a generated one names a site of a composition at index values
 * the selector may override (`ffn_r[layer−1]`), evaluated in the environment of the rule that
 * carries it (§5.2 rule 7).
 */
export function selectSite(selector: PyValue, quantities: Quantities, env: Env): SiteKey {
  if (demand(selector, 'kind') === 'root') return rootSite(pyStr(demand(selector, 'instance')));
  const indices: IndexBinding[] = entries(demand(selector, 'indices')).map((pair) => ({
    name: pair[0],
    value: modelValue(pair[1], quantities, env),
  }));
  return generatedSite(
    pyStr(demand(selector, 'composition')),
    pyStr(demand(selector, 'instance')),
    indices,
  );
}

/**
 * `loop_envs(binding, label)`: "the index environments a rule fires in — its `for_each` grid, kept
 * where its `when` holds. An undecidable guard is a V10 refusal."
 *
 * Feature 1.6c calls it for the parameter, constant and state rules; the refusals it appends are
 * the caller's, which is why the list and the place are parameters rather than a closure.
 */
export function loopEnvs(
  binding: PyValue,
  label: string,
  quantities: Quantities,
  problems: SemanticProblem[],
  at: readonly PathSegment[] = [],
): Env[] {
  let envs: Env[] = [new Map<string, PyValue>()];
  if (has(binding, 'for_each')) {
    const { names, ranges } = indexGrid(demand(binding, 'for_each'), quantities);
    envs = product(ranges).map((combination) => bind(names, combination));
  }
  if (!has(binding, 'when')) return envs;
  const kept: Env[] = [];
  for (const env of envs) {
    const truth = modelCondition(demand(binding, 'when'), quantities, env);
    if (truth === UNRESOLVED) {
      problems.push(
        semanticProblem('V10', `${label}${reprEnv(env)}: \`when\` does not resolve`, at),
      );
    } else if (truthy(truth)) {
      kept.push(env);
    }
  }
  return kept;
}

/**
 * `instance_ports(exposed)`: the ports of a template instance, carrying the kinds and shapes the
 * expanded template resolved, "so that V4 and V5 apply across the boundary" (§4.6).
 */
export function instancePorts(exposed: InterfacePorts): PyRecord {
  const build = (side: ReadonlyMap<string, InterfacePort>): PyRecord => {
    const ports: Record<string, PyValue> = {};
    for (const [name, entry] of side) {
      const port: Record<string, PyValue> = {
        role: 'activation.hidden',
        domain: { kind: falsy(entry.kind) ? 'inherit' : entry.kind, from: { self: true } },
      };
      if (entry.shape !== null) {
        port['shape'] = {
          axes: entry.shape.map(([axis, extent]) => ({
            name: lastSegment(axis),
            axis,
            nature: 'feature',
            extent: { literal: extent },
          })),
        };
      }
      ports[name] = port;
    }
    return ports;
  };
  return { inputs: build(exposed.inputs), outputs: build(exposed.outputs) };
}

/** The whole of `analyse` from the quantities to V19, over a document already normalised. */
function analyseNormalised(
  model: PyRecord,
  library: Library,
  options: GraphOptions,
): GraphAnalysis {
  const depth = options.depth ?? 0;
  const cache = options.cache ?? new Map<string, GraphAnalysis>();
  const problems: SemanticProblem[] = [];
  const advisories: string[] = [];
  const stats = new Map<string, PyValue>();
  const merged = new Map<string, bigint>();

  const fail = (code: `V${number}`, message: string, at: readonly PathSegment[] = []): void => {
    problems.push(semanticProblem(code, message, at));
  };

  const quantities = resolveQuantities(model, options.assignment);
  for (const problem of checkQuantities(model, quantities)) problems.push(problem);

  const value = (expression: PyValue, env?: Env): PyValue =>
    modelValue(expression, quantities, env);
  const staticOf = (one: PyValue, env: Env): PyValue => staticArgument(one, quantities, env);

  // --- the primitive citation graph is acyclic and of bounded depth (§4.6) --
  const templateOf = (definition: PyValue): { path: string; document: PyValue } => {
    const pin = templatePinOf(library, definition);
    if (pin === undefined) {
      const reference = demand(definition, 'template');
      throw new PrimitiveLibraryError(
        `template '${pyStr(demand(reference, 'name'))}' ` +
          `${pyStr(demand(reference, 'version'))} was not resolved at load`,
      );
    }
    return pin;
  };

  const primitiveDependencies = (template: PyValue): Set<string> | null => {
    let document: PyRecord;
    try {
      document = normalise(template);
    } catch (error) {
      if (error instanceof ModelError) return null;
      throw error;
    }
    const names = new Set<string>();
    for (const [, instance] of entries(demand(document, 'instances'))) {
      names.add(pyStr(demand(demand(instance, 'primitive'), 'name')));
    }
    for (const [, composition] of entries(demand(document, 'compositions'))) {
      for (const [, instance] of entries(demand(composition, 'instances'))) {
        names.add(pyStr(demand(demand(instance, 'primitive'), 'name')));
      }
    }
    return names;
  };

  const walk = (name: string, stack: readonly string[], level: number): void => {
    const definition = library.primitives.get(name);
    if (definition === undefined || !has(definition, 'template')) return;
    if (stack.includes(name)) {
      fail('V1', `primitive cycle: ${[...stack, name].join(' -> ')}`);
      return;
    }
    if (level > MAX_PRIMITIVE_DEPTH) {
      fail('V1', `primitive nesting deeper than ${MAX_PRIMITIVE_DEPTH} at '${name}'`);
      return;
    }
    const dependencies = primitiveDependencies(templateOf(definition).document);
    if (dependencies === null) {
      fail('V1', `template not found for template primitive '${name}'`);
      return;
    }
    for (const one of [...dependencies].sort(comparePythonStrings)) {
      walk(one, [...stack, name], level + 1);
    }
  };

  const everyInstance: PyValue[] = [
    ...entries(demand(model, 'instances')).map((pair) => pair[1]),
    ...entries(demand(model, 'compositions')).flatMap(([, composition]) =>
      entries(demand(composition, 'instances')).map((pair) => pair[1]),
    ),
  ];
  const seenPrimitives = new Set<string>();
  for (const instance of everyInstance) {
    const name = pyStr(demand(demand(instance, 'primitive'), 'name'));
    if (!seenPrimitives.has(name)) {
      seenPrimitives.add(name);
      walk(name, [], 1);
    }
  }
  stats.set(
    'composite_primitives',
    BigInt(
      [...seenPrimitives].filter((name) => has(library.primitives.get(name) ?? {}, 'template'))
        .length,
    ),
  );

  // --- expansion: the document is a generator, not the graph ------------
  // "A guarded site that does not fire is not an instance, and is remembered as absent: a binding
  // naming it is not emitted there (§5.2 rule 3), which is not a reference failure. An unknown
  // name or an index outside the ranges still is (V1). A guard that cannot be decided is a
  // refusal."
  const sites = new Map<string, DeclaredSite>();
  const absent = new Map<string, SiteKey>();
  for (const [name, instance] of entries(demand(model, 'instances'))) {
    const key = rootSite(name);
    if (has(instance, 'when')) {
      const truth = modelCondition(demand(instance, 'when'), quantities, new Map());
      if (truth === UNRESOLVED) {
        fail('V10', `${name}: \`when\` does not resolve`, ['instances', name]);
        continue;
      }
      if (!truthy(truth)) {
        absent.set(keyOf(key), key);
        continue;
      }
    }
    sites.set(keyOf(key), { key, instance });
  }
  for (const [composition, declared] of entries(demand(model, 'compositions'))) {
    const { names, ranges } = indexGrid(demand(declared, 'indices'), quantities);
    for (const combination of product(ranges)) {
      const env = bind(names, combination);
      for (const [site, instance] of entries(demand(declared, 'instances'))) {
        const key = generatedSite(
          composition,
          site,
          [...env].map(([name, one]) => ({ name, value: one })),
        );
        if (has(instance, 'when')) {
          const truth = modelCondition(demand(instance, 'when'), quantities, env);
          if (truth === UNRESOLVED) {
            fail('V10', `${composition}.${site}${reprEnv(env)}: \`when\` does not resolve`, [
              'compositions',
              composition,
              'instances',
              site,
            ]);
            continue;
          }
          if (!truthy(truth)) {
            absent.set(keyOf(key), key);
            continue;
          }
        }
        sites.set(keyOf(key), { key, instance });
      }
    }
  }
  stats.set('instances', BigInt(sites.size));

  // --- V1/V2: primitives and arguments -----------------------------------
  const resolved = new Map<string, ResolvedSite>();
  const subResults = new Map<string, GraphAnalysis>();
  const weightsPrefixes = new Map<string, string>();
  for (const [id, { key, instance }] of sites) {
    const reference = demand(instance, 'primitive');
    const name = demand(reference, 'name');
    const at = segmentsOfSite(key);
    let definition = primitiveOf(library, {
      name: pyStr(name),
      version: pyStr(demand(reference, 'version')),
    });
    if (definition === undefined) {
      fail('V1', `primitive absent from primitive library: ${pyStr(name)}`, [...at, 'primitive']);
      continue;
    }
    const env: Env = new Map(key.indices.map((one) => [one.name, one.value]));
    let template: { path: string; document: PyValue } | null = null;
    if (has(instance, 'weights_location_prefix')) {
      if (!has(definition, 'template')) {
        fail(
          'V17',
          `${pyStr(name)} @${whereOfSite(key)}: a weights_location_prefix on an instance that ` +
            `is not a template instance`,
          [...at, 'weights_location_prefix'],
        );
      } else {
        const written = physicalName(
          demand(instance, 'weights_location_prefix'),
          env,
          value,
          new Map(),
        );
        if (written.problem !== null) {
          fail(
            'V17',
            `${pyStr(name)} @${whereOfSite(key)}: weights_location_prefix: ${written.problem}`,
            [...at, 'weights_location_prefix'],
          );
        } else {
          weightsPrefixes.set(id, written.name as string);
        }
      }
    }
    if (has(definition, 'template')) {
      // "A template primitive: its arguments are the template's external quantities, with their
      // types, domains and declared defaults."
      template = templateOf(definition);
      definition = templateInterface(definition, normalise(template.document));
    }
    if (!pyEqual(demand(definition, 'version'), demand(reference, 'version'))) {
      fail(
        'V1',
        `${pyStr(name)}: version ${pyStr(demand(reference, 'version'))} ` +
          `!= primitive library ${pyStr(demand(definition, 'version'))}`,
        [...at, 'primitive', 'version'],
      );
    }
    const description = describeArguments(
      definition,
      demand(instance, 'arguments'),
      (one) => staticOf(one, env),
      at,
    );
    const fromInvariants = new Set(
      description.invariants.flatMap((one) => (one.problem === undefined ? [] : [one.problem])),
    );
    const resolutionProblems = description.problems.filter((one) => !fromInvariants.has(one));
    // `fail(code, f"{name} @{where(key)}: {message}")`, the V8 block included: `describe`'s
    // problems are the resolution's then the invariants', which is `analyse`'s own order.
    for (const problem of description.problems) {
      problems.push(
        withMessage(problem, `${pyStr(name)} @${whereOfSite(key)}: ${problem.message}`),
      );
    }
    if (template !== null && resolutionProblems.length === 0) {
      // Expansion at the call site: the template under this assignment.
      if (depth + 1 > MAX_PRIMITIVE_DEPTH) {
        fail(
          'V1',
          `${pyStr(name)} @${whereOfSite(key)}: primitive nesting deeper than ` +
            `${MAX_PRIMITIVE_DEPTH}`,
          at,
        );
      } else {
        const subAssignment: Record<string, PyValue> = {};
        for (const [argument, one] of Object.entries(description.values)) {
          if (one !== UNRESOLVED) subAssignment[argument] = one;
        }
        const cacheKey = `${template.path}\u0000${assignmentToken(subAssignment)}`;
        let sub = cache.get(cacheKey);
        if (sub === undefined) {
          sub = analyseGraph(template.document, library, {
            assignment: subAssignment,
            depth: depth + 1,
            cache,
          });
          cache.set(cacheKey, sub);
        }
        for (const problem of sub.problems) {
          problems.push(
            withMessage(
              problem,
              `${problem.message}  (in instance ${pyStr(name)} @${whereOfSite(key)})`,
            ),
          );
        }
        for (const line of sub.advisories) {
          advisories.push(`${line}  (in instance ${pyStr(name)})`);
        }
        for (const [counter, one] of sub.stats) {
          if (typeof one === 'bigint') merged.set(counter, (merged.get(counter) ?? 0n) + one);
        }
        definition = { ...(definition as PyRecord), ports: instancePorts(sub.ports) };
        subResults.set(id, sub);
      }
    }
    resolved.set(id, {
      key,
      primitive: name,
      definition,
      args: description.values,
      arguments: description,
      instance,
    });
  }

  // --- V1/V4/V7: value edges --------------------------------------------
  const producers = new Map<string, PortProducer>();
  const consumed = new Map<string, PortReference>();
  const edges: ValueEdge[] = [];
  for (const [bid, binding] of entries(demand(demand(model, 'bindings'), 'values'))) {
    const bindingAt: PathSegment[] = ['bindings', 'values', bid];
    for (const env of loopEnvs(binding, bid, quantities, problems, bindingAt)) {
      const source = demand(binding, 'from');
      const target = demand(binding, 'to');
      const sourceKey = selectSite(demand(source, 'instance'), quantities, env);
      const targetKey = selectSite(demand(target, 'instance'), quantities, env);
      if (absent.has(keyOf(sourceKey)) || absent.has(keyOf(targetKey))) continue; // §5.2 rule 3
      let ok = true;
      const ends = [
        { key: sourceKey, port: demand(source, 'port'), side: 'outputs', label: 'from' },
        { key: targetKey, port: demand(target, 'port'), side: 'inputs', label: 'to' },
      ] as const;
      for (const end of ends) {
        const site = resolved.get(keyOf(end.key));
        if (site === undefined) {
          fail(
            'V1',
            `${bid}${reprEnv(env)}: ${end.label} instance does not exist ${whereOfSite(end.key)}`,
            bindingAt,
          );
          ok = false;
          continue;
        }
        if (!has(demand(demand(site.definition, 'ports'), end.side), pyStr(end.port))) {
          fail(
            'V1',
            `${bid}: ${pyStr(site.primitive)} has no ${end.side.slice(0, -1)} port ` +
              `'${pyStr(end.port)}'`,
            bindingAt,
          );
          ok = false;
        }
      }
      if (!ok) continue;
      // V4: shapes unify by axis identity and exact extent
      const from = resolved.get(keyOf(sourceKey)) as ResolvedSite;
      const to = resolved.get(keyOf(targetKey)) as ResolvedSite;
      const fromPort = demand(source, 'port');
      const toPort = demand(target, 'port');
      const fromShape = portShape(
        demand(demand(demand(from.definition, 'ports'), 'outputs'), pyStr(fromPort)),
        from.args,
      );
      const toShape = portShape(
        demand(demand(demand(to.definition, 'ports'), 'inputs'), pyStr(toPort)),
        to.args,
      );
      if (fromShape !== null && toShape !== null && !shapesAgree(fromShape, toShape)) {
        fail(
          'V4',
          `${bid}: shapes do not unify ${pyStr(from.primitive)}.${pyStr(fromPort)}` +
            `${reprShape(fromShape)} -> ${pyStr(to.primitive)}.${pyStr(toPort)}` +
            `${reprShape(toShape)}`,
          bindingAt,
        );
      }
      const fed = portKeyOf(targetKey, toPort);
      if (producers.has(fed)) {
        fail(
          'V7',
          `input port fed twice: ${whereOfSite(targetKey)}.${pyStr(toPort)} ` +
            `by ${(producers.get(fed) as PortProducer).by} and ${bid}`,
          bindingAt,
        );
      }
      producers.set(fed, { site: targetKey, port: toPort, by: bid });
      consumed.set(portKeyOf(sourceKey, fromPort), { site: sourceKey, port: fromPort });
      edges.push({ from: sourceKey, fromPort, to: targetKey, toPort, binding: bid });
    }
  }
  stats.set('edges', BigInt(edges.length));

  // --- interfaces: edges like any other, seeds of the streams -----------
  const interfaces = demand(model, 'interfaces');
  const inputs = demand(interfaces, 'inputs');
  const outputs = demand(interfaces, 'outputs');
  const fragmented = new Set<string>();
  for (const [name, declared] of entries(inputs)) {
    if (truthy(optional(declared, 'fragmented', false))) {
      fragmented.add(pyStr(optional(declared, 'stream', name)));
    }
  }
  const seeds = new Map<string, PortDomain>();
  const fedShapes = new Map<string, { shape: ShapeIdentity | null; where: string }>();
  for (const [name, declared] of entries(inputs)) {
    const at: PathSegment[] = ['interfaces', 'inputs', name];
    const stream = optional(declared, 'stream', name);
    if (has(declared, 'stream') && !has(inputs, pyStr(demand(declared, 'stream')))) {
      fail('V1', `input ${name}: joins unknown stream '${pyStr(demand(declared, 'stream'))}'`, at);
    }
    for (const endpoint of listOf(demand(declared, 'to'))) {
      const key = selectSite(demand(endpoint, 'instance'), quantities, new Map());
      const site = resolved.get(keyOf(key));
      if (site === undefined) {
        fail('V1', `input ${name}: instance does not exist`, at);
        continue;
      }
      const port = demand(endpoint, 'port');
      if (!has(demand(demand(site.definition, 'ports'), 'inputs'), pyStr(port))) {
        fail('V1', `input ${name}: port '${pyStr(port)}' does not exist`, at);
        continue;
      }
      const fed = portKeyOf(key, port);
      if (producers.has(fed)) {
        fail(
          'V7',
          `input ${name}: port ${whereOfSite(key)}.${pyStr(port)} also fed ` +
            `by ${(producers.get(fed) as PortProducer).by}`,
          at,
        );
      }
      producers.set(fed, { site: key, port, by: `input:${name}` });
      seeds.set(fed, { site: key, port, domain: [demand(declared, 'kind'), stream] });
      // "a public input is an edge like any other (§5.3): the ports it feeds compose (V4)"
      const shape = portShape(
        demand(demand(demand(site.definition, 'ports'), 'inputs'), pyStr(port)),
        site.args,
      );
      let first = fedShapes.get(name);
      if (first === undefined) {
        first = { shape, where: `${whereOfSite(key)}.${pyStr(port)}` };
        fedShapes.set(name, first);
      }
      if (!shapesAgree(shape, first.shape)) {
        fail(
          'V4',
          `input ${name}: feeds ${first.where}${reprShape(demandShape(first.shape))} and ` +
            `${whereOfSite(key)}.${pyStr(port)}${reprShape(demandShape(shape))}, whose shapes differ`,
          at,
        );
      }
    }
  }
  for (const [name, declared] of entries(outputs)) {
    const at: PathSegment[] = ['interfaces', 'outputs', name];
    const from = demand(declared, 'from');
    const key = selectSite(demand(from, 'instance'), quantities, new Map());
    const site = resolved.get(keyOf(key));
    if (site === undefined) {
      fail('V1', `output ${name}: instance does not exist`, at);
      continue;
    }
    const port = demand(from, 'port');
    if (!has(demand(demand(site.definition, 'ports'), 'outputs'), pyStr(port))) {
      fail('V1', `output ${name}: port '${pyStr(port)}' does not exist`, at);
      continue;
    }
    consumed.set(portKeyOf(key, port), { site: key, port });
  }

  // V7: every present input port has a producer; V13: every present output a consumer
  for (const [, site] of resolved) {
    const ports = demand(site.definition, 'ports');
    const at = segmentsOfSite(site.key);
    for (const [name, port] of entries(demand(ports, 'inputs'))) {
      if (present(port, site.args) && !producers.has(portKeyOf(site.key, name))) {
        fail(
          'V7',
          `input port with no producer: ${pyStr(site.primitive)}@${whereOfSite(site.key)}.${name}`,
          at,
        );
      }
    }
    for (const [name, port] of entries(demand(ports, 'outputs'))) {
      if (present(port, site.args) && !consumed.has(portKeyOf(site.key, name))) {
        fail(
          'V13',
          `output consumed by nothing: ${pyStr(site.primitive)}@${whereOfSite(site.key)}.${name}`,
          at,
        );
      }
    }
  }

  // --- V6: acyclicity ----------------------------------------------------
  const adjacency = new Map<string, SiteKey[]>();
  const indegree = new Map<string, number>();
  for (const edge of edges) {
    const from = keyOf(edge.from);
    const to = keyOf(edge.to);
    if (!resolved.has(from) || !resolved.has(to)) continue;
    (adjacency.get(from) ?? setAt(adjacency, from, [])).push(edge.to);
    indegree.set(to, (indegree.get(to) ?? 0) + 1);
  }
  const queue = [...resolved.values()]
    .filter((site) => (indegree.get(keyOf(site.key)) ?? 0) === 0)
    .map((site) => site.key)
    .sort(compareSiteKeys);
  const order: SiteKey[] = [];
  for (let head = 0; head < queue.length; head += 1) {
    const node = queue[head] as SiteKey;
    order.push(node);
    for (const next of adjacency.get(keyOf(node)) ?? []) {
      const id = keyOf(next);
      const left = (indegree.get(id) ?? 0) - 1;
      indegree.set(id, left);
      if (left === 0) queue.push(next);
    }
  }
  if (order.length !== resolved.size) {
    fail('V6', `value cycle: ${resolved.size - order.length} instance(s) in a cycle`);
  }
  stats.set('dag', order.length === resolved.size);

  // --- V5: streams, transforms and the instance's own domain (§5.3) ----
  const domains = new Map<string, PortDomain>(seeds);
  const own = new Map<string, Indexing | null>();
  const incoming = new Map<string, ValueEdge[]>();
  for (const edge of edges) {
    const id = keyOf(edge.to);
    (incoming.get(id) ?? setAt(incoming, id, [])).push(edge);
  }
  for (const node of order) {
    const id = keyOf(node);
    const site = resolved.get(id) as ResolvedSite;
    const at = segmentsOfSite(node);
    const declaredTransforms = listOf(optional(site.definition, 'domain_transforms', []));
    const transforms = new Map<string, PyValue>();
    for (const transform of declaredTransforms) {
      transforms.set(pyStr(demand(transform, 'from_port')), transform);
    }
    for (const edge of incoming.get(id) ?? []) {
      const source = domains.get(portKeyOf(edge.from, edge.fromPort));
      if (source === undefined) {
        fail(
          'V5',
          `${edge.binding}: the domain of ${whereOfSite(edge.from)}.${pyStr(edge.fromPort)} ` +
            `is undetermined`,
          ['bindings', 'values', edge.binding],
        );
        continue;
      }
      domains.set(portKeyOf(node, edge.toPort), {
        site: node,
        port: edge.toPort,
        domain: source.domain,
      });
    }
    const agree = new Map<string, Indexing>();
    const ports = demand(site.definition, 'ports');
    for (const [name, port] of entries(demand(ports, 'inputs'))) {
      if (!present(port, site.args)) continue;
      const entry = domains.get(portKeyOf(node, name));
      if (entry === undefined) continue;
      const carried = entry.domain;
      const declared = demand(demand(port, 'domain'), 'kind');
      if (!pyEqual(declared, 'inherit') && !pyEqual(declared, carried[0])) {
        fail(
          'V5',
          `${pyStr(site.primitive)}@${whereOfSite(node)}.${name} expects ${pyStr(declared)}, ` +
            `receives ${pyStr(carried[0])} (stream '${pyStr(carried[1])}')`,
          at,
        );
      }
      if (!transforms.has(name)) agree.set(indexingToken(carried), carried);
    }
    if (agree.size > 1) {
      fail(
        'V5',
        `${pyStr(site.primitive)}@${whereOfSite(node)}: inputs in different domains ` +
          `${reprIndexings([...agree.values()])}, and no domain_transform declares it`,
        at,
      );
    }
    const mine = agree.size === 0 ? null : ([...agree.values()][0] as Indexing);
    own.set(id, mine);
    const mergedTo = new Map<string, PyValue>();
    for (const transform of declaredTransforms) {
      if (pyEqual(demand(transform, 'relation'), 'merge')) {
        mergedTo.set(pyStr(demand(transform, 'to_port')), transform);
      }
    }
    for (const [name, port] of entries(demand(ports, 'outputs'))) {
      const declared = demand(port, 'domain');
      const kind = demand(declared, 'kind');
      let carried: Indexing | undefined;
      if (pyEqual(kind, 'inherit') && has(optional(declared, 'from', {}), 'port')) {
        carried = domains.get(portKeyOf(node, demand(demand(declared, 'from'), 'port')))?.domain;
      } else if (pyEqual(kind, 'inherit')) {
        carried = mine ?? undefined;
      } else if (mergedTo.has(name)) {
        const source = domains.get(
          portKeyOf(node, demand(mergedTo.get(name) as PyValue, 'from_port')),
        );
        carried = source === undefined ? undefined : [kind, source.domain[1]];
      } else {
        carried = mine === null ? undefined : [kind, mine[1]];
      }
      if (carried === undefined) {
        if (present(port, site.args)) {
          fail(
            'V5',
            `${pyStr(site.primitive)}@${whereOfSite(node)}.${name}: domain undetermined`,
            at,
          );
        }
        continue;
      }
      domains.set(portKeyOf(node, name), { site: node, port: name, domain: carried });
    }
  }
  for (const [name, declared] of entries(outputs)) {
    const from = demand(declared, 'from');
    const key = selectSite(demand(from, 'instance'), quantities, new Map());
    const carried = domains.get(portKeyOf(key, demand(from, 'port')));
    if (carried === undefined) continue;
    if (truthy(demand(declared, 'generative')) && !pyEqual(carried.domain[0], 'token')) {
      fail('V5', `output ${name}: generative, but of kind ${pyStr(carried.domain[0])}`, [
        'interfaces',
        'outputs',
        name,
      ]);
    }
  }
  stats.set('resolved_domains', BigInt(domains.size));

  // --- V19: a joining input joins at a kind the stream carries independently of it (§2.3, §5.3)
  const downstream = new Map<string, SiteKey[]>();
  for (const edge of edges) {
    const id = keyOf(edge.from);
    (downstream.get(id) ?? setAt(downstream, id, [])).push(edge.to);
  }
  for (const [name, declared] of entries(inputs)) {
    if (!has(declared, 'stream')) continue;
    const stream = demand(declared, 'stream');
    if (!has(inputs, pyStr(stream))) continue;
    // everything the joining input feeds, directly or through values
    const descends = new Set<string>();
    const pending = listOf(demand(declared, 'to')).map((endpoint) =>
      selectSite(demand(endpoint, 'instance'), quantities, new Map()),
    );
    while (pending.length > 0) {
      const site = pending.pop() as SiteKey;
      const id = keyOf(site);
      if (descends.has(id) || !resolved.has(id)) continue;
      descends.add(id);
      pending.push(...(downstream.get(id) ?? []));
    }
    // "independent of the input: another input's delivery on the stream, or a value produced by an
    // instance the input does not reach (an elementwise primitive's second operand joins the
    // first's stream, which the first delivers: `residual.add`'s `b` joins `a`)"
    const independent = new Map<string, PyValue>();
    for (const [fed, entry] of domains) {
      if (!pyEqual(entry.domain[1], stream)) continue;
      const reached = descends.has(keyOf(entry.site));
      if (!reached || (seeds.has(fed) && producers.get(fed)?.by !== `input:${name}`)) {
        independent.set(valueToken(entry.domain[0]), entry.domain[0]);
      }
    }
    const kind = demand(declared, 'kind');
    if (!independent.has(valueToken(kind))) {
      fail(
        'V19',
        `input ${name}: joins stream '${pyStr(stream)}' at kind ${pyStr(kind)}, which that ` +
          `stream carries at no value independently of the input (it carries ` +
          `${reprSorted([...independent.values()])}): the join would count elements the stream ` +
          `does not have (§5.3)`,
        ['interfaces', 'inputs', name],
      );
    }
  }

  // What this document exposes to a caller: its interface ports, resolved.
  const exposedInputs = new Map<string, InterfacePort>();
  for (const [name, declared] of entries(inputs)) {
    // `decl['to'][0]`: the grammar requires at least one endpoint, and a document that has none
    // gets Python's own refusal rather than a reading this side invented.
    const endpoints = listOf(demand(declared, 'to'));
    if (endpoints.length === 0) throw new PyIndexError('list index out of range');
    const endpoint = endpoints[0] as PyValue;
    const key = selectSite(demand(endpoint, 'instance'), quantities, new Map());
    const site = resolved.get(keyOf(key));
    exposedInputs.set(name, {
      kind: demand(declared, 'kind'),
      stream: optional(declared, 'stream', name),
      shape:
        site === undefined
          ? null
          : portShape(
              portOf(site.definition, 'inputs', demand(endpoint, 'port')),
              site.args,
            ),
    });
  }
  const exposedOutputs = new Map<string, InterfacePort>();
  for (const [name, declared] of entries(outputs)) {
    const from = demand(declared, 'from');
    const key = selectSite(demand(from, 'instance'), quantities, new Map());
    const carried = domains.get(portKeyOf(key, demand(from, 'port')));
    const site = resolved.get(keyOf(key));
    exposedOutputs.set(name, {
      kind: carried === undefined ? null : carried.domain[0],
      stream: carried === undefined ? null : carried.domain[1],
      shape:
        site === undefined
          ? null
          : portShape(portOf(site.definition, 'outputs', demand(from, 'port')), site.args),
    });
  }

  // "Slots and states of expanded templates count with the caller's (§4.6)"; `located` is counted
  // per instance, only where the instance is prefixed, so it is left out of the merge.
  for (const [counter, one] of merged) {
    const held = stats.get(counter);
    if (counter !== 'located' && typeof held === 'bigint') stats.set(counter, held + one);
  }

  return {
    problems,
    advisories,
    stats,
    ports: { inputs: exposedInputs, outputs: exposedOutputs },
    model,
    quantities,
    sites,
    absent,
    resolved,
    edges,
    domains,
    own,
    order,
    fragmented,
    seeds,
    producers,
    consumed,
    weightsPrefixes,
    subResults,
  };
}

/** `definition['ports'][side].get(name, {})`: the port, or an empty one where there is none. */
function portOf(definition: PyValue, side: 'inputs' | 'outputs', name: PyValue): PyValue {
  const ports = demand(demand(definition, 'ports'), side);
  return has(ports, pyStr(name)) ? demand(ports, pyStr(name)) : {};
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

/** `dict(zip(names, combo))`: the index environment at one point of a grid. */
function bind(names: readonly string[], combination: readonly bigint[]): Env {
  const env = new Map<string, PyValue>();
  names.forEach((name, index) => env.set(name, combination[index] as bigint));
  return env;
}

/** `f"{env}"`: a Python dictionary written into a message, `{'layer': 3}` or `{}`. */
function reprEnv(env: Env): string {
  const written = [...env].map(([name, one]) => `${pyRepr(name)}: ${pyRepr(one)}`);
  return `{${written.join(', ')}}`;
}

/** `f"{sorted(agree)}"`: the domains of an instance's inputs, as the V5 refusal lists them. */
function reprIndexings(indexings: readonly Indexing[]): string {
  const sorted = [...indexings].sort((one, other) => {
    const kind = pyOrder(one[0], other[0]) ?? 0;
    return kind !== 0 ? kind : (pyOrder(one[1], other[1]) ?? 0);
  });
  return `[${sorted.map((one) => reprTuple(one)).join(', ')}]`;
}

/** `f"{sorted(independent)}"`: a set of kinds, sorted and written as a list. */
function reprSorted(values: readonly PyValue[]): string {
  const sorted = [...values].sort((one, other) => pyOrder(one, other) ?? 0);
  return `[${sorted.map((one) => pyRepr(one)).join(', ')}]`;
}

/** An indexing domain as a set member: the tuple's two values, hashed as Python hashes them. */
function indexingToken(indexing: Indexing): string {
  return `${valueToken(indexing[0])}\u0000${valueToken(indexing[1])}`;
}

/**
 * `list(shape)` where the shape may be `None`.
 *
 * The V4 refusal about a public input interpolates both shapes without checking either, so a port
 * that declares no shape beside one that does raises `TypeError: 'NoneType' object is not
 * iterable` there — a document the grammar admits, refused by an exception rather than by a line.
 * Reproduced, not corrected (feature 1.5's rule: where the tools raise, the port raises).
 */
function demandShape(shape: ShapeIdentity | null): ShapeIdentity {
  if (shape === null) throw new PyTypeError("'NoneType' object is not iterable");
  return shape;
}

/** `entry['kind'] or 'inherit'`: Python's truthiness over the kind a template's port exposes. */
function falsy(value: PyValue): boolean {
  return value === null || value === false || value === '' || value === 0n;
}

/** `axis.split('.')[-1]`: the local name `instance_ports` gives an axis of an exposed shape. */
function lastSegment(axis: PyValue): PyValue {
  if (typeof axis !== 'string') return axis;
  const parts = axis.split('.');
  return parts[parts.length - 1] as string;
}

/**
 * `json.dumps(sub_assignment, sort_keys=True, default=str)`: the memo's key beside the file.
 *
 * Only its injectivity matters — two different assignments must not share an entry — so the
 * encoding keeps the integer/float distinction the tools' `json.dumps` keeps (`1` against `1.0`)
 * rather than reproducing its text.
 */
function assignmentToken(assignment: Record<string, PyValue>): string {
  const names = Object.keys(assignment).sort(comparePythonStrings);
  return names.map((name) => `${JSON.stringify(name)}:${deepToken(assignment[name] as PyValue)}`).join(',');
}

/** {@link valueToken} extended over the records and lists an argument value may be. */
function deepToken(value: PyValue): string {
  if (Array.isArray(value)) {
    return `[${(value as readonly PyValue[]).map((one) => deepToken(one)).join(',')}]`;
  }
  if (value !== null && typeof value === 'object') {
    const record = value as PyRecord;
    const names = Object.keys(record).sort(comparePythonStrings);
    return `{${names.map((name) => `${JSON.stringify(name)}:${deepToken(record[name] as PyValue)}`).join(',')}}`;
  }
  // A float and an integer of the same value are two assignments, unlike two dictionary keys.
  return typeof value === 'number' ? `f${String(value)}` : valueToken(value);
}

/** `defaultdict(list)`: the list at a key, created empty when there is none. */
function setAt<T>(map: Map<string, T>, key: string, value: T): T {
  map.set(key, value);
  return value;
}

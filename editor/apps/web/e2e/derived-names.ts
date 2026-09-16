/**
 * The normalisation the phase-2 exit compares through — feature 2.19.
 *
 * Plan §0's third done-criterion asks that `llama3-8b` built from an empty document derive "to the
 * same products as the corpus document (**up to instance names**)", and feature 2.19's block says
 * the comparison is "a normalising comparison". This is that normalisation, written once: a
 * **renaming** of the names a *document* declares, applied to a derived document at the places
 * that hold one.
 *
 * ## What it normalises
 *
 * Seven classes of name, each the document's own to choose:
 *
 * | Class | What it renames | Where the derived document carries it |
 * |---|---|---|
 * | `model` | the model identifier | `/model` |
 * | `compositions` | a composition | inside every node identifier, identity name and graph-split id |
 * | `sites` | a root instance, a composition's site, a template instance | the last segment of a node identifier, and a template prefix |
 * | `indices` | a composition's index | inside the brackets of a node identifier and an identity name, in a layer split's id, and in a state's `instance_key` |
 * | `rules` | a binding rule, by the name §5.2 rule 7 gives its identity | `d1.edges[].rule`, and the head of every identity name |
 * | `families` | a family | `d1.nodes[].families`, and the tail of a family split's id |
 * | `interfaces` | a public input or output, and the stream an input introduces (§2.3) | the interface maps, `input`, `exposed`, `required_for`, and every stream-keyed map |
 * | `quantities` | an external quantity | the keys of `/assignment` |
 *
 * Each class is a map from the name as **this** document writes it to the name the other one
 * does. A name the map does not carry is left as it stands, so a renaming of one class alone is
 * an ordinary use.
 *
 * ## What it does not normalise, deliberately
 *
 * Everything else, and in particular:
 *
 * - **every figure** — elements, bytes, counts, statuses, sizes, granularities: a difference
 *   there is what the comparison exists to find;
 * - **the library's own vocabulary** — port, slot, state and component names, roles, axes,
 *   dtypes, primitive names and versions, evolutions, accesses, sharings, communications,
 *   partition targets, sensitivities, visit wordings: those come from the units, not from the
 *   document, so a difference there is real;
 * - **the order of any array** — `d1.edges`, `d3.tensors`, `d2.values`, `d6.partition_options`
 *   and the rest are compared as they stand. D1's listing is canonical (`d1.py`: "the listing
 *   here is the canonical one … whatever the order of the document's members"), and the others
 *   follow the *document's* own member order, which is a fact about the document the exit test
 *   holds itself to rather than one it normalises away;
 * - **the location of a tensor**, its physical name and its parts — those are the checkpoint's;
 * - **`primitive_libraries`**, the base each document resolves from.
 *
 * ## How it refuses to hide a difference
 *
 * **Every string a document writes is at a declared place.** A string *value* at a place that is
 * in neither table raises, so a derived schema that grew a member forces a decision instead of
 * being renamed by accident or skipped by accident. A *key* at such a place is passed through as
 * a member of the schema — which is what it is, every map of the derived schema being in the
 * tables — and reported by {@link foreignKeys}, so that the reading can be held to the schema's
 * own member names rather than asserted here. (It cannot simply refuse a key that is a name being
 * renamed: `input` is a member of a D2 value *and* a family of `llama3-8b`, and the two are told
 * apart by the place and by nothing else.)
 *
 * And the arithmetic of the comparison is the other guard: a rewrite that renamed too much (a port
 * that happens to share a family's name) or too little makes the deep-equal fail. There is no
 * direction in which a mistake here turns a red test green — only a green one red.
 */

/** A renaming of the names a document declares, class by class. */
export interface Renaming {
  /** The model identifier — `/model`. */
  readonly model?: string;
  readonly compositions?: Readonly<Record<string, string>>;
  /** Root instances, composition sites and template instances: the segments of a node identifier. */
  readonly sites?: Readonly<Record<string, string>>;
  readonly indices?: Readonly<Record<string, string>>;
  /** Binding rules, keyed by the identity name §5.2 rule 7 gives them (`decoder.attn.q`). */
  readonly rules?: Readonly<Record<string, string>>;
  readonly families?: Readonly<Record<string, string>>;
  /** Public inputs and outputs — and so the streams an input introduces (§2.3). */
  readonly interfaces?: Readonly<Record<string, string>>;
  /** External quantities: the keys of the assignment a derived document carries. */
  readonly quantities?: Readonly<Record<string, string>>;
}

/** The form the string at a place is written in. */
type Form =
  | 'model'
  | 'node'
  | 'value'
  | 'identity'
  | 'split'
  | 'rule'
  | 'family'
  | 'interface'
  | 'stream'
  | 'index'
  | 'quantity';

/**
 * The places a derived document writes a name the document declares.
 *
 * A pattern is a JSON path whose segments are read literally, except `#` (any array index), `*`
 * (any map key) and a trailing `$key` (the *keys* of the map at that place rather than the values
 * under them). The first pattern that matches wins, so a specific one is written before a general
 * one.
 */
const NAMES: readonly (readonly [string, Form])[] = [
  ['/model', 'model'],
  ['/assignment/$key', 'quantity'],
  // D1 — the expanded graph.
  ['/d1/nodes/$key', 'node'],
  ['/d1/nodes/*/families/#', 'family'],
  ['/d1/instances/$key', 'node'],
  ['/d1/edges/#/rule', 'rule'],
  ['/d1/edges/#/from/node', 'node'],
  ['/d1/edges/#/to/node', 'node'],
  ['/d1/interfaces/inputs/$key', 'interface'],
  ['/d1/interfaces/inputs/*/stream', 'stream'],
  ['/d1/interfaces/inputs/*/to/#/node', 'node'],
  ['/d1/interfaces/outputs/$key', 'interface'],
  ['/d1/interfaces/outputs/*/node', 'node'],
  ['/d1/topological_order/#', 'node'],
  // D2 — the values, the streams and the graph splits.
  ['/d2/values/#/value', 'value'],
  ['/d2/values/#/to/#', 'value'],
  ['/d2/values/#/input', 'interface'],
  ['/d2/values/#/exposed/#', 'interface'],
  ['/d2/values/#/required_for/#', 'interface'],
  ['/d2/values/#/domain/stream', 'stream'],
  ['/d2/values/#/count/$key', 'stream'],
  ['/d2/streams/$key', 'stream'],
  ['/d2/streams/*/count/$key', 'stream'],
  ['/d2/graph_splits/#/graph_split', 'split'],
  ['/d2/graph_splits/#/payload/#/value', 'value'],
  ['/d2/graph_splits/#/payload/#/count/$key', 'stream'],
  ['/d2/graph_splits/#/bytes_per_invocation/$key', 'stream'],
  ['/d2/peak_live/node', 'node'],
  ['/d2/peak_live/values/#', 'value'],
  ['/d2/peak_live/bytes_per_invocation/$key', 'stream'],
  // D3 — the parameter tensors.
  ['/d3/tensors/#/identity', 'identity'],
  ['/d3/tensors/#/members/#', 'value'],
  // D4 — the state identities.
  ['/d4/states/#/identity', 'identity'],
  ['/d4/states/#/members/#', 'value'],
  ['/d4/states/#/writer', 'value'],
  ['/d4/states/#/stream/stream', 'stream'],
  ['/d4/states/#/instance_key/#', 'index'],
  ['/d4/totals/carried/#', 'identity'],
  // D5 — the arithmetic.
  ['/d5/corrections/#/node', 'node'],
  ['/d5/sparsity/#/node', 'node'],
  ['/d5/graph_splits/#/graph_split', 'split'],
  ['/d5/graph_splits/#/bytes_per_invocation/$key', 'stream'],
  // D6 — the decomposition options.
  ['/d6/graph_splits/#/graph_split', 'split'],
  ['/d6/graph_splits/#/block/#', 'node'],
  ['/d6/graph_splits/#/separated_states/#/identity', 'rule'],
  ['/d6/graph_splits/#/separated_states/#/writer', 'value'],
  ['/d6/graph_splits/#/separated_states/#/first/#', 'value'],
  ['/d6/graph_splits/#/separated_states/#/second/#', 'value'],
  ['/d6/graph_splits/#/separated_states/#/history_needed_by/#', 'value'],
  ['/d6/information_loss/#/node', 'node'],
  ['/d6/partition_options/#/node', 'node'],
];

/**
 * The places a derived document writes something that is **not** a name the document declares.
 *
 * The library's vocabulary (ports, slots, roles, axes, dtypes, primitives), the schemas' own
 * enumerations, the wordings §7 fixes, the checkpoint's physical names, and the two members of
 * the envelope that are neither a name nor a product. A trailing `**` covers the place and
 * everything under it, keys included — which is what an argument map, a location and a partition
 * target need, their shapes being the primitive's rather than this schema's.
 */
const VOCABULARY: readonly string[] = [
  '/schema',
  '/primitive_libraries/#/base',
  '/assignment/*',
  '/d1/nodes/*/primitive/name',
  '/d1/nodes/*/primitive/version',
  '/d1/nodes/*/arguments/**',
  '/d1/instances/*/primitive/name',
  '/d1/instances/*/primitive/version',
  '/d1/instances/*/arguments/**',
  '/d1/instances/*/weights_location_prefix',
  '/d1/edges/#/from/port',
  '/d1/edges/#/to/port',
  '/d1/interfaces/inputs/*/kind',
  '/d1/interfaces/inputs/*/to/#/port',
  '/d1/interfaces/outputs/*/port',
  '/d2/values/#/dtype',
  '/d2/values/#/role',
  '/d2/values/#/domain/kind',
  '/d2/values/#/shape/#/axis',
  '/d2/streams/*/kind',
  '/d2/graph_splits/#/kind',
  '/d3/tensors/#/dtype',
  '/d3/tensors/#/primitive',
  '/d3/tensors/#/role',
  '/d3/tensors/#/sensitivity',
  '/d3/tensors/#/slot',
  '/d3/tensors/#/shape/#/axis',
  '/d3/tensors/#/shape/#/factors/#/axis',
  '/d3/tensors/#/sparsity/axis',
  '/d3/tensors/#/location/**',
  '/d4/states/#/access',
  '/d4/states/#/evolution',
  '/d4/states/#/indexed_by_port',
  '/d4/states/#/operations/#',
  '/d4/states/#/payload/#/component',
  '/d4/states/#/payload/#/dtype',
  '/d4/states/#/payload/#/role',
  '/d4/states/#/payload/#/shape/#/axis',
  '/d4/states/#/primitive',
  '/d4/states/#/sharing',
  '/d4/states/#/state',
  '/d4/states/#/stream/kind',
  '/d4/states/#/visits/read',
  '/d4/states/#/visits/write',
  '/d4/totals/by_evolution/$key',
  '/d5/corrections/#/per',
  '/d5/corrections/#/primitive',
  '/d5/corrections/#/status',
  '/d5/operations/$key',
  '/d5/operations/*/status',
  '/d5/parameters/status',
  '/d5/sparsity/#/primitive',
  '/d5/sparsity/#/union_per_invocation/status',
  '/d5/state/status',
  '/d6/graph_splits/#/kind',
  '/d6/graph_splits/#/separated_states/#/evolution',
  '/d6/graph_splits/#/separated_states/#/sharing',
  '/d6/graph_splits/#/separated_states/#/writer_side',
  '/d6/information_loss/#/axis',
  '/d6/information_loss/#/reason',
  '/d6/information_loss/#/slot',
  '/d6/partition_options/#/communication/#',
  '/d6/partition_options/#/primitive',
  '/d6/partition_options/#/target/**',
];


/** A JSON value, as `JSON.parse` answers one. */
type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** The error a guard raises: a place the tables do not account for. */
export class UndeclaredPlace extends Error {
  override readonly name = 'UndeclaredPlace';

  constructor(
    readonly place: string,
    readonly written: string,
  ) {
    super(`${place}: ${JSON.stringify(written)} is at a place the renaming’s tables do not name`);
  }
}

/** Whether a path matches a pattern, `#` and `*` standing for one segment and `**` for any. */
function matches(pattern: string, path: string): boolean {
  const wanted = pattern.split('/');
  const steps = path.split('/');
  for (const [at, step] of wanted.entries()) {
    if (step === '**') return true;
    if (at >= steps.length) return false;
    if (step === '#' || step === '*') continue;
    if (step !== steps[at]) return false;
  }
  return wanted.length === steps.length;
}

/**
 * The form the tables give a place: a {@link Form} where a name stands there, `null` where the
 * tables call it vocabulary, and `undefined` where neither table names the place at all.
 */
function formOf(path: string): Form | null | undefined {
  for (const [pattern, form] of NAMES) if (matches(pattern, path)) return form;
  for (const pattern of VOCABULARY) if (matches(pattern, path)) return null;
  return undefined;
}

/**
 * What a walk does with one name: answer what stands there instead.
 *
 * The one seam of this module. {@link renameDerived} answers the map's entry, {@link namesIn}
 * records the name and answers it unchanged — so the *reading* of every identifier form is
 * written once and the two callers cannot disagree about which segments of an identifier are
 * names and which are the library's own words.
 */
export type Take = (kind: keyof Renaming, name: string) => string;

/** The indices of an identifier's bracket — `layer=0,head=2` — with their names taken. */
function indices(inner: string, take: Take): string {
  return inner
    .split(',')
    .map((one) => {
      const at = one.indexOf('=');
      return at < 0 ? one : `${take('indices', one.slice(0, at))}=${one.slice(at + 1)}`;
    })
    .join(',');
}

/** One segment of a node identifier — `attn[layer=0]` — with its own name taken by `named`. */
function segment(text: string, named: (name: string) => string, take: Take): string {
  const bracket = /^([^[\]]*)\[([^[\]]*)\]$/.exec(text);
  if (bracket === null) return named(text);
  return `${named(bracket[1] as string)}[${indices(bracket[2] as string, take)}]`;
}

/**
 * A node identifier: `<site>`, `<composition>/<site>[<i>=<v>,…]`, or either prefixed by a
 * template instance (§5.2 rule 2).
 *
 * The last segment is a site; every segment before it is a composition or the template instance
 * that carries it, so both classes are asked — a name in neither is left alone.
 */
function node(text: string, take: Take): string {
  const steps = text.split('/');
  return steps
    .map((step, at) =>
      segment(
        step,
        (name) =>
          at === steps.length - 1 ? take('sites', name) : take('sites', take('compositions', name)),
        take,
      ),
    )
    .join('/');
}

/**
 * A value reference: `<node identifier>.<port or slot>`, or a public input's own name.
 *
 * A node identifier carries no dot (the derived schema's own pattern for one), so the last dot is
 * where the port begins; a string with no dot at all is an interface. The port or slot is the
 * *primitive's* name and is never taken.
 */
function value(text: string, take: Take): string {
  const at = text.lastIndexOf('.');
  if (at < 0) return take('interfaces', text);
  return `${node(text.slice(0, at), take)}.${text.slice(at + 1)}`;
}

/**
 * A binding rule, by the name §5.2 rule 7 gives its identity.
 *
 * Whole, because a rule's name is a `qualified_name` and nothing in the text says where a
 * composition's prefix ends (`decoder.attn.q` is `decoder` ▸ `attn.q` *or* a top-level rule of
 * that name — feature 2.8 met the same ambiguity and refused to split it). Where the class does
 * not carry the whole name and its head is a composition being renamed, the head alone is taken,
 * so that renaming a composition moves the rules it scopes without listing them one by one.
 */
function rule(text: string, take: Take): string {
  const whole = take('rules', text);
  if (whole !== text) return whole;
  const at = text.indexOf('.');
  if (at < 0) return text;
  const head = take('compositions', text.slice(0, at));
  if (head === text.slice(0, at)) return text;
  return `${head}.${take('rules', text.slice(at + 1))}`;
}

/**
 * An identity name: the binding rule's identity with its evaluated indices, prefixed by its
 * template instance when it is inside one.
 */
function identity(text: string, take: Take): string {
  const bracket = /^([^[\]]*)\[([^[\]]*)\]$/.exec(text);
  const head = bracket === null ? text : (bracket[1] as string);
  const tail = bracket === null ? '' : `[${indices(bracket[2] as string, take)}]`;
  const slash = head.lastIndexOf('/');
  const prefix = slash < 0 ? '' : `${node(head.slice(0, slash), take)}/`;
  return `${prefix}${rule(head.slice(slash + 1), take)}${tail}`;
}

/**
 * A graph-split id: a composition's layer prefix (`decoder[layer<=0]`) or a family's
 * (`family:norm`).
 *
 * Neither shape is stated by the derived schema — `graph_split_id` is a bare string — so both are
 * read off the id itself, and an id of neither shape is left exactly as it stands.
 */
function split(text: string, take: Take): string {
  const bracket = /^([^[\]]*)\[([A-Za-z_][A-Za-z0-9_-]*)([^[\]]*)\]$/.exec(text);
  if (bracket !== null) {
    const composition = take('compositions', bracket[1] as string);
    return `${composition}[${take('indices', bracket[2] as string)}${bracket[3] as string}]`;
  }
  const at = text.indexOf(':');
  return at < 0 ? text : `${text.slice(0, at + 1)}${take('families', text.slice(at + 1))}`;
}

/** One string, at the place it stands, under a walk's own {@link Take}. */
function visit(text: string, form: Form, take: Take): string {
  switch (form) {
    case 'model':
      return take('model', text);
    case 'node':
      return node(text, take);
    case 'value':
      return value(text, take);
    case 'identity':
      return identity(text, take);
    case 'split':
      return split(text, take);
    case 'rule':
      return rule(text, take);
    case 'family':
      return take('families', text);
    case 'interface':
    case 'stream':
      return take('interfaces', text);
    case 'index':
      return take('indices', text);
    case 'quantity':
      return take('quantities', text);
  }
}

/**
 * Walk a derived document, taking every name at every place the tables call one.
 *
 * A key at a place neither table names is passed through as a *member of the schema*, and
 * reported to `foreign` so that the claim can be asked of a document rather than asserted here:
 * the derived schema declares every map of its own (`d1.nodes`, the interfaces, the streams, the
 * stream-keyed figures, the arguments, the location, the partition target, the assignment), and
 * all of them are in the tables. A key that is neither is a map nobody classified — which the
 * suite catches by holding every such key to the schema's own member names.
 */
function walk(document: unknown, take: Take, foreign?: (place: string, key: string) => void): unknown {
  const step = (held: Json, path: string): Json => {
    if (Array.isArray(held)) return held.map((one, at) => step(one, `${path}/${String(at)}`));
    if (held !== null && typeof held === 'object') {
      const made: { [key: string]: Json } = {};
      for (const [key, one] of Object.entries(held)) {
        const place = `${path}/$key`;
        const form = formOf(place);
        if (form === undefined) foreign?.(place, key);
        made[form === undefined || form === null ? key : visit(key, form, take)] = step(
          one,
          // A map's key is one segment of the place whatever it holds, and a node identifier
          // holds slashes (`decoder/attn[layer=0]`): escaped as RFC 6901 escapes one, so that a
          // pattern's `*` matches the whole key and never a piece of it.
          `${path}/${key.replaceAll('/', '~1')}`,
        );
      }
      return made;
    }
    if (typeof held === 'string') {
      const form = formOf(path);
      if (form === undefined) throw new UndeclaredPlace(path, held);
      return form === null ? held : visit(held, form, take);
    }
    return held;
  };
  return step(document as Json, '');
}

/**
 * A derived document with the names one document declares replaced by the names of another.
 *
 * The document handed in is not touched: a new tree comes back, with member order preserved so
 * that a reader diffing the two sees only what moved.
 */
export function renameDerived(document: unknown, renaming: Renaming): unknown {
  const take: Take = (kind, name) => {
    if (kind === 'model') return renaming.model ?? name;
    const map = renaming[kind];
    return map !== undefined && Object.prototype.hasOwnProperty.call(map, name)
      ? (map[name] as string)
      : name;
  };
  return walk(document, take);
}

/**
 * Every name a derived document carries, by the class the tables read it under.
 *
 * The same walk as {@link renameDerived} with the map left out, so what it answers is exactly what
 * a renaming can reach — which is how the suite states that a rewrite left nothing behind, and
 * what makes the tables' claim about the document readable rather than asserted.
 */
export function namesIn(document: unknown): Readonly<Record<keyof Renaming, readonly string[]>> {
  const held = new Map<keyof Renaming, Set<string>>();
  const take: Take = (kind, name) => {
    const set = held.get(kind) ?? new Set<string>();
    set.add(name);
    held.set(kind, set);
    return name;
  };
  walk(document, take);
  const made: Record<string, readonly string[]> = {};
  for (const kind of ['model', 'compositions', 'sites', 'indices', 'rules', 'families', 'interfaces', 'quantities'] as const) {
    made[kind] = [...(held.get(kind) ?? [])].sort();
  }
  return made as Readonly<Record<keyof Renaming, readonly string[]>>;
}

/**
 * Every map key a derived document carries at a place neither table names.
 *
 * The walk takes such a key for a member of the schema; this is what lets a suite hold that
 * reading to the schema itself, so that a map the tables do not classify is a decision somebody
 * takes rather than a name the renaming quietly misses.
 */
export function foreignKeys(document: unknown): readonly { place: string; key: string }[] {
  const found: { place: string; key: string }[] = [];
  walk(document, (_kind, name) => name, (place, key) => found.push({ place, key }));
  return found;
}

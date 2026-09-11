/**
 * The **folded** reading of a document: what §4.7's canvas draws, before anything is derived.
 *
 * > The default editor of a model: the *folded* document — root instances, compositions as group
 * > boxes, interfaces as terminal nodes — the reading `--view` gives, made editable.
 *
 * The layout spike (feature 0.4) projected a document into boxes in its own source and said why
 * that could not ship: "reading a document means naming the language's vocabulary … §5.3 says
 * where it belongs instead: **the core answers what an instance is, and feature 2.9 hands the
 * canvas the facts it draws**." This is that answer. Everything the canvas needs *about the
 * document itself* — which boxes there are, which box an edge's end hangs on, which index a
 * boundary handle names, how many iterations a composition runs, what a gesture writes into a
 * `value_endpoint` — is read here, and the interface draws it without naming a member of the
 * grammar.
 *
 * **It is a reading of the document, not of the analysis.** Three states the canvas has to draw
 * say why:
 *
 *  - a document **off the grammar** has no analysis at all (`describe` gates on it, feature 1.6d)
 *    and the canvas still has to show what the author is editing;
 *  - a **template with no assignment** is skipped by every call of the tools (feature 1.11), and
 *    artboard S3 draws exactly that: the box with `layer ∈ [0, layers)` and a count of `×?`;
 *  - a document whose **library was not gathered** resolves no primitive, and a card with no
 *    ports is still a card.
 *
 * So every read below is the tolerant one ({@link members}, {@link get}) and nothing raises: a
 * member the grammar requires and the file does not carry is simply absent from the answer. What
 * the analysis adds — present ports, slot chips, evaluated shapes — is `describe`'s, keyed to
 * these boxes by {@link FoldedNode.where}.
 *
 * **It reads the store's own tree** (D1), lexemes and member order included, because the two
 * things a gesture needs are the *written* expression of an index (an author who wrote
 * `layers − 1` meant the last layer whatever `layers` is, and a canvas that wrote back `31` would
 * have changed the document) and a pointer into the tree the sheets edit.
 *
 * **What is evaluated, and by whom.** A composition's count and a handle's index are expressions,
 * and §7 of the component inventory forbids a component to evaluate one. They are evaluated here,
 * by `expr/model.ts` — the evaluator `analyse` and `d1.emit` use — under the quantities
 * `resolveQuantities` answers, and an expression that does not resolve answers `null` rather than
 * a guess. That is the whole of "the count only when the range resolves — the same rule as
 * `--view`" (§4.7), and the lesson of erratum E10: `llama3-8b`'s `decoder` runs `layer ∈ [0, 32)`
 * because its `stop` *is* the literal 32, and only the template names a quantity there.
 */
import { indexGrid, modelValue, QUANTITIES, resolveQuantities, type Quantities } from '../expr/model.js';
import { toPython, UNRESOLVED, type PyRecord, type PyValue } from '../expr/value.js';
import {
  isJsonArray,
  isJsonObject,
  jsonObject,
  type JsonObject,
  type JsonValue,
} from '../json/index.js';
import { pointerSegment } from '../schema/index.js';
import type { PathSegment } from '../schema/types.js';
import { generatedSite, rootSite, whereOfSite, type IndexBinding, type SiteKey } from '../validate/index.js';

// ---------------------------------------------------------------------------------------------
// The members of a `tensorspine/2.0` document this reading walks. They are the grammar's, and the
// core is where the grammar is read: plan §1 keeps them out of the interface's own source, which
// is the whole reason this module exists rather than a projection in `packages/ui`.
// ---------------------------------------------------------------------------------------------

const INSTANCES = 'instances';
const COMPOSITIONS = 'compositions';
const BINDINGS = 'bindings';
const VALUES = 'values';
const INTERFACES = 'interfaces';
const INPUTS = 'inputs';
const OUTPUTS = 'outputs';
const INDICES = 'indices';
const PRIMITIVE = 'primitive';
const NAME = 'name';
const VERSION = 'version';
const FAMILIES = 'families';
const WHEN = 'when';
const FOR_EACH = 'for_each';
const FROM = 'from';
const TO = 'to';
const PORT = 'port';
const INSTANCE = 'instance';
const COMPOSITION = 'composition';
const KIND = 'kind';
const ROOT = 'root';
const GENERATED = 'generated';
const START = 'start';
const STOP = 'stop';
const STEP = 'step';
const LITERAL = 'literal';

/** The members of an interface's declaration that are endpoints rather than badges. */
const ENDPOINTS: readonly string[] = [TO, FROM];

/** The members of an `instance_definition` the folded card shows on its face (§4.7). */
export const FACE_MEMBERS: readonly string[] = [PRIMITIVE, FAMILIES, WHEN];

/** What a box of the folded canvas stands for. */
export type FoldedKind =
  /** An instance written at the top level of the document. */
  | 'root'
  /** A site of a composition: a card inside a group box. */
  | 'site'
  /** A composition: §4.7's group box, collapsed or expanded in place. */
  | 'composition'
  /** A public input: the terminal on the left. */
  | 'input'
  /** A public output: the terminal on the right. */
  | 'output';

/** One member of an interface's declaration shown beside its name (S1: `token`, `generative`). */
export interface FoldedBadge {
  /** The member's own name, which is what a `true` shows (`generative`, `fragmented`). */
  readonly name: string;
  /** The value, where it is not a truth — `token` for a `kind`, `audio` for a `stream`. */
  readonly value: JsonValue | null;
}

/** One index of a composition, as the header prints it: the bounds as the document writes them. */
export interface FoldedRange {
  readonly name: string;
  readonly start: JsonValue;
  readonly stop: JsonValue;
  /** The step, `null` where the document writes none (the grammar's own default is 1). */
  readonly step: JsonValue | null;
}

/** One index of a selector: written, and — where it resolves — evaluated. */
export interface FoldedIndexValue {
  readonly name: string;
  /** The expression the document writes: `{"literal": 0}`, `{"op": "subtract", …}`. */
  readonly written: JsonValue;
  /** What it evaluates to under the document's quantities, or `null` where it does not resolve. */
  readonly value: PyValue | null;
}

/**
 * Where one end of an edge attaches on the folded canvas.
 *
 * The distinction §4.7 turns on is {@link boundary}: an end inside a composition hangs on the
 * *box*, because "a composition has no ports" (D8) — "the sites and indices that outside edges
 * name are listed on the composition's box as labelled handles (`attn_n[layer=0].input`)".
 */
export interface FoldedHandle {
  /** The pointer of the box the handle is drawn on: the composition where the site is inside one. */
  readonly box: string;
  /** The pointer of the site itself; the same as {@link box} for a root instance or a terminal. */
  readonly site: string;
  /** The site's own name, which is what a boundary handle's label starts with. */
  readonly name: string;
  /** The port, or `''` for a terminal, which has none. */
  readonly port: string;
  /** The indices the selector names, in the order the document writes them. */
  readonly indices: readonly FoldedIndexValue[];
  /** How a refusal and D1 name the site (`decoder/attn_n[layer=0]`); `null` while an index does not resolve. */
  readonly where: string | null;
  /**
   * The identifier D2 gives the value this end carries — `<where>.<port>` — which is what
   * View ▸ Show Edge Types looks the type up by. `null` where {@link where} is.
   */
  readonly value: string | null;
  /** Whether the handle sits on a composition's boundary rather than on the site's own card. */
  readonly boundary: boolean;
}

/** One edge of the folded canvas (§4.7, inventory §3 "Value edge"). */
export interface FoldedEdge {
  /** The binding's own name — the edge's label (`presentation.json`'s `label: "$key"`). */
  readonly rule: string;
  readonly pointer: string;
  readonly segments: readonly PathSegment[];
  /** The producing end, or `null` where the document writes none the canvas can place. */
  readonly from: FoldedHandle | null;
  /** The consuming end. */
  readonly to: FoldedHandle | null;
  /** The binding's guard, as written; `null` where it has none. */
  readonly guard: JsonValue | null;
  /** The binding's `for_each`, as written; `null` where it has none. */
  readonly forEach: JsonValue | null;
  /** Whether the edge is an interface's own wire rather than a value binding. */
  readonly interface: boolean;
}

/** How many entries a container of a composition holds — S3's `6 sites · 8 scoped edges · …`. */
export interface FoldedHeld {
  readonly name: string;
  readonly count: number;
}

/** One box of the folded canvas. */
export interface FoldedNode {
  /** The place, as an RFC 6901 pointer: the box's identity and the sidecar's own key (D6). */
  readonly pointer: string;
  readonly segments: readonly PathSegment[];
  /** The map key: what the box is called. */
  readonly name: string;
  readonly kind: FoldedKind;
  /** The pointer of the box this one sits in — a site's composition; `null` at the top level. */
  readonly parent: string | null;
  /** `primitive.name`, or `null` for a composition and an interface. */
  readonly primitive: string | null;
  /** `primitive.version`. */
  readonly version: string | null;
  readonly families: readonly string[];
  /** The `when` of an instance or a composition, as written; `null` where it has none. */
  readonly guard: JsonValue | null;
  /** An interface's scalar members, shown beside its name (S1's `token`, `generative`). */
  readonly badges: readonly FoldedBadge[];
  /** A composition's sites, in the document's own order; empty for every other box. */
  readonly children: readonly FoldedNode[];
  /** A composition's index ranges, in the document's own order. */
  readonly ranges: readonly FoldedRange[];
  /**
   * How many iterations the ranges resolve to, or `null` where a bound does not resolve.
   *
   * §4.7: "the count only when the range resolves — the same rule as `--view`".
   */
  readonly count: bigint | null;
  /** What a composition holds, container by container: S3's own summary line. */
  readonly held: readonly FoldedHeld[];
  /**
   * The site of the expanded graph this box stands for — the **representative** iteration.
   *
   * A root instance stands for itself; a site stands for the first point of its composition's
   * grid, which is what §4.8 calls "one representative iteration" and what the card's ports,
   * slots and figures are described from. `null` for a composition, for an interface, and for a
   * site whose grid does not resolve.
   */
  readonly site: SiteKey | null;
  /** {@link site} as a refusal and D1 name it — the key `describe`'s `only` filter takes. */
  readonly where: string | null;
}

/** The folded document: its boxes, in the document's own order, and the edges between them. */
export interface FoldedGraph {
  /** The top-level boxes: root instances, compositions, then the interfaces' terminals. */
  readonly nodes: readonly FoldedNode[];
  /** The value bindings and the interfaces' wires, the bindings first. */
  readonly edges: readonly FoldedEdge[];
  /** Every box, a composition's sites included, by its pointer. */
  readonly byPointer: ReadonlyMap<string, FoldedNode>;
  /** The quantities the ranges and the indices were evaluated under. */
  readonly quantities: Quantities;
}

/** What {@link foldedGraph} is given beside the document. */
export interface FoldedOptions {
  /** The assignment the external quantities are read under (§4.6); absent for a closed document. */
  readonly assignment?: PyRecord;
}

// ---------------------------------------------------------------------------------------------
// Reading the tree, tolerantly: a document off the grammar is drawn as far as it goes.
// ---------------------------------------------------------------------------------------------

/** A pointer from path segments, RFC 6901. */
function pointerOf(segments: readonly PathSegment[]): string {
  return segments.map((step) => `/${pointerSegment(String(step))}`).join('');
}

/** The member of an object node, or `null` where there is none — and where it is not one. */
function member(value: JsonValue | null, name: string): JsonValue | null {
  if (value === null || !isJsonObject(value)) return null;
  const found = value.members.find((one) => one.name === name);
  return found === undefined ? null : found.value;
}

/** Whether an object node writes that member at all (`null` is a value, absence is not). */
function writes(value: JsonValue | null, name: string): boolean {
  return value !== null && isJsonObject(value) && value.members.some((one) => one.name === name);
}

/** The members of a map, in the document's own order; nothing where the place is not a map. */
function entriesOf(value: JsonValue | null): { name: string; value: JsonValue }[] {
  if (value === null || !isJsonObject(value)) return [];
  return value.members.map((one) => ({ name: one.name, value: one.value }));
}

/** A member that must be a name, or `null`. */
function nameOf(value: JsonValue | null, name: string): string | null {
  const found = member(value, name);
  return typeof found === 'string' ? found : null;
}

/** The strings of a list member, ignoring whatever is not one (a document off the grammar). */
function stringsOf(value: JsonValue | null): string[] {
  if (value === null || !isJsonArray(value)) return [];
  return value.filter((one): one is string => typeof one === 'string');
}

/** The quantities of a document, resolved — empty where the document has none to resolve. */
function quantitiesOf(document: JsonValue, assignment?: PyRecord): Quantities {
  const declared = member(document, QUANTITIES);
  if (declared === null || !isJsonObject(declared)) return new Map<string, PyValue>();
  try {
    return resolveQuantities({ [QUANTITIES]: toPython(declared) }, assignment);
  } catch {
    // A quantity declaration off the grammar. The schema stage reports it; the canvas draws the
    // boxes it can, and every expression then simply fails to resolve rather than being guessed.
    return new Map<string, PyValue>();
  }
}

/** An expression under the document's quantities, or `null` where it does not resolve. */
function resolve(expression: JsonValue | null, quantities: Quantities): PyValue | null {
  if (expression === null) return null;
  try {
    const value = modelValue(toPython(expression), quantities);
    return value === UNRESOLVED ? null : value;
  } catch {
    return null;
  }
}

/** The grid of a composition's indices, or `null` where a bound does not resolve. */
function grid(indices: JsonValue | null, quantities: Quantities): { names: string[]; ranges: bigint[][] } | null {
  if (indices === null || !isJsonObject(indices)) return null;
  try {
    return indexGrid(toPython(indices), quantities);
  } catch {
    return null;
  }
}

/** The first point of a grid, in the order `generatedSite` sorts them; `null` where it is empty. */
function firstPoint(grid: { names: string[]; ranges: bigint[][] } | null): IndexBinding[] | null {
  if (grid === null) return null;
  const point: IndexBinding[] = [];
  for (const [at, name] of grid.names.entries()) {
    const values = grid.ranges[at];
    if (values === undefined || values.length === 0) return null;
    point.push({ name, value: values[0] as PyValue });
  }
  return point;
}

/** How many points a grid has; `null` where it does not resolve. */
function countOf(grid: { names: string[]; ranges: bigint[][] } | null): bigint | null {
  if (grid === null) return null;
  let count = 1n;
  for (const values of grid.ranges) count *= BigInt(values.length);
  return count;
}

/** The ranges a composition declares, as written. */
function rangesOf(indices: JsonValue | null): FoldedRange[] {
  return entriesOf(indices).map((one) => ({
    name: one.name,
    start: member(one.value, START) ?? null,
    stop: member(one.value, STOP) ?? null,
    step: writes(one.value, STEP) ? member(one.value, STEP) : null,
  }));
}

/** What a composition holds, container by container — S3's summary line. */
function heldBy(composition: JsonValue): FoldedHeld[] {
  const held: FoldedHeld[] = [
    { name: INSTANCES, count: entriesOf(member(composition, INSTANCES)).length },
  ];
  for (const one of entriesOf(member(composition, BINDINGS))) {
    held.push({ name: one.name, count: entriesOf(one.value).length });
  }
  return held;
}

/** The scalar members of an interface's declaration: what the terminal shows beside its name. */
function badgesOf(declaration: JsonValue): FoldedBadge[] {
  const badges: FoldedBadge[] = [];
  for (const one of entriesOf(declaration)) {
    if (ENDPOINTS.includes(one.name)) continue;
    const value = one.value;
    if (value === true) badges.push({ name: one.name, value: null });
    else if (value === false || value === null || isJsonObject(value) || isJsonArray(value)) continue;
    else badges.push({ name: one.name, value });
  }
  return badges;
}

/** One instance of the document, folded: a root instance, or a site of a composition. */
function instanceNode(
  name: string,
  declaration: JsonValue,
  segments: readonly PathSegment[],
  kind: 'root' | 'site',
  parent: string | null,
  site: SiteKey | null,
): FoldedNode {
  const reference = member(declaration, PRIMITIVE);
  return {
    pointer: pointerOf(segments),
    segments,
    name,
    kind,
    parent,
    primitive: nameOf(reference, NAME),
    version: nameOf(reference, VERSION),
    families: stringsOf(member(declaration, FAMILIES)),
    guard: writes(declaration, WHEN) ? member(declaration, WHEN) : null,
    badges: [],
    children: [],
    ranges: [],
    count: null,
    held: [],
    site,
    where: site === null ? null : whereOfSite(site),
  };
}

/**
 * The folded document, as §4.7 draws it.
 *
 * The nodes come in the schema's own order — instances, compositions, then the interfaces — which
 * is the order the Model explorer walks them in (§4.5) and the order the layout breaks ties with
 * (`respectOrder`, feature 0.4).
 */
export function foldedGraph(tree: JsonValue, options: FoldedOptions = {}): FoldedGraph {
  const quantities = quantitiesOf(tree, options.assignment);
  const nodes: FoldedNode[] = [];
  const byPointer = new Map<string, FoldedNode>();
  /** Which box a site's pointer hangs on: itself for a root instance, its composition for a site. */
  const boxOf = new Map<string, string>();

  const keep = (node: FoldedNode, box: string): void => {
    byPointer.set(node.pointer, node);
    boxOf.set(node.pointer, box);
  };

  for (const one of entriesOf(member(tree, INSTANCES))) {
    const node = instanceNode(one.name, one.value, [INSTANCES, one.name], ROOT, null, rootSite(one.name));
    nodes.push(node);
    keep(node, node.pointer);
  }

  for (const one of entriesOf(member(tree, COMPOSITIONS))) {
    const segments: PathSegment[] = [COMPOSITIONS, one.name];
    const pointer = pointerOf(segments);
    const indices = member(one.value, INDICES);
    const points = grid(indices, quantities);
    const first = firstPoint(points);
    const children: FoldedNode[] = [];
    for (const site of entriesOf(member(one.value, INSTANCES))) {
      const key = first === null ? null : generatedSite(one.name, site.name, first);
      children.push(
        instanceNode(
          site.name,
          site.value,
          [...segments, INSTANCES, site.name],
          'site',
          pointer,
          key,
        ),
      );
    }
    const node: FoldedNode = {
      pointer,
      segments,
      name: one.name,
      kind: 'composition',
      parent: null,
      primitive: null,
      version: null,
      families: stringsOf(member(one.value, FAMILIES)),
      guard: writes(one.value, WHEN) ? member(one.value, WHEN) : null,
      badges: [],
      children,
      ranges: rangesOf(indices),
      count: countOf(points),
      held: heldBy(one.value),
      site: null,
      where: null,
    };
    nodes.push(node);
    keep(node, node.pointer);
    for (const child of children) keep(child, pointer);
  }

  const interfaces = member(tree, INTERFACES);
  for (const [side, kind] of [
    [INPUTS, 'input'],
    [OUTPUTS, 'output'],
  ] as const) {
    for (const one of entriesOf(member(interfaces, side))) {
      const segments: PathSegment[] = [INTERFACES, side, one.name];
      const node: FoldedNode = {
        pointer: pointerOf(segments),
        segments,
        name: one.name,
        kind,
        parent: null,
        primitive: null,
        version: null,
        families: [],
        guard: null,
        badges: badgesOf(one.value),
        children: [],
        ranges: [],
        count: null,
        held: [],
        site: null,
        where: null,
      };
      nodes.push(node);
      keep(node, node.pointer);
    }
  }

  /** The handle one `value_endpoint` hangs on, or `null` where it names nothing drawable. */
  const handleOf = (endpoint: JsonValue | null): FoldedHandle | null => {
    const selector = member(endpoint, INSTANCE);
    const port = nameOf(endpoint, PORT);
    if (port === null || selector === null || !isJsonObject(selector)) return null;
    const instance = nameOf(selector, INSTANCE);
    if (instance === null) return null;
    const composition = nameOf(selector, COMPOSITION);
    // The `kind` says which selector it is; a document off the grammar may write neither, and a
    // `composition` beside the name is then what tells the two apart (the same reading
    // `presentation.json`'s `without: ["composition"]` states for the reference index).
    const rooted = member(selector, KIND) === ROOT || (composition === null && member(selector, KIND) !== GENERATED);
    const segments: PathSegment[] = rooted
      ? [INSTANCES, instance]
      : [COMPOSITIONS, composition ?? '', INSTANCES, instance];
    const site = pointerOf(segments);
    const box = boxOf.get(site) ?? site;
    const indices: FoldedIndexValue[] = rooted
      ? []
      : entriesOf(member(selector, INDICES)).map((one) => ({
          name: one.name,
          written: one.value,
          value: resolve(one.value, quantities),
        }));
    const key = siteKeyOf(rooted, instance, composition ?? '', indices);
    const where = key === null ? null : whereOfSite(key);
    return {
      box,
      site,
      name: instance,
      port,
      indices,
      where,
      value: where === null ? null : `${where}.${port}`,
      boundary: box !== site,
    };
  };

  const edges: FoldedEdge[] = [];
  for (const one of entriesOf(member(member(tree, BINDINGS), VALUES))) {
    const segments: PathSegment[] = [BINDINGS, VALUES, one.name];
    edges.push({
      rule: one.name,
      pointer: pointerOf(segments),
      segments,
      from: handleOf(member(one.value, FROM)),
      to: handleOf(member(one.value, TO)),
      guard: writes(one.value, WHEN) ? member(one.value, WHEN) : null,
      forEach: writes(one.value, FOR_EACH) ? member(one.value, FOR_EACH) : null,
      interface: false,
    });
  }

  // The interfaces' own wires (§4.7: "their edges are the `to`/`from` endpoints"). A public input
  // names several endpoints and each is a wire of its own; a public output names one.
  for (const one of entriesOf(member(interfaces, INPUTS))) {
    const segments: PathSegment[] = [INTERFACES, INPUTS, one.name];
    const terminal = byPointer.get(pointerOf(segments));
    const listed = member(one.value, TO);
    const items = listed !== null && isJsonArray(listed) ? listed : [];
    items.forEach((endpoint, at) => {
      edges.push({
        rule: one.name,
        pointer: pointerOf([...segments, TO, at]),
        segments: [...segments, TO, at],
        from: terminal === undefined ? null : terminalHandle(terminal),
        to: handleOf(endpoint),
        guard: null,
        forEach: null,
        interface: true,
      });
    });
  }
  for (const one of entriesOf(member(interfaces, OUTPUTS))) {
    const segments: PathSegment[] = [INTERFACES, OUTPUTS, one.name];
    const terminal = byPointer.get(pointerOf(segments));
    edges.push({
      rule: one.name,
      pointer: pointerOf([...segments, FROM]),
      segments: [...segments, FROM],
      from: handleOf(member(one.value, FROM)),
      to: terminal === undefined ? null : terminalHandle(terminal),
      guard: null,
      forEach: null,
      interface: true,
    });
  }

  return { nodes, edges, byPointer, quantities };
}

/** The site a handle names, or `null` where an index did not resolve. */
function siteKeyOf(
  rooted: boolean,
  instance: string,
  composition: string,
  indices: readonly FoldedIndexValue[],
): SiteKey | null {
  if (rooted) return rootSite(instance);
  if (indices.some((one) => one.value === null)) return null;
  return generatedSite(
    composition,
    instance,
    indices.map((one) => ({ name: one.name, value: one.value })),
  );
}

/** A terminal's own handle: the box is the terminal, and an interface has no port to name. */
function terminalHandle(node: FoldedNode): FoldedHandle {
  return {
    box: node.pointer,
    site: node.pointer,
    name: node.name,
    port: '',
    indices: [],
    where: null,
    value: null,
    boundary: false,
  };
}

/**
 * The `value_endpoint` a gesture writes — the inverse of `selectSite`.
 *
 * §4.7's connection gesture builds one, and the canvas may not: "what a `value_endpoint` looks
 * like is the schema's" (feature 2.1's `connect`), and the interface's source names no member of
 * the grammar (§1). So it is written here, from the handle the drag ended on.
 *
 * The indices are written as the document writes them. An author who wrote `layers − 1` meant the
 * last layer whatever `layers` is, and an endpoint that wrote back the `31` it resolved to would
 * have changed the document while claiming to keep it — which is why {@link FoldedHandle} carries
 * the written expression beside the value.
 */
export function valueEndpoint(handle: FoldedHandle): JsonObject {
  if (handle.boundary || handle.indices.length > 0) {
    return jsonObject([
      {
        name: INSTANCE,
        value: jsonObject([
          { name: KIND, value: GENERATED },
          { name: COMPOSITION, value: compositionOf(handle) },
          { name: INSTANCE, value: handle.name },
          {
            name: INDICES,
            value: jsonObject(handle.indices.map((one) => ({ name: one.name, value: one.written }))),
          },
        ]),
      },
      { name: PORT, value: handle.port },
    ]);
  }
  return jsonObject([
    {
      name: INSTANCE,
      value: jsonObject([
        { name: KIND, value: ROOT },
        { name: INSTANCE, value: handle.name },
      ]),
    },
    { name: PORT, value: handle.port },
  ]);
}

/** The composition a handle's site belongs to, read back out of its pointer. */
function compositionOf(handle: FoldedHandle): string {
  const segment = handle.site.split('/')[2] ?? '';
  return segment.replace(/~1/g, '/').replace(/~0/g, '~');
}

/**
 * The skeleton a dropped primitive becomes — D5's "placeholders", §4.7's palette drop.
 *
 * > Adding an instance writes `{primitive, arguments: {}, families: [<proposed>]}` — on the
 * > grammar, refused by V2 (missing required arguments), which the Problems panel lists.
 *
 * The family is the *proposal* §9 Q4 decided on ("the primitive name's first segment"), which the
 * author changes freely: a proposal, not a default the language would call silent.
 */
export function instanceSkeleton(primitive: string, version: string, family?: string): JsonObject {
  return jsonObject([
    {
      name: PRIMITIVE,
      value: jsonObject([
        { name: NAME, value: primitive },
        { name: VERSION, value: version },
      ]),
    },
    { name: 'arguments', value: jsonObject([]) },
    { name: FAMILIES, value: [family ?? proposedFamily(primitive)] },
  ]);
}

/** §9 Q4's family proposal: the primitive name's first segment (`attention` for `attention.dense`). */
export function proposedFamily(primitive: string): string {
  const [first] = primitive.split('.');
  return first ?? primitive;
}

/** §9 Q4's name proposal: the primitive name's last segment (`dense` for `attention.dense`). */
export function proposedName(primitive: string): string {
  const parts = primitive.split('.');
  return parts[parts.length - 1] ?? primitive;
}

/** The map a value binding of the folded canvas is written into: the document's own. */
export const VALUE_BINDINGS: readonly PathSegment[] = [BINDINGS, VALUES];

/** The map a root instance is written into. */
export const ROOT_INSTANCES: readonly PathSegment[] = [INSTANCES];

/** The member of a value binding that names the fed end, for `connect`'s replacement rule (V7). */
export const FED_END = TO;

/** The member that names the producing end. */
export const PRODUCING_END = FROM;

/** An index written as a literal — what a boundary handle a gesture makes names. */
export function literalIndex(value: bigint): JsonObject {
  return jsonObject([
    { name: LITERAL, value: { kind: 'number', value: Number(value), real: false, lexeme: value.toString() } },
  ]);
}

/** Every box the canvas draws, a composition's sites included, in the document's order. */
export function foldedBoxes(graph: FoldedGraph): FoldedNode[] {
  const boxes: FoldedNode[] = [];
  for (const node of graph.nodes) {
    boxes.push(node);
    for (const child of node.children) boxes.push(child);
  }
  return boxes;
}

/** Every site the canvas draws a card for, by the identifier `describe`'s `only` filter takes. */
export function foldedSites(graph: FoldedGraph): string[] {
  const wanted = new Set<string>();
  for (const box of foldedBoxes(graph)) if (box.where !== null) wanted.add(box.where);
  return [...wanted];
}

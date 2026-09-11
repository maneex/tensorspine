/**
 * Reading the document against its schema: which subschemas describe a place, and what they
 * require of it.
 *
 * The governing rule (plan §1) is that an item of information a schema can state is never
 * hard-coded, and this module is how the store obeys it. Two readers use it:
 *
 * - the reference index (`./references.ts`), which has to know that `{"quantity": "d"}` names a
 *   quantity *because the schema says the member `quantity` holds a name*, not because someone
 *   wrote the word down;
 * - the delete cascade (`./commands.ts`), which has to know that a public input's `to` may not
 *   be emptied *because the schema says `minItems: 1`*, not because someone remembered it.
 *
 * A place is described by more than one subschema whenever a union is on the way down
 * (`argument_value` is a `scalar_expression` or a `record_argument`; `location` is four shapes),
 * so a {@link Shape} carries a *set* of candidates and a descent takes the union. The union is
 * safe to read tags from because the grammar closes every object
 * (`additionalProperties: false`) and tells its alternatives apart by their required keys: a
 * member the document writes is declared by whichever alternatives admit it, and no two
 * alternatives of these schemas declare one member name with two meanings.
 *
 * The union is **not** safe to read *constraints* from, and the model schema says why in its own
 * root: `anyOf: [{properties: {instances: {minProperties: 1}}}, {properties: {compositions:
 * {minProperties: 1}}}]` — a document needs *one* of the two non-empty, and a reading that took
 * the union would have both, which would make removing the last composition of a document that
 * has instances impossible. So a Shape keeps two readings apart:
 *
 * - `all` is every candidate: the node, its `$ref` chain, its `allOf`, and every alternative
 *   under it — what a descent follows and what says which definition a member *is*;
 * - `direct` is what the place is *by definition*: the same chain with no alternative in it, and
 *   reached from its parent's `direct` alone, so that a constraint written inside an `anyOf`
 *   branch never becomes a constraint of the place it mentions.
 *
 * {@link constraintsOf} reads `direct` and nothing else, and it is the only reader of it.
 */
import {
  isSchemaObject,
  nodeAtPointer,
  parseAnchor,
  pointerSegment,
  type PathSegment,
  type SchemaObject,
  type SchemaRegistry,
} from '@tensorspine/lang';

/** One candidate subschema, and where it is written. */
export interface Place {
  /** The `$id` of the schema the node belongs to. */
  readonly schema: string;
  /** The JSON pointer of the node inside that schema: `#/$defs/identifier`. */
  readonly pointer: string;
  /** `<$id>#<pointer>` — the anchor a presentation binding names (plan §1). */
  readonly anchor: string;
  /** The node itself. */
  readonly node: SchemaObject;
}

/** The subschemas that describe one place of a document. */
export interface Shape {
  /** The node and what it is by definition: its `$ref` chain and its `allOf` members. */
  readonly direct: readonly Place[];
  /** `direct` and every alternative under it: what a descent follows. */
  readonly all: readonly Place[];
}

/** What a container's own schema requires of it, read from `direct` alone. */
export interface Constraints {
  /** The member names the schema requires: one of these cannot be removed. */
  readonly required: ReadonlySet<string>;
  /** The fewest members an object may have. */
  readonly minProperties: number;
  /** The fewest items an array may have. */
  readonly minItems: number;
}

/** The empty shape: what a place the schema describes in no way answers. */
const EMPTY: Shape = { direct: [], all: [] };

/**
 * The schemas as one document-reading context: `$ref` resolution, memoised.
 *
 * A schema is read many times over one document — `scalar_expression` alone is reached at every
 * argument, every extent and every index override — so following a `$ref` and expanding a union
 * are cached by anchor. The cache is per context and a context is built per registry, so a
 * workspace that reloads its schemas gets a new one.
 */
export class SchemaShapes {
  private readonly resolved = new Map<string, Place[]>();
  private readonly expanded = new Map<string, Place[]>();
  /** One `Shape` object per set of candidates, so that a descent can be memoised on identity. */
  private readonly interned = new Map<string, Shape>();
  private readonly descents = new WeakMap<Shape, Map<string, Shape>>();

  constructor(private readonly registry: SchemaRegistry) {}

  /**
   * The shape of a whole document of that role — `model`, `primitive-library-unit`.
   *
   * The root is followed like any other place. It matters: the unit schema reaches its
   * `definition` through the root's `allOf: [{if, then}, …]` — one branch per `kind` — and a
   * reading that took the root node alone left `definition` as `{"type": "object"}`, so no
   * `propertyNames` of the unit schema was reachable and `referenceTags` answered nothing at all
   * (found by feature 2.2, stated as a suite, closed here). `follow` flattens the `$ref` chain
   * and the `allOf` into `direct`; the branches under each of them are alternatives, so the
   * conditional `then` is something a descent sees and `constraintsOf` still does not.
   */
  root(role: string): Shape {
    const schema = this.registry.locate(role);
    if (schema === undefined) return EMPTY;
    return this.shapeOf(this.follow(schema.id, '', schema.document));
  }

  /**
   * The shape of the place an anchor names: `<$id>#<JSON pointer>`, as a presentation binding
   * writes it and as a form is asked for one (`…/model.json#/$defs/instance_definition`).
   *
   * A generated form starts at a `$def` rather than at a document root — the argument sheet at
   * `argument_value`, the location editor at `location`, the primitive editor at
   * `primitive_primitive` — so the same reading has to be reachable from an anchor. Empty when
   * the anchor names nothing, which is what a presentation key that resolves nowhere answers.
   */
  at(anchor: string): Shape {
    const parsed = parseAnchor(anchor);
    if (parsed === null) return EMPTY;
    const document = this.registry.byId(parsed.schema)?.document;
    if (document === undefined) return EMPTY;
    const node = objectAtPointer(document, parsed.pointer);
    if (node === null) return EMPTY;
    return this.shapeOf(this.follow(parsed.schema, parsed.pointer, node));
  }

  /** The shape of the member `name` of an object whose shape is `shape`. */
  member(shape: Shape, name: string): Shape {
    return this.memoised(shape, `p ${name}`, () => this.memberShape(shape, name));
  }

  private memberShape(shape: Shape, name: string): Shape {
    const under = (candidates: readonly Place[]): Place[] => {
      const found: Place[] = [];
      for (const place of candidates) {
        const properties = place.node['properties'];
        if (properties !== null && typeof properties === 'object' && !Array.isArray(properties)) {
          const declared = (properties as Record<string, unknown>)[name];
          if (declared !== undefined) {
            found.push(
              ...this.follow(place.schema, `${place.pointer}/properties/${pointerSegment(name)}`, declared),
            );
            continue;
          }
        }
        const additional = place.node['additionalProperties'];
        if (additional !== undefined && additional !== true && additional !== false) {
          found.push(...this.follow(place.schema, `${place.pointer}/additionalProperties`, additional));
        }
      }
      return found;
    };
    return this.shapeOf(under(shape.direct), under(shape.all));
  }

  /**
   * The shape of the values of a map: its `additionalProperties`.
   *
   * The values of `quantities`, of `instances` and of an `argument_map` are reached this way and
   * no other, so a walk of what a document can hold has to ask for them by name.
   */
  values(shape: Shape): Shape {
    return this.memoised(shape, 'v', () => {
      const under = (candidates: readonly Place[]): Place[] => {
        const found: Place[] = [];
        for (const place of candidates) {
          const additional = place.node['additionalProperties'];
          if (additional !== undefined && additional !== true && additional !== false) {
            found.push(...this.follow(place.schema, `${place.pointer}/additionalProperties`, additional));
          }
        }
        return found;
      };
      return this.shapeOf(under(shape.direct), under(shape.all));
    });
  }

  /** The shape of the item at `index` of an array whose shape is `shape`. */
  item(shape: Shape, index: number): Shape {
    return this.memoised(shape, `i ${String(index)}`, () => this.itemShape(shape, index));
  }

  private itemShape(shape: Shape, index: number): Shape {
    const under = (candidates: readonly Place[]): Place[] => {
      const found: Place[] = [];
      for (const place of candidates) {
        const prefix = place.node['prefixItems'];
        if (Array.isArray(prefix) && index < prefix.length) {
          found.push(
            ...this.follow(place.schema, `${place.pointer}/prefixItems/${String(index)}`, prefix[index]),
          );
          continue;
        }
        const items = place.node['items'];
        if (items !== undefined && items !== true && items !== false) {
          found.push(...this.follow(place.schema, `${place.pointer}/items`, items));
        }
      }
      return found;
    };
    return this.shapeOf(under(shape.direct), under(shape.all));
  }

  /**
   * The member names the schema declares at this place, in the order it writes them.
   *
   * Declaration order is the author's order and the order the interface shows a form in
   * (plan D7: "`doc.priority` is not carried over — declaration order is the library author's
   * order"); here it is also the order a command writes a new object's members in, so that a
   * document the editor makes reads like one the corpus writes. A member a `required` list names
   * comes first, in that list's own order, because that is the order the schema states the
   * object's own shape in.
   */
  propertyOrder(shape: Shape): readonly string[] {
    const found: string[] = [];
    const add = (name: string): void => {
      if (!found.includes(name)) found.push(name);
    };
    for (const place of shape.direct) {
      const required = place.node['required'];
      if (Array.isArray(required)) {
        for (const name of required) if (typeof name === 'string') add(name);
      }
    }
    for (const place of shape.all) {
      const properties = place.node['properties'];
      if (properties !== null && typeof properties === 'object' && !Array.isArray(properties)) {
        for (const name of Object.keys(properties)) add(name);
      }
    }
    return found;
  }

  /** One step down, by a member name or an array index. */
  step(shape: Shape, step: PathSegment): Shape {
    return typeof step === 'number' ? this.item(shape, step) : this.member(shape, step);
  }

  /** The shape of the *names* of a map: its `propertyNames`. Empty where the schema states none. */
  keys(shape: Shape): Shape {
    return this.memoised(shape, 'k', () => this.keyShape(shape));
  }

  private keyShape(shape: Shape): Shape {
    const under = (candidates: readonly Place[]): Place[] => {
      const found: Place[] = [];
      for (const place of candidates) {
        const names = place.node['propertyNames'];
        if (names !== undefined && names !== true && names !== false) {
          found.push(...this.follow(place.schema, `${place.pointer}/propertyNames`, names));
        }
      }
      return found;
    };
    return this.shapeOf(under(shape.direct), under(shape.all));
  }

  /**
   * What the schema requires of the container at this place.
   *
   * Read from `direct` alone, for the reason the module's header states: an alternative's
   * constraint is not the place's.
   */
  constraintsOf(shape: Shape): Constraints {
    const required = new Set<string>();
    let minProperties = 0;
    let minItems = 0;
    for (const place of shape.direct) {
      const names = place.node['required'];
      if (Array.isArray(names)) {
        for (const name of names) if (typeof name === 'string') required.add(name);
      }
      const properties = place.node['minProperties'];
      if (typeof properties === 'number') minProperties = Math.max(minProperties, properties);
      const items = place.node['minItems'];
      if (typeof items === 'number') minItems = Math.max(minItems, items);
    }
    return { required, minProperties, minItems };
  }

  /** The identity of a shape: the anchors of its candidates, which decide every descent from it. */
  keyOfShape(shape: Shape): string {
    return shape.all.map((place) => place.anchor).join(' ');
  }

  private memoised(shape: Shape, step: string, build: () => Shape): Shape {
    let byStep = this.descents.get(shape);
    if (byStep === undefined) {
      byStep = new Map<string, Shape>();
      this.descents.set(shape, byStep);
    }
    const known = byStep.get(step);
    if (known !== undefined) return known;
    const built = build();
    byStep.set(step, built);
    return built;
  }

  private shapeOf(direct: readonly Place[], candidates: readonly Place[] = direct): Shape {
    if (candidates.length === 0) return EMPTY;
    const all: Place[] = [];
    const seen = new Set<string>();
    for (const place of candidates) {
      for (const alternative of this.alternatives(place)) {
        if (seen.has(alternative.anchor)) continue;
        seen.add(alternative.anchor);
        all.push(alternative);
      }
    }
    const identity = `${direct.map((place) => place.anchor).join(' ')} | ${all
      .map((place) => place.anchor)
      .join(' ')}`;
    const known = this.interned.get(identity);
    if (known !== undefined) return known;
    const shape: Shape = { direct, all };
    this.interned.set(identity, shape);
    return shape;
  }

  /** A node reached under a keyword: its `$ref` chain followed and its `allOf` flattened. */
  private follow(schema: string, pointer: string, node: unknown): Place[] {
    if (node === null || typeof node !== 'object' || Array.isArray(node)) return [];
    const place = this.place(schema, pointer, node as SchemaObject);
    const known = this.resolved.get(place.anchor);
    if (known !== undefined) return known;
    const found: Place[] = [];
    this.gather(place, found, new Set<string>());
    this.resolved.set(place.anchor, found);
    return found;
  }

  /** The place itself, what its `$ref` names, and what its `allOf` members name. */
  private gather(place: Place, into: Place[], seen: Set<string>): void {
    if (seen.has(place.anchor)) return;
    seen.add(place.anchor);
    into.push(place);
    const ref = place.node['$ref'];
    if (typeof ref === 'string') {
      const target = this.target(place.schema, ref);
      if (target !== null) this.gather(target, into, seen);
    }
    const all = place.node['allOf'];
    if (Array.isArray(all)) {
      for (const [position, member] of all.entries()) {
        if (member === null || typeof member !== 'object' || Array.isArray(member)) continue;
        this.gather(
          this.place(place.schema, `${place.pointer}/allOf/${String(position)}`, member as SchemaObject),
          into,
          seen,
        );
      }
    }
  }

  /** A place and every alternative below it: `oneOf`, `anyOf` and the two branches of an `if`. */
  private alternatives(place: Place): Place[] {
    const known = this.expanded.get(place.anchor);
    if (known !== undefined) return known;
    const found: Place[] = [];
    this.branch(place, found, new Set<string>());
    this.expanded.set(place.anchor, found);
    return found;
  }

  private branch(place: Place, into: Place[], seen: Set<string>): void {
    if (seen.has(place.anchor)) return;
    seen.add(place.anchor);
    into.push(place);
    for (const keyword of ['oneOf', 'anyOf'] as const) {
      const list = place.node[keyword];
      if (!Array.isArray(list)) continue;
      for (const [position, member] of list.entries()) {
        if (member === null || typeof member !== 'object' || Array.isArray(member)) continue;
        for (const resolved of this.follow(
          place.schema,
          `${place.pointer}/${keyword}/${String(position)}`,
          member,
        )) {
          this.branch(resolved, into, seen);
        }
      }
    }
    // `if`/`then`/`else` is a conditional shape: `quantity_definition` requires `domain` when the
    // source is external (plan §1). Both branches describe places a document can be in, so both
    // are followed for a descent — and neither is read for a constraint, which is what keeps the
    // conditional `required` out of `constraintsOf`.
    for (const keyword of ['then', 'else'] as const) {
      const branch = place.node[keyword];
      if (branch === undefined || branch === true || branch === false) continue;
      for (const resolved of this.follow(place.schema, `${place.pointer}/${keyword}`, branch)) {
        this.branch(resolved, into, seen);
      }
    }
  }

  /** The place a `$ref` names, in this schema or in another of the registry. */
  private target(schema: string, ref: string): Place | null {
    const hash = ref.indexOf('#');
    const id = hash <= 0 ? schema : ref.slice(0, hash);
    const pointer = hash < 0 ? '' : ref.slice(hash + 1);
    const document = id === schema ? this.registry.byId(schema)?.document : this.registry.byId(id)?.document;
    if (document === undefined) return null;
    const node = objectAtPointer(document, pointer);
    return node === null ? null : this.place(id, pointer, node);
  }

  private place(schema: string, pointer: string, node: SchemaObject): Place {
    return { schema, pointer, anchor: `${schema}#${pointer}`, node };
  }
}

/**
 * The object a JSON pointer names inside a schema document, or `null`.
 *
 * The walk is the core's ({@link nodeAtPointer}, feature 2.2's `schema/pointer.ts`): resolving
 * `<$id>#<pointer>` is the registry's business and this module reads the answer. What it adds is
 * the one thing a shape needs of it — that the place be an *object*, since `true` and `false`
 * are schemas that describe no place a form or a cascade can stand at.
 */
function objectAtPointer(document: SchemaObject, pointer: string): SchemaObject | null {
  const node = nodeAtPointer(document, pointer);
  return node !== null && isSchemaObject(node) ? node : null;
}

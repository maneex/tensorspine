/**
 * What a presentation binding says, and what it may not say.
 *
 * The plan's governing rule (§1) is that an item of information a schema can state is never
 * hard-coded in the interface. What a schema *cannot* state — that an `instance_definition` is
 * drawn as a node, a `value_binding` as an edge, that `bytes` is shown in MiB, that `add` prints
 * as `+`, that a `shape` gets the axis-row editor — lives in **one** data file,
 * `packages/ui/src/presentation.json`, keyed by JSON pointers into the schemas.
 *
 * These are the shapes that file is read into. Three things about them are deliberate:
 *
 * - **A member name is closed; a member value is open.** The loader refuses a binding member it
 *   does not know, because a mistyped `widgit` would otherwise do nothing at all and nobody
 *   would learn of it. It does *not* refuse an unknown `widget`, `role` or `format`: plan §1's
 *   "unknown constructs get the generic widget" is the rule for a construct the interface has no
 *   rendering for, and the same answer is the right one for a rendering the interface does not
 *   know yet. `editor/schemas/tensorspine-editor-presentation.schema.json` states the admissible
 *   values and the audit layer holds the shipped file to it, so a typo fails the build without
 *   the startup path having to carry a vocabulary of its own.
 * - **No member here carries a value of the four schemas.** `symbols` is keyed by whatever the
 *   enumeration or the union at the binding's own anchor admits — the audit checks exactly
 *   that — and the keys live in the data file, never in this source, which is what catching
 *   rule §1 (b) asks of every file of `packages/ui/src` but the one.
 * - **A reference rule is stated, not guessed.** Feature 2.1 found that the schemas say *which*
 *   members hold a name and no keyword says what the name refers to: nothing links the member
 *   `quantity` to the map `quantities`, or a generated selector's `instance` to a site of the
 *   composition written beside it. That pairing is a presentation binding, keyed by the same
 *   anchors the store computes, and {@link ReferenceRule} is how it is written down.
 */

/** The symbol an operator, or an alternative of a union, prints as (plan §4.13). */
export interface SymbolBinding {
  /** What is printed: `+`, `div`, `abs`, `and`. */
  readonly text: string;
  /**
   * Where it is printed: between its arguments, before its argument, or as a function name.
   *
   * Not inferable from the arity: `negate` and `absolute` are both unary and print `-a` and
   * `abs(a)`, and `add` and `min` are both n-ary and print `a + b` and `min(a, b)`. An operator
   * with no binding at all prints as `name(args)` (plan §4.13), which is why `min` and `max`
   * carry none.
   */
  readonly form: string;
}

/**
 * How the occurrences of a name are found, once its declaration is known.
 *
 * The store's reference index answers *occurrences* — each with the member it is written under
 * (its `tag`), whether it is a map key or a tagged value, and the names written beside it. A
 * rule says which of those occurrences refer to a declaration of this kind.
 */
export interface ReferenceRule {
  /** The member the name is written under: `quantity`, `site`, `indices`. */
  readonly tag: string;
  /** `key` for a map's own name, `tagged` for a tagged member's; absent, both. */
  readonly kind?: string;
  /**
   * `scope`: only the occurrences at or below the enclosing declaration's place.
   *
   * A site of `decoder` is named `{"site": …}` inside `decoder`'s own bindings and nowhere else,
   * so the rule is scoped by the composition the declaration belongs to.
   */
  readonly under?: string;
  /**
   * The qualifier that must carry the enclosing declaration's *name*.
   *
   * `{"kind": "generated", "composition": "decoder", "instance": "attn_n"}` names a site of
   * `decoder` from outside it: the occurrence's tag is `instance` and the qualifier
   * `composition` says which composition's site it is.
   */
  readonly scopedBy?: string;
  /**
   * The qualifiers that must be *absent*.
   *
   * A root instance is named by an `instance` written with no `composition` beside it, which is
   * how a root selector is told from a generated one without naming either.
   */
  readonly without?: readonly string[];
}

/** One binding: what the schema at this anchor cannot say. */
export interface Binding {
  /** How the construct is drawn on the canvas: a node, a group, an edge, a terminal. */
  readonly role?: string;
  /** Which side of the canvas a terminal sits on. */
  readonly side?: string;
  /** The members shown on a node's face, in the order they are shown. */
  readonly face?: readonly string[];
  /** Whether the node's face carries the summary of its structural arguments. */
  readonly structuralSummary?: boolean;
  /** What labels the construct: `$key` is the name of the map member it is written under. */
  readonly label?: string;
  /** The editor bound to this construct, where the generic walker has no reading of its own. */
  readonly widget?: string;
  /** The referents a picker inside that editor offers, by the tag each is named under. */
  readonly references?: readonly string[];
  /** The list a select at this place is filled from, when it is not an enumeration. */
  readonly picker?: string;
  /** The label of the "New …" action beside that picker. */
  readonly create?: string;
  /** A symbol per value of the enumeration, or per tag of the union, at this anchor. */
  readonly symbols?: ReadonlyMap<string, SymbolBinding>;
  /** How a figure at this place is shown: as a size, a count, a status, a shape. */
  readonly format?: string;
  /** The kind of thing the map at this anchor declares: a quantity, a site, an index. */
  readonly declares?: string;
  /** The anchor of the declaration this one lives inside, when its references are scoped. */
  readonly scope?: string;
  /** Where a name declared here is referred to. */
  readonly refers?: readonly ReferenceRule[];
}

/** The bindings, by the anchor each one is keyed by. */
export interface Presentation {
  /** Every anchor a binding is written for, in the order the file writes them. */
  readonly anchors: readonly string[];
  /** The binding at that anchor, or `undefined`. */
  at(anchor: string): Binding | undefined;
  /**
   * The binding of the first of those anchors that has one.
   *
   * A place of a document is described by a chain of schema nodes — the place itself, then what
   * its `$ref` names — and the store answers that chain most specific first. So a binding
   * written for a definition (`#/$defs/count`) serves every place that refers to it, and one
   * written for a place (`#/$defs/composition_definition/properties/indices`) wins over it
   * there.
   */
  firstOf(anchors: readonly string[]): Binding | undefined;
}

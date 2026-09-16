/**
 * The binding gestures of §4.11 and §4.14, as commands of the store (2.1, D13).
 *
 * | Gesture | Effect (a JSON edit) |
 * |---|---|
 * | **Bind privately** | the slot leaves the identity it is in and gets one of its own, `<site>.<slot>` |
 * | **Tie to…** / **Share with…** | the slot joins the identity a rule declares, leaving the one it was in |
 * | a slot chip dragged onto another (§4.7) | the same, with the identity named by the chip it was dropped on |
 *
 * **One gesture, one command** (D13): a slot *moves*, so the member it leaves and the member it
 * joins are written by one command with one undo — which is also how `check` reads the candidate
 * ("a member candidate is a slot **joining** an identity, not a second binding of it", feature
 * 1.6d), so the verdict shown during the gesture is the verdict of what is made.
 *
 * **Nothing here writes a member of the grammar.** What a parameter or state endpoint looks like
 * is the schema's, and the core writes it (`identityMember`, `identitySymbol`) — feature 2.9's
 * rule for `value_endpoint`, one binding along. What is decided here is what §4.11 decides: the
 * *name* a private identity is proposed under (`<site>.<slot>`, uniquified), and where the rule is
 * written — inside the composition the site belongs to, which is what makes it one tensor per
 * iteration (§5.2 rule 7), and at the top level for a root instance.
 *
 * **Nothing is refused** (Q5): a tie V15 will refuse is made, and the refusal lands in Problems.
 */
import {
  identityMember,
  identitySymbol,
  jsonObject,
  MEMBERS,
  PARAMETER_BINDINGS,
  STATE_BINDINGS,
  STATE_SYMBOL,
  TENSOR_SYMBOL,
  type JsonValue,
  type SiteKey,
} from '@tensorspine/lang';
import {
  addToMap,
  insertItem,
  nodeAt,
  objectAt,
  remove,
  setMemberAt,
  shapeAt,
  unique,
  type Command,
  type EditContext,
  type Path,
} from '@tensorspine/store';

/** Which slot a gesture is about: the site of the expanded graph, and the slot or port it names. */
export interface SlotTarget {
  readonly site: SiteKey;
  readonly slot: string;
  /** Whether it is a state port (D4's identities) rather than a parameter slot (D3's). */
  readonly state: boolean;
}

/** Where a member sits today: the rule that holds it, and its position in the written list. */
export interface HeldMember {
  /** The place the rule is written at, as a path of the document (`identityReadings`' pointer). */
  readonly rule: Path;
  /** Its position in that rule's `members`; `null` where the core could not name one. */
  readonly at: number | null;
}

/** What a binding gesture is given. */
export interface BindRequest {
  readonly target: SlotTarget;
  /** Where the slot is bound today, and `null` where nothing binds it (V7's unbound chip). */
  readonly held?: HeldMember | null;
  /** The rule the slot joins — "Tie to…", "Share with…", a chip dropped on another. */
  readonly into?: Path;
  /** The name proposed for a new private identity; `<site>.<slot>` otherwise. */
  readonly name?: string;
  /**
   * A second slot that joins the new identity at once — §4.7's chip dropped on an **unbound** chip.
   *
   * "Creates or extends the identity": where the chip dropped on belongs to no identity, the one
   * created holds both, which is one command and one undo. A valid document has no such chip (V7
   * binds every present slot exactly once), so this is the answer to a document already carrying
   * that refusal, not a second way to tie.
   */
  readonly joining?: { readonly target: SlotTarget; readonly held?: HeldMember | null };
}

/**
 * "Bind privately" — §4.11: *creates `<site>.<slot>`*.
 *
 * The rule is written where the site lives: inside the composition for a site of one, at the top
 * level for a root instance. A scoped rule needs no symbol (§5.2 rule 7 names the identity
 * `<composition>.<rule>`), and a top-level one requires it, which is what the grammar says and
 * what {@link identitySymbol} writes.
 */
export function bindPrivately(edit: EditContext, request: BindRequest): Command & { readonly name: string } {
  const { target } = request;
  const place = bindingsPlace(target);
  const proposed = request.name ?? `${target.site.name}.${target.slot}`;
  const name = unique(proposed, namesUnder(edit, place.path));
  const member = identityMember(target.site, target.slot, {
    state: target.state,
    scope: place.scope,
  });
  const joining = request.joining;
  const members = [member];
  if (joining !== undefined) {
    members.push(
      identityMember(joining.target.site, joining.target.slot, {
        state: joining.target.state,
        scope: place.scope,
      }),
    );
  }
  const values: Record<string, JsonValue> = { [MEMBERS]: members };
  // A top-level rule declares the symbol its identity is named by; a scoped one may leave it out,
  // and the corpus's own scoped rules do (§5.2 rule 7).
  if (place.scope === '') {
    values[target.state ? STATE_SYMBOL : TENSOR_SYMBOL] = identitySymbol(name);
  }
  const label = `Bind ${target.site.name}.${target.slot} privately`;
  const added =
    objectAt(edit.tree, place.path) === undefined
      ? firstRule(edit, place.path, name, values, label)
      : addToMap(edit, { path: place.path, name, values, label });
  // The new rule first and the old members after, for {@link sequence}'s reason: both places are
  // computed against the tree as it is, and a map takes its new entry by name whatever else goes.
  const left = [leaving(edit, request), joining === undefined ? null : leaving(edit, joining)];
  return { ...sequence(label, [added, ...left]), name };
}

/**
 * "Tie to…" / "Share with…" — the slot joins the identity a rule already declares.
 *
 * The member is appended to that rule's own list, in the form the *place it is written in* asks
 * for: a rule scoped to the site's own composition names the site alone, and any other names it
 * with a selector at the point the chip stands for (`identityMember`).
 */
export function tieTo(edit: EditContext, request: BindRequest & { readonly into: Path }): Command {
  const { target, into } = request;
  const scope = scopeOf(into);
  const member = identityMember(target.site, target.slot, { state: target.state, scope });
  const members = [...into, MEMBERS] as Path;
  const list = nodeAt(edit.tree, members);
  const label = `${target.state ? 'Share' : 'Tie'} ${target.site.name}.${target.slot} into ${String(
    into[into.length - 1],
  )}`;
  const joining =
    list === undefined
      ? setMemberAt(edit, { path: into, name: MEMBERS, value: [member], label })
      : insertItem(edit, { path: members, value: member, label });
  return sequence(label, [joining, leaving(edit, request)]);
}

/**
 * What the slot leaves: the member it is written as today, or nothing where it is unbound.
 *
 * The removal is the store's own `remove`, so the collapse rule applies: a `members` list left
 * below the `minItems: 1` the grammar declares takes its rule with it, which is exactly what an
 * identity whose last member left should do — and the confirmation the cascade carries says so.
 */
function leaving(
  edit: EditContext,
  request: { readonly target: SlotTarget; readonly held?: HeldMember | null },
): Command | null {
  const held = request.held ?? null;
  if (held === null || held.at === null) return null;
  const at = [...held.rule, MEMBERS, held.at] as Path;
  if (nodeAt(edit.tree, at) === undefined) return null;
  return remove(edit, { path: at, label: `Unbind ${request.target.slot}` });
}

/**
 * Two edits as one gesture: the join first, the leave after.
 *
 * Their places are computed against the same tree, so the order matters — the removal may take a
 * whole rule with it, and a list that lost an item moves what follows. The join is therefore
 * applied first and is never inside what the leave removes: a slot does not leave the identity it
 * is joining (the compatibility list never offers a slot its own identity, feature 1.6d).
 */
function sequence(label: string, commands: readonly (Command | null)[]): Command {
  const made = commands.filter((one): one is Command => one !== null);
  const cascade = made.find((one) => one.cascade !== undefined)?.cascade;
  return {
    label,
    moves: made.flatMap((one) => one.moves),
    ...(cascade === undefined ? {} : { cascade }),
    edit(draft) {
      for (const one of made) one.edit(draft);
    },
  };
}

/**
 * Where a new rule for this site goes, and the composition it is scoped to.
 *
 * A site of a composition is bound **inside** it whether or not the composition already has a map
 * of that kind: that is what makes the rule one tensor per iteration (§5.2 rule 7), and a rule
 * written at the top level instead would name one iteration's slot and leave the other thirty-one
 * unbound (V7). The map is created where it is missing ({@link firstRule}), exactly as a scoped
 * value rule's is — feature 2.14 does the same one map along. Until feature 2.19 built a document
 * from nothing, every composition of the corpus already had both maps and the difference could not
 * be seen.
 */
export function bindingsPlace(target: SlotTarget): { readonly path: Path; readonly scope: string } {
  const map = target.state ? STATE_BINDINGS : PARAMETER_BINDINGS;
  if (target.site.kind === 'gen') {
    return {
      path: ['compositions', target.site.composition, BINDINGS, map] as Path,
      scope: target.site.composition,
    };
  }
  return { path: [BINDINGS, map] as Path, scope: '' };
}

/**
 * The first rule of a map that is not there yet: the map, and everything above it, created holding
 * it.
 *
 * The grammar makes a composition's `bindings` optional and each of its four maps optional in
 * turn, so a composition may be missing one, the other, or both. What is written is the chain from
 * the deepest place that exists down to the rule, with the rule's own members in the schema's
 * property order — which is what {@link addToMap} does where the map is there.
 */
function firstRule(
  edit: EditContext,
  path: Path,
  name: string,
  values: Readonly<Record<string, JsonValue>>,
  label: string,
): Command {
  let at = path.length;
  while (at > 0 && objectAt(edit.tree, path.slice(0, at)) === undefined) at -= 1;
  const shape = edit.shapes.values(shapeAt(edit, path));
  const order = edit.shapes.propertyOrder(shape);
  const members = Object.keys(values).sort(
    (left, right) => positionOf(order, left) - positionOf(order, right),
  );
  let value: JsonValue = jsonObject([
    { name, value: jsonObject(members.map((member) => ({ name: member, value: values[member] as JsonValue }))) },
  ]);
  for (let depth = path.length - 1; depth > at; depth -= 1) {
    value = jsonObject([{ name: path[depth] as string, value }]);
  }
  return setMemberAt(edit, {
    path: path.slice(0, at),
    name: path[at] as string,
    value,
    label,
  });
}

/** Where a member stands in the schema's own order; the ones it does not name come last. */
function positionOf(order: readonly string[], name: string): number {
  const at = order.indexOf(name);
  return at < 0 ? order.length : at;
}

/**
 * Whether a rule is written inside a composition, and which one.
 *
 * Read off the *place*, which is where a scope is (feature 2.2's rule: "the scope is the caller's
 * to supply … because the document is where a scope is"). A rule at
 * `/compositions/decoder/bindings/parameters/attn.q` is `decoder`'s; one at
 * `/bindings/parameters/embed.weight` is nobody's.
 */
export function scopeOf(rule: Path): string {
  const composition = rule.length >= 5 && rule[0] === 'compositions' ? rule[1] : undefined;
  return typeof composition === 'string' ? composition : '';
}

/** The names a map already carries. */
function namesUnder(edit: EditContext, path: Path): string[] {
  const map = objectAt(edit.tree, path);
  return map === undefined ? [] : map.members.map((member) => member.name);
}

/** The member of a document the binding rules are written under. */
const BINDINGS = 'bindings';

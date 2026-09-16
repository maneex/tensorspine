/**
 * The catalog a primitive is chosen from — plan §4.6's palette list, feature 2.21.
 *
 * Three places choose a primitive, and §4.6 already said they are one choice: the Identity row of
 * §4.11, the drop of §4.7 and `Model ▸ Add Instance…`. So the list they offer is **one** list,
 * answered here, and every one of them reads it through this module. Nothing below knows an
 * identity of the library: `library.byId` is the catalog and the core projects it
 * ({@link PrimitiveIdentity}), which is what catching rule §1 (b) requires — `attention.dense` is
 * data a test types, never a literal a component holds.
 *
 * **One field over two members.** `primitive_reference` has a `name` and a `version`, both
 * required; the library's own form for the pair is `identityKey`'s `name@version`, and its note
 * says `@` occurs in neither half — which is exactly what makes one field over the two total
 * rather than a convention. {@link readIdentity} is the read, {@link identityKey} and
 * `splitIdentity` are the core's.
 *
 * **Nothing is refused** (§9 Q5). A name the catalog lacks is not a verdict of the language — V1
 * says `primitive absent from primitive library` and it says it about a document, not about a
 * keystroke — so the answer here is a *verdict about what the author meant*
 * ({@link IdentityVerdict}) and the caller asks. A suggestion is a suggestion: the list never
 * limits what may be written.
 */
import { identityKey, splitIdentity, type PrimitiveIdentity } from '@tensorspine/lang';

export { identityKey, splitIdentity, type PrimitiveIdentity };

/**
 * The pickers `presentation.json` names, each beside the list it is filled from.
 *
 * A picker names a list the **editor** fills — which is why it is the presentation file's word
 * and not a schema's (§1) — and the names are written here, once, next to the answers, the way
 * `tokens.ts` writes the widget it draws. `tests/audit/presentation.test.ts` holds the shipped
 * file to this set, so a picker named in the file and filled by nobody is a failed build rather
 * than an empty list.
 */
export const PICKER = {
  /** Every identity the gathered bases carry — this feature's own. */
  primitives: 'primitives',
  /** The versions the library carries of the primitive an instance pins (feature 2.10). */
  versions: 'primitive-versions',
  /** The family names the open document already writes (feature 2.10). */
  families: 'families',
  /** The axes a base declares — the primitive editor's, feature 3.4. */
  axes: 'axes',
  /** The precision roles a base declares — the primitive editor's, feature 3.4. */
  roles: 'precision-roles',
} as const;

/**
 * The editor bound at `#/$defs/primitive_reference`: one field over the reference's two members.
 *
 * Named here and nowhere else in the interface, beside the editor it names — `tokens.ts`'s own
 * pattern, and for its reason: it is no word of the four schemas but the presentation file's, and
 * the audit holds that file to a schema listing it.
 */
export const PRIMITIVE_IDENTITY = 'primitive-identity';

/** Whether a widget is that editor. */
export function editsIdentity(widget: string | undefined): boolean {
  return widget === PRIMITIVE_IDENTITY;
}

/** What a chooser offers, as this module reads it: the catalog, and one name's versions. */
export interface IdentityOffers {
  /** Every identity of the gathered bases, in the loader's own order (the `primitives` picker). */
  readonly identities: readonly PrimitiveIdentity[];
  /** The versions a name carries (the `primitive-versions` picker, feature 2.10's list). */
  readonly versions: (name: string) => readonly string[];
}

/** The offers of a catalog alone, where no caller has the version list beside it. */
export function offersOf(identities: readonly PrimitiveIdentity[]): IdentityOffers {
  return {
    identities,
    versions: (name) => identities.filter((one) => one.name === name).map((one) => one.version),
  };
}

/**
 * What the suggest list holds for a text being typed.
 *
 * Two readings, because a `name@version` field is two questions one after another:
 *
 *  - the text carries an `@`, so the **name half is settled** and what is still open is the
 *    version: the offers are that name's versions, which is the list feature 2.10 already fills
 *    (`primitive-versions`), joined back into identities;
 *  - it does not, so the name is what is being typed: the identities whose text begins with it
 *    first, those that merely carry it after, each in the catalog's own order.
 *
 * The empty text offers the whole catalog, which is what a reader who has typed nothing wants.
 */
export function suggestIdentities(offers: IdentityOffers, typed: string): string[] {
  const text = typed.trim();
  const at = text.indexOf('@');
  if (at >= 0) {
    const name = text.slice(0, at);
    const offered = offers.versions(name).map((version) => identityKey(name, version));
    if (offered.length > 0) return offered;
  }
  const ids = offers.identities.map((one) => one.id);
  if (text === '') return ids;
  const lower = text.toLowerCase();
  const begins = ids.filter((id) => id.toLowerCase().startsWith(lower));
  const carries = ids.filter(
    (id) => !id.toLowerCase().startsWith(lower) && id.toLowerCase().includes(lower),
  );
  return [...begins, ...carries];
}

/** A primitive reference as the grammar writes it: the two members of `primitive_reference`. */
export interface Reference {
  readonly name: string;
  readonly version: string;
}

/**
 * What one field over two members reads: `name@version`, the library's own form.
 *
 * A reference with neither half written reads as nothing at all rather than as `@`: a field the
 * author has not filled in shows them an empty field, and `@` alone is no identity of anything.
 */
export function identityOf(held: Reference): string {
  if (held.name === '' && held.version === '') return '';
  return identityKey(held.name, held.version);
}

/**
 * What the author meant by a text, once they have finished typing it.
 *
 * `known` is an identity of the catalog and is written without a word: the two members are the
 * catalog's own spelling, so choosing one writes what the corpus writes. `unknown` is not a
 * refusal — Q5 forbids one — but a **question**: a typo of `norm.rms`, or a primitive that does
 * not exist yet, and the caller is what asks (feature 2.21's confirm).
 *
 * A text with no `@` names the primitive alone and **keeps the version the document holds**: the
 * two halves are edited in one field and a reader who retypes the name has not said anything
 * about the version. A text with no name half at all (`@1.0.0`, the empty string) is no identity
 * and reverts, which is the field's own Escape.
 */
export type IdentityVerdict =
  | { readonly kind: 'known'; readonly reference: Reference }
  | {
      readonly kind: 'unknown';
      readonly reference: Reference;
      /** The nearest identity of the catalog, where one is near enough to be a typo of it. */
      readonly nearest: string | null;
    }
  | { readonly kind: 'nothing' };

export function readIdentity(
  offers: IdentityOffers,
  typed: string,
  held: Reference,
): IdentityVerdict {
  const text = typed.trim();
  if (text === '') return { kind: 'nothing' };
  const split = splitIdentity(text);
  // A name alone keeps the version beside it; anything with an `@` says both halves itself.
  const reference: Reference | null =
    split !== null
      ? split
      : text.includes('@')
        ? null
        : { name: text, version: held.version };
  if (reference === null) return { kind: 'nothing' };
  const id = identityOf(reference);
  if (offers.identities.some((one) => one.id === id)) return { kind: 'known', reference };
  return { kind: 'unknown', reference, nearest: nearestIdentity(offers, reference) };
}

/**
 * The identity a drag carries — §4.7's "drop a primitive from the palette", read off the catalog.
 *
 * The palette that will carry it is feature 3.1's, and what it carries is a *name*: which version
 * of that name the drop pins is a fact about the **library**, so it is read here rather than sent
 * with the drag. A drag that names a version keeps it, an identity the catalog does not carry
 * still lands (Q5, and V1 is what reports it), and a name the catalog carries at one version gets
 * that version without anybody typing it — which is what "no instance needs a name typed by hand"
 * asks of the drop.
 */
export function carriedIdentity(
  offers: IdentityOffers,
  carried: { readonly primitive: string; readonly version?: string },
): Reference {
  const versions = offers.versions(carried.primitive);
  if (carried.version !== undefined && carried.version !== '') {
    return { name: carried.primitive, version: carried.version };
  }
  const pinned = versions[versions.length - 1];
  return { name: carried.primitive, version: pinned ?? '' };
}

/**
 * The identity a mistyped one is nearest to, or `null` where nothing is near enough.
 *
 * The comparison is over the **name** half alone, because the version half is chosen from a list
 * and the name is what is typed; and it is bounded, because a nearest match with nothing near is
 * noise offered as help — `zzz` is not a typo of `moe`. The bound is a quarter of what was
 * typed, at least one: `attention.dens` reaches `attention.dense`, `zzz` reaches nothing.
 *
 * Levenshtein, iteratively over one row, because the catalog is tens of names long and the field
 * asks once per commit — measured in the suite beside it.
 */
export function nearestIdentity(offers: IdentityOffers, wanted: Reference): string | null {
  const bound = Math.max(1, Math.floor(wanted.name.length / 4));
  let best: { id: string; distance: number } | null = null;
  const names = new Set(offers.identities.map((one) => one.name));
  for (const name of names) {
    const distance = editDistance(wanted.name, name, bound);
    if (distance > bound) continue;
    if (best !== null && distance >= best.distance) continue;
    const versions = offers.versions(name);
    const version = versions.includes(wanted.version) ? wanted.version : versions[0];
    if (version === undefined) continue;
    best = { id: identityKey(name, version), distance };
  }
  return best?.id ?? null;
}

/** The edit distance of two words, stopped as soon as it is past `bound`. */
function editDistance(left: string, right: string, bound: number): number {
  if (Math.abs(left.length - right.length) > bound) return bound + 1;
  let row = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let i = 1; i <= left.length; i += 1) {
    const next = [i];
    let least = i;
    for (let j = 1; j <= right.length; j += 1) {
      const substitute = (row[j - 1] ?? 0) + (left[i - 1] === right[j - 1] ? 0 : 1);
      const remove = (row[j] ?? 0) + 1;
      const insert = (next[j - 1] ?? 0) + 1;
      const best = Math.min(substitute, remove, insert);
      next.push(best);
      least = Math.min(least, best);
    }
    if (least > bound) return bound + 1;
    row = next;
  }
  return row[right.length] ?? bound + 1;
}

/**
 * §5.2 rule 7 read backwards: where a hoisted place was **written**.
 *
 * `normalise` expands every composition-scoped binding into exactly one top-level rule named
 * `<composition>.<rule>` before any other rule applies, so the validator, D1, the viewer and the
 * linter all read one form. Every place those stages name is therefore a place of the *normalised*
 * document — and `[V5] decoder.attn.norm_in` carries `/bindings/values/decoder.attn.norm_in`,
 * which the file on disk does not have: what it has is
 * `/compositions/decoder/bindings/values/attn.norm_in`.
 *
 * That was feature 2.7's finding, left for the feature where navigation to a problem is decided.
 * This is the answer: while it hoists, `normalise` records where each place it writes came from,
 * and {@link writtenPlace} reads that record backwards. **It is not a second reading of the rule**
 * — the record is a by-product of the hoist itself, written member by member as the hoist writes
 * them, so a change to the expansion moves the map with it or fails to compile.
 *
 * **Why the name cannot be split instead.** A binding rule is keyed by a `qualified_name`, and
 * the corpus writes dotted rule names everywhere (`attn.carry` inside `enc`, `enc.entry.a` at the
 * top level of `colbert-v2`), so `enc.entry.a` is `enc` ▸ `entry.a` or a top-level rule of that
 * name and nothing in the text says which. `normalise` refuses the collision
 * (`binding '…' is declared both in composition '…' and at the top level`), which makes the two
 * exclusive but does not make either recoverable from the name. The map is built from the
 * document's own compositions, so there is nothing to guess.
 *
 * **Three kinds of place, because the hoist does three things.** Some members are *copied* — a
 * guard, a port name, a declared `tensor`, the composition's own `indices` — and there the whole
 * subtree corresponds, so a longer pointer keeps its tail. Some are *rebuilt*: the rule, an
 * endpoint, a member of an identity. They denote what the author wrote, member for member they do
 * not, so the place itself answers exactly and a pointer below it falls back to it. And some the
 * hoist *invents*: the generated instance selector, an index selecting itself, an identity §5.2
 * rule 7 names `C.R` for a rule that declared none. Those have no written counterpart at all, and
 * the answer is the nearest place that does, with {@link WrittenPlace.exact} saying so rather than
 * naming a place the document has not got.
 */
import { pointerOf, type PathSegment } from '../schema/types.js';

/** What the hoist did at a place. */
export type HoistKind =
  /** The written subtree, copied: a longer pointer keeps its tail. */
  | 'copied'
  /** Rebuilt from the written place: the place answers, a longer pointer falls back to it. */
  | 'rebuilt'
  /** Invented: the written place is the nearest one, and no pointer here is exact. */
  | 'invented';

/** One place of the normalised document, and where it came from. */
export interface HoistedPlace {
  /** Where it was written, as an RFC 6901 pointer into the document as the author wrote it. */
  readonly path: string;
  readonly kind: HoistKind;
}

/** What {@link HoistRecorder} collected, keyed by the place in the normalised document. */
export interface Hoisting {
  readonly places: ReadonlyMap<string, HoistedPlace>;
}

/** A hoisting that expanded nothing: a document with no composition-scoped binding. */
export const NO_HOISTING: Hoisting = { places: new Map() };

/** Whether a hoisting expanded anything at all. */
export function hoisted(hoisting: Hoisting): boolean {
  return hoisting.places.size > 0;
}

/** Where a place of the normalised document was written. */
export interface WrittenPlace {
  /** The place in the document as the author wrote it, as an RFC 6901 pointer. */
  readonly path: string;
  /**
   * Whether the pointer itself has that place, rather than falling back to the nearest one above.
   *
   * `false` where the tail named something the hoist built or invented — the `kind` of a generated
   * selector, the `name` of an implicit identity — and the answer is the nearest written place.
   */
  readonly exact: boolean;
}

/**
 * Where a place of the normalised document was written, or `null` where it was written as it
 * stands.
 *
 * `null` is the ordinary answer: a pointer into `/quantities`, `/instances` or a top-level binding
 * names the same place in both readings, and a caller that finds nothing here has nothing to
 * rewrite.
 */
export function writtenPlace(hoisting: Hoisting, pointer: string): WrittenPlace | null {
  const found = longestPrefix(hoisting.places, pointer);
  if (found === null) return null;
  const tail = pointer.slice(found.key.length);
  if (found.place.kind === 'copied') return { path: `${found.place.path}${tail}`, exact: true };
  if (found.place.kind === 'rebuilt') return { path: found.place.path, exact: tail === '' };
  return { path: found.place.path, exact: false };
}

/** The longest key of a map that is the pointer or a prefix of it ending at a segment boundary. */
function longestPrefix(
  places: ReadonlyMap<string, HoistedPlace>,
  pointer: string,
): { key: string; place: HoistedPlace } | null {
  let found: { key: string; place: HoistedPlace } | null = null;
  for (const [key, place] of places) {
    if (pointer !== key && !pointer.startsWith(`${key}/`)) continue;
    if (found !== null && found.key.length >= key.length) continue;
    found = { key, place };
  }
  return found;
}

/**
 * The record a hoist writes as it goes.
 *
 * It is handed to `normalise` and filled there; nothing here decides anything about the
 * expansion, which is why the two cannot disagree.
 */
export class HoistRecorder {
  private readonly held = new Map<string, HoistedPlace>();

  /** A place whose subtree is the written one. */
  copy(at: readonly PathSegment[], from: readonly PathSegment[]): void {
    this.put(at, from, 'copied');
  }

  /** A place the hoist rebuilt out of the written one: the two denote the same thing. */
  rebuild(at: readonly PathSegment[], from: readonly PathSegment[]): void {
    this.put(at, from, 'rebuilt');
  }

  /** A place the hoist invented, and the nearest written place it was invented from. */
  invent(at: readonly PathSegment[], from: readonly PathSegment[]): void {
    this.put(at, from, 'invented');
  }

  /** What was collected. */
  hoisting(): Hoisting {
    return { places: this.held };
  }

  private put(at: readonly PathSegment[], from: readonly PathSegment[], kind: HoistKind): void {
    this.held.set(pointerOf(at), { path: pointerOf(from), kind });
  }
}

/**
 * The identities a document declares, read as the editor's sheets need them (plan §4.11).
 *
 * > **Identity** (parameter / constant / state) | Name (`tensor` / `identity` symbol with indices)
 * > · members … · Derived: the D3 tensor rows or D4 state rows of the identity's instances
 * > (`decoder.attn.q[layer=3]`, …), totals
 *
 * A sheet knows the **place** the author selected — `/bindings/parameters/embed.weight`, or
 * `/compositions/decoder/bindings/parameters/attn.q` — and the products name their rows by the
 * **identity instance** (`decoder.attn.q[layer=0]`). Nothing in the interface may join the two:
 * the name a scoped rule's identity takes is §5.2 rule 7's own (“a scoped parameter or state rule
 * without a declared `tensor` / `identity` names it `C.R`, indexed by the composition's indices”),
 * and a second reading of that rule in a component is exactly what plan §1 forbids and what the
 * ledger's own findings keep catching.
 *
 * So the core answers it, the way feature 2.7's {@link QuantityReading} answers where a quantity
 * is declared and what it resolves to: **one reading per rule, keyed by the pointer of the place
 * the author wrote**, carrying the identity the products will name. It is a by-product of the
 * hoist and not a copy of it — `normalise` is run for its own record (feature 2.8's
 * {@link HoistRecorder}), so a change to the expansion moves this with it or fails to compile.
 */
import { entries, get } from '../library/access.js';
import { pyStr } from '../library/repr.js';
import { normalise, HoistRecorder, writtenPlace, type Hoisting } from '../model/index.js';
import { pointerSegment } from '../schema/pointer.js';
import type { PyValue } from '../expr/value.js';

/**
 * One identity a binding rule declares.
 *
 * The order is the normalised document's: the top-level rules as they are written, then each
 * composition's, which is the order §5.2 rule 7 hoists them in.
 */
export interface IdentityReading {
  /** The rule as the normalised document names it: `embed.weight`, `decoder.attn.q`. */
  readonly rule: string;
  /**
   * Where the rule is **written**, as an RFC 6901 pointer into the document as authored.
   *
   * `/bindings/parameters/embed.weight` for a top-level rule,
   * `/compositions/decoder/bindings/parameters/attn.q` for a scoped one — which is the place a
   * selection, the explorer's outline and the canvas all carry.
   */
  readonly pointer: string;
  /** The map of `bindings` the rule is written in: the word that says what kind of identity it is. */
  readonly kind: string;
  /**
   * The identity the products name their instances after, without indices.
   *
   * The symbol the rule declares (`tensor.name`, `identity.name`), the constant it names, or the
   * name §5.2 rule 7 gives a scoped rule that declares none (`<composition>.<rule>`).
   */
  readonly identity: string;
  /** The identity's own index names, in the order they are written. */
  readonly indices: readonly string[];
  /** Whether D4 carries its instances (a state identity) rather than D3. */
  readonly state: boolean;
}

/** The member of a document the binding rules are written under, as the core reads them. */
export const BINDINGS = 'bindings';

/**
 * The maps of `bindings` whose rules declare an identity, with the member each names it by.
 *
 * `values` is not one: a value rule is an edge and its name is a label (feature 2.2's own reading
 * of why nothing refers to it). The three below are what D3 and D4 are built from, and the member
 * each carries is the one `normalise` writes the identity under.
 */
const IDENTITY_MAPS: readonly { readonly map: string; readonly symbol: string; readonly state: boolean }[] = [
  { map: 'parameters', symbol: 'tensor', state: false },
  { map: 'constants', symbol: 'constant', state: false },
  { map: 'states', symbol: 'identity', state: true },
];

/** Every identity a document declares, read as {@link IdentityReading} describes. */
export function identityReadings(model: PyValue): IdentityReading[] {
  const recorder = new HoistRecorder();
  const normalised = normalise(model, recorder);
  const hoisting = recorder.hoisting();
  const readings: IdentityReading[] = [];
  const bindings = get(normalised, BINDINGS);
  for (const { map, symbol, state } of IDENTITY_MAPS) {
    const rules = get(bindings, map);
    if (rules === null) continue;
    for (const [rule, declaration] of entries(rules)) {
      const at = `/${BINDINGS}/${pointerSegment(map)}/${pointerSegment(rule)}`;
      readings.push({
        rule,
        pointer: writtenAt(hoisting, at),
        kind: map,
        identity: identityName(get(declaration, symbol)),
        indices: indexNames(get(declaration, symbol)),
        state,
      });
    }
  }
  return readings;
}

/** The reading of the place a selection names, or `undefined` where it names no identity. */
export function identityAt(
  readings: readonly IdentityReading[],
  pointer: string,
): IdentityReading | undefined {
  return readings.find((one) => one.pointer === pointer);
}

/**
 * Where a place of the normalised document was written.
 *
 * A rule the author wrote at the top level is its own answer — the hoist has no record of a place
 * it did not move — and a scoped one is the rule `normalise` rebuilt, whose record points at the
 * place inside the composition.
 */
function writtenAt(hoisting: Hoisting, pointer: string): string {
  return writtenPlace(hoisting, pointer)?.path ?? pointer;
}

/**
 * The name the identity's instances carry, from the symbol the rule declares.
 *
 * A parameter and a state rule declare a `symbol_instance` — a name and, where the identity has
 * several instances, the indices that tell them apart; a constant rule names the constant itself,
 * which is a name and nothing else. Both are read here rather than in two places.
 */
function identityName(symbol: PyValue): string {
  if (typeof symbol === 'string') return symbol;
  const name = get(symbol, 'name');
  return name === null ? '' : pyStr(name);
}

/** The index names a symbol instance is indexed by, in the order the document writes them. */
function indexNames(symbol: PyValue): string[] {
  if (typeof symbol === 'string') return [];
  const indices = get(symbol, 'indices');
  return indices === null ? [] : entries(indices).map(([name]) => name);
}

/**
 * The rows the editor produces itself — §4.17's `editor` source, at the `notice` severity.
 *
 * | Source | Severity | Produced by |
 * |---|---|---|
 * | editor | notice | a binding naming an absent slot after a structural change (with a fix), an unused quantity, a sidecar key dropped, a presentation binding rendered generically, a schema mismatch, a primitive without a witness in the loaded manifest |
 *
 * These are the **only** rows of the panel the core does not answer — "the panel never invents a
 * problem: every row but the editor's own notices is a verdict of the core" — and the type they
 * arrive as has said so since feature 1.1: `editor` and `notice` are declared in the core's own
 * `Problem` with the comment that this feature is what produces them.
 *
 * **Three of the six are here, and the reasons the other three are not.**
 *
 *  - *a binding naming an absent slot after a structural change* needs `describe` to say which
 *    slots an instance's arguments create, which the argument sheet is what asks for (2.10). A
 *    notice built without it would be the editor re-deriving V7, which §1 forbids in as many
 *    words: "the GUI packages implement no rule of §6".
 *  - *a presentation binding rendered generically* is a fact about the **editor**, not about the
 *    document: 121 enumerated values and 84 union members of the loaded schemas have no binding,
 *    and every document would carry the same row. §1 says where it goes — "renders generically,
 *    never fails, and is listed in the log" — and the startup audit writes it there.
 *  - *a primitive without a witness* needs a reference-generator manifest, which nothing loads
 *    (finding F11, out of this plan): "the editor's only obligation is the `no witness` badge …
 *    from the absence of a manifest entry **when a manifest is loaded**".
 *
 * **What a notice may read.** The document's own text and the editor's own events, and nothing
 * else. An unused quantity is a question about *references* — which the store's index answers for
 * the rename and the delete cascade already (2.1, 2.2) — and not a rule of §6: `--validate` says
 * nothing about a quantity nothing reads, and neither does `--lint`.
 */
import type { Facts, Problem } from '@tensorspine/lang/api';
import { PROBLEM_SEVERITY, PROBLEM_SOURCE } from '@tensorspine/lang/api';
import { QUANTITIES } from '@tensorspine/lang';
import {
  isUnder,
  matches,
  pointerOf,
  type Occurrence,
  type Path,
  type ReferenceIndex,
  type ReferenceSelector,
} from '@tensorspine/store';

import { presentation } from '../presentation/index.js';
import { referenceSelectors } from '../presentation/selectors.js';
import { textWith } from '../shell/strings.js';
import type { OutlineRow } from '../explorer/outline.js';

/** The rule each notice names, so a filter and a fix can key on it without reading the message. */
export const NOTICE = {
  /** A quantity the document declares and nothing reads. */
  unusedQuantity: 'unused-quantity',
  /** A sidecar key that named a place the document no longer has, dropped (D6). */
  droppedKey: 'dropped-key',
  /** The workspace's own schemas differ from the ones the build vendored (§1). */
  schemaMismatch: 'schema-mismatch',
  /** A binding naming a slot or a state port the instance's arguments no longer create (§3). */
  absentSlot: 'absent-slot',
  /** A primitive the author asked the project to declare, that no gathered base carries (2.21). */
  wantedPrimitive: 'wanted-primitive',
} as const;

/** The fix each notice carries, by the name a provider is keyed on. */
export const FIX = {
  /** Declare the wanted primitive as a unit of a base of this model (Q6) — features 3.2, 3.3. */
  createPrimitive: 'create-primitive',
} as const;

/** One notice, built the same way every row of the panel is. */
function notice(rule: string, message: string, place: { path?: string; file?: string }): Problem {
  return {
    code: '',
    message,
    path: place.path ?? '',
    rule,
    ...(place.file === undefined ? {} : { file: place.file }),
    severity: PROBLEM_SEVERITY.notice,
    source: PROBLEM_SOURCE.editor,
  };
}

/** What {@link unusedQuantities} reads: the document's outline, and its reference index. */
export interface QuantityReading {
  /** The outline of the document, every row of it (2.7's `openAll`). */
  readonly rows: readonly OutlineRow[];
  readonly index: ReferenceIndex;
  /** The document's own path, so the row names the file it is about. */
  readonly file?: string;
}

/** The place the quantities of a document are declared — the core's own member name, never ours. */
const QUANTITIES_AT = `/${QUANTITIES}`;

/**
 * A notice per quantity the document declares and nothing reads.
 *
 * The reference rules are `presentation.json`'s, read at the anchor the outline row says declared
 * it — the same rules the rename rewrites by and the delete cascades over, so a quantity this
 * calls unread is a quantity a rename would have found nothing to rewrite for. Quantities "form
 * one flat namespace: each is declared once and referenced by name everywhere" (O0.4), so there is
 * no scope to read them against.
 */
export function unusedQuantities(reading: QuantityReading): Problem[] {
  const found: Problem[] = [];
  const bindings = presentation();
  // The rules are the same for every quantity — one flat namespace (O0.4) — so they are read once
  // per anchor rather than once per declaration.
  const rules = new Map<string, readonly ReferenceSelector[]>();
  for (const row of reading.rows) {
    if (row.kind !== 'entry' || !row.named) continue;
    if (!row.pointer.startsWith(`${QUANTITIES_AT}/`)) continue;
    if (row.pointer.slice(QUANTITIES_AT.length + 1).includes('/')) continue;
    if (row.declaredAt === undefined) continue;
    let selectors = rules.get(row.declaredAt);
    if (selectors === undefined) {
      const binding = bindings.at(row.declaredAt);
      selectors = binding === undefined ? [] : referenceSelectors(binding, row.scope);
      rules.set(row.declaredAt, selectors);
    }
    if (selectors.length === 0) continue;
    // `of` and not `select`: what is asked is whether *anything* names it, and `select` answers
    // the occurrences in document order, which costs a walk of the whole index per question —
    // measured at 27 ms over `deepseek-v4-pro`'s 1 245 occurrences and 22 quantities.
    const named = selectors.some((selector) =>
      reading.index
        .of(selector.tag, row.label)
        .some((one) => matches(one, selector) && pointerOf(one.path) !== row.pointer),
    );
    if (named) continue;
    found.push(
      notice(
        NOTICE.unusedQuantity,
        textWith('quantity {} is declared and read nowhere', row.label),
        { path: row.pointer, ...(reading.file === undefined ? {} : { file: reading.file }) },
      ),
    );
  }
  return found;
}

/** One sidecar key the pruning dropped, as the layout store answers it. */
export interface DroppedSidecarKey {
  /** Which member it was under. */
  readonly where: string;
  /** The key itself: a place of the document that no longer resolves. */
  readonly key: string;
}

/**
 * A notice per sidecar key dropped (D6: "a sidecar key that names a path the document no longer
 * has is dropped with a log line").
 *
 * The Log keeps the line, because the Log is the chronological record of what the editor did; the
 * panel keeps the standing fact, because a position silently lost is the kind of thing a reader
 * notices a week later. Both, and neither instead of the other.
 */
export function droppedKeys(
  dropped: readonly DroppedSidecarKey[],
  file?: string,
): Problem[] {
  return dropped.map((one) =>
    notice(
      NOTICE.droppedKey,
      textWith(
        'layout: the sidecar’s {} names a place the document no longer has; dropped',
        `${one.where} '${one.key}'`,
      ),
      { path: one.key, ...(file === undefined ? {} : { file }) },
    ),
  );
}

/** One difference between the workspace's schemas and the vendored ones (§1's mismatch warning). */
export interface SchemaMismatch {
  readonly path: string;
  readonly message: string;
}

/**
 * A notice per schema the workspace carries that the build's own does not match.
 *
 * §1: "a workspace that carries its own `schemas/` overrides them, and a mismatch of `$id` or
 * content between the two is a warning in the log". The workspace's copy is the one in force —
 * this changes no verdict — and the row is how a reader learns that the rules being applied are
 * not the ones the build shipped.
 */
export function schemaMismatches(differences: readonly SchemaMismatch[]): Problem[] {
  return differences.map((one) =>
    notice(NOTICE.schemaMismatch, one.message, { file: one.path }),
  );
}

/**
 * A binding that names a slot or a state port the instance's arguments no longer create.
 *
 * §4.17's first editor notice, and plan §3's own case: "turning `output_gate` on replaces the slot
 * `q` by `q_gated`; `kv_source: shared` removes `k` and `v` … the editor lists the bindings that
 * name absent slots or ports as Problems with a one-click fix ('Rebind q → q_gated'), and it never
 * edits them silently."
 *
 * **The validator says nothing about it, which is why the row is the editor's.** `check_parameters`
 * and `check_states` skip a slot that is not present *before they read anything of it*, so a
 * binding left over from a structural change is neither refused nor reported — and, for the same
 * reason, `SlotDescription.boundBy` is `null` there: the analysis never recorded the binding. The
 * two halves of the condition are therefore the core's and the document's: `describe` says the
 * declared slot is **absent**, and the document's own reference index says a binding **names** it.
 * No rule of §6 is re-derived, which is why feature 2.8 left this notice to this one.
 *
 * **The row names the binding as the document writes it.** `SlotDescription.boundBy` is the rule of
 * the *normalised* document (`decoder.attn.q`), and the file has `attn.q` inside the composition.
 * The written name is found the way the delete cascade finds a rule: the nearest place the
 * reference index records as a **key** above the place the slot's name is written at — the
 * document's own reading, with nothing about §5.2 rule 7 restated here.
 */
export function absentSlotBindings(reading: AbsentSlotReading): Problem[] {
  const found: Problem[] = [];
  // Which `(site, slot)` pairs the document names at all, in one pass: the walk below asks only
  // about the absent slots, and a document names none of them in the ordinary case.
  const named = new Set<string>();
  for (const one of reading.index.all) {
    if (one.kind !== 'key') {
      for (const qualifier of Object.values(one.qualifiers)) {
        named.add(`${qualifier}\u0000${one.name}`);
      }
    }
  }
  for (const site of reading.sites) {
    for (const slot of site.slots) {
      if (slot.present || !named.has(`${site.name}\u0000${slot.name}`)) continue;
      const place = bindingPlaceOf(reading.index, site.name, slot.name);
      if (place === null) continue;
      found.push({
        ...notice(
          NOTICE.absentSlot,
          textWith('binding {} names a slot the arguments no longer create', place.rule),
          { path: place.path, ...(reading.file === undefined ? {} : { file: reading.file }) },
        ),
        node: `${site.where}.${slot.name}`,
      });
    }
  }
  return found;
}

/**
 * A primitive the author asked the project to gain — feature 2.21's *add to the project*.
 *
 * Declared here rather than in the document store because this is the module that reads it: the
 * want is an editor **event**, like a dropped sidecar key, and the document has no member for one
 * (D6's rule one level up — "the grammar closes every object … the language would refuse it").
 */
export interface WantedPrimitive {
  /** The `primitive` member the confirm was raised on, as a JSON pointer. */
  readonly pointer: string;
  readonly name: string;
  readonly version: string;
}

/** What {@link wantedPrimitives} reads: the wants, the catalog, and what the document now writes. */
export interface WantedReading {
  /** What the confirm recorded, in the order it was recorded. */
  readonly wanted: readonly WantedPrimitive[];
  /** Every identity the gathered bases carry, keyed `name@version` — the core's own projection. */
  readonly catalog: ReadonlySet<string>;
  /** The reference the document now writes at a want's place, or `null` where it writes none. */
  readonly written: (pointer: string) => { readonly name: string; readonly version: string } | null;
  readonly file?: string;
}

/**
 * A notice per primitive the author asked the project to declare and no base provides yet.
 *
 * **What it adds to V1, and why it is not V1 again.** The core already says the document names a
 * primitive the library lacks, in the tools' own words; what it cannot say is that the author
 * *meant* to declare it, because that is an answer to a question the editor asked (§9 Q5's
 * confirm) and nothing of the document records it. So this row carries the intent and the repair,
 * and the verdict stays the core's.
 *
 * **It goes when it stops being true**, on both counts, so nothing has to remember to withdraw it:
 * a base that gains the identity takes the row with it, and a place that no longer writes that
 * identity — retyped, or deleted with its instance — takes it too.
 *
 * The fix is declared and not made: the base is `New Base…` (3.2) and the unit is the primitive
 * editor (3.3), so the panel draws what the repair *will be* rather than a button that pretends
 * (§4.17's "every fix is shown before it is applied", with nothing to apply yet).
 */
export function wantedPrimitives(reading: WantedReading): Problem[] {
  const found: Problem[] = [];
  const seen = new Set<string>();
  for (const one of reading.wanted) {
    const id = `${one.name}@${one.version}`;
    if (reading.catalog.has(id)) continue;
    const written = reading.written(one.pointer);
    if (written === null || written.name !== one.name || written.version !== one.version) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    found.push({
      ...notice(
        NOTICE.wantedPrimitive,
        textWith('primitive {} is to be declared in a base of this model', id),
        { path: one.pointer, ...(reading.file === undefined ? {} : { file: reading.file }) },
      ),
      node: one.name,
      detail: [
        {
          message:
            'a base of your own is New Base… and the unit is the primitive editor; ' +
            'neither is built yet, so the repair below is what it will be',
          path: one.pointer,
        },
      ],
      fixes: [
        {
          kind: FIX.createPrimitive,
          title: textWith('Create primitive {} in a base of this model', `‘${one.name}’`),
        },
      ],
    });
  }
  return found;
}

/** One site as {@link absentSlotBindings} reads it: the slots `describe` answered for it. */
export interface AbsentSlotSite {
  /** The site's own name, as the document's map keys it — what a binding's member names. */
  readonly name: string;
  /** How D1 and a refusal name it, for the row's `node`. */
  readonly where: string;
  /** Every declared slot and state port, present or not, with what bound it. */
  readonly slots: readonly AbsentSlotCandidate[];
}

/** One declared slot or state port, as the notice and the rebind read it. */
export interface AbsentSlotCandidate {
  readonly name: string;
  /** Which map of the primitive declares it: a rebind may only name a slot of the same kind. */
  readonly kind: string;
  readonly present: boolean;
  /** The binding rule that bound it in the normalised document, or `null`. */
  readonly boundBy: string | null;
}

/**
 * The sites `describe` answered for, as the notice and its fix read them.
 *
 * One reading for both, because the fix's condition must be the notice's own fact and never a
 * second look at the document: the row exists because a bound slot is absent, and the pill exists
 * because exactly one present slot of that kind is unbound.
 */
export function slotSites(facts: Facts | null): AbsentSlotSite[] {
  if (facts === null) return [];
  const sites: AbsentSlotSite[] = [];
  for (const site of facts.sites.values()) {
    const slots: AbsentSlotCandidate[] = [];
    for (const slot of [...site.parameters, ...site.constants]) {
      slots.push({ name: slot.name, kind: slot.kind, present: slot.present, boundBy: slot.boundBy });
    }
    for (const state of site.states) {
      slots.push({ name: state.name, kind: STATE_KIND, present: state.present, boundBy: state.boundBy });
    }
    sites.push({ name: site.key.name, where: site.where, slots });
  }
  return sites;
}

/**
 * What a state port's kind is called here.
 *
 * `SlotDescription.kind` is the core's own word for which map of the primitive declares a slot
 * (`parameter`, `constant`); a state port is declared in a third and the core gives it no such
 * member, so the reading names it — a key of this module's own, never compared with anything of
 * the language.
 */
const STATE_KIND = 'state-port';

/** What {@link absentSlotBindings} reads: the described sites, and the document's own names. */
export interface AbsentSlotReading {
  readonly sites: readonly AbsentSlotSite[];
  readonly index: ReferenceIndex;
  readonly file?: string;
}

/** Where a binding names one slot of one site, and the rule it stands in, as written. */
export interface BindingPlace {
  /** The rule's own name, as the document writes it: `attn.q` inside `decoder`. */
  readonly rule: string;
  /** The rule's place, as an RFC 6901 pointer into the document as written. */
  readonly path: string;
  /** Where the slot's name itself is written — what a rebind sets. */
  readonly slotPath: Path;
}

/**
 * The place a binding names `(site, slot)`, found in the document as the author wrote it.
 *
 * Two readings of the reference index, neither of which names a member of the grammar: the slot's
 * own occurrence is a **tagged** name equal to the slot, standing beside a name equal to the site
 * (the selector's `site` or `instance`, which is what a qualifier is); the rule is the nearest
 * **key** occurrence above it, which is how the delete cascade finds the unit a reference costs.
 */
export function bindingPlaceOf(
  index: ReferenceIndex,
  site: string,
  slot: string,
): BindingPlace | null {
  const named = index.all.find(
    (one) =>
      one.kind === 'tagged' &&
      one.name === slot &&
      Object.values(one.qualifiers).includes(site),
  );
  if (named === undefined) return null;
  let rule: Occurrence | null = null;
  for (const one of index.all) {
    if (one.kind !== 'key' || !isUnder(named.path, one.path)) continue;
    if (rule === null || one.path.length > rule.path.length) rule = one;
  }
  if (rule === null) return null;
  return { rule: rule.name, path: pointerOf(rule.path), slotPath: named.path };
}

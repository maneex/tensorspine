/**
 * The fix-action framework — §4.17, plan §3, artboard S10's `.fix` pills.
 *
 * > Problems whose fix is mechanical carry a fix action — rebind a slot renamed by a structural
 * > change, remove a binding naming an absent instance, replace the older edge of a twice-fed
 * > input, expose an unconsumed output, name a scoped identity, create a missing axis or role in
 * > an owned base — and **every fix is shown before it is applied**. A later increment may add an
 * > **autofix** that applies such fixes in one pass, previewed as a list; it is noted here as a
 * > candidate, not scheduled.
 *
 * The framework is the machinery: a provider answers the {@link FixAction}s a row carries, the
 * panel draws them as pills, and choosing one *offers* the command it makes — the confirmation
 * dialog of §4.4 lists what the edit would do before anything is written, and the command lands in
 * the undo log named after the fix (D13). `FixAction` is the core's own declared shape, "declared
 * so that the producer arrives without the contract moving"; this is the producer.
 *
 * **What is shipped, and what each remaining fix waits for.** A fix must be an unambiguous, whole
 * edit, and the condition under which it is offered must be a fact the editor has — never a
 * reading of the core's message, which "would be a second reading of the wording the parity job
 * pins", and never a rule of §6 re-derived in the interface (§1). That leaves exactly one of
 * §4.17's six for this feature, and it is the one whose *problem* the editor produced itself:
 *
 * | §4.17's fix | Offered on | Owed by |
 * |---|---|---|
 * | remove an unused quantity | the editor's own `unused-quantity` notice | **here** |
 * | rebind a slot a structural change renamed | `describe`'s present slots | **here** (feature 2.10) |
 * | remove a binding naming an absent instance | the same, once a sheet knows the binding's slots | 2.13 |
 * | replace the older edge of a twice-fed input | the canvas's own drop (Q5) | 2.9 |
 * | expose an unconsumed output | the interfaces sheet | 2.12 |
 * | name a scoped identity | the identities sheet | 2.13 |
 * | create a missing axis or role in an owned base | the primitive editor | 3.4 |
 * | create a wanted primitive in a base of this model | the editor's own `wanted-primitive` notice | 3.2, 3.3 |
 *
 * Each of those needs a fact this feature has no call for — which slots an instance has under its
 * arguments, which edge a drop replaced, which base the user owns — and inventing the condition
 * from the message would put the panel in the business of re-deciding what the core decided.
 *
 * **A row may declare a fix nobody can make yet, and the panel draws it as such.** Feature 2.21's
 * last row is the case: the *condition* is known here and now — the author answered the confirm
 * and said the primitive is to be declared — and the *edit* is a unit in a base of the model,
 * which needs `New Base…` (3.2) and the primitive editor (3.3). So the notice carries the action
 * in `Problem.fixes`, where feature 1.1 put it, and `Problems.tsx` draws such an action as the
 * words of the repair rather than as a button that would do nothing. No provider below claims it,
 * and the feature that can make it becomes its provider without the row moving.
 */
import type { FixAction, Problem } from '@tensorspine/lang/api';
import type { Command, EditContext, ReferenceIndex } from '@tensorspine/store';
import { remove, setValue } from '@tensorspine/store';

import { presentation } from '../presentation/index.js';
import { referenceSelectors } from '../presentation/selectors.js';
import { textWith } from '../shell/strings.js';
import type { OutlineRow } from '../explorer/outline.js';
import { bindingPlaceOf, NOTICE, type AbsentSlotSite } from './notices.js';

/** What a fix knows how to do, once the panel has decided to do it. */
export interface Fix {
  /** What the pill says, and what distinguishes it (the core's declared shape). */
  readonly action: FixAction;
  /** The command it makes, against the document's own edit context. */
  readonly make: (context: EditContext) => Command;
}

/** What a provider reads: the document as the outline sees it, and nothing else. */
export interface FixReading {
  /** Every row of the outline, whatever is open (2.7's `openAll`). */
  readonly rows: readonly OutlineRow[];
  /**
   * What a structural change did to the slots of each site — `describe`'s own answer.
   *
   * Feature 2.10's half of the table below: the rebind is offered where the core says a binding
   * names an absent slot and which present slots of the same kind nothing binds, and the interface
   * decides nothing about either. It is the notice's own reading (`slotSites`), so a pill cannot
   * appear on a row whose condition has moved.
   */
  readonly slots?: readonly AbsentSlotSite[];
  /** Where each `(site, slot)` is written, for the edit the rebind makes. */
  readonly index?: ReferenceIndex;
}

/** One family of fixes: what it is called, and what it offers for a row. */
export interface FixProvider {
  readonly kind: string;
  offer(problem: Problem, reading: FixReading): Fix | null;
}

/**
 * Remove a declaration the editor itself found nothing naming.
 *
 * The condition is the notice's own rule, not a reading of its words, and the edit is the delete
 * command the tree and the sheet already make — with its cascade, which for a declaration nothing
 * names is the declaration and nothing else. That is what makes it mechanical: there is no second
 * place for it to touch, by the very fact that produced the notice.
 */
const removeUnread: FixProvider = {
  kind: 'remove-declaration',
  offer(problem, reading) {
    if (problem.rule !== NOTICE.unusedQuantity) return null;
    const row = reading.rows.find((one) => one.pointer === problem.path);
    if (row === undefined || !row.named) return null;
    const what = row.declares ?? '';
    const label = what === '' ? row.label : `${what} ${row.label}`;
    const binding = row.declaredAt === undefined ? undefined : presentation().at(row.declaredAt);
    const references = binding === undefined ? [] : referenceSelectors(binding, row.scope);
    return {
      action: {
        kind: this.kind,
        title: textWith('Remove {}', label),
      },
      make: (context) => remove(context, { path: row.path, references, label: `Remove ${label}` }),
    };
  },
};

/**
 * Rebind a slot a structural change renamed — §4.17's second fix, plan §3's `Rebind q → q_gated`.
 *
 * **Offered only where it is one edit and there is only one of it.** "A fix action exists only
 * where the fix is *one creation*": turning `output_gate` on takes `q` away and puts `q_gated`
 * there, so the binding has exactly one present, unbound slot of its own kind to name and the
 * repair is that one word. Where a change took two slots away (`kv_source: shared` removes `k` and
 * `v`) or left two free, nothing here can say which goes with which, and the row carries the
 * notice without a pill rather than a guess — which is the rule §4.17 states and the reason this
 * feature ships one provider and not two.
 *
 * The edit writes the slot's own place, which the notice already found in the document as written
 * (`bindingPlaceOf`), so the rule keeps the name its author gave it (2.9's finding: a rule's name
 * is the author's) and the cascade has nothing to do.
 */
const rebindSlot: FixProvider = {
  kind: 'rebind-slot',
  offer(problem, reading) {
    if (problem.rule !== NOTICE.absentSlot) return null;
    const node = problem.node ?? '';
    const dot = node.lastIndexOf('.');
    if (dot < 0 || reading.index === undefined) return null;
    const slot = node.slice(dot + 1);
    const where = node.slice(0, dot);
    const site = reading.slots?.find((one) => one.where === where);
    const gone = site?.slots.find((one) => one.name === slot && !one.present);
    if (site === undefined || gone === undefined) return null;
    const free = site.slots.filter(
      (one) => one.kind === gone.kind && one.present && one.boundBy === null,
    );
    if (free.length !== 1) return null;
    const arrived = (free[0] as { name: string }).name;
    const place = bindingPlaceOf(reading.index, site.name, slot);
    if (place === null) return null;
    return {
      action: { kind: this.kind, title: textWith('Rebind {}', `${slot} → ${arrived}`) },
      make: (context) =>
        setValue(context, {
          path: place.slotPath,
          value: arrived,
          label: `Rebind ${slot} → ${arrived}`,
        }),
    };
  },
};

/** Every provider, in the order a row's pills are drawn. */
export const FIX_PROVIDERS: readonly FixProvider[] = [removeUnread, rebindSlot];

/** The fixes a row carries — none for most rows, which is the honest answer (§4.17). */
export function fixesFor(
  problem: Problem,
  reading: FixReading,
  providers: readonly FixProvider[] = FIX_PROVIDERS,
): Fix[] {
  const found: Fix[] = [];
  for (const provider of providers) {
    const fix = provider.offer(problem, reading);
    if (fix !== null) found.push(fix);
  }
  return found;
}

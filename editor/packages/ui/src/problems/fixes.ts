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
 * | rebind a slot a structural change renamed | `describe`'s present slots | 2.10 |
 * | remove a binding naming an absent instance | the same, once a sheet knows the binding's slots | 2.13 |
 * | replace the older edge of a twice-fed input | the canvas's own drop (Q5) | 2.9 |
 * | expose an unconsumed output | the interfaces sheet | 2.12 |
 * | name a scoped identity | the identities sheet | 2.13 |
 * | create a missing axis or role in an owned base | the primitive editor | 3.4 |
 *
 * Each of those needs a fact this feature has no call for — which slots an instance has under its
 * arguments, which edge a drop replaced, which base the user owns — and inventing the condition
 * from the message would put the panel in the business of re-deciding what the core decided.
 */
import type { FixAction, Problem } from '@tensorspine/lang/api';
import type { Command, EditContext } from '@tensorspine/store';
import { remove } from '@tensorspine/store';

import { presentation } from '../presentation/index.js';
import { referenceSelectors } from '../presentation/selectors.js';
import { textWith } from '../shell/strings.js';
import type { OutlineRow } from '../explorer/outline.js';
import { NOTICE } from './notices.js';

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

/** Every provider, in the order a row's pills are drawn. */
export const FIX_PROVIDERS: readonly FixProvider[] = [removeUnread];

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

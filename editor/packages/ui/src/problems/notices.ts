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
import type { Problem } from '@tensorspine/lang/api';
import { PROBLEM_SEVERITY, PROBLEM_SOURCE } from '@tensorspine/lang/api';
import { QUANTITIES } from '@tensorspine/lang';
import { matches, pointerOf, type ReferenceIndex, type ReferenceSelector } from '@tensorspine/store';

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

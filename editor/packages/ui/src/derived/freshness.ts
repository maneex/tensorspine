/**
 * What the Derived panel's header says about its figures — plan §4.18's freshness.
 *
 * > The header shows freshness (`fresh` / `stale since <edit>` / `derivation failed: …`) and the
 * > assignment used.
 *
 * §5.4 states the rule the panel obeys: "freshness is tracked per product against the document's
 * revision counter; anything computed for an older revision is shown dimmed with `stale`". So the
 * one fact this module reads is the pair of revisions the pipeline already keeps, and the one
 * thing it adds is *what* moved the document — the name of the last command, which is D13's
 * ("every gesture is a named command … the Edit menu names them") and which is what makes the
 * artboard's `stale since kv_heads 8 → 3` a sentence rather than a colour.
 *
 * A document nothing has derived yet is **not** stale, and that distinction is feature 2.6's own:
 * "a document nothing has derived yet is undone, which is a different word". A refusal stays a
 * refusal whichever revision it was computed for, because the figures beside it do not exist at
 * all.
 */
import type { Problem } from '@tensorspine/lang/api';

import type { Reading } from '../documents/pipeline.js';

/** What the header shows, and whether the figures under it are dimmed. */
export interface DerivedFreshness {
  /** Whether a derived document is there to show at all. */
  readonly derived: boolean;
  /** Whether it was computed for an older revision than the document's (§5.4). */
  readonly stale: boolean;
  /** The command that moved the document past it — D13's own name for the gesture. */
  readonly since: string | null;
  /** The derivation's refusal, in the core's own words. */
  readonly failure: Problem | null;
  /** Whether the core is deriving, or waiting for the debounce, right now. */
  readonly running: boolean;
  /** Whether a refusal of the validation is why there is no derivation (§5.4's gate). */
  readonly skipped: boolean;
}

/** The freshness of one reading, with the name of the edit that invalidated it. */
export function freshnessOf(reading: Reading, since: string | null): DerivedFreshness {
  const derived = reading.derived !== null;
  const behind = reading.derivedAt !== reading.revision;
  return {
    derived,
    stale: derived && behind,
    since: derived && behind ? since : null,
    failure: reading.failure,
    running: reading.derivation === RUNNING || reading.checking,
    skipped: reading.derivation === SKIPPED && reading.failure === null,
  };
}

/** The two states of a derivation this module tells apart, as the pipeline names them. */
const RUNNING = 'running';
const SKIPPED = 'skipped';

/**
 * What every {@link DraftStore} agrees on — feature 2.4.
 *
 * §4.3 autosaves a document "every 30 s and on blur to the platform's draft store", and offers to
 * restore it "on reopening a document with a newer draft". Two implementations exist (IndexedDB
 * in the browser, memory in the stub) and a third will (the desktop's), so the key a draft is held
 * under and the summary a listing shows are written once, here, rather than three times.
 *
 * The policy — when to autosave, and what "newer" decides — is feature 2.6's; this is the shape
 * it will write through.
 */
import { normalise, type WorkspacePath } from './paths.js';
import { byteLength } from './readonly.js';
import type { Draft, DraftSummary } from './types.js';

/**
 * The key one draft is held under: the workspace it belongs to and the path inside it.
 *
 * Joined with a NUL, which is injective because neither a workspace identity nor a path can carry
 * one — the same reason the core joins its own composite keys that way.
 */
export function draftKey(workspace: string, path: WorkspacePath): string {
  return `${workspace}\u0000${normalise(path)}`;
}

/** A draft as a listing shows it: everything but the text, and how many bytes the text was. */
export function summaryOf(draft: Draft): DraftSummary {
  return {
    workspace: draft.workspace,
    path: draft.path,
    revision: draft.revision,
    savedAt: draft.savedAt,
    bytes: byteLength(draft.text),
  };
}

/** Newest first, which is the order a list of drafts is offered in. */
export function byNewest(a: DraftSummary, b: DraftSummary): number {
  return b.savedAt.localeCompare(a.savedAt);
}

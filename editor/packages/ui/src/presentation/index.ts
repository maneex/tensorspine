/**
 * Presentation bindings: the one data file of the interface, and the audit that keeps it honest.
 *
 * Plan §1 — the schemas are the source of truth — leaves exactly one thing to the interface:
 * what a schema cannot say. That lives in `packages/ui/src/presentation.json`, keyed by JSON
 * pointers into the schemas, and nowhere else; catching rule (b) is a test over every other file
 * of `packages/ui/src` and `packages/store/src`, and catching rule (a) is
 * {@link startPresentation}, which the shell runs when the registry is loaded.
 */
import type { SchemaRegistry } from '@tensorspine/lang';

import { auditPresentation, presentationLines, type PresentationAudit } from './audit.js';
import { presentation } from './load.js';
import type { Presentation } from './types.js';

export { auditPresentation, genericLines, presentationLines, SCOPE } from './audit.js';
export type { GenericConstruct, PresentationAudit, PresentationProblem } from './audit.js';
export { presentation, PresentationError, readPresentation } from './load.js';
export { referenceSelectors, type Scope } from './selectors.js';
export type {
  Binding,
  Presentation,
  ReferenceRule,
  StatusBarField,
  SymbolBinding,
} from './types.js';

/** What the shell holds after startup: the bindings, and what the audit said about them. */
export interface StartedPresentation {
  readonly bindings: Presentation;
  readonly audit: PresentationAudit;
}

/**
 * Reads the bindings, audits them against the loaded schemas, and writes the audit to the log.
 *
 * "The audit runs at startup (logged) and in CI": this is the startup half. The log sink is the
 * shell's — the Log tab of the bottom panel (§4.2) — and defaults to none, so that a caller with
 * no log yet (a test, a worker) still gets the report.
 *
 * It **logs** rather than refuses, and that is the difference between this half and the CI one.
 * The schemas the audit resolves against are whatever was loaded: the vendored ones, or a
 * workspace's own, which plan §1 admits ("a workspace that carries its own `schemas/` overrides
 * them, and a mismatch … is a warning in the log"). A user whose workspace moved a definition
 * must still get an editor, with the binding that no longer resolves named in the log; the
 * repository's own schemas are held to a stricter rule by `tests/audit/presentation.test.ts`,
 * where an unresolved key fails the build.
 */
export function startPresentation(
  registry: SchemaRegistry,
  log?: (line: string) => void,
): StartedPresentation {
  const bindings = presentation();
  const audit = auditPresentation(registry, bindings);
  if (log !== undefined) for (const line of presentationLines(audit)) log(line);
  return { bindings, audit };
}

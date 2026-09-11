/**
 * `describe` and `check`: the editor-facing surface of the language core (plan §5.3).
 *
 * The interface asks its questions here. The inventory's §7 lists them — "which ports, slots and
 * states exist", "whether an argument applies", "what a shape evaluates to", "whether an edge may
 * be made", "whether an identity may take a member" — under one rule: *a component that shows a
 * fact never computes it*. Everything below is the validator's own answer, lifted out of
 * `validate.analyse`'s closures where the tools keep it (plan §7 F2: "the facts the editor needs
 * per instance live in `validate.analyse`'s closures … the port lifts them into `describe`").
 *
 * **`describe` gates on the grammar.** "Meaning assumes grammar", as the tools stage it and as
 * feature 1.2 recorded of this port: where the tools raise, the port raises, and `analyse` raises
 * on a document the schema would have refused. The editor calls `describe` on every keystroke
 * (§5.4), so a document off the grammar must come back as a *report* and not as an exception:
 * {@link describe} runs the schema stage first and answers the schema's problems with no facts at
 * all, exactly as `--validate` and `--derive` stop there. What it does not promise is that a
 * document *on* the grammar cannot raise: two places in the tools do (feature 1.5's enum-typed
 * quantity with an interval domain, feature 1.6a's unorderable domain bound), and the port
 * reproduces them rather than inventing a verdict the tools do not give.
 *
 * **What has facts, and what has none.** A site is described where its primitive *resolved*: the
 * ports, slots, states and shapes are read off a definition, and a site whose `primitive` V1
 * refused has none to read. Such a site is still on the canvas — the document draws it — and the
 * V1 line in `analysis.problems` says why it carries no facts; the same holds for a site a guard
 * removed, which the analysis lists under `absent` (§5.2 rule 3, "not a reference failure").
 *
 * **What travels, and what does not.** A {@link Description} is data: it survives the structured
 * clone into the worker (§5.3) and carries the whole analysis beside the per-site facts, so the
 * Problems panel, the canvas and the sheets read one answer. Two facts are deliberately *not*
 * materialised per site: the reason an identity is **absent** from a compatibility list (that is
 * {@link check}'s answer on that one candidate, §7's "with the reason a candidate is excluded"),
 * and anything of D1–D6 — the products are their own features, and no figure is computed twice.
 */
import { toPython, type PyRecord } from '../expr/value.js';
import type { JsonValue } from '../json/index.js';
import type { Library } from '../library/index.js';
import type { SchemaRegistry, StructuralProblem } from '../schema/index.js';
import { analyse, type Analysis } from '../validate/index.js';

import { attachCompatibility } from './compatibility.js';
import { checkCandidate, type Candidate, type Verdict } from './check.js';
import { describeSite, type Mutable, type SiteDescription } from './site.js';

export type { DescribedAxis, DescribedShape } from './shape.js';
export { describeShape } from './shape.js';
export type {
  CompatibleIdentity,
  CostDescription,
  Mutable,
  PartitionDescription,
  PortDescription,
  SiteDescription,
  SlotDescription,
  StateComponent,
  StateDescription,
} from './site.js';
export { describeSite } from './site.js';
export { attachCompatibility } from './compatibility.js';
export {
  checkCandidate,
  type Candidate,
  type EdgeCandidate,
  type LocationCandidate,
  type MemberCandidate,
  type PortRef,
  type SlotRef,
  type Verdict,
} from './check.js';

/** What `describe` needs beside the document. */
export interface DescribeOptions {
  /** The schemas, for the grammar stage `describe` gates on. */
  readonly schemas: SchemaRegistry;
  /** The gathered primitive library, as `loadLibrary` answers it. */
  readonly library: Library;
  /** The assignment the external quantities are read under (§4.6); absent for a closed document. */
  readonly assignment?: PyRecord;
}

/** Everything the interface asks of one document (plan §5.3, `describe`). */
export interface Description {
  /** Whether the document is on the grammar. Nothing below it was computed when it is not. */
  readonly conforms: boolean;
  /** The schema stage's problems, in `--validate`'s own words (feature 1.1). */
  readonly structural: readonly StructuralProblem[];
  /**
   * The whole semantic stage — refusals, counters, the expanded graph, the bindings — or `null`
   * where the grammar refused the document.
   */
  readonly analysis: Analysis | null;
  /** One entry per resolved site, keyed as the analysis keys its own maps (`keyOf`). */
  readonly sites: ReadonlyMap<string, SiteDescription>;
}

/**
 * `describe(tree, path, assignment?)`: the editor's reading of one document.
 *
 * The tree is the lexeme-preserving one the store holds (D1) — the same object the JSON view
 * edits and the serializer writes back — because that is what the grammar stage validates and
 * what a pointer into a problem addresses.
 */
export function describe(tree: JsonValue, options: DescribeOptions): Description {
  const structural = options.schemas.structural(tree, 'model');
  if (structural.length > 0) {
    return { conforms: false, structural, analysis: null, sites: new Map() };
  }
  const analysis = analyse(toPython(tree), options.library, {
    ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
  });
  return { conforms: true, structural, analysis, sites: describedSites(analysis) };
}

/**
 * The same over a document already analysed, for a caller that has one.
 *
 * The validation pipeline of §5.4 runs `validate` and `describe` on the same revision; this is
 * what lets the second read the first's answer instead of analysing twice.
 */
export function describeAnalysis(analysis: Analysis): Description {
  return { conforms: true, structural: [], analysis, sites: describedSites(analysis) };
}

/**
 * `check(tree, candidate)`: the verdict on one candidate edit, and its reason.
 *
 * It takes the {@link Description} rather than the tree: the editor holds one per revision (§5.4)
 * and a drag asks about a handle every few milliseconds, which re-analysing the document could
 * not answer inside §5.6's 20 ms. The rules are the same either way — the description carries the
 * analysis the verdict is read against.
 */
export function check(description: Description, candidate: Candidate): Verdict {
  if (description.analysis === null) {
    return {
      ok: false,
      problems: [],
      unknown: 'the document is off the grammar, so nothing about a candidate edit is decidable',
    };
  }
  return checkCandidate(description.analysis, candidate);
}

/** Every resolved site described, with the compatibility lists filled in. */
function describedSites(analysis: Analysis): ReadonlyMap<string, SiteDescription> {
  const sites = new Map<string, Mutable<SiteDescription>>();
  for (const [id, site] of analysis.resolved) {
    sites.set(id, describeSite(analysis, site));
  }
  attachCompatibility(analysis, sites);
  return sites;
}

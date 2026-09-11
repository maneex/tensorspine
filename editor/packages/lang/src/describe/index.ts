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
import { analyse, whereOfSite, type Analysis } from '../validate/index.js';

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
export { describeSite, sortedEffects } from './site.js';
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
export {
  FACE_MEMBERS,
  FED_END,
  foldedBoxes,
  foldedGraph,
  foldedSites,
  instanceSkeleton,
  literalIndex,
  PRODUCING_END,
  proposedFamily,
  proposedName,
  ROOT_INSTANCES,
  VALUE_BINDINGS,
  valueEndpoint,
  type FoldedBadge,
  type FoldedEdge,
  type FoldedGraph,
  type FoldedHandle,
  type FoldedHeld,
  type FoldedIndexValue,
  type FoldedKind,
  type FoldedNode,
  type FoldedOptions,
  type FoldedRange,
} from './folded.js';
export {
  boxOfSite,
  derivedFacts,
  noDerivedFacts,
  slotKey,
  splitMember,
  tiedCount,
  type DerivedFacts,
  type IdentityLink,
  type IdentityMembership,
  type IdentityPlace,
  type SiteFigures,
} from './figures.js';

/** What `describe` needs beside the document. */
export interface DescribeOptions {
  /** The schemas, for the grammar stage `describe` gates on. */
  readonly schemas: SchemaRegistry;
  /** The gathered primitive library, as `loadLibrary` answers it. */
  readonly library: Library;
  /** The assignment the external quantities are read under (§4.6); absent for a closed document. */
  readonly assignment?: PyRecord;
  /**
   * Describe only these sites, by the identifier §5.2 rule 2 gives them (`whereOfSite`).
   *
   * Plan §3 asks for `describe` "for the selected instance", which is what a sheet needs and what
   * its 20 ms round trip is budgeted for; a site nobody is looking at costs its ports, its slots,
   * its states, its shapes and its compatibility lists to describe and the same again to carry
   * out of the worker (feature 1.11, `editor/spikes/timings.md`). The analysis is the whole
   * document's either way — a compatibility list is answered from every identity of the graph —
   * so what this narrows is the describing, not the reading.
   *
   * Absent, every resolved site is described, which is what the corpus suites read.
   */
  readonly only?: readonly string[];
  /**
   * Describe one site per *declared* instance: the folded canvas of §4.7, and nothing more.
   *
   * The canvas draws the document as its author edits it — "root instances, compositions as group
   * boxes" — so a composition's thirty-two iterations are one card, drawn over "one
   * representative iteration" (§4.8). The representative is the **first point of the grid where
   * the site's own guard fires**, which the caller cannot name: a guard removes iterations
   * (§5.2 rule 3) and only the analysis knows which, so `only` would ask for a site that is not
   * there and the card would lose its ports for a reason that is not a defect.
   *
   * It narrows for the same reason `only` does, and by the same amount on the corpus: 9 sites
   * instead of 195 on `llama3-8b`, 35 instead of 552 on `deepseek-v4-pro`.
   */
  readonly folded?: boolean;
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
  return {
    conforms: true,
    structural,
    analysis,
    sites: describedSites(analysis, options.only, options.folded),
  };
}

/**
 * The same over a document already analysed, for a caller that has one.
 *
 * The validation pipeline of §5.4 runs `validate` and `describe` on the same revision; this is
 * what lets the second read the first's answer instead of analysing twice.
 */
export function describeAnalysis(
  analysis: Analysis,
  only?: readonly string[],
  folded?: boolean,
): Description {
  return { conforms: true, structural: [], analysis, sites: describedSites(analysis, only, folded) };
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

/**
 * Every resolved site described, with the compatibility lists filled in — or only the named ones.
 *
 * The filter is applied on `whereOfSite`, which is what the identifiers a caller has are: §5.2
 * rule 2's, the ones D1 lists and every refusal prints. `attachCompatibility` reads the identity
 * instances off the analysis and fills whatever map it is given, so a narrowed map is filled from
 * the whole graph exactly as the full one is.
 */
function describedSites(
  analysis: Analysis,
  only?: readonly string[],
  folded?: boolean,
): ReadonlyMap<string, SiteDescription> {
  const wanted = only === undefined ? null : new Set(only);
  const drawn = new Set<string>();
  const sites = new Map<string, Mutable<SiteDescription>>();
  for (const [id, site] of analysis.resolved) {
    if (wanted !== null && !wanted.has(whereOfSite(site.key))) continue;
    if (folded === true) {
      // One card per declared site, over the first iteration that fired. `analysis.resolved` is
      // filled in the document's own order — the instances, then each composition's grid point by
      // point — so the first entry a declared site has is the representative §4.8 asks for.
      const declared = `${site.key.composition}\u0000${site.key.name}`;
      if (drawn.has(declared)) continue;
      drawn.add(declared);
    }
    sites.set(id, describeSite(analysis, site));
  }
  attachCompatibility(analysis, sites);
  return sites;
}

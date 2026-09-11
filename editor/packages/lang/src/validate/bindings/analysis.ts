/**
 * What the bindings stage reads, and what it answers.
 *
 * `tools/validate.py` has one function: `analyse` walks the graph, then the parameter and state
 * bindings, then writes out the interface ports a caller reads back. Feature 1.6b ported the
 * first part and stopped at the comment that opens the bindings; this feature is the rest, and it
 * runs *where the tools run it* — after V19 and before the interface-port block, so that the
 * refusals land in the order the parity contract fixes and the counters it adds are in `stats`
 * before an expanded template's are merged into them.
 *
 * That is what {@link GraphStage} and {@link BindingsStage} are for: `graph.ts` calls the stage at
 * exactly that point and hands it the state `analyse`'s closures share — the lists it appends
 * refusals and advisories to, the counter map it fills, and the graph the earlier blocks built.
 * The types live here rather than in `graph.ts` so that neither module states the other's answer;
 * both imports are type-only, so nothing circular survives compilation.
 */
import type { Env, Quantities } from '../../expr/model.js';
import type { PyRecord, PyValue } from '../../expr/value.js';
import type { Library } from '../../library/load.js';
import type { GraphAnalysis, Indexing, PortDomain, ResolvedSite, SiteKey } from '../graph.js';
import type { SemanticProblem } from '../problems.js';

/**
 * A physical name a location addresses whole, or the region of one it slices.
 *
 * `evaluate_location`'s answer (D3): names substituted, axes resolved to their position, a stack
 * expanded over its axis, a slice with its offset and the slot's extent. The tools build plain
 * dictionaries; the shape is written down here because D3 writes it into the derived document and
 * V17 walks it.
 */
export type EvaluatedLocation =
  | { readonly tensor: string }
  | { readonly stack: EvaluatedParts }
  | { readonly concat: EvaluatedParts }
  | { readonly slice: EvaluatedSlice };

/** `{"axis": …, "dim": …, "parts": […]}`: a stack over its coordinates, or a concatenation. */
export interface EvaluatedParts {
  readonly axis: PyValue;
  /** The axis's position in the slot's stored shape — `axes.index(axis)`. */
  readonly dim: bigint;
  readonly parts: readonly EvaluatedLocation[];
}

/** `{"tensor", "axis", "dim", "offset", "extent"}`: a region of one physical tensor. */
export interface EvaluatedSlice {
  readonly tensor: string;
  readonly axis: PyValue;
  readonly dim: bigint;
  readonly offset: bigint;
  /** The slot's extent along the axis, which is what the region covers. */
  readonly extent: bigint;
}

/** One region of a physical tensor: `(offset, extent, identity)`, as V17 collects them. */
export interface PhysicalSlice {
  readonly offset: bigint;
  readonly extent: bigint;
  /** The identity instance that bound it, as a message names it. */
  readonly identity: string;
}

/**
 * The physical names the document binds: `analyse`'s `physical` answer.
 *
 * A caller reads it for two things — V17's own check that no name is bound twice, and the
 * prefixed names of an expanded template, which the calling document binds like every other
 * (§3.4).
 */
export interface PhysicalNames {
  /** Physical name to the identity instance that binds it whole. */
  readonly whole: ReadonlyMap<string, string>;
  /** Physical name to the regions of it that identities slice, in the order they were bound. */
  readonly slices: ReadonlyMap<string, readonly PhysicalSlice[]>;
}

/** One member of an identity, as the tools carry it: the site, and the slot or port it names. */
export interface IdentityMember {
  readonly site: SiteKey;
  /** The parameter slot or the state port. */
  readonly name: PyValue;
}

/** One parameter identity instance — one per rule per index environment, for D3. */
export interface TensorInstance {
  /** `instance_name(binding['tensor'], env)`: `wq[layer=3]`, or `tied_embeddings`. */
  readonly identity: string;
  /** The binding that declared it. */
  readonly rule: string;
  /**
   * The index environment the rule fired in — one of `loop_envs`' answers.
   *
   * The tools keep it in a loop variable and lose it; a location is written *in* it (`{"index":
   * "layer"}` in a physical name), so feature 1.6d's `check` on a candidate location needs it to
   * evaluate the name where the rule would.
   */
  readonly env: Env;
  /** The members whose site resolved; a member naming a site that did not is left out. */
  readonly members: readonly IdentityMember[];
  /** The `dtype` selector as written, or `null` when the binding declares none. */
  readonly dtype: PyValue | null;
  /** Where the identity's tensor is stored, evaluated; absent when it is unlocated. */
  readonly location?: EvaluatedLocation;
}

/** One state identity instance — one per rule per index environment, for D4. */
export interface StateInstance {
  readonly identity: string;
  readonly rule: string;
  readonly members: readonly IdentityMember[];
  readonly dtype: PyValue | null;
  /** `sorted(binding['identity']['indices'])`: the identity's own index names. */
  readonly indices: readonly string[];
  /** The one member that writes it (V20), `null` when none or several do. */
  readonly writer: IdentityMember | null;
}

/** What the bindings stage answers: `analyse`'s three outputs, and the state D3 and D4 read. */
export interface BindingsAnalysis {
  /**
   * `instance_keys`: the instance key of every state identity instance the stage decided.
   *
   * Keyed as the tools key it — the binding's name, with the index environment appended where
   * there is one (`decoder.attn.kv{'layer': 3}`) — and carrying the identity's own indices
   * followed by the port's `key_axes` (§4.4).
   */
  readonly instanceKeys: ReadonlyMap<string, readonly PyValue[]>;
  /**
   * `carried`: the states that survive between the fragments of their stream (§5.3).
   *
   * Keyed by the state binding that bound the port, or by the site when nothing bound it, and
   * carrying the indexing domain the state grows along.
   */
  readonly carried: ReadonlyMap<string, Indexing | null>;
  /** The physical names the document's own identities bind, with the prefixed ones merged in. */
  readonly physical: PhysicalNames;
  /** Every parameter slot bound, by {@link slotKeyOf}, to the binding that bound it. */
  readonly slots: ReadonlyMap<string, SlotBinding>;
  /** Every state port bound, keyed the same way. */
  readonly stateSlots: ReadonlyMap<string, SlotBinding>;
  /** One entry per parameter identity instance, in the order the bindings emitted them (D3). */
  readonly tensorInstances: readonly TensorInstance[];
  /** One entry per state identity instance, in the same order (D4). */
  readonly stateInstances: readonly StateInstance[];
}

/** One bound slot: the rule that bound it, beside the `(site, slot)` a string key cannot give back. */
export interface SlotBinding extends IdentityMember {
  /** The binding's name, which the "bound twice" refusal prints. */
  readonly rule: string;
}

/**
 * The state of `analyse` where the bindings blocks stand: what its closures share with them.
 *
 * The three lists are *live* — the stage appends to them, as `fail` and `advisories.append` do —
 * and `stats` is the map the counters are written into, before the merge that closes `analyse`.
 * Everything else is what the graph half decided (feature 1.6b's {@link GraphAnalysis}).
 */
export interface GraphStage {
  /** The document, normalised: `model.load`'s answer. */
  readonly model: PyRecord;
  /** The gathered primitive library; the stage reads its precision policy (V14). */
  readonly library: Library;
  /** The quantities under the assignment in force. */
  readonly quantities: Quantities;
  /** `errors`, as {@link SemanticProblem}s: the stage appends its refusals here. */
  readonly problems: SemanticProblem[];
  /** `advisories`: what the validator noticed and does not refuse (`--lint` prints them). */
  readonly advisories: string[];
  /** `stats`: the counters, in the order the tools write them. */
  readonly stats: Map<string, PyValue>;
  /** Every site whose primitive resolved, by `keyOf`. */
  readonly resolved: ReadonlyMap<string, ResolvedSite>;
  /** The sites a guard removed: a binding naming one is not emitted there (§5.2 rule 3). */
  readonly absent: ReadonlyMap<string, SiteKey>;
  /** Each instance's own indexing domain, or `null` where it has none. */
  readonly own: ReadonlyMap<string, Indexing | null>;
  /** The indexing domain of every `(site, port)`, by `portKeyOf`. */
  readonly domains: ReadonlyMap<string, PortDomain>;
  /** The streams a fragmented public input delivers (§5.3). */
  readonly fragmented: ReadonlySet<string>;
  /** The evaluated `weights_location_prefix` of every template instance that carries one. */
  readonly weightsPrefixes: ReadonlyMap<string, string>;
  /** The expansion of each template instance, by the key of its site. */
  readonly subResults: ReadonlyMap<string, GraphAnalysis>;
}

/** The bindings stage itself: `analyse`'s second half, run where the tools run it. */
export type BindingsStage = (stage: GraphStage) => BindingsAnalysis;

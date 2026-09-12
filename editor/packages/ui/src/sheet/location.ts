/**
 * The evaluated preview under a location — §4.14, artboard S8's `.evald` block.
 *
 * > `tensor`: a physical-name token editor …, **with the evaluated names for the current preview
 * > index shown beneath** (from D3 when fresh, else from the core's evaluation).
 *
 * S8 draws four lines for thirty-two layers — `layer = 0`, `layer = 1`, `… 30 more`, `layer = 31`
 * — and this is that reading, as data: the head, the count of what is not drawn, and the last.
 * Every name in it is the core's ({@link IdentityFacts}), evaluated by `evaluate_location`, which
 * is the same value D3 writes; nothing here evaluates, formats or adds anything, and the region of
 * a slice carries its own end for exactly that reason.
 *
 * **A preview is not a verdict.** Whether a name is bound twice, whether a slice fits, whether the
 * document locates everything: V17's lines, in Problems, with their pointers (§4.17). What this
 * says is what the document *would name*, which is the question an author editing tokens is asking.
 */
import type { IdentityFacts, IdentityInstance } from '@tensorspine/lang';

/** One line of the preview: one identity instance and the names it evaluates to. */
export interface PreviewLine {
  /** The identity instance: `decoder.attn.q[layer=0]`. */
  readonly identity: string;
  /** The index environment it fired in, as `layer = 0`; `''` where the rule fires once. */
  readonly indices: string;
  /** The physical names it binds whole, at most {@link NAMES} of them. */
  readonly names: readonly string[];
  /** How many further names the instance binds and the line does not draw. */
  readonly moreNames: number;
  /** The regions it slices, as `name[offset : end]`. */
  readonly slices: readonly string[];
  /** Whether the location evaluated for this instance at all. */
  readonly located: boolean;
}

/** The preview of one identity's location. */
export interface LocationPreview {
  readonly lines: readonly PreviewLine[];
  /** How many instances stand between the head and the last, undrawn — S8's `… 30 more`. */
  readonly more: number;
  /** How many instances the rule produced. */
  readonly instances: number;
  /** How many of them the document locates. */
  readonly located: number;
}

/** How many instances are drawn before the elision, and how many names one line carries. */
const HEAD = 2;
const NAMES = 4;

/** The preview of a described identity; empty where the core answered none. */
export function locationPreview(facts: IdentityFacts | null): LocationPreview {
  if (facts === null) return { lines: [], more: 0, instances: 0, located: 0 };
  const instances = facts.instances;
  const located = instances.filter((one) => one.located).length;
  if (instances.length <= HEAD + 2) {
    return { lines: instances.map((one) => lineOf(one)), more: 0, instances: instances.length, located };
  }
  const head = instances.slice(0, HEAD).map((one) => lineOf(one));
  const last = instances[instances.length - 1];
  return {
    lines: last === undefined ? head : [...head, lineOf(last)],
    more: instances.length - HEAD - 1,
    instances: instances.length,
    located,
  };
}

/** One instance as a line: its indices, its names and its regions, each as the core gave them. */
function lineOf(instance: IdentityInstance): PreviewLine {
  return {
    identity: instance.identity,
    indices: instance.indices.map((one) => `${one.name} = ${one.value}`).join(', '),
    names: instance.names.slice(0, NAMES),
    moreNames: Math.max(0, instance.names.length - NAMES),
    slices: instance.slices.map((one) => `${one.tensor}[${one.offset} : ${one.end}]`),
    located: instance.located,
  };
}

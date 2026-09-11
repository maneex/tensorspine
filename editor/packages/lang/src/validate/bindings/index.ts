/**
 * The bindings: the second half of `validate.analyse`, and with it the whole of the model
 * validator (feature 1.6c).
 *
 * Feature 1.6b ported `analyse` as far as V19 and left the function open at the comment that
 * opens the parameter bindings; {@link bindingsStage} is what stands there. `graph.ts` calls it at
 * exactly that point, so the refusals of V7, V9, V14, V15, V16, V17, V18 and V20 land in the
 * order the parity contract fixes, the counters they add reach `stats` before an expanded
 * template's are merged in, and the interface-port block still closes the walk.
 *
 * {@link analyse} is therefore the whole function: the graph half with the bindings half hooked
 * into it, recursing into *itself* at a template call site — "a template primitive is expanded
 * here at every call site (§4.6): the template is analysed under the assignment the arguments
 * make, its own bindings are checked for totality, and its parameter and state slots are counted
 * into the caller's. Two invocations share nothing."
 *
 * `analyseGraph` without the hook stays what it was, because the two are held to different
 * expectations: the graph suite compares against the tools' function *cut*, this one against the
 * tools' function whole.
 */
import type { PyValue } from '../../expr/value.js';
import type { Library } from '../../library/load.js';

import { analyseGraph, analyseGraphText, type GraphAnalysis, type GraphOptions } from '../graph.js';

import type { BindingsAnalysis, GraphStage } from './analysis.js';
import { bindingsOf } from './context.js';
import { checkParameters } from './parameters.js';
import { checkStates } from './states.js';

export type {
  BindingsAnalysis,
  BindingsStage,
  EvaluatedLocation,
  EvaluatedParts,
  EvaluatedSlice,
  GraphStage,
  IdentityMember,
  PhysicalNames,
  PhysicalSlice,
  SlotBinding,
  StateInstance,
  TensorInstance,
} from './analysis.js';
export {
  bindPhysicalNames,
  dtypeValues,
  physicalNameProblems,
  signatureOf,
  slotKeyOf,
  tyingProblems,
  type Signature,
} from './parameters.js';
export {
  applicableRule,
  compareStateMember,
  componentsOf,
  emptyAgreement,
  streamOf,
  type Component,
  type StateAgreement,
  type StreamSource,
} from './states.js';
export {
  declaredMultiplicity,
  evaluateLocation,
  locationNames,
  storageShape,
  wholeCount,
  STORAGE_AXIS,
  type LocationResult,
  type LocationUse,
  type LocationValue,
  type SlicedRegion,
} from './locations.js';

/** What `analyse` answers where the document could not be read at all: `empty`'s own bindings. */
export const EMPTY_BINDINGS: BindingsAnalysis = {
  instanceKeys: new Map(),
  carried: new Map(),
  physical: { whole: new Map(), slices: new Map() },
  slots: new Map(),
  stateSlots: new Map(),
  tensorInstances: [],
  stateInstances: [],
};

/**
 * The bindings stage: `analyse`'s parameter block, its first derivation, and its state blocks.
 *
 * The parameters come first because the states read what they counted (`precisions_checked` is
 * one counter over both) and because the tools order them so.
 */
export function bindingsStage(stage: GraphStage): BindingsAnalysis {
  const bindings = bindingsOf(stage);
  checkParameters(bindings);
  checkStates(bindings);
  return {
    instanceKeys: bindings.instanceKeys,
    carried: bindings.carried,
    physical: { whole: bindings.physicalWhole, slices: bindings.physicalSlices },
    slots: bindings.slots,
    stateSlots: bindings.stateSlots,
    tensorInstances: bindings.tensorInstances,
    stateInstances: bindings.stateInstances,
  };
}

/** `analyse`'s answer, with the bindings stage's beside the graph's — never `null` here. */
export interface Analysis extends GraphAnalysis {
  readonly bindings: BindingsAnalysis;
}

/**
 * `analyse(model_path, cat, assignment)`: the whole semantic stage over a document already parsed.
 *
 * The reading is the editor's — the tree exists before any validation — and the scoped bindings
 * of §5.2 rule 7 are hoisted here as `model.load` hoists them.
 */
export function analyse(
  document: PyValue,
  library: Library,
  options: GraphOptions = {},
): Analysis {
  return settled(analyseGraph(document, library, { ...options, beyond: bindingsStage }));
}

/** The same over a document read from its text, which is what the tools are given. */
export function analyseText(
  text: string,
  library: Library,
  options: GraphOptions = {},
): Analysis {
  return settled(analyseGraphText(text, library, { ...options, beyond: bindingsStage }));
}

/**
 * `empty` carries `instance_keys: {}` and `carried: {}` rather than nothing at all, so a caller
 * of `analyse` reads empty maps where the document was refused before the stage could run — and
 * `physical` is empty too, which is the reading its one consumer takes (a template that locates
 * none of its identities is never asked for its names).
 */
function settled(answer: GraphAnalysis): Analysis {
  return answer.bindings === null ? { ...answer, bindings: EMPTY_BINDINGS } : (answer as Analysis);
}

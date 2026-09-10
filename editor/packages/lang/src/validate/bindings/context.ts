/**
 * The closures the bindings blocks share, and the lists they fill.
 *
 * `analyse` is one function, so its blocks read `fail`, `value`, `select`, `loop_envs`,
 * `instance_name` and a dozen dictionaries from the same scope. The port splits the blocks across
 * modules, so what the scope held travels as this record — built once by the stage and handed to
 * each block, which keeps every call the tools make a call to the same function here.
 */
import type { Env } from '../../expr/model.js';
import { modelValue } from '../../expr/model.js';
import { type PyValue } from '../../expr/value.js';
import { has, listOf, optional } from '../../library/access.js';
import { demand } from '../../library/access.js';
import { pyStr } from '../../library/repr.js';
import { comparePythonStrings } from '../../schema/index.js';
import type { PathSegment } from '../../schema/types.js';

import { loopEnvs, selectSite, type Indexing, type SiteKey } from '../graph.js';
import { semanticProblem, type RuleCode } from '../problems.js';
import type {
  GraphStage,
  PhysicalSlice,
  SlotBinding,
  StateInstance,
  TensorInstance,
} from './analysis.js';

/** A recorded answer while it is still being filled in. */
export type Mutable<T> = { -readonly [K in keyof T]: T[K] };

/** What the parameter and state blocks are given: `analyse`'s scope, under names. */
export interface Bindings {
  /** The graph half's answer and the live lists `analyse` appends to. */
  readonly stage: GraphStage;
  /** `fail(code, message)`, with the place in the document the panel needs beside the words. */
  readonly fail: (code: RuleCode, message: string, at?: readonly PathSegment[]) => void;
  /** `value(e, env)`: the model-level evaluator under the quantities in force. */
  readonly value: (expression: PyValue, env: Env) => PyValue;
  /** `select(sel, env)`: the site an instance selector designates. */
  readonly select: (selector: PyValue, env: Env) => SiteKey;
  /** `loop_envs(binding, label)`: the index environments a rule fires in (V10). */
  readonly loopEnvs: (binding: PyValue, label: string, at: readonly PathSegment[]) => Env[];
  /** `instance_name(symbol, env)`: `wq[layer=3]`, the name a refusal and D3 give an identity. */
  readonly instanceName: (symbol: PyValue, env: Env) => string;
  /** `slots`: every parameter slot bound, and by which rule. */
  readonly slots: Map<string, SlotBinding>;
  /** `state_slots`: the same for state ports. */
  readonly stateSlots: Map<string, SlotBinding>;
  /** `tensor_instances`: one per parameter identity instance, for D3. */
  readonly tensorInstances: Mutable<TensorInstance>[];
  /** `state_instances`: one per state identity instance, for D4 and V20. */
  readonly stateInstances: Mutable<StateInstance>[];
  /** `instance_keys`: the derived instance key of each state identity instance (§4.4). */
  readonly instanceKeys: Map<string, readonly PyValue[]>;
  /** `carried`: the states that survive between the fragments of their stream (§5.3). */
  readonly carried: Map<string, Indexing | null>;
  /** `carried_on`: per instance, the streams its states carry across fragments (V18 reads it). */
  readonly carriedOn: Map<string, Set<string>>;
  /** `physical_whole`: physical name to the identity instance that binds it whole. */
  physicalWhole: Map<string, string>;
  /** `physical_slices`: physical name to the regions of it identities slice. */
  physicalSlices: Map<string, PhysicalSlice[]>;
  /** `checked`: the precision comparisons V14 made, over parameters and states alike. */
  checked: bigint;
}

/** The record above, built from the state the graph half hands over. */
export function bindingsOf(stage: GraphStage): Bindings {
  const value = (expression: PyValue, env: Env = new Map()): PyValue =>
    modelValue(expression, stage.quantities, env);
  return {
    stage,
    fail: (code, message, at = []) => {
      stage.problems.push(semanticProblem(code, message, at));
    },
    value,
    select: (selector, env) => selectSite(selector, stage.quantities, env),
    loopEnvs: (binding, label, at) =>
      loopEnvs(binding, label, stage.quantities, stage.problems, at),
    instanceName: (symbol, env) => instanceName(symbol, env, value),
    slots: new Map(),
    stateSlots: new Map(),
    tensorInstances: [],
    stateInstances: [],
    instanceKeys: new Map(),
    carried: new Map(),
    carriedOn: new Map(),
    physicalWhole: new Map(),
    physicalSlices: new Map(),
    checked: 0n,
  };
}

/**
 * `instance_name(symbol, env)`: the name an identity instance carries.
 *
 * `tied_embeddings` where the symbol has no indices, `wq[layer=3]` where it has — the indices
 * sorted by name and their expressions evaluated in the rule's environment, which is what makes
 * two rules naming one identity name it the same way.
 */
function instanceName(
  symbol: PyValue,
  env: Env,
  value: (expression: PyValue, env: Env) => PyValue,
): string {
  const name = pyStr(demand(symbol, 'name'));
  if (!has(symbol, 'indices')) return name;
  const indices = demand(symbol, 'indices');
  const written = listOf(indices)
    .map((one) => pyStr(one))
    .sort(comparePythonStrings)
    .map((index) => `${index}=${pyStr(value(optional(indices, index, null), env))}`);
  // `if not indices` is Python's truthiness: an empty map of indices is no bracket at all.
  return written.length === 0 ? name : `${name}[${written.join(',')}]`;
}

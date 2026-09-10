/**
 * The parameter bindings: `analyse`'s `# --- V7/V14/V15: parameters ---` block and the first
 * derivation that follows it (`# --- D5, first derivation ---`).
 *
 * What it decides:
 *
 * - **V7** — "every parameter, constant and state slot present under its `present_when` is bound
 *   exactly once; a slot absent by its condition is bound by nothing. A present slot's declared
 *   multiplicity resolves to a positive integer: a slot with no copies is absent by its condition,
 *   never by arithmetic (§3.4)."
 * - **V14** — "a dtype selected for a parameter identity is admissible for the role of every
 *   member", and a selector that names no quantity, or one that is not an enum over dtypes, is
 *   refused before any member is read.
 * - **V15** — "the members' shapes are compared *as stored* — the storage axis of a declared
 *   multiplicity before the slot's axes — so equal per-copy shapes with different counts, or a
 *   declared count against none, do not share an identity"; and a slot the primitive declares
 *   exclusive cannot be tied at all.
 * - **V17** — the locations: total or absent, a physical name bound by one identity, slices that
 *   do not overlap and do not coexist with a whole binding, and the prefixed names of an expanded
 *   template bound like every other (§3.4).
 * - **V1** — a member naming a site that does not exist.
 *
 * Then the counters `--validate` prints and D5 reads: the resident parameter elements — "each
 * tensor identity once (tied tensors once)" — and the operations per element of each `per`, "two
 * operations per weight element per element of the output domain, scaled by the activated fraction
 * of a sparsity unit (§4.5)".
 *
 * **`loop_envs` is called twice over the parameter bindings**, once here and once for the resident
 * count, so a rule whose `when` cannot be decided is refused twice — the same V10 line, in two
 * places of the list. That is the tools' own behaviour and the parity contract carries it.
 */
import { pyAdd, pyDivide, pyEqual, pyMultiply, pyOrder } from '../../expr/arithmetic.js';
import { PyTypeError } from '../../expr/errors.js';
import { primitiveCondition, primitiveValue } from '../../expr/primitive.js';
import { truthy, UNRESOLVED, type PyRecord, type PyValue } from '../../expr/value.js';
import { demand, entries, has, listOf, optional } from '../../library/access.js';
import { pyRepr, pyStr } from '../../library/repr.js';
import type { PathSegment } from '../../schema/types.js';
import { comparePythonStrings } from '../../schema/index.js';

import {
  keyOf,
  portKeyOf,
  present,
  reprShape,
  segmentsOfSite,
  shapeIdentity,
  shapesAgree,
  whereOfSite,
  type ResolvedSite,
  type ShapeIdentity,
} from '../graph.js';
import { pyInt } from '../conformance.js';
import type { IdentityMember, PhysicalSlice, TensorInstance } from './analysis.js';
import type { Bindings } from './context.js';
import {
  declaredMultiplicity,
  evaluateLocation,
  locationNames,
  storageShape,
  wholeCount,
} from './locations.js';

/** One member's signature, as V15 compares them: `(name, slot, role, stored shape, sharing)`. */
interface Signature {
  readonly primitive: PyValue;
  readonly slot: PyValue;
  readonly role: PyValue;
  readonly shape: ShapeIdentity;
  readonly sharing: PyValue;
}

/**
 * `_dtype_values(model, d)`: "possible values of a dtype expression: a literal, or the values of
 * an enum quantity. A string marker when the set cannot be bounded."
 *
 * The two markers become the V14 line `dtype selector is unknown` and `dtype selector is not an
 * enum`, which is `values.lower().replace('_', ' ')` of the marker itself.
 */
export function dtypeValues(model: PyRecord, declared: PyValue | null): PyValue[] | string | null {
  if (declared === null) return null;
  if (typeof declared === 'string') return [declared];
  const quantity = optional(demand(model, 'quantities'), pyStr(demand(declared, 'quantity')), null);
  if (quantity === null) return 'UNKNOWN';
  if (!pyEqual(demand(demand(quantity, 'type'), 'kind'), 'enum')) return 'NOT_AN_ENUM';
  const source = optional(quantity, 'source', {});
  if (pyEqual(optional(source, 'kind', null), 'literal')) return [demand(source, 'value')];
  return listOf(demand(demand(quantity, 'type'), 'values'));
}

/** The parameter half of the bindings stage, in the order `analyse` walks it. */
export function checkParameters(bindings: Bindings): void {
  const { stage } = bindings;
  const declared = demand(demand(stage.model, 'bindings'), 'parameters');
  const policy = stage.library.precision;
  // `located`: "a document with one located parameter identity locates every parameter identity
  // instance" — and a `weights_location_prefix` alone makes a document located (§3.4, V17).
  const located =
    entries(declared).some((pair) => has(pair[1], 'location')) || stage.weightsPrefixes.size > 0;
  const physicalWhole = new Map<string, string>();
  const physicalSlices = new Map<string, PhysicalSlice[]>();
  let ties = 0;
  let tensorIdentities = 0;

  for (const [rule, binding] of entries(declared)) {
    const at: PathSegment[] = ['bindings', 'parameters', rule];
    let values = dtypeValues(stage.model, optional(binding, 'dtype', null));
    if (typeof values === 'string') {
      bindings.fail(
        'V14',
        `${rule}: dtype selector is ${values.toLowerCase().replaceAll('_', ' ')}`,
        [...at, 'dtype'],
      );
      values = null;
    }
    for (const env of bindings.loopEnvs(binding, rule, at)) {
      const members: IdentityMember[] = listOf(demand(binding, 'members')).map((member) => ({
        site: bindings.select(demand(member, 'instance'), env),
        name: demand(member, 'parameter'),
      }));
      if (members.some((member) => stage.absent.has(keyOf(member.site)))) continue; // §5.2 rule 3
      tensorIdentities += 1;
      const signatures: Signature[] = [];
      const identity = bindings.instanceName(demand(binding, 'tensor'), env);
      const instance: {
        -readonly [K in keyof TensorInstance]: TensorInstance[K];
      } = {
        identity,
        rule,
        members: members.filter((member) => stage.resolved.has(keyOf(member.site))),
        dtype: optional(binding, 'dtype', null),
      };
      bindings.tensorInstances.push(instance);

      if (has(binding, 'location')) {
        // The location is written against the *first* resolved member's slot: one identity, one
        // stored shape (V15 is what makes the others equal to it).
        const first = members.find((member) => stage.resolved.has(keyOf(member.site)));
        const site = first === undefined ? undefined : stage.resolved.get(keyOf(first.site));
        const slot =
          first === undefined || site === undefined
            ? null
            : optional(demand(site.definition, 'parameters'), pyStr(first.name), null);
        if (slot !== null && site !== undefined) {
          const answer = evaluateLocation(
            demand(binding, 'location'),
            env,
            storageShape(slot),
            site.args,
            bindings.value,
          );
          for (const problem of answer.problems) {
            bindings.fail('V17', `${identity}: ${problem}`, [...at, 'location']);
          }
          if (answer.evaluated !== null) {
            instance.location = answer.evaluated;
            const used = locationNames(answer.evaluated);
            for (const name of used.whole) {
              const bound = physicalWhole.get(name);
              if (bound !== undefined) {
                bindings.fail(
                  'V17',
                  `${identity}: physical tensor '${name}' already bound by ${bound}`,
                  [...at, 'location'],
                );
              }
              physicalWhole.set(name, identity);
            }
            for (const region of used.slices) {
              addSlice(physicalSlices, region.name, {
                offset: region.offset,
                extent: region.extent,
                identity,
              });
            }
          }
        }
      } else if (located) {
        bindings.fail('V17', `${identity}: no location, while the document locates its weights`, at);
      }

      for (const member of members) {
        const site = stage.resolved.get(keyOf(member.site));
        if (site === undefined) {
          bindings.fail(
            'V1',
            `parameter ${rule}: instance does not exist ${whereOfSite(member.site)}`,
            at,
          );
          continue;
        }
        const name = site.primitive;
        const slotName = pyStr(member.name);
        const slot = optional(demand(site.definition, 'parameters'), slotName, null);
        if (slot === null) {
          bindings.fail('V7', `parameter ${rule}: ${pyStr(name)} has no parameter '${slotName}'`, at);
          continue;
        }
        if (!present(slot, site.args)) {
          bindings.fail(
            'V7',
            `parameter ${rule}: slot '${slotName}' absent for these arguments`,
            at,
          );
          continue;
        }
        const multiplicity = declaredMultiplicity(slot);
        if (multiplicity !== null) {
          const count = primitiveValue(multiplicity, site.args);
          if (wholeCount(count) === null) {
            bindings.fail(
              'V7',
              `parameter ${rule}: slot '${slotName}' declares a multiplicity that resolves to ` +
                `${pyRepr(count)} — a present slot has a positive integer number of copies; ` +
                'none is `present_when` (§3.4)',
              at,
            );
          }
        }
        const bound = bindings.slots.get(slotKeyOf(member));
        if (bound !== undefined) {
          bindings.fail(
            'V7',
            `slot ${pyStr(name)}.${slotName} bound twice (${bound.rule}, ${rule})`,
            at,
          );
        }
        bindings.slots.set(slotKeyOf(member), { ...member, rule });
        // "the count is in it (V15)": the shape compared is the stored one.
        signatures.push({
          primitive: name,
          slot: member.name,
          role: demand(slot, 'role'),
          shape: shapeIdentity(storageShape(slot), site.args),
          sharing: demand(slot, 'sharing'),
        });
        const admissible = policy.get(pyStr(demand(slot, 'role')));
        if (admissible !== undefined && values !== null && values.length > 0) {
          const allowed = demand(admissible, 'admissible');
          const bad = values.filter((one) => !contains(allowed, one));
          if (bad.length > 0) {
            bindings.fail(
              'V14',
              `${rule}: precision ${pyRepr(bad)} outside the admissible set of ` +
                `role '${pyStr(demand(slot, 'role'))}' ${pyStr(allowed)}`,
              [...at, 'dtype'],
            );
          }
          bindings.checked += 1n;
        }
      }
      if (signatures.length > 1) {
        ties += 1;
        checkTying(bindings, rule, signatures, at);
      }
    }
  }

  // "every parameter … slot present under its `present_when` is bound exactly once" (V7)
  for (const [, site] of stage.resolved) {
    for (const [slotName, slot] of entries(demand(site.definition, 'parameters'))) {
      if (present(slot, site.args) && !bindings.slots.has(slotKeyOf({ site: site.key, name: slotName }))) {
        bindings.fail(
          'V7',
          `unbound parameter slot: ${pyStr(site.primitive)}@${whereOfSite(site.key)}.${slotName}`,
          segmentsOfSite(site.key),
        );
      }
    }
  }
  stage.stats.set('parameter_slots', BigInt(bindings.slots.size));
  stage.stats.set('tensors', BigInt(tensorIdentities));
  stage.stats.set('shared', BigInt(ties));

  checkPhysicalNames(bindings, physicalWhole, physicalSlices);
  // "V17 across template instances (§3.4): a located document instantiates only templates that
  // locate their identities, under a prefix each instance supplies; the instance's physical names,
  // prefixed, are bound once like every other."
  stage.stats.set(
    'located',
    BigInt(bindings.tensorInstances.filter((one) => one.location !== undefined).length),
  );
  checkTemplateLocations(bindings, located, physicalWhole, physicalSlices);

  bindings.physicalWhole = physicalWhole;
  bindings.physicalSlices = physicalSlices;
  countElements(bindings);
}

/** "Parameter identity compatibility (§3.4)": every member against every other (V15). */
function checkTying(
  bindings: Bindings,
  rule: string,
  signatures: readonly Signature[],
  at: readonly PathSegment[],
): void {
  for (const [index, mine] of signatures.entries()) {
    if (!pyEqual(demand(mine.sharing, 'kind'), 'shareable')) {
      bindings.fail(
        'V15',
        `${rule}: ${pyStr(mine.primitive)}.${pyStr(mine.slot)} is exclusive, it cannot be tied`,
        at,
      );
    }
    for (const [other, theirs] of signatures.entries()) {
      if (other === index) continue; // `if o is s: continue`
      if (!contains(optional(mine.sharing, 'roles', []), theirs.role)) {
        bindings.fail(
          'V15',
          `${rule}: ${pyStr(mine.primitive)}.${pyStr(mine.slot)} does not share with ` +
            `role '${pyStr(theirs.role)}'`,
          at,
        );
      }
      if (!shapesAgree(theirs.shape, mine.shape)) {
        bindings.fail(
          'V15',
          `${rule}: incompatible shapes ${reprShape(mine.shape)} vs ${reprShape(theirs.shape)}`,
          at,
        );
      }
    }
  }
}

/** "A physical name is bound by one identity; the slices of one physical tensor do not overlap." */
function checkPhysicalNames(
  bindings: Bindings,
  whole: ReadonlyMap<string, string>,
  slices: ReadonlyMap<string, readonly PhysicalSlice[]>,
): void {
  const at: PathSegment[] = ['bindings', 'parameters'];
  for (const [name, intervals] of slices) {
    const bound = whole.get(name);
    if (bound !== undefined) {
      bindings.fail(
        'V17',
        `physical tensor '${name}' is bound whole by ${bound} and sliced by ` +
          `${(intervals[0] as PhysicalSlice).identity}`,
        at,
      );
    }
    const ordered = [...intervals].sort(compareSlices);
    for (let index = 0; index + 1 < ordered.length; index += 1) {
      const one = ordered[index] as PhysicalSlice;
      const next = ordered[index + 1] as PhysicalSlice;
      if (next.offset < one.offset + one.extent) {
        bindings.fail(
          'V17',
          `physical tensor '${name}': slices of ${one.identity} ` +
            `[${one.offset}, ${one.offset + one.extent}) and ${next.identity} ` +
            `[${next.offset}, ${next.offset + next.extent}) overlap`,
          at,
        );
      }
    }
  }
}

/**
 * The template instances of a located document (§3.4, V17).
 *
 * "A document that locates its weights instantiates only templates that locate their identities,
 * and gives each instance a `weights_location_prefix` (`[]` is one); a prefix locates the
 * instance's tensors, so a document carrying one locates its weights and every other identity
 * needs its location; the prefixed names of an instance are bound once like every other, so two
 * instances under one prefix collide."
 */
function checkTemplateLocations(
  bindings: Bindings,
  located: boolean,
  whole: Map<string, string>,
  slices: Map<string, PhysicalSlice[]>,
): void {
  const { stage } = bindings;
  for (const [id, sub] of stage.subResults) {
    const site = stage.resolved.get(id) as ResolvedSite;
    const where = whereOfSite(site.key);
    const count = sub.stats.get('located');
    const templateLocated = typeof count === 'bigint' && count > 0n;
    const prefix = stage.weightsPrefixes.get(id);
    if (located && !templateLocated) {
      bindings.fail(
        'V17',
        `${pyStr(site.primitive)} @${where}: the document locates its weights, but the template ` +
          'locates none of its identities',
        segmentsOfSite(site.key),
      );
    } else if (located && prefix === undefined) {
      bindings.fail(
        'V17',
        `${pyStr(site.primitive)} @${where}: a located template instance needs a ` +
          'weights_location_prefix',
        segmentsOfSite(site.key),
      );
    }
    if (!located || !templateLocated || prefix === undefined) continue;
    const held = stage.stats.get('located');
    stage.stats.set('located', pyAdd(typeof held === 'bigint' ? held : 0n, count));
    const physical = sub.bindings;
    if (physical === null) continue;
    for (const [name, identity] of physical.physical.whole) {
      const full = prefix + name;
      const bound = whole.get(full);
      if (bound !== undefined) {
        bindings.fail(
          'V17',
          `${where}/${identity}: physical tensor '${full}' already bound by ${bound}`,
          segmentsOfSite(site.key),
        );
      }
      whole.set(full, `${where}/${identity}`);
    }
    for (const [name, intervals] of physical.physical.slices) {
      for (const region of intervals) {
        addSlice(slices, prefix + name, { ...region, identity: `${where}/${region.identity}` });
      }
    }
  }
}

/**
 * "D5, first derivation: elements, operations per element (§4.1)".
 *
 * "Two operations per weight element per element of the output domain, scaled by the activated
 * fraction of a sparsity unit (§4.5); a primitive adds only the corrections the inventory cannot
 * see, every applying one." The resident count is per *identity*, so a tied tensor counts once.
 */
function countElements(bindings: Bindings): void {
  const { stage } = bindings;
  const ops = new Map<string, PyValue>();
  const add = (per: string, value: PyValue): void => {
    ops.set(per, pyAdd(ops.get(per) ?? 0n, value));
  };
  for (const [, site] of stage.resolved) {
    const fraction = new Map<string, PyValue>();
    for (const unit of listOf(optional(site.definition, 'sparsity', []))) {
      const activated = primitiveValue(demand(unit, 'activated_per_element'), site.args);
      const declared = demand(unit, 'unit');
      for (const slotName of listOf(demand(declared, 'parameters'))) {
        const slot = optional(demand(site.definition, 'parameters'), pyStr(slotName), null);
        if (slot === null) continue;
        let extent: PyValue = null;
        for (const axis of listOf(demand(demand(slot, 'shape'), 'axes'))) {
          if (pyEqual(demand(axis, 'axis'), demand(declared, 'axis'))) {
            extent = primitiveValue(demand(axis, 'extent'), site.args);
          }
        }
        if (activated !== UNRESOLVED && truthy(extent)) {
          fraction.set(pyStr(slotName), pyDivide(activated, extent));
        }
      }
    }
    for (const [slotName, slot] of entries(demand(site.definition, 'parameters'))) {
      if (!present(slot, site.args)) continue;
      const count = elementsOf(demand(slot, 'shape'), site.args, declaredMultiplicity(slot));
      if (count === null) continue;
      add('element', pyMultiply(pyMultiply(2n, count), fraction.get(slotName) ?? 1n));
    }
    for (const entry of listOf(optional(site.definition, 'logical_cost', []))) {
      if (has(entry, 'when') && !truthy(primitiveCondition(demand(entry, 'when'), site.args))) {
        continue;
      }
      const value = primitiveValue(demand(entry, 'expression'), site.args);
      if (value !== UNRESOLVED && value !== null) add(pyStr(demand(entry, 'per')), value);
    }
  }
  // "Resident elements count each tensor identity once (tied tensors once)."
  let resident: PyValue = 0n;
  const declared = demand(demand(stage.model, 'bindings'), 'parameters');
  for (const [rule, binding] of entries(declared)) {
    const at: PathSegment[] = ['bindings', 'parameters', rule];
    for (const env of bindings.loopEnvs(binding, rule, at)) {
      const member = listOf(demand(binding, 'members'))[0] as PyValue;
      const site = stage.resolved.get(keyOf(bindings.select(demand(member, 'instance'), env)));
      if (site === undefined) continue;
      const slot = optional(
        demand(site.definition, 'parameters'),
        pyStr(demand(member, 'parameter')),
        null,
      );
      if (slot === null) continue;
      const count = elementsOf(demand(slot, 'shape'), site.args, declaredMultiplicity(slot));
      if (count !== null) resident = pyAdd(resident, count);
    }
  }
  stage.stats.set('parameter_elements', countOf(resident));
  // `ops` is a `Counter`, so a `per` nothing counted reads as the integer zero without being
  // written into it — the four counters exist whatever the document declares.
  for (const per of ['element', 'cached_position', 'sequence', 'invocation'] as const) {
    stage.stats.set(
      per === 'element' ? 'ops_per_element' : `ops_per_${per}`,
      countOf(ops.get(per) ?? 0n),
    );
  }
}

/** `int(v)` on a counter these blocks summed: a bool is an int in Python, as it is everywhere. */
function countOf(value: PyValue): bigint {
  if (typeof value === 'boolean') return value ? 1n : 0n;
  if (typeof value === 'bigint' || typeof value === 'number') return pyInt(value);
  throw new PyTypeError(
    `int() argument must be a string, a bytes-like object or a real number, not '${
      value === null ? 'NoneType' : Array.isArray(value) ? 'list' : 'dict'
    }'`,
  );
}

/** `elements(shape, args, multiplicity)`: the product of the extents, `None` where one is not one. */
function elementsOf(shape: PyValue, args: PyRecord, multiplicity: PyValue | null): PyValue | null {
  let count: PyValue = 1n;
  for (const axis of listOf(demand(shape, 'axes'))) {
    const extent = primitiveValue(demand(axis, 'extent'), args);
    if (extent === UNRESOLVED || !isNumber(extent)) return null;
    count = pyMultiply(count, extent);
  }
  if (multiplicity !== null) {
    const copies = primitiveValue(multiplicity, args);
    if (copies === UNRESOLVED || !isNumber(copies)) return null;
    count = pyMultiply(count, copies);
  }
  return count;
}

/** `isinstance(v, (int, float))`, where a `bool` is an `int` as it is everywhere in Python. */
function isNumber(value: PyValue): boolean {
  return typeof value === 'bigint' || typeof value === 'number' || typeof value === 'boolean';
}

/** `v in container` over a list the definition writes, by Python's `==`. */
function contains(container: PyValue, value: PyValue): boolean {
  return listOf(container).some((one) => pyEqual(one, value));
}

/** `defaultdict(list)[name].append(region)`. */
function addSlice(
  slices: Map<string, PhysicalSlice[]>,
  name: string,
  region: PhysicalSlice,
): void {
  const held = slices.get(name);
  if (held === undefined) slices.set(name, [region]);
  else held.push(region);
}

/** `sorted(intervals)` over `(offset, extent, identity)` tuples: Python compares in that order. */
function compareSlices(one: PhysicalSlice, other: PhysicalSlice): number {
  const offset = pyOrder(one.offset, other.offset) ?? 0;
  if (offset !== 0) return offset;
  const extent = pyOrder(one.extent, other.extent) ?? 0;
  if (extent !== 0) return extent;
  return comparePythonStrings(one.identity, other.identity);
}

/**
 * A `(site, slot)` pair as one key, which is how `slots` and `state_slots` are keyed.
 *
 * The tools key both dictionaries by a tuple of exactly the shape a port is keyed by, so the
 * port's own encoding serves — one implementation of the pair, and the two key spaces stay
 * apart because the maps do.
 */
export function slotKeyOf(member: IdentityMember): string {
  return portKeyOf(member.site, member.name);
}

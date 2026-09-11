/**
 * The two facts D1 carries about an instance that the graph itself does not: the arguments a
 * consumer reads without knowing the primitive library's defaults, and whether the instance reads
 * across the positions of its stream.
 *
 * Both are `tools/d1.py`'s — `_record_defaults` and `_across_positions` — and neither is the
 * validator's resolution. `validate.resolve_arguments` builds a *typed* map, refusing what does
 * not conform and leaving {@link UNRESOLVED} where a value did not resolve; the emitter builds a
 * *written* one, applying the declared defaults and nothing else, and the derivation checks that
 * the two agree node by node (`derive._consistent`). Keeping them apart is the point: D1 is what a
 * consumer reads, and it must not depend on the validator having run.
 */
import { PyValueError } from '../expr/errors.js';
import { argumentReferences, primitiveCondition, primitiveValue } from '../expr/primitive.js';
import {
  hasKey,
  isRecord,
  member,
  truthy,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../expr/value.js';
import { put } from '../json/tree.js';
import { demand, entries, has, optional } from '../library/access.js';
import { comparePythonStrings } from '../schema/repr.js';

/**
 * `_record_defaults`: a record field the document leaves out, declared with a default and
 * applicable, gets the default — recursively.
 *
 * "D1 is fully resolved, records included, so no consumer has to know the primitive library's
 * defaults" (`tools/d1.py`, finding 12 of 30 Aug 2026). The paths a default or a `present_when`
 * reads are absolute — `rope.scaling.kind` — so the scope is `root`, the instance's whole
 * argument map, at every depth; the record is filled **in place**, which is what lets a field's
 * condition read a sibling the same pass has just written.
 *
 * A default that does not resolve, and one that resolves to `None`, are left out rather than
 * written: an absent member says "the primitive decides", a written `null` would not.
 */
export function recordDefaults(declared: PyValue, values: PyRecord, root: PyRecord): void {
  for (const [name, declaration] of entries(declared)) {
    const type = optional(declaration, 'type', {});
    const held = member(values, name) ?? null;
    if (optional(type, 'kind', null) !== 'record' || !isRecord(held)) continue;
    const record = held;
    const fields = demand(type, 'fields');
    for (const [field, fieldDeclaration] of entries(fields)) {
      if (hasKey(record, field) || !has(fieldDeclaration, 'default')) continue;
      if (
        has(fieldDeclaration, 'present_when') &&
        !truthy(primitiveCondition(demand(fieldDeclaration, 'present_when'), root))
      ) {
        continue;
      }
      const value = primitiveValue(demand(fieldDeclaration, 'default'), root);
      if (value !== UNRESOLVED && value !== null) put(record, field, value);
    }
    recordDefaults(fields, record, root);
  }
}

/**
 * `_across_positions`: whether the instance reads positions of its stream beyond those of the
 * element it produces — the primitive's `effects.across_positions` condition (§4.1, O9.5) on its
 * resolved arguments, as V18 evaluates it.
 *
 * "False when the primitive declares none": absent, the primitive reads its own element alone.
 *
 * The undecidable case is the one that matters. `primitive_condition` answers *false* to what it
 * cannot decide, which is right for a guard — "the guard simply does not fire" — and wrong for a
 * fact: a document that left an argument of the condition unresolved would be given `false`, and a
 * batching runtime would read that as licence. So the arguments the condition names are looked up
 * first, and an {@link UNRESOLVED} one is a refusal, never a guess.
 */
export function acrossPositions(definition: PyValue, args: PyRecord, where: string): boolean {
  const effect = optional(optional(definition, 'effects', {}), 'across_positions', null);
  // `if not effect`: absent, or an empty object, is no effect at all.
  if (!truthy(effect)) return false;
  const condition = demand(effect, 'when');
  for (const path of [...argumentReferences(condition)].sort(comparePythonStrings)) {
    let cursor: PyValue = args;
    for (const part of path.split('.')) {
      // `cur = cur.get(part) if isinstance(cur, dict) else None`
      cursor = isRecord(cursor) ? (member(cursor, part) ?? null) : null;
    }
    if (cursor === UNRESOLVED) {
      throw new PyValueError(
        `${where}: across_positions is undecidable — its condition reads argument ` +
          `'${path}', which does not resolve`,
      );
    }
  }
  const settled: PyRecord = {};
  for (const [name, value] of Object.entries(args)) {
    if (value !== UNRESOLVED) put(settled, name, value);
  }
  return truthy(primitiveCondition(condition, settled));
}

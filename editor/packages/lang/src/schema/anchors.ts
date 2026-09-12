/**
 * Where the two expression languages are written in the schemas — the anchors an editor starts a
 * form or a printing at when the value it holds is not at a place of a document.
 *
 * Plan §4.13: "The same editor serves guards, index ranges, derivations, defaults, extents, domain
 * bounds, location offsets, `present_when`, rule `when`s, invariants, cost entries and
 * granularities", on both sides of the language. A value reached through a *document* carries its
 * place with it and the schema reading follows the path (`SchemaShapes.step`, which is how the
 * canvas prints a guard). A value reached any other way does not: an argument declaration's
 * `present_when` comes out of a **generated argument schema**, an invariant's condition out of a
 * unit, and neither is a place of the document the editor holds.
 *
 * So the anchor has to be named, and it is named **here**, for the reason `FACE_MEMBERS` and
 * `QUANTITIES` are: naming a place of a schema in the interface is naming something the schema
 * states, which §1 forbids there and gives to the core. The identities are the registry's, so a
 * workspace that carries its own `schemas/` is answered with its own (§1's override).
 */
import type { SchemaRegistry } from './registry.js';

/** The role of the model grammar, as `schema.locate` names it. */
const MODEL = 'model';
/** The role of the closed vocabulary. */
const UNIT = 'primitive-library-unit';

/** The `$def` of each: `scalar_expression`/`condition` on the model side, their two on the unit's. */
const MODEL_EXPRESSION = '/$defs/scalar_expression';
const MODEL_CONDITION = '/$defs/condition';
const UNIT_EXPRESSION = '/$defs/expression';
const UNIT_CONDITION = '/$defs/condition';

/** The four anchors an expression or a condition editor is opened at (plan §4.13). */
export interface LanguageAnchors {
  /** A document's scalar expression: a literal, a quantity, an index, an operation, an `if`. */
  readonly expression: string;
  /** A document's condition. */
  readonly condition: string;
  /** A unit's expression, whose leaves are argument paths rather than quantities. */
  readonly unitExpression: string;
  /** A unit's condition, which adds `present`. */
  readonly unitCondition: string;
}

/** Every anchor of {@link LanguageAnchors}; `''` where the registry holds no schema of that role. */
export function languageAnchors(registry: SchemaRegistry): LanguageAnchors {
  const model = registry.locate(MODEL)?.id ?? '';
  const unit = registry.locate(UNIT)?.id ?? '';
  return {
    expression: model === '' ? '' : `${model}#${MODEL_EXPRESSION}`,
    condition: model === '' ? '' : `${model}#${MODEL_CONDITION}`,
    unitExpression: unit === '' ? '' : `${unit}#${UNIT_EXPRESSION}`,
    unitCondition: unit === '' ? '' : `${unit}#${UNIT_CONDITION}`,
  };
}

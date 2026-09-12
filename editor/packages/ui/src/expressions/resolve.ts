/**
 * What a model expression resolves to — artboard S7's `quantity d 4096` and `resolves to 128`.
 *
 * **The evaluator is the core's** (feature 1.2, plan §5.3's `evaluate`). This module binds it to a
 * document: it reads the document's own quantities the way `view.py` does — `resolve_quantities`
 * over the `quantities` member, which is the same call the explorer's figures go through (feature
 * 2.7) — and then asks `modelValue` for each expression. The interface evaluates nothing; the
 * component inventory's §7 puts "what a shape evaluates to" and every figure on the core's side of
 * the line, and an expression's value is the same kind of fact.
 *
 * **It answers `null` rather than raising.** The editor calls it on every keystroke, and a
 * half-typed expression is exactly the input the evaluator's own rules leave undefined: "meaning
 * assumes grammar", so `_apply` raises on an argument list too short for its operator and on a
 * comparison without its members (feature 1.2 recorded which errors propagate). A row that cannot
 * be resolved says nothing, which is what S7 draws beside an operator.
 *
 * **An index resolves to nothing here, and that is right.** A guard inside a composition is
 * written against `$layer`, which has a value per iteration and none at the place the guard is
 * written; the drill-in's scrubber (§4.8, feature 2.14) is where one iteration is chosen. The
 * environment is therefore empty unless a caller supplies one.
 */
import {
  isJsonObject,
  modelValue,
  pyStr,
  QUANTITIES,
  resolveQuantities,
  toPython,
  UNRESOLVED,
  type Env,
  type JsonObject,
  type JsonValue,
  type PyRecord,
  type PyValue,
  type Quantities,
} from '@tensorspine/lang';

/** What an expression resolves to, printed as the core prints it; `null` where it does not. */
export type Resolver = (value: JsonValue, anchor: string) => string | null;

/** The resolver of a document: its quantities read once, every expression asked of the core. */
export function documentResolver(
  tree: JsonValue,
  options?: { readonly assignment?: PyRecord; readonly env?: Env },
): Resolver {
  const quantities = quantitiesOf(tree, options?.assignment);
  return (value: JsonValue): string | null => {
    try {
      const resolved = modelValue(toPython(value), quantities, options?.env);
      return resolved === UNRESOLVED ? null : pyStr(resolved);
    } catch {
      // Every refusal of the evaluator is a fact about a value the author is still writing.
      return null;
    }
  };
}

/** The document's quantities, resolved by the core, or an empty map where it has none. */
function quantitiesOf(tree: JsonValue, assignment?: PyRecord): Quantities {
  if (!isJsonObject(tree)) return new Map<string, PyValue>();
  const written = memberOf(tree, QUANTITIES);
  if (written === undefined || !isJsonObject(written)) return new Map<string, PyValue>();
  try {
    return resolveQuantities({ [QUANTITIES]: toPython(written) }, assignment);
  } catch {
    return new Map<string, PyValue>();
  }
}

/** One member of an object node, by name. */
function memberOf(node: JsonObject, name: string): JsonValue | undefined {
  return node.members.find((one) => one.name === name)?.value;
}

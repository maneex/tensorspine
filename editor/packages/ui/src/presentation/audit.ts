/**
 * The startup audit of plan §1 (a).
 *
 * > A startup audit resolves every key of `presentation.json` against the loaded schemas; a key
 * > that resolves nowhere is an error in the log, and every enum value and `oneOf` member of the
 * > loaded schemas that no binding names is listed as "rendered generically".
 *
 * Both halves matter, and they fail in opposite directions. A key that resolves nowhere is a
 * binding that does nothing: the schema it names moved, or the pointer was mistyped, and the
 * interface would go on rendering generically without anyone learning of it. A construct that no
 * binding names is *not* an error — most of the language renders generically on purpose, a dtype
 * being a plain select and `min` a function of its own name — but it has to be **listed**, so
 * that "the editor has no reading for this yet" is a fact somebody can look up rather than a
 * discovery made in front of a user.
 *
 * Three checks go beyond the letter of (a), each one the same failure one level down: a binding
 * whose `symbols` name a value the enumeration at that anchor does not admit (the plan's own
 * Appendix B sketch does this — it gives `binary_operation_expression` the symbols of `add` and
 * `multiply`, which are `nary_operation_expression`'s), a `face` naming a member the definition
 * does not declare, and a scoped reference rule with no scope to be read against. Each would
 * otherwise be a binding that resolves and still says nothing.
 *
 * Nothing here reads a keyword of a schema by name except `properties`: the enumerations and the
 * unions come from the core's `vocabulary()`, which is the one reading of them (plan §1, D3).
 */
import {
  alternativeLabel,
  factsOf,
  followAnchor,
  parseAnchor,
  type SchemaNode,
  type SchemaRegistry,
  type Vocabulary,
} from '@tensorspine/lang';

import type { Binding, Presentation } from './types.js';

/** What the audit found wrong with one binding. */
export interface PresentationProblem {
  /** Which kind of failure it is. */
  readonly code: string;
  /** The key of the binding it is about. */
  readonly anchor: string;
  /** What is wrong, without repeating the anchor. */
  readonly message: string;
}

/** One construct of the loaded schemas that no binding names. */
export interface GenericConstruct {
  /** `enumeration` for a value of an `enum`, `alternative` for a member of a `oneOf`. */
  readonly kind: string;
  /** Where it is written: the enumeration's anchor, or the alternative's own. */
  readonly anchor: string;
  /** The value, or the tags a chooser would label the alternative by; `''` when it has none. */
  readonly name: string;
}

/** What the startup audit answers. */
export interface PresentationAudit {
  /** How many bindings the file carries. */
  readonly bindings: number;
  /** How many of them name a place of the loaded schemas. */
  readonly resolved: number;
  /** The errors, in the order the file writes the bindings they are about. */
  readonly problems: readonly PresentationProblem[];
  /** Everything that renders generically, in the order the schemas write it. */
  readonly generic: readonly GenericConstruct[];
  /** The `$id`s the audit resolved against. */
  readonly schemas: readonly string[];
}

const ENUMERATION = 'enumeration';
const ALTERNATIVE = 'alternative';

/** The marker a scoped reference rule writes: the enclosing declaration. */
export const SCOPE = 'scope';

/** The code of a keyword that introduces a member the definition does not declare. */
const KEYWORD = 'keyword';

/** The code of a product named at a definition that declares nothing to show. */
const PRODUCT = 'product';

/** The code of a naming binding at a place that admits no name. */
const NAMES = 'names';

/** Resolves every key of the bindings against the registry, and lists what nothing names. */
export function auditPresentation(
  registry: SchemaRegistry,
  bindings: Presentation,
): PresentationAudit {
  const vocabulary = registry.vocabulary();
  const problems: PresentationProblem[] = [];
  let resolved = 0;

  for (const anchor of bindings.anchors) {
    const binding = bindings.at(anchor);
    if (binding === undefined) continue;
    const parsed = parseAnchor(anchor);
    if (parsed === null) {
      problems.push({
        code: 'malformed-key',
        anchor,
        message: 'is not an anchor: a schema identity, `#`, and a JSON pointer',
      });
      continue;
    }
    if (registry.byId(parsed.schema) === undefined) {
      problems.push({
        code: 'unknown-schema',
        anchor,
        message: `names no schema of the registry: ${parsed.schema}`,
      });
      continue;
    }
    const place = followAnchor(registry, anchor);
    if (place === null) {
      problems.push({
        code: 'unresolved',
        anchor,
        message: `names nothing in ${parsed.schema}`,
      });
      continue;
    }
    resolved += 1;
    problems.push(...symbolProblems(anchor, binding, place.anchor, vocabulary));
    problems.push(...faceProblems(anchor, binding, place.node));
    problems.push(...keywordProblems(anchor, binding, place.node));
    problems.push(...productProblems(anchor, binding, place.node));
    problems.push(...namingProblems(anchor, binding, place.node));
    problems.push(...scopeProblems(anchor, binding, bindings));
  }

  return {
    bindings: bindings.anchors.length,
    resolved,
    problems,
    generic: genericConstructs(vocabulary, bindings),
    schemas: registry.schemas.map((schema) => schema.id),
  };
}

/** The symbols of a binding, against what the enumeration or the union at its anchor admits. */
function symbolProblems(
  anchor: string,
  binding: Binding,
  place: string,
  vocabulary: Vocabulary,
): PresentationProblem[] {
  const symbols = binding.symbols;
  if (symbols === undefined) return [];
  const admitted = new Set<string>();
  const enumeration = vocabulary.enumAt(place);
  for (const one of enumeration?.values ?? []) admitted.add(String(one));
  const union = vocabulary.unionAt(place);
  for (const alternative of union?.alternatives ?? []) {
    for (const tag of alternative.tags) admitted.add(tag);
  }
  if (enumeration === undefined && union === undefined) {
    return [
      {
        code: 'symbol',
        anchor,
        message: 'carries symbols, but names neither an enumeration nor a union',
      },
    ];
  }
  const found: PresentationProblem[] = [];
  for (const name of symbols.keys()) {
    if (admitted.has(name)) continue;
    found.push({
      code: 'symbol',
      anchor,
      message: `gives a symbol to '${name}', which it does not admit`,
    });
  }
  return found;
}

/** The face of a node, against the members the definition declares. */
function faceProblems(anchor: string, binding: Binding, node: unknown): PresentationProblem[] {
  const face = binding.face;
  if (face === undefined) return [];
  const names = declaredMembers(node);
  if (names === null) {
    return [{ code: 'face', anchor, message: 'has a face, but declares no members' }];
  }
  return face
    .filter((member) => !names.has(member))
    .map((member) => ({
      code: 'face',
      anchor,
      message: `shows '${member}' on its face, which it does not declare`,
    }));
}

/**
 * The keywords of a text form, against the members the definition declares.
 *
 * The same failure as a `face` naming a member that is not there, one level down: a keyword
 * written for a member the definition does not declare introduces an operand nothing writes, so
 * the text form would print a word with nothing after it and the parser would ask for one.
 */
function keywordProblems(anchor: string, binding: Binding, node: unknown): PresentationProblem[] {
  const keywords = binding.keywords;
  if (keywords === undefined) return [];
  const names = declaredMembers(node);
  if (names === null) {
    return [{ code: KEYWORD, anchor, message: 'introduces members, but declares none' }];
  }
  return [...keywords.keys()]
    .filter((member) => !names.has(member))
    .map((member) => ({
      code: KEYWORD,
      anchor,
      message: `introduces '${member}', which it does not declare`,
    }));
}

/**
 * A derived product, against the definition it names.
 *
 * The Derived panel builds a tab's body out of the members the product declares (§4.18: "an array
 * of objects becomes a table whose columns are the schema's properties in order"), so a `product`
 * at a definition that declares none is a tab with nothing in it — the same failure a `face`
 * naming an absent member is, one construct along.
 */
function productProblems(anchor: string, binding: Binding, node: unknown): PresentationProblem[] {
  if (binding.product === undefined) return [];
  const names = declaredMembers(node);
  if (names !== null && names.size > 0) return [];
  return [{ code: PRODUCT, anchor, message: 'is named as a product, but declares no members' }];
}

/**
 * A naming binding, against what the definition admits.
 *
 * `names` says what a *string* written at this place stands for, so a place that admits no string
 * at all is a binding that resolves and still says nothing: the panel would look for a link and
 * for a subject where neither can be written.
 */
function namingProblems(anchor: string, binding: Binding, node: unknown): PresentationProblem[] {
  if (binding.names === undefined) return [];
  return factsOf(node as SchemaNode).holdsText
    ? []
    : [{ code: NAMES, anchor, message: 'says what a name there stands for, but admits no text' }];
}

/** The member names a definition declares, or `null` where it declares none. */
function declaredMembers(node: unknown): ReadonlySet<string> | null {
  const declared =
    typeof node === 'object' && node !== null ? (node as Record<string, unknown>)['properties'] : undefined;
  if (typeof declared !== 'object' || declared === null || Array.isArray(declared)) return null;
  return new Set(Object.keys(declared));
}

/** The scope a binding's reference rules read against. */
function scopeProblems(
  anchor: string,
  binding: Binding,
  bindings: Presentation,
): PresentationProblem[] {
  const found: PresentationProblem[] = [];
  const scope = binding.scope;
  if (scope !== undefined) {
    const enclosing = bindings.at(scope);
    if (enclosing === undefined || enclosing.declares === undefined) {
      found.push({
        code: SCOPE,
        anchor,
        message: `is scoped by ${scope}, which declares nothing`,
      });
    }
  }
  for (const rule of binding.refers ?? []) {
    if (rule.under !== undefined && rule.under !== SCOPE) {
      found.push({
        code: SCOPE,
        anchor,
        message: `restricts '${rule.tag}' to '${rule.under}', which names no place`,
      });
    }
    if ((rule.under !== undefined || rule.scopedBy !== undefined) && scope === undefined) {
      found.push({
        code: SCOPE,
        anchor,
        message: `reads '${rule.tag}' against an enclosing declaration and names none`,
      });
    }
  }
  return found;
}

/** Every enum value and every `oneOf` member of the loaded schemas that no binding names. */
function genericConstructs(
  vocabulary: Vocabulary,
  bindings: Presentation,
): readonly GenericConstruct[] {
  const found: GenericConstruct[] = [];
  for (const enumeration of vocabulary.enums) {
    const symbols = bindings.at(enumeration.pointer)?.symbols;
    for (const one of enumeration.values) {
      const name = String(one);
      if (symbols?.has(name) === true) continue;
      found.push({ kind: ENUMERATION, anchor: enumeration.pointer, name });
    }
  }
  for (const union of vocabulary.unions) {
    const symbols = bindings.at(union.pointer)?.symbols;
    for (const alternative of union.alternatives) {
      const anchor = `${union.schema}${alternative.place}`;
      const named =
        alternative.tags.some((tag) => symbols?.has(tag) === true) ||
        bindings.at(anchor) !== undefined ||
        (alternative.target !== null && bindings.at(alternative.target) !== undefined);
      if (named) continue;
      // What a chooser would label it by — the core's own formula (`alternativeLabel`), so that
      // this list and the chooser a form generates say the same thing about the same alternative.
      found.push({ kind: ALTERNATIVE, anchor, name: alternativeLabel(alternative) });
    }
  }
  return found;
}

/** The lines the audit writes to the log at startup: every error, then what renders generically. */
export function presentationLines(audit: PresentationAudit): readonly string[] {
  const lines = audit.problems.map((problem) => `presentation: ${problem.anchor}: ${problem.message}`);
  const values = audit.generic.filter((one) => one.kind === ENUMERATION).length;
  const alternatives = audit.generic.length - values;
  lines.push(
    `presentation: ${String(audit.resolved)} of ${String(audit.bindings)} bindings resolved; ` +
      `${String(values)} enumerated values and ${String(alternatives)} union members of ` +
      `${String(audit.schemas.length)} schemas render generically`,
  );
  return lines;
}

/** One line per construct that renders generically: what the log shows when it is unfolded. */
export function genericLines(audit: PresentationAudit): readonly string[] {
  return audit.generic.map(
    (one) => `presentation: ${one.anchor}: ${one.kind} '${one.name}' renders generically`,
  );
}

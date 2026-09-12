/**
 * The text form's grammar, derived from the schemas and `presentation.json` — plan §4.13.
 *
 * §4.13 asks for two views of the same tagged unions, "both generated", and for a text form whose
 * "round-trip is exact: text → JSON → text is the identity". Exactness is a property of a *pair*
 * of programs, and a pair that reads two tables drifts: a symbol added to one, a precedence
 * changed in the other, and the printer writes what the parser will not read. So there is one
 * table, and this module is it. The printer (`print.ts`), the parser (`parse.ts`) and the tree
 * editor (`tree.ts`) all ask for the {@link Grammar} at an anchor and render, read or edit what it
 * answers; none of them knows an operator, a tag or a member name of the language.
 *
 * **What a production is.** One way a value at an anchor is written, one per alternative of the
 * union there and one per operator name the alternative's enumeration admits — because an
 * operator's symbol, its form and its arity are per name, not per alternative. Four shapes cover
 * the two languages, and each is *recognised*, never listed:
 *
 * | Shape | How it is recognised | What it writes |
 * |---|---|---|
 * | {@link PAYLOAD} | one member holding a scalar | `{"quantity": "d"}`, `{"literal": 4096}`, `{"present": "chunk"}` |
 * | {@link APPLICATION} | a member whose binding carries `symbols` over an enumeration, beside its operands | `{"op": "add", "args": […]}`, `{"compare": {…}}`, `{"all": […]}` |
 * | {@link KEYWORD} | a binding at the alternative's own anchor carrying `keywords` | `{"if": …, "then": …, "else": …}` |
 * | {@link GENERIC} | anything else | `label(a, b)` — plan §1's "unknown constructs get the generic widget" |
 *
 * **Which alternative a value is, is the grammar's own verdict** (`registry.accepts`, feature
 * 2.3): `unary_operation_expression` and `nary_operation_expression` require the same two members
 * and are told apart only by the operators their enumerations admit, so a tag match would answer
 * both. An application production adds one reading on top of that — the operator its own name is —
 * which is a comparison of data with data.
 *
 * **Every member name a production reads or writes is the schema's**, taken from
 * `SchemaShapes.propertyOrder` at the moment the grammar is built. This module is therefore the
 * one place where the shape of a written expression is known, and `write`/`parts` are the two
 * directions of that one reading — which is what makes the round trip a property of the code
 * rather than of two agreeing authors.
 */
import {
  getMember,
  isJsonArray,
  isJsonObject,
  jsonObject,
  type JsonMember,
  type JsonValue,
  type SchemaFacts,
  type SchemaRegistry,
} from '@tensorspine/lang';
import type { SchemaShapes, Shape } from '@tensorspine/store';

import { alternationAt, type Alternative } from '../forms/alternatives.js';
import { factsOfShape } from '../forms/walk.js';
import {
  boundAlong,
  type Binding,
  type Presentation,
  type SymbolBinding,
} from '../presentation/index.js';

/** One member holding a scalar: a name the editor picks, or a literal it edits. */
export const PAYLOAD = 'payload';
/** An operator with its operands, infix, prefix, function or `name(args)`. */
export const APPLICATION = 'application';
/** Several operands, each introduced by a keyword: `if … then … else …`. */
export const KEYWORD = 'keyword';
/** A shape nothing above answers for: printed and parsed as `label(a, b)`. */
export const GENERIC = 'generic';

/**
 * The two editors §4.13 owns, by the name `presentation.json` binds them under.
 *
 * Named here and nowhere else in the interface, beside the editor they name — the pattern
 * `widget.ts` follows for the generic widgets. Neither word is vocabulary of the four schemas;
 * both are the presentation file's own, and the audit holds the file to a schema that lists them.
 */
export const EXPRESSION = 'expression';
export const CONDITION = 'condition';

/** Whether a widget is one of those two: what tells a shaped mode from a record's section. */
export function editsExpression(widget: string | undefined): boolean {
  return widget === EXPRESSION || widget === CONDITION;
}

/** The `form` an operator printed between its operands carries. */
export const INFIX = 'infix';
/** The `form` an operator printed before its operand carries. */
export const PREFIX = 'prefix';
/** The `form` an operator printed around its operands carries. */
export const FUNCTION = 'function';

/** Where one operand of a production stands. */
export interface Operand {
  /** The anchor it is written at — the union the operand itself belongs to. */
  readonly anchor: string;
  /** The keyword that introduces it, in a keyword form. */
  readonly keyword?: string;
  /** The member it is written under, for a tree row's label. */
  readonly member: string;
}

/** A list of operands written in one member: what the schema admits of its length. */
export interface OperandList {
  readonly anchor: string;
  readonly minimum: number;
  /** `Infinity` where the schema states no maximum — an n-ary operator. */
  readonly maximum: number;
}

/** What a payload production holds. */
export interface Payload {
  /** Whether it is a name the editor picks rather than a value it types. */
  readonly named: boolean;
  /** What kind of name, by the member it is written under: the `references` binding's own word. */
  readonly referent?: string;
  /** What is printed before it, so a text form tells an index from a quantity (`$layer`). */
  readonly prefix: string;
  /** What the schema asserts about the value, for the widget that edits it. */
  readonly facts: SchemaFacts;
}

/** One way a value at an anchor is written in the text form. */
export interface Production {
  /** {@link PAYLOAD}, {@link APPLICATION}, {@link KEYWORD} or {@link GENERIC}. */
  readonly kind: string;
  /** The alternative of the union it writes. */
  readonly alternative: Alternative;
  /** What a tree row calls it: the operator's name, or the alternative's own label. */
  readonly label: string;
  /** The operator's name, where the alternative enumerates operators. */
  readonly operator?: string;
  /** The symbol `presentation.json` gives it; absent where it has none (§4.13's `name(args)`). */
  readonly symbol?: SymbolBinding;
  /** The operands written one after another, in the schema's own member order. */
  readonly operands: readonly Operand[];
  /** The operands written as a list in one member, where they are. */
  readonly list?: OperandList;
  /** The scalar it holds, where it holds one. */
  readonly payload?: Payload;
  /** The parts of a value written this way, in this production's order; `null` when it is not. */
  parts(value: JsonValue): readonly JsonValue[] | null;
  /** A value written this way, from its parts — the same reading, the other way round. */
  write(parts: readonly JsonValue[]): JsonValue;
}

/** The text form at one anchor. */
export interface Grammar {
  /** The union the grammar is for. */
  readonly anchor: string;
  /** Its productions, in the order the schema writes the alternatives and their enumerations. */
  readonly productions: readonly Production[];
  /** Every symbol, keyword and prefix reachable from here: what the lexer has to cut. */
  readonly marks: readonly string[];
  /** The production a value is written by, with its parts; `null` when nothing writes it. */
  chosen(value: JsonValue): { readonly production: Production; readonly parts: readonly JsonValue[] } | null;
}

/** What a grammar is built from: the schemas, the shapes and the bindings (plan §1). */
export interface LanguageContext {
  readonly registry: SchemaRegistry;
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
}

/** The grammar at an anchor, built once per registry, per bindings and per anchor. */
export function grammarAt(context: LanguageContext, anchor: string): Grammar {
  let byBindings = cache.get(context.shapes);
  if (byBindings === undefined) {
    byBindings = new WeakMap<Presentation, Map<string, Grammar>>();
    cache.set(context.shapes, byBindings);
  }
  let byAnchor = byBindings.get(context.bindings);
  if (byAnchor === undefined) {
    byAnchor = new Map<string, Grammar>();
    byBindings.set(context.bindings, byAnchor);
  }
  const known = byAnchor.get(anchor);
  if (known !== undefined) return known;
  const built = build(context, anchor);
  byAnchor.set(anchor, built);
  return built;
}

const cache = new WeakMap<SchemaShapes, WeakMap<Presentation, Map<string, Grammar>>>();

/**
 * The anchor a place is read as a union by: the first of its chain that *is* one.
 *
 * A member holding an expression is reached through its own property anchor
 * (`…/comparison_condition/properties/compare/properties/left`), which names no union at all; the
 * union is one `$ref` along, at `#/$defs/scalar_expression`. Taking the first anchor of the chain
 * would leave every nested application undiscriminated, and the operator's symbol with it.
 */
export function unionAnchorOf(context: LanguageContext, shape: Shape): string {
  const vocabulary = context.registry.vocabulary();
  for (const place of shape.all) {
    if (vocabulary.unionAt(place.anchor) !== undefined) return place.anchor;
  }
  return shape.all[0]?.anchor ?? '';
}

/** The binding of a place, most specific first, as every reader of `presentation.json` takes it. */
function bindingOf(bindings: Presentation, shape: Shape): Binding | undefined {
  return bindings.firstOf(shape.all.map((place) => place.anchor));
}

function build(context: LanguageContext, anchor: string): Grammar {
  const { registry, shapes } = context;
  const alternation = alternationAt(registry.vocabulary(), shapes, anchor);
  const productions: Production[] = [];
  for (const alternative of alternation?.alternatives ?? []) {
    productions.push(...productionsOf(context, alternative));
  }
  // The marks are gathered lazily, because gathering them walks the operand anchors and an
  // expression's operands are expressions: the closure asks for this very grammar, which the
  // cache holds only once `build` has returned.
  let marks: readonly string[] | null = null;
  return {
    anchor,
    productions,
    get marks(): readonly string[] {
      marks ??= reachableMarks(context, anchor);
      return marks;
    },
    chosen(value: JsonValue) {
      for (const production of productions) {
        const parts = production.parts(value);
        if (parts !== null) return { production, parts };
      }
      return null;
    },
  };
}

/**
 * The productions one alternative contributes: one, or one per operator its enumeration admits.
 *
 * The binding is looked up along the alternative's **ancestry** and not at the anchor the grammar
 * was asked for, which is feature 2.3's own rule ("it keeps the chain each alternative was
 * reached through, so that a presentation binding written for the *union* serves its
 * alternatives"). It is what makes the grammar at `argument_value` — where a sheet and a card
 * reach an expression — the same grammar as the one at `scalar_expression`: the `references` and
 * the connectives' symbols are written at the second and read from the first.
 */
function productionsOf(context: LanguageContext, alternative: Alternative): Production[] {
  const { registry, shapes, bindings } = context;
  const shape = shapes.at(alternative.anchor);
  const order = shapes.propertyOrder(shape);
  const accepts = (value: JsonValue): boolean => registry.accepts(value, alternative.anchor);
  const along = <K extends 'keywords' | 'symbols' | 'references'>(
    member: K,
  ): ReturnType<typeof boundAlong<K>> => boundAlong(bindings, alternative.ancestry, member);

  // A keyword form: the alternative's own binding says which word introduces each of its members.
  const keywords = along('keywords');
  if (keywords !== undefined) {
    const operands = order
      .filter((member) => keywords.has(member))
      .map((member) => ({
        member,
        keyword: keywords.get(member) as string,
        anchor: unionAnchorOf(context, shapes.member(shape, member)),
      }));
    return [positional(KEYWORD, alternative, alternative.label, operands, accepts)];
  }

  // A tagged application: the *union*'s binding gives a symbol to the member this alternative is
  // written under — `all` prints `and`, `not` prints `not`, `present` prints `present(path)`.
  const tags = along('symbols');
  for (const member of order) {
    const symbol = tags?.get(member);
    if (symbol === undefined) continue;
    return [tagged(context, alternative, member, symbol, along('references'), accepts)];
  }

  // An application: the member whose own binding carries the symbols of an enumeration is the
  // operator, and the operands are what the other members hold.
  for (const member of order) {
    const memberShape = shapes.member(shape, member);
    const facts = factsOfShape(memberShape);
    const symbols = bindingOf(bindings, memberShape)?.symbols;
    if (facts.choices === null) continue;
    const rest = order.filter((one) => one !== member);
    return facts.choices
      .filter((choice): choice is string => typeof choice === 'string')
      .map((operator) =>
        applied(context, alternative, shape, member, operator, symbols?.get(operator), rest, accepts),
      );
  }

  // A member holding an object whose own members carry the operator: `{"compare": {…}}`. The
  // alternative's one member *is* the application, so the production reaches through it.
  if (order.length === 1) {
    const member = order[0] as string;
    const inner = shapes.member(shape, member);
    const innerOrder = shapes.propertyOrder(inner);
    for (const name of innerOrder) {
      const facts = factsOfShape(shapes.member(inner, name));
      if (facts.choices === null) continue;
      const symbols = bindingOf(bindings, shapes.member(inner, name))?.symbols;
      const rest = innerOrder.filter((one) => one !== name);
      return facts.choices
        .filter((choice): choice is string => typeof choice === 'string')
        .map((operator) =>
          nested(context, alternative, member, inner, name, operator, symbols?.get(operator), rest, accepts),
        );
    }
    // One member holding a scalar: a name the editor picks, or a literal it types.
    const facts = factsOfShape(inner);
    if (!facts.holdsObject && !facts.holdsArray && facts.members.length === 0) {
      return [payload(context, alternative, member, inner, facts, along('references'), accepts)];
    }
  }

  return [
    positional(
      GENERIC,
      alternative,
      alternative.label,
      order.map((member) => ({
        member,
        anchor: unionAnchorOf(context, shapes.member(shape, member)),
      })),
      accepts,
    ),
  ];
}

/** A production whose operands are written one per member, in the schema's order. */
function positional(
  kind: string,
  alternative: Alternative,
  label: string,
  operands: readonly Operand[],
  accepts: (value: JsonValue) => boolean,
): Production {
  return {
    kind,
    alternative,
    label,
    operands,
    parts(value: JsonValue) {
      if (!isJsonObject(value) || !accepts(value)) return null;
      const found: JsonValue[] = [];
      for (const operand of operands) {
        const held = getMember(value, operand.member);
        if (held === undefined) return null;
        found.push(held);
      }
      return found;
    },
    write(parts: readonly JsonValue[]) {
      return jsonObject(
        operands.map((operand, at): JsonMember => ({ name: operand.member, value: parts[at] ?? null })),
      );
    },
  };
}

/** A production whose one member holds a scalar the editor picks or types. */
function payload(
  context: LanguageContext,
  alternative: Alternative,
  member: string,
  shape: Shape,
  facts: SchemaFacts,
  references: readonly string[] | undefined,
  accepts: (value: JsonValue) => boolean,
): Production {
  const referent = references?.includes(member) === true ? member : undefined;
  return {
    kind: PAYLOAD,
    alternative,
    label: alternative.label,
    operands: [],
    payload: {
      named: referent !== undefined,
      ...(referent === undefined ? {} : { referent }),
      prefix: bindingOf(context.bindings, shape)?.prefix ?? '',
      facts,
    },
    parts(value: JsonValue) {
      if (!isJsonObject(value) || !accepts(value)) return null;
      const held = getMember(value, member);
      return held === undefined ? null : [held];
    },
    write(parts: readonly JsonValue[]) {
      return jsonObject([{ name: member, value: parts[0] ?? null }]);
    },
  };
}

/** A production the *union*'s binding names: `{"all": […]}`, `{"not": …}`, `{"present": "…"}`. */
function tagged(
  context: LanguageContext,
  alternative: Alternative,
  member: string,
  symbol: SymbolBinding,
  references: readonly string[] | undefined,
  accepts: (value: JsonValue) => boolean,
): Production {
  const { shapes, bindings } = context;
  const shape = shapes.member(shapes.at(alternative.anchor), member);
  const facts = factsOfShape(shape);
  const listed = facts.holdsArray || facts.listed;
  const shaped = facts.holdsObject || facts.members.length > 0 || facts.alternation;
  const referent = references?.includes(member) === true ? member : undefined;
  const list: OperandList | undefined = listed
    ? {
        anchor: unionAnchorOf(context, shapes.item(shape, 0)),
        minimum: facts.minItems ?? 0,
        maximum: facts.maxItems ?? Infinity,
      }
    : undefined;
  const operands: readonly Operand[] =
    !listed && shaped ? [{ member, anchor: unionAnchorOf(context, shape) }] : [];
  const held: Payload | undefined =
    listed || shaped
      ? undefined
      : {
          named: referent !== undefined,
          ...(referent === undefined ? {} : { referent }),
          prefix: bindingOf(bindings, shape)?.prefix ?? '',
          facts,
        };
  return {
    kind: APPLICATION,
    alternative,
    label: member,
    operator: member,
    symbol,
    operands,
    ...(list === undefined ? {} : { list }),
    ...(held === undefined ? {} : { payload: held }),
    parts(value: JsonValue) {
      if (!isJsonObject(value) || !accepts(value)) return null;
      const written = getMember(value, member);
      if (written === undefined) return null;
      return listed ? (isJsonArray(written) ? [...written] : null) : [written];
    },
    write(parts: readonly JsonValue[]) {
      return jsonObject([{ name: member, value: listed ? [...parts] : (parts[0] ?? null) }]);
    },
  };
}

/** An application: an operator member and the operands beside it. */
function applied(
  context: LanguageContext,
  alternative: Alternative,
  shape: Shape,
  operatorMember: string,
  operator: string,
  symbol: SymbolBinding | undefined,
  rest: readonly string[],
  accepts: (value: JsonValue) => boolean,
): Production {
  const { shapes } = context;
  const order = [operatorMember, ...rest];
  const listMember = rest.find((member) => {
    const facts = factsOfShape(shapes.member(shape, member));
    return facts.holdsArray || facts.listed;
  });
  const listShape = listMember === undefined ? null : shapes.member(shape, listMember);
  const listFacts = listShape === null ? null : factsOfShape(listShape);
  return {
    kind: APPLICATION,
    alternative,
    label: operator,
    operator,
    ...(symbol === undefined ? {} : { symbol }),
    operands:
      listMember === undefined
        ? rest.map((member) => ({ member, anchor: unionAnchorOf(context, shapes.member(shape, member)) }))
        : [],
    ...(listShape === null || listFacts === null
      ? {}
      : {
          list: {
            anchor: unionAnchorOf(context, shapes.item(listShape, 0)),
            minimum: listFacts.minItems ?? 0,
            maximum: listFacts.maxItems ?? Infinity,
          },
        }),
    parts(value: JsonValue) {
      if (!isJsonObject(value) || !accepts(value)) return null;
      if (getMember(value, operatorMember) !== operator) return null;
      if (listMember === undefined) {
        const found: JsonValue[] = [];
        for (const member of rest) {
          const held = getMember(value, member);
          if (held === undefined) return null;
          found.push(held);
        }
        return found;
      }
      const held = getMember(value, listMember);
      return held !== undefined && isJsonArray(held) ? [...held] : null;
    },
    write(parts: readonly JsonValue[]) {
      const members: JsonMember[] = order.map((member): JsonMember => {
        if (member === operatorMember) return { name: member, value: operator };
        if (member === listMember) return { name: member, value: [...parts] };
        return { name: member, value: parts[rest.indexOf(member)] ?? null };
      });
      return jsonObject(members);
    },
  };
}

/** An application written one object in: `{"compare": {"operator": …, "left": …, "right": …}}`. */
function nested(
  context: LanguageContext,
  alternative: Alternative,
  member: string,
  inner: Shape,
  operatorMember: string,
  operator: string,
  symbol: SymbolBinding | undefined,
  rest: readonly string[],
  accepts: (value: JsonValue) => boolean,
): Production {
  const { shapes } = context;
  const order = [operatorMember, ...rest];
  return {
    kind: APPLICATION,
    alternative,
    label: operator,
    operator,
    ...(symbol === undefined ? {} : { symbol }),
    operands: rest.map((name) => ({
      member: name,
      anchor: unionAnchorOf(context, shapes.member(inner, name)),
    })),
    parts(value: JsonValue) {
      if (!isJsonObject(value) || !accepts(value)) return null;
      const held = getMember(value, member);
      if (held === undefined || !isJsonObject(held)) return null;
      if (getMember(held, operatorMember) !== operator) return null;
      const found: JsonValue[] = [];
      for (const name of rest) {
        const one = getMember(held, name);
        if (one === undefined) return null;
        found.push(one);
      }
      return found;
    },
    write(parts: readonly JsonValue[]) {
      const members: JsonMember[] = order.map((name): JsonMember =>
        name === operatorMember
          ? { name, value: operator }
          : { name, value: parts[rest.indexOf(name)] ?? null },
      );
      return jsonObject([{ name: member, value: jsonObject(members) }]);
    },
  };
}

/**
 * Every symbol, keyword and prefix reachable from an anchor, which is what the lexer has to cut.
 *
 * A comparison's symbols are written at the *condition* anchor and its operands at the
 * expression one, so a lexing of a condition has to know both — the closure is over the operand
 * anchors, gathered from the grammars themselves rather than from a list. The prefixes are here
 * for the same reason a symbol is: `$` is a mark of the text form wherever `$layer` can stand,
 * and no rule of the lexer would find it.
 */
function reachableMarks(context: LanguageContext, anchor: string): readonly string[] {
  const found = new Set<string>();
  const seen = new Set<string>([anchor]);
  const pending = [anchor];
  while (pending.length > 0) {
    const one = pending.pop() as string;
    for (const production of grammarAt(context, one).productions) {
      if (production.symbol !== undefined) found.add(production.symbol.text);
      if (production.payload !== undefined && production.payload.prefix !== '') {
        found.add(production.payload.prefix);
      }
      for (const operand of production.operands) {
        if (operand.keyword !== undefined) found.add(operand.keyword);
        if (!seen.has(operand.anchor)) {
          seen.add(operand.anchor);
          pending.push(operand.anchor);
        }
      }
      const list = production.list;
      if (list !== undefined && !seen.has(list.anchor)) {
        seen.add(list.anchor);
        pending.push(list.anchor);
      }
    }
  }
  return [...found];
}

/**
 * How tightly a keyword form binds: less than every symbol, so that one nested in an application
 * is parenthesised. The number is not `presentation.json`'s — a keyword form has no infix symbol
 * to carry one — it is the statement that `if … then … else …` reaches as far as it can, which is
 * what makes `(if c then a else b) + 1` the only reading of that document's text.
 */
export const LOOSEST = 0;

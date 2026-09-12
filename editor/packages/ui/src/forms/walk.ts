/**
 * The generic schema walker: a `$def` and a value become rows.
 *
 * This is plan §1's first consequence made true rather than aspirational — "**Every form is
 * generated.** One schema walker renders any `$def`" — and it is the only walker there is. The
 * instance sheet, the argument sheet (§4.12), every sheet of §4.11, the location editor, the
 * dialogs of §4.4 and the primitive editor over the unit schema's `primitive_primitive` (§4.22,
 * D15) are this function with a different anchor.
 *
 * **What it reads, and from where.**
 *
 * | Question | Answered by |
 * |---|---|
 * | which subschemas describe a place, through `$ref` and `allOf` | the store's `SchemaShapes` (feature 2.1) |
 * | what one of them asserts — type, `enum`, `const`, bounds, `description` | the core's `factsOf` |
 * | which alternatives a union offers and what labels each | the core's `vocabulary()` |
 * | which alternative a value *is*, and whether a conditional branch holds | the core's `registry.accepts` — Ajv on that place of that schema |
 * | which editor, picker, referent or figure format a place gets | `presentation.json` (feature 2.2) |
 *
 * Nothing else. No member name, no enum value and no operator of the language is written in this
 * file or in any other of `packages/ui/src` but the one data file §1 admits — catching rule (b)
 * is a scan that says so — and the four cases the plan puts hardest (`location`'s four-way
 * recursion, `quantity_definition`'s conditional `domain`, `argument_value`'s recursion, a map
 * with `propertyNames`) are rendered by the same walk as everything else.
 *
 * **What it does not do.** It states no verdict: `describe` says whether an argument applies and
 * what its default resolves to, `validate` says what is wrong, and both are the core's (inventory
 * §7 — "where a component must ask, not compute"). A row carries the problem it was handed and
 * the shape of the place; it never re-derives either. And it refuses nothing: a value that
 * matches no alternative, a member no schema declares and a place the schema says nothing about
 * all produce a row (Q5, D5 — the author wires first and fixes afterwards).
 */
import {
  isJsonArray,
  isJsonNumber,
  isJsonObject,
  mergeFacts,
  type JsonValue,
  type PathSegment,
  type SchemaFacts,
  type SchemaObject,
  type SchemaRegistry,
  type VocabularyValue,
} from '@tensorspine/lang';
import { pointerOf, SchemaShapes, type Shape } from '@tensorspine/store';

import { boundAlong, presentation } from '../presentation/load.js';
import type { Binding, Presentation } from '../presentation/types.js';
import {
  alternationAt,
  chosenOf,
  resembling,
  type Alternation,
  type Alternative,
} from './alternatives.js';
import type {
  Form,
  FormBounds,
  FormCondition,
  FormKeys,
  FormMode,
  FormNote,
  FormOption,
  FormProblem,
  FormRow,
} from './types.js';
import { editsOneValue, JSON_EDITOR, widgetOf } from './widget.js';

/** What a form is generated against: the schemas, their reading, and the bindings. */
export interface FormContext {
  readonly registry: SchemaRegistry;
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
  /** The flattened unions by anchor, built once per context as the shapes are. */
  readonly unions: Map<string, Alternation | null>;
}

/**
 * A context over a registry, with the readings that cost something built once.
 *
 * `SchemaShapes` memoises every `$ref` it follows and every union it expands, the registry keeps
 * its vocabulary (feature 2.2 measured that walk at 1–2.5 ms — nothing at startup, a trap per
 * render), and the flattened unions are kept here. A context is made once per registry and
 * shared by every sheet; it is the only supported way to build one.
 */
export function formContext(registry: SchemaRegistry, bindings?: Presentation): FormContext {
  return {
    registry,
    shapes: new SchemaShapes(registry),
    bindings: bindings ?? presentation(),
    unions: new Map<string, Alternation | null>(),
  };
}

/** One context per registry, so every sheet, table and command shares the walker's caches. */
const CONTEXTS = new WeakMap<SchemaRegistry, FormContext>();

/**
 * The context of a registry, built once and kept.
 *
 * `formContext` makes a new one — the shapes, the vocabulary's readings and the flattened unions
 * with it — and a component that built one per render would pay feature 2.2's measured 1–2.5 ms
 * for the vocabulary alone on every keystroke. The sheets, the tables and the commands of §4.4 all
 * read the same schemas, so they read the same context.
 */
export function formsFor(registry: SchemaRegistry, bindings?: Presentation): FormContext {
  const held = CONTEXTS.get(registry);
  if (held !== undefined) return held;
  const made = formContext(registry, bindings);
  CONTEXTS.set(registry, made);
  return made;
}

/** What a form is asked for. */
export interface FormRequest {
  /** Where it starts: `<$id>#<pointer>`, as a presentation binding is keyed. */
  readonly anchor?: string;
  /** Or the role of a whole document — `model`, `primitive-library-unit`. */
  readonly role?: string;
  /**
   * Or the reading of a place already stepped down to, which is what a sheet of one *value* has.
   *
   * The argument sheet asks for the form of the value at `/…/instances/attn/arguments/heads`, and
   * the way it knows that place is `argument_value` is by having walked there from the document's
   * root (`SchemaShapes.step`, as the canvas does for a guard) — never by writing the anchor down,
   * which would be an item of information the schema states (§1). A caller with a `Shape` hands it
   * over; a caller with an anchor or a role names one.
   */
  readonly shape?: Shape;
  /** The value it edits, as the ordered tree; absent for the form of a `$def` alone. */
  readonly value?: JsonValue;
  /** The label of the root row; the last segment of the anchor otherwise. */
  readonly label?: string;
  /** The problems to put on the rows, by JSON pointer relative to the value. */
  readonly problems?: readonly FormProblem[];
  /** How deep the walk goes. A value is finite, so this is a backstop and not the rule. */
  readonly limit?: number;
}

/** How deep a walk goes before it stops, whatever the value says. */
const LIMIT = 24;

/** A union that cannot be told apart by its labels, and nothing is bound to edit it. */
export const UNDISCRIMINATED = 'undiscriminated';
/** An alternative with no required key, no type and no `$ref`: nothing to label a chooser with. */
export const UNTAGGED = 'untagged';
/** A place whose schema states nothing a widget can be chosen from. */
export const UNREADABLE = 'unreadable';
/** A member of the value no schema of the place declares. */
export const UNDECLARED = 'undeclared';
/** A value more than one alternative of a union accepts. */
export const AMBIGUOUS = 'ambiguous';

/** The rows a `$def` and a value become, and what the walker had no reading for. */
export function formOf(context: FormContext, request: FormRequest): Form {
  const rows: FormRow[] = [];
  const notes: FormNote[] = [];
  const shape =
    request.shape ??
    (request.anchor !== undefined
      ? context.shapes.at(request.anchor)
      : context.shapes.root(request.role ?? ''));
  new Walk(context, rows, notes, request).visit({
    shape,
    value: request.value,
    steps: [],
    label:
      request.label ??
      nameOf(request.anchor ?? request.shape?.direct[0]?.anchor ?? request.role ?? ''),
    pinned: request.label !== undefined,
    required: false,
    entered: [],
  });
  return { rows, notes };
}

/** A mode, and whether a presentation binding is what names it. */
interface Named {
  readonly mode: FormMode;
  readonly editor: boolean;
}

/**
 * Modes one editor owns are one mode.
 *
 * They arrive with that editor's name as their tag (see `modeOf`), so the grouping is by name and
 * the anchors accumulate: one `expression` mode over `scalar_expression`'s four shaped forms, one
 * `condition` mode over a condition's four. **Only an editor collapses anything.** A generic
 * widget never does — `location`'s four shapes are all sections and stay four modes — and two
 * alternatives a generic chooser would label the same stay two, which is the state the row
 * reports rather than hides.
 */
function collapse(modes: readonly Named[]): readonly FormMode[] {
  const found: FormMode[] = [];
  for (const { mode, editor } of modes) {
    const already = editor
      ? found.find((one) => !one.inline && one.tag === mode.tag && one.widget === mode.widget)
      : undefined;
    if (already === undefined) {
      found.push(mode);
      continue;
    }
    const into = already as Mutable<FormMode>;
    into.anchors = [...already.anchors, ...mode.anchors];
    into.unbound = [...already.unbound, ...mode.unbound];
  }
  return found;
}

/** The last segment of an anchor or a role: what a form is called when the caller says nothing. */
function nameOf(anchor: string): string {
  const hash = anchor.indexOf('#');
  const pointer = hash < 0 ? anchor : anchor.slice(hash + 1);
  const slash = pointer.lastIndexOf('/');
  return slash < 0 ? pointer : pointer.slice(slash + 1);
}

/** The scalar a tree node is, or `undefined` when it is an object or an array. */
function scalarOf(value: JsonValue | undefined): VocabularyValue | undefined {
  if (value === undefined) return undefined;
  if (isJsonNumber(value)) return value.value;
  if (isJsonObject(value) || isJsonArray(value)) return undefined;
  return value;
}

/** The member of an object tree, or `undefined`. */
function memberOf(value: JsonValue | undefined, name: string): JsonValue | undefined {
  if (value === undefined || !isJsonObject(value)) return undefined;
  return value.members.find((member) => member.name === name)?.value;
}

/** What a place is asked to render. */
interface Visit {
  readonly shape: Shape;
  readonly value: JsonValue | undefined;
  readonly steps: readonly PathSegment[];
  readonly label: string;
  /** Whether the caller fixed the label, so the schema's own `title` does not replace it. */
  readonly pinned?: boolean;
  readonly required: boolean;
  readonly conditions?: readonly FormCondition[];
  /** The anchors already on the way down, so a schematic walk stops where a definition repeats. */
  readonly entered: readonly string[];
}

/** One branch of a conditional or of an `anyOf`, with what it adds to the place. */
interface Branch {
  readonly anchor: string;
  readonly test: string;
  readonly holds: boolean | null;
  readonly facts: SchemaFacts;
}

/** A row under construction: the rows themselves are read-only. */
type Mutable<T> = { -readonly [K in keyof T]: T[K] };

class Walk {
  private readonly limit: number;

  constructor(
    private readonly context: FormContext,
    private readonly rows: FormRow[],
    private readonly notes: FormNote[],
    private readonly request: FormRequest,
  ) {
    this.limit = request.limit ?? LIMIT;
  }

  visit(visit: Visit): void {
    const shape = visit.shape;
    const path = pointerOf(visit.steps);
    // The chain, which falls back to the alternatives where a place has no reading of its own: a
    // member declared only inside an alternative — `theta` under an argument's `record`, `unit`
    // under `argument_type` — is described by those and by nothing else, and a row that took
    // `direct` alone would carry no anchors, hence no binding and no source modes at all.
    const anchors = chainOf(shape).map((place) => place.anchor);
    const facts = this.factsOfShape(shape);
    const binding = this.context.bindings.firstOf(anchors);
    const alternation = this.alternationOf(anchors);
    const widget = binding?.widget ?? widgetOf(facts, alternation !== undefined);
    const chosen =
      alternation === undefined ? null : chosenOf(this.context.registry, alternation, visit.value);
    // Which alternative a value **is** is Ajv's answer; which alternative's rows a form *draws*
    // is that answer where there is one, and the alternative the value resembles where there is
    // none. A chooser writes the blank of the alternative it was set to, and a blank identifier
    // is no `stack` to Ajv: without this, choosing a form would hide the very rows that repair it.
    const drawn =
      chosen?.alternative ??
      (alternation === undefined ? undefined : resembling(alternation, visit.value));

    const row: Mutable<FormRow> = {
      path,
      steps: visit.steps,
      depth: visit.steps.length,
      label: visit.pinned === true ? visit.label : (facts.title ?? visit.label),
      widget,
      anchors,
      required: visit.required,
      present: visit.value !== undefined,
    };
    if (visit.conditions !== undefined && visit.conditions.length > 0) {
      row.conditions = visit.conditions;
    }
    if (facts.description !== null) row.description = facts.description;
    if (facts.choices !== null) row.options = optionsOf(facts.choices);
    if (facts.fixed && facts.constant !== undefined) row.constant = facts.constant;
    const bounds = boundsOf(facts);
    if (bounds !== undefined) row.bounds = bounds;
    if (facts.named) {
      const keys = this.keysOf(shape);
      if (keys !== undefined) row.keys = keys;
    }
    if (binding?.picker !== undefined) row.picker = binding.picker;
    if (binding?.create !== undefined) row.create = binding.create;
    if (binding?.format !== undefined) row.format = binding.format;
    if (alternation !== undefined) {
      const modes = collapse(alternation.alternatives.map((one) => this.modeOf(one, binding)));
      row.modes = modes;
      if (drawn !== undefined) {
        const mode = modes.find((one) => one.anchors.includes(drawn.anchor));
        if (mode !== undefined) row.mode = mode.tag;
        if (mode?.referent !== undefined) row.referent = mode.referent;
        if (mode?.inline === true && mode.member !== undefined) {
          const written = scalarOf(memberOf(visit.value, mode.member));
          if (written !== undefined) row.written = written;
        }
      }
    } else if (editsOneValue(widget)) {
      const written = scalarOf(visit.value);
      if (written !== undefined) row.written = written;
    }
    const problem = this.request.problems?.find((one) => one.path === path);
    if (problem !== undefined) row.error = problem.message;
    const generic = this.genericReason(path, anchors, facts, alternation, row.modes, chosen, binding);
    if (generic !== undefined) row.generic = generic;

    this.rows.push(row);

    if (binding?.widget !== undefined) return; // the bound editor owns its subtree.
    if (visit.steps.length >= this.limit) return;
    if (alternation !== undefined) {
      this.descendIntoMode(visit, row.modes ?? [], drawn);
      return;
    }
    this.descend(visit, facts, widget);
  }

  /**
   * The facts of a shape: its chain, most specific first.
   *
   * `direct` is what the place is by definition, and it is empty for a member only a branch
   * declares — `unit` under `argument_type`, which exists when `kind` is `physical` and nowhere
   * else. There the branches are the only reading there is, so `all` answers; the store's own
   * rule (a constraint written inside a branch is not the place's) is about `constraintsOf`, and
   * is untouched.
   */
  private factsOfShape(shape: Shape): SchemaFacts {
    return factsOfShape(shape);
  }

  /** The union at the first anchor of the chain that names one, flattened and kept. */
  private alternationOf(anchors: readonly string[]): Alternation | undefined {
    const vocabulary = this.context.registry.vocabulary();
    for (const anchor of anchors) {
      const known = this.context.unions.get(anchor);
      if (known !== undefined) {
        if (known !== null) return known;
        continue;
      }
      const found = alternationAt(vocabulary, this.context.shapes, anchor) ?? null;
      this.context.unions.set(anchor, found);
      if (found !== null) return found;
    }
    return undefined;
  }

  /**
   * One alternative as a source mode (plan §4.12).
   *
   * A mode written as one scalar member is edited **on the row itself** — `{"quantity": "d"}` is
   * the row's value and its mode at once, which is what §4.12's table and the S6 row-states key
   * draw. A shaped one is edited by the widget bound to the union it came from, and takes that
   * editor's name: which is how `scalar_expression`'s three operation forms and its conditional,
   * the four the vocabulary cannot tell apart by their required keys (feature 1.1), become the
   * single `expression` mode §4.12 lists. Nothing else is edited by a name the schema does not
   * carry.
   */
  private modeOf(alternative: Alternative, outer: Binding | undefined): Named {
    const { shapes, bindings } = this.context;
    const shape = shapes.at(alternative.anchor);
    const facts = this.factsOfShape(shape);
    const only = facts.members.length === 1 ? facts.members[0] : undefined;
    let member: { widget: string; facts: SchemaFacts } | undefined;
    if (only !== undefined) {
      const inner = shapes.member(shape, only);
      const innerAnchors = inner.direct.map((place) => place.anchor);
      const innerFacts = this.factsOfShape(inner);
      member = {
        facts: innerFacts,
        widget:
          bindings.firstOf(innerAnchors)?.widget ??
          widgetOf(innerFacts, this.alternationOf(innerAnchors) !== undefined),
      };
    }
    const inline =
      only !== undefined && facts.closed && member !== undefined && editsOneValue(member.widget);
    // The editor is asked for *along* the chain rather than taken from its first binding: an
    // alternative may carry a binding of its own — `conditional_expression`'s keywords — while
    // the editor that owns it is bound at the union one step along (§4.13).
    const editor = inline ? undefined : boundAlong(bindings, alternative.ancestry, 'widget');
    const tag = alternative.tags.length === 1 ? alternative.tags[0] : undefined;
    const symbols = bindings.at(alternative.union)?.symbols;
    // The same three conditions the presentation audit calls "named" (plan §1 (a)): a symbol at
    // the union for one of its tags, a binding at the alternative's own place, or one at what its
    // `$ref` names. Read here so that a chooser and the log never disagree about one alternative.
    const named =
      bindings.at(alternative.anchor) !== undefined ||
      bindings.at(alternative.place) !== undefined ||
      alternative.tags.some((one) => symbols?.has(one) === true);
    const mode: Mutable<FormMode> = {
      tag: editor ?? alternative.label,
      anchors: [alternative.anchor],
      union: alternative.union,
      widget: inline ? (member?.widget ?? JSON_EDITOR) : (editor ?? widgetOf(facts, false)),
      inline,
      unbound: named ? [] : [alternative.anchor],
    };
    if (only !== undefined) mode.member = only;
    const references = boundAlong(bindings, alternative.ancestry, 'references') ?? outer?.references;
    if (tag !== undefined && references?.includes(tag) === true) mode.referent = tag;
    if (inline && member?.facts.choices != null) mode.options = optionsOf(member.facts.choices);
    return { mode, editor: editor !== undefined };
  }

  /** What a map's names must be, read from its `propertyNames`. */
  private keysOf(shape: Shape): FormKeys | undefined {
    const keys = this.context.shapes.keys(shape);
    const place = keys.direct[keys.direct.length - 1];
    if (place === undefined) return undefined;
    const facts = mergeFacts(keys.direct.map((one) => one.node));
    const found: Mutable<FormKeys> = { anchor: place.anchor };
    if (facts.pattern !== null) found.pattern = facts.pattern;
    if (facts.minLength !== null) found.minLength = facts.minLength;
    if (facts.maxLength !== null) found.maxLength = facts.maxLength;
    return found;
  }

  /** A chooser descends into the alternative the value is, and into no other. */
  private descendIntoMode(
    visit: Visit,
    modes: readonly FormMode[],
    alternative: Alternative | undefined,
  ): void {
    if (alternative === undefined) return;
    const mode = modes.find((one) => one.anchors.includes(alternative.anchor));
    if (mode === undefined || mode.inline) return;
    const binding = this.context.bindings.firstOf(alternative.ancestry);
    if (binding?.widget !== undefined) return; // the expression or the condition editor owns it.
    const shape = this.context.shapes.at(alternative.anchor);
    this.descend({ ...visit, shape }, this.factsOfShape(shape), mode.widget);
  }

  /** The rows below a place: its members, its map entries or its elements. */
  private descend(visit: Visit, facts: SchemaFacts, widget: string): void {
    const { shapes } = this.context;
    const shape = visit.shape;
    const value = visit.value;
    const entered = [...visit.entered, ...chainOf(shape).map((place) => place.anchor)];
    if (facts.keyed) {
      if (value === undefined || !isJsonObject(value)) return;
      const inner = shapes.values(shape);
      for (const member of value.members) {
        this.visit({
          shape: inner,
          value: member.value,
          steps: [...visit.steps, member.name],
          label: member.name,
          pinned: true,
          required: false,
          entered,
        });
      }
      return;
    }
    if (facts.holdsArray || facts.listed) {
      if (value === undefined || !isJsonArray(value)) return;
      value.forEach((element, index) => {
        this.visit({
          shape: shapes.item(shape, index),
          value: element,
          steps: [...visit.steps, index],
          label: String(index),
          pinned: true,
          required: false,
          entered,
        });
      });
      return;
    }
    if (widget === JSON_EDITOR) return;
    const order = memberOrder(shape);
    const branches = this.branchesOf(shape, value);
    const required = shapes.constraintsOf(shape).required;
    for (const name of order) {
      const member = shapes.member(shape, name);
      // With no value the walk is schematic and a recursive definition would never end; with one
      // it ends because the value does. `location`'s `stack.part` is a `location` again, and the
      // corpus writes one.
      if (value === undefined && chainOf(member).some((place) => entered.includes(place.anchor))) {
        continue;
      }
      const conditions = conditionsOf(branches, name, facts, required.has(name));
      this.visit({
        shape: member,
        value: memberOf(value, name),
        steps: [...visit.steps, name],
        label: name,
        required: required.has(name),
        ...(conditions.length > 0 ? { conditions } : {}),
        entered,
      });
    }
    if (value === undefined || !isJsonObject(value)) return;
    for (const member of value.members) {
      if (order.includes(member.name)) continue;
      this.undeclared(shape, visit.steps, member.name);
    }
  }

  /** A member the value writes and no schema of the place declares: the generic JSON row. */
  private undeclared(shape: Shape, steps: readonly PathSegment[], name: string): void {
    const path = pointerOf([...steps, name]);
    this.rows.push({
      path,
      steps: [...steps, name],
      depth: steps.length + 1,
      label: name,
      widget: JSON_EDITOR,
      anchors: [],
      required: false,
      present: true,
      generic: UNDECLARED,
    });
    this.notes.push({
      code: UNDECLARED,
      path,
      anchor: shape.direct[0]?.anchor ?? '',
      message: `'${name}' is written here and the schema declares no such member`,
    });
  }

  /** The `if`/`then`/`else` and `anyOf` branches of a place, with whether each one holds. */
  private branchesOf(shape: Shape, value: JsonValue | undefined): readonly Branch[] {
    const found: Branch[] = [];
    const holds = (test: string, negated: boolean): boolean | null => {
      if (value === undefined) return null;
      const accepted = this.context.registry.accepts(value, test);
      return negated ? !accepted : accepted;
    };
    for (const place of shape.direct) {
      if (place.node['if'] !== undefined) {
        const test = `${place.anchor}/if`;
        for (const keyword of ['then', 'else'] as const) {
          const branch = place.node[keyword];
          if (branch === undefined || branch === true || branch === false) continue;
          const anchor = `${place.anchor}/${keyword}`;
          found.push({
            anchor,
            test,
            holds: holds(test, keyword === 'else'),
            facts: this.factsAt(anchor),
          });
        }
      }
      const any = place.node['anyOf'];
      if (Array.isArray(any)) {
        any.forEach((_, index) => {
          const anchor = `${place.anchor}/anyOf/${String(index)}`;
          found.push({
            anchor,
            test: anchor,
            holds: holds(anchor, false),
            facts: this.factsAt(anchor),
          });
        });
      }
    }
    return found;
  }

  /** The facts of the place an anchor names, its `$ref` chain and its `allOf` followed. */
  private factsAt(anchor: string): SchemaFacts {
    return this.factsOfShape(this.context.shapes.at(anchor));
  }

  /** Why a row renders generically, when it does (plan §1's last consequence). */
  private genericReason(
    path: string,
    anchors: readonly string[],
    facts: SchemaFacts,
    alternation: Alternation | undefined,
    modes: readonly FormMode[] | undefined,
    chosen: { readonly ambiguous: boolean } | null,
    binding: Binding | undefined,
  ): string | undefined {
    if (alternation !== undefined && modes !== undefined) {
      const untagged = modes.filter((one) => one.tag === '');
      if (untagged.length > 0) {
        this.note(UNTAGGED, path, alternation.anchor, () => {
          const count = `${String(untagged.length)} of ${String(modes.length)}`;
          return `${count} alternatives carry no required key and no type, so a chooser has nothing to label them by`;
        });
        return UNTAGGED;
      }
      const tags = new Set(modes.map((one) => one.tag));
      if (tags.size !== modes.length) {
        this.note(UNDISCRIMINATED, path, alternation.anchor, () =>
          'its alternatives share their required keys and no member fixes a constant, so no chooser tells them apart',
        );
        return UNDISCRIMINATED;
      }
      if (chosen?.ambiguous === true) {
        this.note(AMBIGUOUS, path, alternation.anchor, () =>
          'more than one alternative accepts the value written here',
        );
        return AMBIGUOUS;
      }
      return undefined;
    }
    if (!facts.states && binding?.widget === undefined) {
      this.note(UNREADABLE, path, anchors[0] ?? '', () =>
        'the schema states nothing here, so the place gets the generic JSON row',
      );
      return UNREADABLE;
    }
    return undefined;
  }

  /**
   * One note per construct, not per occurrence.
   *
   * Plan §1 lists what renders generically so that somebody can look it up, and a construct that
   * appears at three hundred places of a document is one thing to look up, not three hundred.
   * The path recorded is the first place it was met, which is where to go and see it.
   */
  private note(code: string, path: string, anchor: string, message: () => string): void {
    if (this.notes.some((one) => one.code === code && one.anchor === anchor)) return;
    this.notes.push({ code, path, anchor, message: message() });
  }
}

/**
 * The chain of a shape: what the place is by definition, or what its branches say when it is
 * nothing by definition.
 *
 * `primitive_definition` is the case: it declares no member of its own and reaches
 * `template_primitive` or `primitive_primitive` through `if`/`then`/`else`, so `direct` names
 * only the definition itself and every member of the form is reached through `all`. Reading the
 * branches there is the only reading available — and it is what the walk enters, so a definition
 * that comes round again is met and stopped.
 */
export function chainOf(shape: Shape): readonly { readonly anchor: string; readonly node: SchemaObject }[] {
  return shape.direct.length > 0 ? shape.direct : shape.all;
}

/**
 * What a place asserts, read from its chain most specific first.
 *
 * Exported because the expression grammar of §4.13 asks the same question of the same places —
 * whether a member holds a list, an object or a scalar, what its enumeration is, what bounds its
 * arity has — and a second reading of a chain is a second answer waiting to disagree with this
 * one.
 */
export function factsOfShape(shape: Shape): SchemaFacts {
  return mergeFacts(chainOf(shape).map((place) => place.node));
}

/** The options of a select: the schema's `enum`, in its own order, each labelled by its value. */
function optionsOf(choices: readonly VocabularyValue[]): readonly FormOption[] {
  return choices.map((value) => ({
    value,
    label: typeof value === 'string' ? value : JSON.stringify(value),
  }));
}

/** The bounds a row shows, or `undefined` when the schema states none. */
function boundsOf(facts: SchemaFacts): FormBounds | undefined {
  const bounds: Mutable<FormBounds> = {};
  if (facts.minimum !== null) {
    bounds.minimum = facts.minimum;
    if (facts.minimumExcluded) bounds.minimumExcluded = true;
  }
  if (facts.maximum !== null) {
    bounds.maximum = facts.maximum;
    if (facts.maximumExcluded) bounds.maximumExcluded = true;
  }
  if (facts.minLength !== null) bounds.minLength = facts.minLength;
  if (facts.maxLength !== null) bounds.maxLength = facts.maxLength;
  if (facts.pattern !== null) bounds.pattern = facts.pattern;
  if (facts.minItems !== null) bounds.minItems = facts.minItems;
  if (facts.maxItems !== null) bounds.maxItems = facts.maxItems;
  if (facts.uniqueItems) bounds.uniqueItems = true;
  if (facts.minProperties !== null) bounds.minProperties = facts.minProperties;
  if (facts.maxProperties !== null) bounds.maxProperties = facts.maxProperties;
  return Object.keys(bounds).length === 0 ? undefined : bounds;
}

/**
 * The members of a place, in the order the schema declares them.
 *
 * Declaration order, and not the store's `propertyOrder`: a command writes a new object's
 * required members first, which is the order the schema states the object's *shape* in, while a
 * form keeps "declaration order … the library author's order" (D7, §4.12). A member a branch
 * requires without declaring comes last, since the schema gives it no place of its own.
 */
export function memberOrder(shape: Shape): readonly string[] {
  const found: string[] = [];
  const add = (name: string): void => {
    if (!found.includes(name)) found.push(name);
  };
  for (const place of shape.all) for (const name of memberNamesOf(place.node)) add(name);
  for (const place of shape.all) {
    const required = place.node['required'];
    if (Array.isArray(required)) {
      for (const name of required) if (typeof name === 'string') add(name);
    }
  }
  return found;
}

/** The names a node's `properties` declares, in the order it writes them. */
function memberNamesOf(node: SchemaObject): readonly string[] {
  const declared = node['properties'];
  if (declared === null || typeof declared !== 'object' || Array.isArray(declared)) return [];
  return Object.keys(declared);
}

/**
 * The conditions under which a member is declared or required.
 *
 * Only what a branch *adds* is a condition: a member the place itself declares and requires is
 * required, full stop, even when a branch says so again. That is what keeps
 * `quantity_definition`'s `domain` — declared by the place, required only by the `then` — the one
 * conditional row of that form, and `type` and `source` plain required ones.
 */
function conditionsOf(
  branches: readonly Branch[],
  name: string,
  facts: SchemaFacts,
  required: boolean,
): readonly FormCondition[] {
  const found: FormCondition[] = [];
  for (const branch of branches) {
    const declares = branch.facts.members.includes(name);
    const requires = branch.facts.required.includes(name);
    if (!declares && !requires) continue;
    if (requires && required) continue;
    if (declares && !requires && facts.members.includes(name)) continue;
    found.push({ branch: branch.anchor, test: branch.test, holds: branch.holds, requires, declares });
  }
  return found;
}

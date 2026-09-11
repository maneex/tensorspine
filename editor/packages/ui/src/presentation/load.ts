/**
 * Reading `presentation.json`.
 *
 * The file is the one data file plan §1 admits in the interface, and it is read once at startup:
 * {@link readPresentation} turns its text (already parsed) into the bindings, refusing anything
 * the format does not admit, and {@link presentation} is the shipped file read that way.
 *
 * What it refuses and what it lets through is the decision stated in `./types.ts`: a member name
 * the format does not carry is a refusal, because a mistyped member would silently do nothing;
 * a member *value* the interface does not know is not, because "unknown constructs get the
 * generic widget" (§1) is the same answer for an unknown rendering as for an unknown construct.
 * The audit of §1 (a) is what then says whether each key names a place of the loaded schemas.
 */
import source from '../presentation.json';

import type {
  Binding,
  Presentation,
  ReferenceRule,
  StatusBarField,
  SymbolBinding,
} from './types.js';

/** Raised when `presentation.json` is not a presentation file. */
export class PresentationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PresentationError';
  }
}

/** The members a binding may carry. A name outside this set is a refusal. */
const MEMBERS = [
  'role',
  'side',
  'face',
  'structuralSummary',
  'label',
  'widget',
  'references',
  'picker',
  'create',
  'symbols',
  'format',
  'statusBar',
  'declares',
  'scope',
  'refers',
] as const;

/** The members a reference rule may carry. */
const RULE_MEMBERS = ['tag', 'kind', 'under', 'scopedBy', 'without'] as const;

/** The members a symbol may carry; both are required. */
const SYMBOL_MEMBERS = ['text', 'form'] as const;

/** The members a status-bar mark may carry; both are required. */
const FIELD_MEMBERS = ['order', 'label'] as const;

/** The members whose value is a list of strings. */
const LISTS = new Set<string>(['face', 'references']);

/** The members whose value is a string. */
const STRINGS = new Set<string>([
  'role',
  'side',
  'label',
  'widget',
  'picker',
  'create',
  'format',
  'declares',
  'scope',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** A list of strings, or a refusal naming where it was expected. */
function stringsOf(value: unknown, where: string): readonly string[] {
  if (!Array.isArray(value)) throw new PresentationError(`${where}: expected a list of names`);
  return value.map((one, index) => {
    if (typeof one !== 'string') {
      throw new PresentationError(`${where}[${String(index)}]: expected a name`);
    }
    return one;
  });
}

/** One symbol: what is printed, and where. */
function symbolOf(value: unknown, where: string): SymbolBinding {
  if (!isRecord(value)) throw new PresentationError(`${where}: expected a symbol`);
  for (const member of Object.keys(value)) {
    if (!(SYMBOL_MEMBERS as readonly string[]).includes(member)) {
      throw new PresentationError(`${where}: a symbol carries no member '${member}'`);
    }
  }
  const printed: unknown = value['text'];
  const form: unknown = value['form'];
  if (typeof printed !== 'string' || printed === '') {
    throw new PresentationError(`${where}: a symbol needs the text it prints as`);
  }
  if (typeof form !== 'string' || form === '') {
    throw new PresentationError(`${where}: a symbol needs the place it prints in`);
  }
  return { text: printed, form };
}

/** One status-bar mark: where the figure sits in the bar, and what is written beside it. */
function fieldOf(value: unknown, where: string): StatusBarField {
  if (!isRecord(value)) throw new PresentationError(`${where}: expected a status-bar field`);
  for (const member of Object.keys(value)) {
    if (!(FIELD_MEMBERS as readonly string[]).includes(member)) {
      throw new PresentationError(`${where}: a status-bar field carries no member '${member}'`);
    }
  }
  const order: unknown = value['order'];
  const label: unknown = value['label'];
  if (typeof order !== 'number' || !Number.isInteger(order) || order < 1) {
    throw new PresentationError(`${where}: a status-bar field needs the place it sits in`);
  }
  if (typeof label !== 'string' || label === '') {
    throw new PresentationError(`${where}: a status-bar field needs the label written beside it`);
  }
  return { order, label };
}

/** One reference rule. */
function ruleOf(value: unknown, where: string): ReferenceRule {
  if (!isRecord(value)) throw new PresentationError(`${where}: expected a reference rule`);
  for (const member of Object.keys(value)) {
    if (!(RULE_MEMBERS as readonly string[]).includes(member)) {
      throw new PresentationError(`${where}: a reference rule carries no member '${member}'`);
    }
  }
  const tag: unknown = value['tag'];
  if (typeof tag !== 'string' || tag === '') {
    throw new PresentationError(`${where}: a reference rule needs the member a name is written under`);
  }
  const beside: { -readonly [K in Exclude<keyof ReferenceRule, 'tag'>]?: ReferenceRule[K] } = {};
  for (const member of ['kind', 'under', 'scopedBy'] as const) {
    const one: unknown = value[member];
    if (one === undefined) continue;
    if (typeof one !== 'string' || one === '') {
      throw new PresentationError(`${where}: '${member}' expects a name`);
    }
    beside[member] = one;
  }
  if (value['without'] !== undefined) {
    beside.without = stringsOf(value['without'], `${where}.without`);
  }
  return { tag, ...beside };
}

/** One binding. */
function bindingOf(value: unknown, anchor: string): Binding {
  if (!isRecord(value)) throw new PresentationError(`${anchor}: expected a binding`);
  const members = Object.keys(value);
  if (members.length === 0) throw new PresentationError(`${anchor}: a binding says nothing`);
  const binding: Record<string, unknown> = {};
  for (const member of members) {
    if (!(MEMBERS as readonly string[]).includes(member)) {
      throw new PresentationError(`${anchor}: a binding carries no member '${member}'`);
    }
    const one: unknown = value[member];
    const where = `${anchor}.${member}`;
    if (STRINGS.has(member)) {
      if (typeof one !== 'string' || one === '') {
        throw new PresentationError(`${where}: expected a name`);
      }
      binding[member] = one;
    } else if (LISTS.has(member)) {
      binding[member] = stringsOf(one, where);
    } else if (member === 'structuralSummary') {
      if (one !== true && one !== false) {
        throw new PresentationError(`${where}: expected true or false`);
      }
      binding[member] = one;
    } else if (member === 'statusBar') {
      binding[member] = fieldOf(one, where);
    } else if (member === 'symbols') {
      if (!isRecord(one)) throw new PresentationError(`${where}: expected symbols by name`);
      const symbols = new Map<string, SymbolBinding>();
      for (const [name, symbol] of Object.entries(one)) {
        symbols.set(name, symbolOf(symbol, `${where}.${name}`));
      }
      if (symbols.size === 0) throw new PresentationError(`${where}: names no symbol`);
      binding[member] = symbols;
    } else {
      if (!Array.isArray(one)) throw new PresentationError(`${where}: expected reference rules`);
      binding[member] = one.map((rule, index) => ruleOf(rule, `${where}[${String(index)}]`));
    }
  }
  return binding;
}

/** The bindings of a parsed presentation file, or a refusal saying what is wrong with it. */
export function readPresentation(parsed: unknown): Presentation {
  if (!isRecord(parsed)) {
    throw new PresentationError('a presentation file is an object of bindings by anchor');
  }
  const bindings = new Map<string, Binding>();
  const anchors: string[] = [];
  for (const [anchor, value] of Object.entries(parsed)) {
    if (bindings.has(anchor)) throw new PresentationError(`${anchor}: bound twice`);
    bindings.set(anchor, bindingOf(value, anchor));
    anchors.push(anchor);
  }
  return {
    anchors,
    at: (anchor) => bindings.get(anchor),
    firstOf(candidates) {
      for (const anchor of candidates) {
        const found = bindings.get(anchor);
        if (found !== undefined) return found;
      }
      return undefined;
    },
  };
}

let loaded: Presentation | undefined;

/** The bindings the application ships, read once. */
export function presentation(): Presentation {
  loaded ??= readPresentation(source);
  return loaded;
}

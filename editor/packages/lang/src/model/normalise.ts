/**
 * Reading a model document: the port of `tools/model.py`.
 *
 * "A composition may carry its own bindings, written against its sites (§5.2). They are syntactic
 * sugar: each scoped rule is exactly one top-level rule whose `for_each` is the composition's
 * indices and whose endpoints select the generated instance of the current index. `normalise`
 * performs that expansion, so that validation, D1, the viewer and the linter read one form and the
 * denotation has one definition."
 *
 *     composition C, indices {i}, scoped value rule R {from: {site: a, port: p},
 *                                                      to:   {site: b, port: q, indices: {i: i-1}}}
 *     ==  top-level rule "C.R" {for_each: C.indices,
 *                               from: {instance: generated(C, a, {i: i}), port: p},
 *                               to:   {instance: generated(C, b, {i: i-1}), port: q}}
 *
 * Rule 7 of the specification's §5.2 says the same and adds what makes it worth doing: "nothing is
 * expressible in the scoped form that is not expressible at the top level; the scoped form only
 * removes the repetition of the composition's ranges and selectors, and **every reading expands it
 * before any other rule applies**."
 *
 * **What this operates on.** A document as CPython's `json` reads one — `expr/value.ts`'s
 * `PyValue`, where an `int` is a `bigint` and a `float` a `number` — because that is what the
 * tools normalise and what every later stage reads: `validate.analyse`, `d1.emit` and
 * `derive.products` all take the dictionary `model.load` answers. The lexeme-preserving tree of
 * `json/` is the *document*, which the store holds and the serializer writes (D1, D12); this is a
 * reading of it, and a hoisted rule has no place in the document to point at anyway. `toPython`
 * and `toJsonValue` are the two conversions, and the round trip through them is byte-exact on
 * every file of the corpus (feature 1.2), so a parity suite can still compare the normalised
 * document's *bytes* with the ones `json.dumps(model.load(path))` writes.
 *
 * **Nothing is modified.** Python deep-copies the document and mutates the copy; a `PyValue` is
 * immutable by convention here, so the port builds new records and shares every subtree it does
 * not change — which is what `copy.deepcopy` exists to make safe, not an effect a reader could
 * observe. Member order is Python's dictionary order in both directions: a name already present
 * keeps its place when it is assigned again, a new one is appended, and JavaScript's plain objects
 * agree on both for every name the grammar admits (an integer-like member name would enumerate
 * first, and neither an identifier nor a qualified name can be one).
 *
 * **Where it raises.** Meaning assumes grammar, here as everywhere: `model.py` reads
 * `comp['indices']`, `rule['members']` and `model['bindings'][kind]` with `[]`, so a document that
 * has not crossed the grammar raises `KeyError` — a `PyKeyError` here — rather than being read
 * with a guess (I7). The four texts it refuses with are {@link ModelError}'s. The three readings
 * that make the difference — `d['x']` raising, `d.get('x')` answering, `'x' in d` asking — are
 * `library/access.ts`'s, written for `primitive_library.py` and used here for `model.py`: they are
 * how this port reads any document, not something the loader owns, and they will move beside
 * `expr/value.ts` when `validate/` gives them a third caller.
 */
import { PyTypeError } from '../expr/errors.js';
import {
  hasKey,
  isRecord,
  member,
  pythonTypeName,
  toPython,
  truthy,
  type PyRecord,
  type PyValue,
} from '../expr/value.js';
import { put } from '../json/tree.js';
import type { PathSegment } from '../schema/types.js';
import { JsonParseError, parse } from '../json/parse.js';
import { demand, entries, has, optional } from '../library/access.js';
import { pyStr } from '../library/repr.js';

import { ModelError } from './errors.js';
import { HoistRecorder, NO_HOISTING, type Hoisting } from './hoisting.js';

/** The member of a top-level rule that names the slot, per kind of binding. */
const SLOT: PyRecord = {
  parameters: 'parameter',
  states: 'state',
  constants: 'constant',
};

/** Python's `d[name] = value` on a copy: an existing name keeps its place, a new one is appended. */
function withKey(record: PyRecord, name: string, value: PyValue): PyRecord {
  const out: Record<string, PyValue> = {};
  for (const [key, held] of entries(record)) put(out, key, held);
  put(out, name, value);
  return out;
}

/** Python's `del d[name]` on a copy — here `comp.pop('bindings')`, whose value is read first. */
function withoutKey(record: PyRecord, name: string): PyRecord {
  const out: Record<string, PyValue> = {};
  for (const [key, held] of entries(record)) {
    if (key !== name) put(out, key, held);
  }
  return out;
}

/** `_current`: the composition's indices, each selecting itself — `{i: {"index": i}}`. */
function current(composition: PyValue): PyRecord {
  const out: Record<string, PyValue> = {};
  for (const [name] of entries(demand(composition, 'indices'))) put(out, name, { index: name });
  return out;
}

/**
 * `_selector`: the instance selector an endpoint denotes — a site of the composition, at the
 * current indices unless overridden, or any explicit selector.
 */
function selector(
  compositionName: string,
  composition: PyValue,
  endpoint: PyValue,
  ruleName: string,
  where: Where,
): PyValue {
  // Where the selector lands in the hoisted rule, and where the endpoint it denotes was written.
  const at = [...where.at, 'instance'];
  if (has(endpoint, 'instance')) {
    // An explicit selector is copied as it stands, so its whole subtree corresponds.
    where.record?.copy(at, [...where.from, 'instance']);
    return demand(endpoint, 'instance');
  }
  const site = demand(endpoint, 'site');
  const instances = demand(composition, 'instances');
  // `site not in comp['instances']`: a dictionary answers by key, and every key of an instance map
  // is a name, so anything else is absent — except a list or a record, which Python cannot hash.
  if (Array.isArray(site) || isRecord(site)) {
    throw new PyTypeError(`unhashable type: '${pythonTypeName(site)}'`);
  }
  if (!isRecord(instances)) {
    throw new PyTypeError(`argument of type '${pythonTypeName(instances)}' is not iterable`);
  }
  if (typeof site !== 'string' || !hasKey(instances, site)) {
    throw new ModelError(
      'site',
      `composition '${compositionName}', binding '${ruleName}': no site named '${pyStr(site)}'`,
    );
  }
  let indices = current(composition);
  const overridden = new Set<string>();
  for (const [name, expression] of entries(optional(endpoint, 'indices', {}))) {
    if (!hasKey(indices, name)) {
      throw new ModelError(
        'index',
        `composition '${compositionName}', binding '${ruleName}': ` +
          `'${name}' is not an index of the composition`,
      );
    }
    // An override keeps the index's place, as assigning an existing key of a dictionary does.
    indices = withKey(indices, name, expression);
    overridden.add(name);
  }
  if (where.record !== undefined) {
    // The selector itself is built out of the endpoint; the site name and every overridden index
    // expression inside it are the endpoint's own text, and the rest is the composition's.
    where.record.invent(at, where.from);
    where.record.copy([...at, 'instance'], [...where.from, 'site']);
    for (const [name] of entries(indices)) {
      if (overridden.has(name)) {
        where.record.copy([...at, 'indices', name], [...where.from, 'indices', name]);
      } else {
        where.record.invent([...at, 'indices', name], [...where.compositionAt, 'indices', name]);
      }
    }
  }
  return { kind: 'generated', composition: compositionName, instance: site, indices };
}

/**
 * Where one hoisted rule sits, and where it was written: what the recorder is told about.
 *
 * `at` is the place of the rule in the *normalised* document, `from` the place of the scoped rule
 * in the document as written, and `compositionAt` the composition itself — the `for_each` a hoist
 * writes is the composition's own `indices` and nothing of the rule's.
 */
interface Where {
  readonly at: readonly PathSegment[];
  readonly from: readonly PathSegment[];
  readonly compositionAt: readonly PathSegment[];
  readonly record: HoistRecorder | undefined;
}

/**
 * `_hoist`: one scoped rule as the top-level rule it denotes.
 *
 * The member order is the one Python's assignments make, and it is what the normalised document is
 * written with: `for_each`, then `when` if the rule carries one, then the rule's own members —
 * `from` and `to` for a value rule; `members`, `dtype`, and then `tensor` and `location`,
 * `identity`, or `constant` for the three others. It is not the order the scoped rule was written
 * in: a rule writing `when` before `from` normalises to the same document as one that did not.
 */
function hoist(
  compositionName: string,
  composition: PyValue,
  kind: string,
  ruleName: string,
  rule: PyValue,
  where: Where,
): PyRecord {
  const record = where.record;
  // The rule as a whole is rebuilt: it denotes the rule the author wrote, member for member it
  // is not, so a pointer below it falls back to it unless one of the members below answers.
  record?.rebuild(where.at, where.from);
  // `copy.deepcopy(comp['indices'])`: the copy is Python's defence against a shared mutable, and
  // there is nothing to defend here — a value of the reading is never written into.
  let top: PyRecord = { for_each: demand(composition, 'indices') };
  record?.copy([...where.at, 'for_each'], [...where.compositionAt, 'indices']);
  if (has(rule, 'when')) {
    top = withKey(top, 'when', demand(rule, 'when'));
    record?.copy([...where.at, 'when'], [...where.from, 'when']);
  }
  const qualified = `${compositionName}.${ruleName}`;
  if (kind === 'values') {
    for (const side of ['from', 'to'] as const) {
      const endpoint = demand(rule, side);
      const endpointAt = [...where.at, side];
      const endpointFrom = [...where.from, side];
      // The endpoint is rebuilt around the selector; its port is the written one.
      record?.rebuild(endpointAt, endpointFrom);
      record?.copy([...endpointAt, 'port'], [...endpointFrom, 'port']);
      top = withKey(top, side, {
        instance: selector(compositionName, composition, endpoint, ruleName, {
          ...where,
          at: endpointAt,
          from: endpointFrom,
        }),
        port: demand(endpoint, 'port'),
      });
    }
    return top;
  }
  // `{'parameters': 'parameter', …}[kind]`: a kind the grammar does not carry is a `KeyError`.
  const slot = demand(SLOT, kind) as string;
  const declared = demand(rule, 'members');
  if (!Array.isArray(declared)) {
    throw new PyTypeError(`'${pythonTypeName(declared)}' object is not iterable`);
  }
  const members = declared as readonly PyValue[];
  record?.rebuild([...where.at, 'members'], [...where.from, 'members']);
  top = withKey(
    top,
    'members',
    members.map((one, index) => {
      const memberAt = [...where.at, 'members', index];
      const memberFrom = [...where.from, 'members', index];
      record?.rebuild(memberAt, memberFrom);
      record?.copy([...memberAt, slot], [...memberFrom, slot]);
      return {
        instance: selector(compositionName, composition, one, ruleName, {
          ...where,
          at: memberAt,
          from: memberFrom,
        }),
        [slot]: demand(one, slot),
      };
    }),
  );
  // Parameter and state identities select a dtype. A constant rule carries none on the grammar,
  // and the tools copy one all the same when a document holds it: the line is theirs, not the
  // schema's, so the port copies it too.
  if (has(rule, 'dtype')) {
    top = withKey(top, 'dtype', demand(rule, 'dtype'));
    record?.copy([...where.at, 'dtype'], [...where.from, 'dtype']);
  }
  if (kind === 'parameters') {
    // "A scoped parameter or state rule without a declared `tensor` / `identity` names it `C.R`,
    // indexed by the composition's indices" (§5.2 rule 7).
    top = withKey(
      top,
      'tensor',
      has(rule, 'tensor')
        ? demand(rule, 'tensor')
        : { name: qualified, indices: current(composition) },
    );
    named(record, where, 'tensor', has(rule, 'tensor'));
    // Where the identity's tensor is stored (§3.4).
    if (has(rule, 'location')) {
      top = withKey(top, 'location', demand(rule, 'location'));
      record?.copy([...where.at, 'location'], [...where.from, 'location']);
    }
  } else if (kind === 'states') {
    top = withKey(
      top,
      'identity',
      has(rule, 'identity')
        ? demand(rule, 'identity')
        : { name: qualified, indices: current(composition) },
    );
    named(record, where, 'identity', has(rule, 'identity'));
  } else {
    top = withKey(top, 'constant', demand(rule, 'constant'));
    record?.copy([...where.at, 'constant'], [...where.from, 'constant']);
  }
  return top;
}

/**
 * The identity a parameter or state rule names, recorded.
 *
 * Declared, it is the written member and its subtree corresponds; undeclared, §5.2 rule 7 names it
 * `C.R` indexed by the composition's indices — a value the hoist builds, which stands for the rule
 * that did not write one.
 */
function named(
  record: HoistRecorder | undefined,
  where: Where,
  member: string,
  declared: boolean,
): void {
  if (record === undefined) return;
  if (declared) record.copy([...where.at, member], [...where.from, member]);
  else record.invent([...where.at, member], where.from);
}

/**
 * The document with every composition-scoped binding hoisted to the top level under the name
 * `<composition>.<rule>`. Idempotent; the input is not modified.
 *
 * A composition's `bindings` member goes whether or not it holds a rule, because the tools pop it
 * before they ask: a composition writing `"bindings": {}` — which the grammar refuses,
 * `scoped_bindings` requiring one member — normalises to a composition without one.
 */
export function normalise(model: PyValue, record?: HoistRecorder): PyRecord {
  // `model.get('compositions', {})`, which is where a document that is not a document is refused.
  const declared = optional(model, 'compositions', {});
  let compositions: PyRecord | null = null;
  let bindings: PyRecord | null = null;
  for (const [compositionName, value] of entries(declared)) {
    if (!isRecord(value)) {
      throw new PyTypeError(`'${pythonTypeName(value)}' object has no attribute 'pop'`);
    }
    if (!hasKey(value, 'bindings')) continue;
    const scoped = member(value, 'bindings') as PyValue;
    const composition = withoutKey(value, 'bindings');
    compositions = withKey(compositions ?? (declared as PyRecord), compositionName, composition);
    if (!truthy(scoped)) continue;
    for (const [kind, rules] of entries(scoped)) {
      for (const [ruleName, rule] of entries(rules)) {
        const qualified = `${compositionName}.${ruleName}`;
        // `model['bindings'][kind]`: read only where a scoped rule of that kind exists, as the
        // tools read it, so a document without the map is refused only where they refuse it.
        bindings ??= demand(model, 'bindings') as PyRecord;
        const map = demand(bindings, kind);
        if (has(map, qualified)) {
          throw new ModelError(
            'collision',
            `binding '${qualified}' is declared both in composition ` +
              `'${compositionName}' and at the top level`,
          );
        }
        if (!isRecord(map)) {
          throw new PyTypeError(
            `'${pythonTypeName(map)}' object does not support item assignment`,
          );
        }
        bindings = withKey(
          bindings,
          kind,
          withKey(
            map,
            qualified,
            hoist(compositionName, composition, kind, ruleName, rule, {
              at: ['bindings', kind, qualified],
              from: ['compositions', compositionName, 'bindings', kind, ruleName],
              compositionAt: ['compositions', compositionName],
              record,
            }),
          ),
        );
      }
    }
  }
  let result = model as PyRecord;
  if (compositions !== null) result = withKey(result, 'compositions', compositions);
  if (bindings !== null) result = withKey(result, 'bindings', bindings);
  return result;
}

/**
 * What `normalise` expanded, and where each expanded place was written (§5.2 rule 7, backwards).
 *
 * The hoist is run for its record: `normalise` is the one implementation of the rule, so the map
 * cannot say something the expansion does not do. A document that carries no composition-scoped
 * binding answers {@link NO_HOISTING}, which every lookup misses — the ordinary case, and the
 * cheap one.
 *
 * It raises what `normalise` raises, because it *is* `normalise`: a caller that has not yet been
 * able to read the document has nothing to map either.
 */
export function hoistingOf(model: PyValue): Hoisting {
  const record = new HoistRecorder();
  normalise(model, record);
  const hoisting = record.hoisting();
  return hoisting.places.size === 0 ? NO_HOISTING : hoisting;
}

/**
 * `model.load`: the document of a text, normalised.
 *
 * "A duplicate member name is a refusal (V12), never the last value silently kept" — the parser of
 * 0.3 is that refusal, and it is re-raised here as the {@link ModelError} the tools raise, so that
 * a caller catching `ModelError` catches what `except model_mod.ModelError` catches. A text that
 * is not JSON at all raises the `JsonParseError` standing for CPython's `JSONDecodeError`:
 * `model.load` does not catch that one either, and `validate.structural` has no line for it
 * (feature 1.1 states it under V12 at the schema stage, the same hole read the same way).
 */
export function loadModel(text: string): PyRecord {
  let tree;
  try {
    tree = parse(text);
  } catch (error) {
    if (error instanceof JsonParseError && error.duplicateMember !== null) {
      throw new ModelError('duplicate', error.message);
    }
    throw error;
  }
  return normalise(toPython(tree));
}

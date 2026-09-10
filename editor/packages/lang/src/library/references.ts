/**
 * `primitive_library.primitive_references`: "what the schema cannot see".
 *
 * A unit that satisfies the primitive-library-unit schema can still cite an axis no base holds, a
 * precision role that does not exist, a port that is not a port, or an argument it never declares.
 * The grammar cannot see any of that — a shape's `axis` is a qualified name, a `present_when`'s
 * `argument` is a path, and both are well-formed strings whatever they name — so the loader
 * resolves them once the bases are gathered, and refuses the unit naming the file (§7 of the unit
 * guide, I7).
 *
 * Two of the checks are not resolutions but decidability rules, and they are the reason the
 * primitive-side evaluator may answer `false` to what it cannot decide (§4.3):
 *
 * - a `domain` bound may name another argument, and that argument must be required or defaulted,
 *   so the bound always resolves at a call site;
 * - a condition may compare an argument that may be absent only under a `present` test of it —
 *   `all[present X, …]` for a field, `any[not present X, …]` for the invariant that holds
 *   vacuously — since a comparison with nothing is undecidable, and undecidable read as false
 *   would silently admit an instance the primitive means to refuse.
 *
 * The order the problems come out in is the tools' order, member by member and index by index,
 * because it is the order the lines of one refusal are printed in, and a refusal's whole text is
 * what `tests/rejections/primitive-library.json` matches against. Every problem carries the place
 * in the definition it speaks of, which the tools have no room for and §4.22 needs: a Problems row
 * with somewhere to click.
 *
 * Meaning assumes grammar. Every reader here is the tools' own — `d['arguments']`, `port['role']`,
 * `rule['indexed_by']` — so a definition that never crossed the schema stage raises here as it
 * raises there. The one difference is the exception's own type: a condition that is not an object
 * raises `PyTypeError` here where CPython would raise `AttributeError`, since nothing calls this
 * before the schema stage and no message of either is a contract.
 */
import { hasKey, isRecord, member, truthy, type PyRecord, type PyValue } from '../expr/value.js';
import { PyTypeError } from '../expr/errors.js';
import { pyEqual } from '../expr/arithmetic.js';
import type { PathSegment } from '../schema/types.js';
import { comparePythonStrings } from '../schema/repr.js';
import { demand, entries, has, listOf as list, optional } from './access.js';
import { detailAt, type LibraryDetail } from './problems.js';
import { pyRepr, pyStr } from './repr.js';

/** What `primitive_references` reads of the gathered bases: the axes and the precision roles. */
export interface ReferenceLibrary {
  readonly axes: ReadonlyMap<string, PyValue>;
  readonly precision: ReadonlyMap<string, PyValue>;
}

/** `a + b` over two lists, as `_condition_paths` concatenates `all` and `any`. */
function concat(left: PyValue, right: PyValue): PyValue[] {
  const one = (value: PyValue): PyValue[] => {
    if (!Array.isArray(value)) {
      throw new PyTypeError(`can only concatenate list (not "${typeof value}") to list`);
    }
    return [...(value as readonly PyValue[])];
  };
  return [...one(left), ...one(right)];
}

/**
 * A name the definition writes, as the message interpolates it.
 *
 * Every one of these is an `identifier` or a `qualified_name` once the schema stage has passed, so
 * this is the identity on every reachable path. A value that is not a string is written as
 * Python's `f"{v}"` would write it rather than refused, so that a definition reaching this
 * function without the grammar behind it still gets a message instead of an exception.
 */
const text = pyStr;

// --- the two walks that find argument paths --------------------------------

/**
 * `_expression_paths`: the argument paths an expression reads, its conditionals included.
 *
 * A generator, and the duplicates matter: two occurrences of one undeclared path in one expression
 * are two lines of the refusal, as they are in the tools.
 */
export function expressionPaths(expression: PyValue): string[] {
  const out: string[] = [];
  if (!isRecord(expression)) return out;
  if (hasKey(expression, 'argument')) out.push(text(member(expression, 'argument') as PyValue));
  for (const one of list(optional(expression, 'args', []))) out.push(...expressionPaths(one));
  for (const branch of ['then', 'else'] as const) {
    if (hasKey(expression, branch)) out.push(...expressionPaths(member(expression, branch) as PyValue));
  }
  if (hasKey(expression, 'if')) out.push(...conditionPaths(member(expression, 'if') as PyValue));
  return out;
}

/** `_condition_paths`: the argument paths a condition tests. */
export function conditionPaths(condition: PyValue): string[] {
  const out: string[] = [];
  const node = asCondition(condition);
  if (hasKey(node, 'not')) out.push(...conditionPaths(member(node, 'not') as PyValue));
  for (const one of concat(optional(node, 'all', []), optional(node, 'any', []))) {
    out.push(...conditionPaths(one));
  }
  if (hasKey(node, 'present')) out.push(text(member(node, 'present') as PyValue));
  if (hasKey(node, 'compare')) {
    const comparison = member(node, 'compare') as PyValue;
    out.push(...expressionPaths(demand(comparison, 'left')));
    out.push(...expressionPaths(demand(comparison, 'right')));
  }
  return out;
}

/** A condition node. `_condition_paths` has no `isinstance` guard; nor, in effect, has this. */
function asCondition(condition: PyValue): PyRecord {
  if (isRecord(condition)) return condition;
  throw new PyTypeError(`a condition is an object, not ${pyRepr(condition)}`);
}

/**
 * `_absent_comparisons`: the paths a condition compares or computes with although they may be
 * absent, outside a `present` test of them.
 *
 * `all` gathers the `present` tests among its own parts as guards for the others; `any` gathers
 * the `not present` tests, which is the dual — `any[not present X, P(X)]` reads `X` only where it
 * is present. A `present` node guards nothing beyond itself and is not descended into.
 */
export function absentComparisons(
  condition: PyValue,
  guarded: ReadonlySet<string>,
  optionalPaths: ReadonlySet<string>,
): string[] {
  const out: string[] = [];
  if (!isRecord(condition)) return out;
  if (hasKey(condition, 'present')) return out;
  if (hasKey(condition, 'all')) {
    const parts = list(member(condition, 'all') as PyValue);
    const wider = new Set(guarded);
    for (const part of parts) {
      if (isRecord(part) && hasKey(part, 'present')) wider.add(text(member(part, 'present') as PyValue));
    }
    for (const part of parts) out.push(...absentComparisons(part, wider, optionalPaths));
    return out;
  }
  if (hasKey(condition, 'any')) {
    const parts = list(member(condition, 'any') as PyValue);
    const wider = new Set(guarded);
    for (const part of parts) {
      if (!isRecord(part)) continue;
      const negated = member(part, 'not');
      if (negated !== undefined && isRecord(negated) && hasKey(negated, 'present')) {
        wider.add(text(member(negated, 'present') as PyValue));
      }
    }
    for (const part of parts) out.push(...absentComparisons(part, wider, optionalPaths));
    return out;
  }
  if (hasKey(condition, 'not')) {
    return absentComparisons(member(condition, 'not') as PyValue, guarded, optionalPaths);
  }
  if (hasKey(condition, 'compare')) {
    const comparison = member(condition, 'compare') as PyValue;
    for (const side of ['left', 'right'] as const) {
      for (const path of expressionPaths(demand(comparison, side))) {
        if (unguarded(path, guarded, optionalPaths)) out.push(path);
      }
    }
  }
  return out;
}

/** `_unguarded`: a path that is, or is under, a maybe-absent argument and under no guard. */
function unguarded(
  path: string,
  guarded: ReadonlySet<string>,
  optionalPaths: ReadonlySet<string>,
): boolean {
  const under = (names: ReadonlySet<string>): boolean => {
    for (const name of names) {
      if (path === name || path.startsWith(`${name}.`)) return true;
    }
    return false;
  };
  return under(optionalPaths) && !under(guarded);
}

/** `_declared_paths`: every argument path a set of declarations makes addressable. */
export function declaredPaths(args: PyValue, prefix = ''): string[] {
  const out: string[] = [];
  for (const [name, declaration] of entries(args)) {
    out.push(prefix + name);
    const type = demand(declaration, 'type');
    if (pyEqual(demand(type, 'kind'), 'record')) {
      out.push(...declaredPaths(demand(type, 'fields'), `${prefix + name}.`));
    }
  }
  return out;
}

/** `_optional_paths`: the paths that may be absent at an instance — optional, without a default. */
export function optionalPaths(args: PyValue, prefix = ''): string[] {
  const out: string[] = [];
  for (const [name, declaration] of entries(args)) {
    if (!truthy(demand(declaration, 'required')) && !has(declaration, 'default')) {
      out.push(prefix + name);
    }
    const type = demand(declaration, 'type');
    if (pyEqual(demand(type, 'kind'), 'record')) {
      out.push(...optionalPaths(demand(type, 'fields'), `${prefix + name}.`));
    }
  }
  return out;
}

// --- the checker -----------------------------------------------------------

/** One condition the primitive writes, with what guards it and where it stands. */
interface PendingCondition {
  readonly label: string;
  readonly condition: PyValue;
  readonly guarded: ReadonlySet<string>;
  readonly at: readonly PathSegment[];
}

/**
 * The problems of one primitive definition against the primitive library it lives in.
 *
 * A template primitive declares none of this — its arguments are the template's external
 * quantities (§4.6) — so it has nothing to resolve and answers no problems.
 */
export function primitiveReferences(
  definition: PyValue,
  library: ReferenceLibrary,
): LibraryDetail[] {
  if (has(definition, 'template')) return [];
  const out: LibraryDetail[] = [];
  const paths = new Set(declaredPaths(demand(definition, 'arguments')));
  const optionals = new Set(optionalPaths(demand(definition, 'arguments')));
  const conditions: PendingCondition[] = [];
  const say = (message: string, at: readonly PathSegment[]): void => {
    out.push(detailAt(message, at));
  };

  /** Every axis and every extent of one shape, in the order `_shape_problems` reads them. */
  const shapeProblems = (shape: PyValue, label: string, at: readonly PathSegment[]): void => {
    const axes = list(demand(shape, 'axes'));
    for (const [index, axis] of axes.entries()) {
      const here = [...at, 'axes', index];
      const named = text(demand(axis, 'axis'));
      if (!library.axes.has(named)) {
        say(`${label}: axis '${named}' is not in the primitive_library`, [...here, 'axis']);
      }
      for (const [position, factor] of list(optional(axis, 'factors', [])).entries()) {
        const there = [...here, 'factors', position];
        const factorAxis = text(demand(factor, 'axis'));
        if (!library.axes.has(factorAxis)) {
          say(`${label}: factor axis '${factorAxis}' is not in the primitive_library`, [
            ...there,
            'axis',
          ]);
        }
        for (const path of expressionPaths(demand(factor, 'extent'))) {
          if (!paths.has(path)) {
            say(`${label}: extent reads undeclared argument '${path}'`, [...there, 'extent']);
          }
        }
      }
      for (const path of expressionPaths(demand(axis, 'extent'))) {
        if (!paths.has(path)) {
          say(`${label}: extent reads undeclared argument '${path}'`, [...here, 'extent']);
        }
      }
    }
  };

  /** `enum_checks`: one declaration, its defaults, its enum values, its domain and its guard. */
  const declarationProblems = (
    declaration: PyValue,
    label: string,
    path: string,
    at: readonly PathSegment[],
  ): void => {
    const type = demand(declaration, 'type');
    // Inside a record, the record itself is present by construction: a field's condition is only
    // ever evaluated when its record is.
    const enclosing = new Set<string>();
    const parts = path.split('.');
    for (let depth = 1; depth < parts.length; depth += 1) {
      enclosing.add(parts.slice(0, parts.length - depth).join('.'));
    }
    if (pyEqual(demand(type, 'kind'), 'enum')) {
      const values = demand(type, 'values');
      const fallback = optional(declaration, 'default', null);
      if (has(declaration, 'default') && has(fallback, 'literal')) {
        const literal = member(fallback as PyRecord, 'literal') as PyValue;
        if (!list(values).some((one) => pyEqual(one, literal))) {
          say(`${label}: default ${pyRepr(literal)} is not among ${pyRepr(values)}`, [
            ...at,
            'default',
            'literal',
          ]);
        }
      }
      for (const [described] of entries(optional(declaration, 'value_descriptions', {}))) {
        if (!list(values).some((one) => pyEqual(one, described))) {
          say(`${label}: value_descriptions names '${described}', not an enum value`, [
            ...at,
            'value_descriptions',
            described,
          ]);
        }
      }
    } else if (has(declaration, 'value_descriptions')) {
      say(`${label}: value_descriptions on a non-enum type`, [...at, 'value_descriptions']);
    }
    for (const path_ of expressionPaths(optional(declaration, 'default', {}))) {
      if (!paths.has(path_)) {
        say(`${label}: default reads undeclared argument '${path_}'`, [...at, 'default']);
      }
    }
    const domain = optional(declaration, 'domain', null);
    if (truthy(domain) && pyEqual(demand(domain, 'kind'), 'interval')) {
      // A bound may name another argument; it must be declared and always resolve, so a
      // maybe-absent argument (optional, no default) is refused rather than read as no bound.
      for (const edge of ['lower', 'upper'] as const) {
        const bound = optional(domain, edge, null);
        if (!truthy(bound)) continue;
        const value = demand(bound, 'value');
        if (!isRecord(value) || !hasKey(value, 'argument')) continue;
        const named = text(member(value, 'argument') as PyValue);
        const where = [...at, 'domain', edge, 'value'];
        if (!paths.has(named)) {
          say(`${label}: domain ${edge} bound reads undeclared argument '${named}'`, where);
        } else if (optionals.has(named)) {
          say(
            `${label}: domain ${edge} bound reads '${named}', which may be absent — a bound must ` +
              'always resolve (§4.6)',
            where,
          );
        }
      }
    }
    if (has(declaration, 'present_when')) {
      conditions.push({
        label: `${label} present_when`,
        condition: member(declaration as PyRecord, 'present_when') as PyValue,
        guarded: enclosing,
        at: [...at, 'present_when'],
      });
    }
    if (pyEqual(demand(type, 'kind'), 'record')) {
      for (const [field, declared] of entries(demand(type, 'fields'))) {
        declarationProblems(declared, `${label}.${field}`, `${path}.${field}`, [
          ...at,
          'type',
          'fields',
          field,
        ]);
      }
    }
  };

  for (const [name, declaration] of entries(demand(definition, 'arguments'))) {
    declarationProblems(declaration, `argument '${name}'`, name, ['arguments', name]);
  }

  for (const [index, invariant] of list(optional(definition, 'invariants', [])).entries()) {
    // An invariant's condition is a primitive condition: every argument it reads must be declared,
    // and one that may be absent must be guarded by a `present` test of it (§4.3) — exactly as
    // every other condition of the primitive is checked below.
    conditions.push({
      label: `invariant ${String(index)} ('${text(demand(invariant, 'description'))}')`,
      condition: demand(invariant, 'holds'),
      guarded: new Set(),
      at: ['invariants', index, 'holds'],
    });
  }

  const ports = new Map<string, string>();
  const declaredPorts = demand(definition, 'ports');
  for (const side of ['inputs', 'outputs'] as const) {
    for (const [name, port] of entries(demand(declaredPorts, side))) {
      ports.set(name, side);
      const label = `port '${name}'`;
      const at: PathSegment[] = ['ports', side, name];
      const role = text(demand(port, 'role'));
      if (!library.precision.has(role)) {
        say(`${label}: role '${role}' has no precision rule`, [...at, 'role']);
      }
      if (has(port, 'shape')) {
        shapeProblems(member(port as PyRecord, 'shape') as PyValue, label, [...at, 'shape']);
      }
      if (has(port, 'present_when')) {
        conditions.push({
          label: `${label} present_when`,
          condition: member(port as PyRecord, 'present_when') as PyValue,
          guarded: new Set(),
          at: [...at, 'present_when'],
        });
      }
      const from = optional(demand(port, 'domain'), 'from', {});
      if (has(from, 'port')) {
        const named = text(member(from as PyRecord, 'port') as PyValue);
        const where = [...at, 'domain', 'from', 'port'];
        const target = optional(demand(declaredPorts, 'inputs'), named, null);
        if (target === null) {
          say(`${label}: domain inherited from unknown input port '${named}'`, where);
        } else if (has(optional(demand(target, 'domain'), 'from', {}), 'port')) {
          say(
            `${label}: domain inherited from '${named}', which itself inherits from a port`,
            where,
          );
        }
      }
    }
  }

  for (const section of ['parameters', 'constants'] as const) {
    for (const [name, slot] of entries(demand(definition, section))) {
      const label = `${section.slice(0, -1)} '${name}'`;
      const at: PathSegment[] = [section, name];
      const role = text(demand(slot, 'role'));
      if (!library.precision.has(role)) {
        say(`${label}: role '${role}' has no precision rule`, [...at, 'role']);
      }
      const shape = demand(slot, 'shape');
      shapeProblems(shape, label, [...at, 'shape']);
      for (const [index, axis] of list(demand(shape, 'axes')).entries()) {
        if (pyEqual(demand(axis, 'name'), 'multiplicity')) {
          say(
            `${label}: axis name 'multiplicity' is reserved — the storage axis of a slot that ` +
              'declares a multiplicity (§3.4)',
            [...at, 'shape', 'axes', index, 'name'],
          );
        }
      }
      if (has(slot, 'present_when')) {
        conditions.push({
          label: `${label} present_when`,
          condition: member(slot as PyRecord, 'present_when') as PyValue,
          guarded: new Set(),
          at: [...at, 'present_when'],
        });
      }
      const sharing = optional(slot, 'sharing', {});
      if (pyEqual(optional(sharing, 'kind', null), 'shareable')) {
        for (const [index, shared] of list(optional(sharing, 'roles', [])).entries()) {
          const named = text(shared);
          if (!library.precision.has(named)) {
            say(`${label}: sharing role '${named}' has no precision rule`, [
              ...at,
              'sharing',
              'roles',
              index,
            ]);
          }
        }
      }
      for (const path of expressionPaths(optional(slot, 'multiplicity', {}))) {
        if (!paths.has(path)) {
          say(`${label}: multiplicity reads undeclared argument '${path}'`, [...at, 'multiplicity']);
        }
      }
    }
  }

  for (const [name, port] of entries(demand(definition, 'state_ports'))) {
    const label = `state '${name}'`;
    const at: PathSegment[] = ['state_ports', name];
    conditions.push({
      label: `${label} present_when`,
      condition: demand(port, 'present_when'),
      guarded: new Set(),
      at: [...at, 'present_when'],
    });
    if (has(port, 'written_when')) {
      conditions.push({
        label: `${label} written_when`,
        condition: member(port as PyRecord, 'written_when') as PyValue,
        guarded: new Set(),
        at: [...at, 'written_when'],
      });
    }
    if (has(port, 'carried_across')) {
      conditions.push({
        label: `${label} carried_across`,
        condition: demand(member(port as PyRecord, 'carried_across') as PyValue, 'when'),
        guarded: new Set(),
        at: [...at, 'carried_across', 'when'],
      });
    }
    const evolutions = new Set<string>();
    for (const rule of list(demand(port, 'rules'))) evolutions.add(text(demand(rule, 'evolution')));
    evolutions.delete('fixed');
    const growing = [...evolutions].sort(comparePythonStrings);
    for (const [component, payload] of entries(demand(port, 'payload'))) {
      const componentLabel = `${label}.${component}`;
      const here: PathSegment[] = [...at, 'payload', component];
      const role = text(demand(payload, 'role'));
      if (!library.precision.has(role)) {
        say(`${componentLabel}: role '${role}' has no precision rule`, [...here, 'role']);
      }
      const shape = demand(payload, 'shape');
      shapeProblems(shape, componentLabel, [...here, 'shape']);
      const positional = list(demand(shape, 'axes')).some((axis) =>
        pyEqual(demand(axis, 'axis'), 'sequence.position'),
      );
      if (growing.length > 0 && positional) {
        say(
          `${componentLabel}: a sequence.position axis under a ${growing.join('/')} rule ` +
            '— a payload is declared per position (§4.3)',
          [...here, 'shape'],
        );
      }
    }
    for (const [index, axis] of list(demand(port, 'key_axes')).entries()) {
      const named = text(axis);
      const where = [...at, 'key_axes', index];
      const declared = library.axes.get(named);
      if (declared === undefined) {
        say(`${label}: key axis '${named}' is not in the primitive_library`, where);
      } else if (!pyEqual(demand(declared, 'space'), 'instance')) {
        say(`${label}: key axis '${named}' is a value axis, not an instance axis`, where);
      }
    }
    for (const [index, rule] of list(demand(port, 'rules')).entries()) {
      const ruleLabel = `${label} rule ${String(index)}`;
      conditions.push({
        label: ruleLabel,
        condition: demand(rule, 'when'),
        guarded: new Set(),
        at: [...at, 'rules', index, 'when'],
      });
      for (const field of ['span', 'stride'] as const) {
        for (const path of expressionPaths(optional(rule, field, {}))) {
          if (!paths.has(path)) {
            say(`${ruleLabel}: ${field} reads undeclared argument '${path}'`, [
              ...at,
              'rules',
              index,
              field,
            ]);
          }
        }
      }
      const indexedBy = demand(rule, 'indexed_by');
      if (has(indexedBy, 'port')) {
        const named = text(member(indexedBy as PyRecord, 'port') as PyValue);
        if (ports.get(named) !== 'inputs') {
          say(`${ruleLabel}: indexed_by '${named}', not an input port`, [
            ...at,
            'rules',
            index,
            'indexed_by',
            'port',
          ]);
        }
      }
    }
  }

  const effects = demand(definition, 'effects');
  for (const [side, key] of [
    ['inputs', 'reads'],
    ['outputs', 'writes'],
  ] as const) {
    for (const [index, named] of list(demand(effects, key)).entries()) {
      const name = text(named);
      if (ports.get(name) !== side) {
        say(`effects.${key}: '${name}' is not an ${side.slice(0, -1)} port`, [
          'effects',
          key,
          index,
        ]);
      }
    }
  }

  const partitions = list(demand(definition, 'partition_options'));
  for (const [index, partition] of partitions.entries()) {
    const label = `partition ${String(index)}`;
    const at: PathSegment[] = ['partition_options', index];
    const target = demand(partition, 'target');
    if (has(target, 'argument_axis')) {
      const named = text(member(target as PyRecord, 'argument_axis') as PyValue);
      if (!library.axes.has(named)) {
        say(`${label}: axis '${named}' is not in the primitive_library`, [
          ...at,
          'target',
          'argument_axis',
        ]);
      }
    }
    if (has(target, 'instance_key_axis')) {
      const named = text(member(target as PyRecord, 'instance_key_axis') as PyValue);
      if (!library.axes.has(named)) {
        say(`${label}: axis '${named}' is not in the primitive_library`, [
          ...at,
          'target',
          'instance_key_axis',
        ]);
      }
    }
    if (has(target, 'payload_axis')) {
      const payloadAxis = member(target as PyRecord, 'payload_axis') as PyValue;
      const state = text(demand(payloadAxis, 'state'));
      const component = text(demand(payloadAxis, 'component'));
      const axis = text(demand(payloadAxis, 'axis'));
      const payload = optional(
        optional(optional(demand(definition, 'state_ports'), state, {}), 'payload', {}),
        component,
        null,
      );
      if (payload === null) {
        say(`${label}: no payload '${state}.${component}'`, [...at, 'target', 'payload_axis']);
      } else {
        const named = list(demand(demand(payload, 'shape'), 'axes')).map((one) =>
          text(demand(one, 'axis')),
        );
        if (!named.includes(axis)) {
          say(`${label}: '${axis}' is not an axis of '${state}.${component}'`, [
            ...at,
            'target',
            'payload_axis',
            'axis',
          ]);
        }
      }
    }
    if (has(partition, 'when')) {
      conditions.push({
        label,
        condition: member(partition as PyRecord, 'when') as PyValue,
        guarded: new Set(),
        at: [...at, 'when'],
      });
    }
    for (const path of expressionPaths(optional(partition, 'granularity', {}))) {
      if (!paths.has(path)) {
        say(`${label}: granularity reads undeclared argument '${path}'`, [...at, 'granularity']);
      }
    }
  }

  for (const [index, transform] of list(optional(definition, 'domain_transforms', [])).entries()) {
    const label = `domain_transform ${String(index)}`;
    const at: PathSegment[] = ['domain_transforms', index];
    const from = text(demand(transform, 'from_port'));
    const to = text(demand(transform, 'to_port'));
    if (ports.get(from) !== 'inputs') {
      say(`${label}: from_port '${from}' is not an input port`, [...at, 'from_port']);
    }
    if (ports.get(to) !== 'outputs') {
      say(`${label}: to_port '${to}' is not an output port`, [...at, 'to_port']);
    }
    if (pyEqual(demand(transform, 'relation'), 'merge') && !has(transform, 'factor')) {
      say(`${label}: a merge declares its factor`, [...at]);
    }
    for (const path of expressionPaths(optional(transform, 'factor', {}))) {
      if (!paths.has(path)) {
        say(`${label}: factor reads undeclared argument '${path}'`, [...at, 'factor']);
      }
    }
  }

  for (const [index, cost] of list(optional(definition, 'logical_cost', [])).entries()) {
    const label = `logical_cost ${String(index)}`;
    const at: PathSegment[] = ['logical_cost', index];
    if (has(cost, 'when')) {
      conditions.push({
        label,
        condition: member(cost as PyRecord, 'when') as PyValue,
        guarded: new Set(),
        at: [...at, 'when'],
      });
    }
    for (const path of expressionPaths(demand(cost, 'expression'))) {
      if (!paths.has(path)) {
        say(`${label}: reads undeclared argument '${path}'`, [...at, 'expression']);
      }
    }
  }

  for (const [index, sparsity] of list(optional(definition, 'sparsity', [])).entries()) {
    const label = `sparsity ${String(index)}`;
    const at: PathSegment[] = ['sparsity', index];
    const unit = demand(sparsity, 'unit');
    const axis = text(demand(unit, 'axis'));
    for (const [position, named] of list(demand(unit, 'parameters')).entries()) {
      const slotName = text(named);
      const slot = optional(demand(definition, 'parameters'), slotName, null);
      if (slot === null) {
        say(`${label}: unit parameter '${slotName}' is not a slot of this primitive`, [
          ...at,
          'unit',
          'parameters',
          position,
        ]);
      } else {
        const declared = list(demand(demand(slot, 'shape'), 'axes')).map((one) =>
          text(demand(one, 'axis')),
        );
        if (!declared.includes(axis)) {
          say(`${label}: unit axis '${axis}' is not an axis of slot '${slotName}'`, [
            ...at,
            'unit',
            'axis',
          ]);
        }
      }
    }
    if (!library.axes.has(axis)) {
      say(`${label}: axis '${axis}' is not in the primitive_library`, [...at, 'unit', 'axis']);
    }
    const policy = demand(sparsity, 'policy');
    if (has(policy, 'argument')) {
      const named = text(member(policy as PyRecord, 'argument') as PyValue);
      if (!paths.has(named)) {
        say(`${label}: policy names undeclared argument '${named}'`, [...at, 'policy', 'argument']);
      }
    }
    if (has(policy, 'port')) {
      const named = text(member(policy as PyRecord, 'port') as PyValue);
      if (ports.get(named) !== 'inputs') {
        say(`${label}: policy names '${named}', not an input port`, [...at, 'policy', 'port']);
      }
    }
    const fields: [string, PyValue, PathSegment[]][] = [
      ['activated_per_element', demand(sparsity, 'activated_per_element'), [...at, 'activated_per_element']],
      [
        'union_per_invocation',
        demand(demand(sparsity, 'union_per_invocation'), 'expression'),
        [...at, 'union_per_invocation', 'expression'],
      ],
    ];
    for (const [field, expression, where] of fields) {
      for (const path of expressionPaths(expression)) {
        if (!paths.has(path)) {
          say(`${label}.${field}: reads undeclared argument '${path}'`, where);
        }
      }
    }
  }

  for (const pending of conditions) {
    for (const path of conditionPaths(pending.condition)) {
      if (!paths.has(path)) {
        say(`${pending.label}: tests undeclared argument '${path}'`, pending.at);
      }
    }
    for (const path of absentComparisons(pending.condition, pending.guarded, optionals)) {
      say(
        `${pending.label}: compares '${path}', which may be absent, outside a present test of it ` +
          '(§4.3)',
        pending.at,
      );
    }
  }

  for (const [index, partition] of partitions.entries()) {
    const label = `partition ${String(index)}`;
    const at: PathSegment[] = ['partition_options', index];
    const communication = demand(partition, 'communication');
    const communications: PyValue[] = Array.isArray(communication)
      ? [...(communication as readonly PyValue[])]
      : [communication];
    if (has(demand(partition, 'target'), 'none')) {
      if (!(communications.length === 1 && pyEqual(communications[0] as PyValue, 'none'))) {
        say(`${label}: a \`none\` target implies no communication`, [...at, 'communication']);
      }
      if (partitions.length > 1) {
        say(`${label}: \`none\` cannot stand beside other partition_options`, [...at, 'target']);
      }
    }
  }

  return out;
}

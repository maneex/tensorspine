"""Expression and condition evaluation, shared by every command.

There are two evaluators here, and they do not read the same thing:

  * `contract_value` / `contract_condition` evaluate a CONTRACT expression
    against resolved arguments. Declared defaults, `present_when` guards and
    shape extents are written in that language.
  * `model_value` / `model_condition` evaluate a MODEL expression against the
    document's quantities and the composition indices currently in scope.

`UNRESOLVED` marks a value that cannot be decided statically: an external
quantity that no assignment supplies, or an operand that was refused. It is not an error
by itself — each caller decides whether an unresolved value is acceptable at
that point. What is forbidden is deciding silently, so the sentinel is
propagated rather than replaced by a guess (I7).
"""

UNRESOLVED = object()


class Unassigned(ValueError):
    """Raised when a bound cannot be evaluated because an external quantity has
    no value. Not a defect of the document: a template denotes one graph
    per admissible assignment (§4.6), so reading it alone requires one."""

_COMPARISONS = {
    'equal': lambda l, r: l == r,
    'not_equal': lambda l, r: l != r,
    'greater': lambda l, r: l > r,
    'less': lambda l, r: l < r,
    'greater_or_equal': lambda l, r: l >= r,
    'less_or_equal': lambda l, r: l <= r,
}


def _apply(op, a):
    """One operator over evaluated operands. An operand of the wrong kind, or
    a division by zero, yields UNRESOLVED: the value it would have fed is not
    decidable, and the argument that caused it has been refused already (V3)."""
    try:
        return _apply_raw(op, a)
    except (TypeError, ZeroDivisionError):
        return UNRESOLVED


def _apply_raw(op, a):
    if op == 'add': return sum(a)
    if op == 'multiply':
        r = 1
        for v in a: r *= v
        return r
    if op == 'subtract': return a[0] - a[1]
    if op == 'divide': return a[0] / a[1]
    if op == 'floor_divide': return a[0] // a[1]
    if op == 'ceil_divide': return -((-a[0]) // a[1])
    if op == 'modulo': return a[0] % a[1]
    if op == 'min': return min(a)
    if op == 'max': return max(a)
    if op == 'negate': return -a[0]
    if op == 'absolute': return abs(a[0])
    return UNRESOLVED


# --- contract side: expressions over resolved arguments -------------------

def contract_value(e, args):
    """Value of a contract expression against resolved arguments."""
    if 'literal' in e: return e['literal']
    if 'argument' in e:
        cur = args
        for part in e['argument'].split('.'):
            if not isinstance(cur, dict) or part not in cur: return None
            cur = cur[part]
        return cur
    if 'op' in e:
        a = [contract_value(x, args) for x in e['args']]
        if any(v is None or v is UNRESOLVED for v in a): return UNRESOLVED
        return _apply(e['op'], a)
    return UNRESOLVED


def contract_condition(c, args):
    """Truth of a contract condition. An undecidable comparison is false, not
    an exception: the guard it protects simply does not fire."""
    if 'boolean' in c: return c['boolean']
    if 'not' in c: return not contract_condition(c['not'], args)
    if 'all' in c: return all(contract_condition(x, args) for x in c['all'])
    if 'any' in c: return any(contract_condition(x, args) for x in c['any'])
    if 'present' in c:
        cur = args
        for part in c['present'].split('.'):
            if not isinstance(cur, dict) or part not in cur: return False
            cur = cur[part]
        return True
    cp = c['compare']
    l = contract_value(cp['left'], args)
    r = contract_value(cp['right'], args)
    if l is UNRESOLVED or r is UNRESOLVED or l is None or r is None: return False
    return _COMPARISONS[cp['operator']](l, r)


# --- model side: expressions over quantities and indices ------------------

def resolve_quantities(model, assignment=None):
    """Statically known quantities: the literal ones, plus the external ones
    the assignment supplies. An external quantity left unassigned is absent,
    and every expression that reads it evaluates to UNRESOLVED."""
    assignment = assignment or {}
    q = {}
    pending = []
    for name, decl in model['quantities'].items():
        src = decl['source']
        if src['kind'] == 'literal':
            q[name] = src['value']
        elif src['kind'] == 'external' and src['name'] in assignment:
            q[name] = assignment[src['name']]
        elif src['kind'] == 'external' and 'default' in src:
            pending.append((name, src['default']))
    # Declared defaults may read other quantities, in any order, acyclically (§4.6).
    while pending:
        progress = False
        for name, default in list(pending):
            v = model_value(default, q)
            if v is not UNRESOLVED:
                q[name] = v
                pending.remove((name, default))
                progress = True
        if not progress:
            break
    return q


def model_value(e, quantities, env=None):
    """Value of a model expression against quantities and loop indices."""
    env = env or {}
    if 'literal' in e: return e['literal']
    if 'quantity' in e: return quantities.get(e['quantity'], UNRESOLVED)
    if 'index' in e: return env.get(e['index'], UNRESOLVED)
    if 'op' in e:
        a = [model_value(x, quantities, env) for x in e['args']]
        if UNRESOLVED in a: return UNRESOLVED
        return _apply(e['op'], a)
    return UNRESOLVED


def model_condition(c, quantities, env=None):
    """Truth of a model condition — the `when` guard of a generated site."""
    if 'boolean' in c: return c['boolean']
    if 'not' in c: return not model_condition(c['not'], quantities, env)
    if 'all' in c: return all(model_condition(x, quantities, env) for x in c['all'])
    if 'any' in c: return any(model_condition(x, quantities, env) for x in c['any'])
    cp = c['compare']
    l = model_value(cp['left'], quantities, env)
    r = model_value(cp['right'], quantities, env)
    if l is UNRESOLVED or r is UNRESOLVED: return False
    return _COMPARISONS[cp['operator']](l, r)


def static_argument(v, quantities, env=None):
    """Static value of an occurrence argument: a literal, a resolved quantity,
    or a record of those."""
    if isinstance(v, dict) and 'record' in v:
        return {k: static_argument(x, quantities, env) for k, x in v['record'].items()}
    return model_value(v, quantities, env)


def external_names(model, with_defaults=True):
    """External quantity names the document expects an assignment to supply;
    `with_defaults=False` leaves out those a declared default can stand for."""
    return {q['source']['name'] for q in model['quantities'].values()
            if q['source']['kind'] == 'external'
            and (with_defaults or 'default' not in q['source'])}


def missing_assignment(model, assignment=None):
    """External names the assignment leaves unset, in order."""
    return sorted(external_names(model, with_defaults=False) - set(assignment or {}))


def index_grid(indices, quantities):
    """The (name, values) pairs a set of index declarations unrolls to, in
    lexicographic order of the index names (§11.4)."""
    names = sorted(indices)
    ranges = []
    for k in names:
        bounds = []
        for edge in ('start', 'stop', 'step'):
            v = model_value(indices[k][edge], quantities)
            if v is UNRESOLVED:
                raise Unassigned(f"index '{k}': {edge} does not resolve to a value")
            bounds.append(v)
        ranges.append(range(*bounds))
    return names, ranges

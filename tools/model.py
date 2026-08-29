"""Reading a model document: the form every command works on.

A composition may carry its own bindings, written against its sites (§5.2).
They are syntactic sugar: each scoped rule is exactly one top-level rule whose
`for_each` is the composition's indices and whose endpoints select the
generated occurrence of the current index. `normalise` performs that
expansion, so that validation, D1, the viewer and the linter read one form
and the denotation has one definition.

    composition C, indices {i}, scoped value rule R {from: {site: a, port: p},
                                                     to:   {site: b, port: q, indices: {i: i-1}}}
    ==  top-level rule "C.R" {for_each: C.indices,
                              from: {occurrence: generated(C, a, {i: i}), port: p},
                              to:   {occurrence: generated(C, b, {i: i-1}), port: q}}

A scoped parameter or state rule without a declared `tensor` / `identity`
names it `C.R`, indexed by the composition's indices.
"""
import copy
import json


class ModelError(ValueError):
    """A document that cannot be read as written: a scoped rule colliding with
    a top-level one, or an endpoint selecting no site."""


def _pairs(pairs):
    seen = set()
    for k, _v in pairs:
        if k in seen:
            raise ModelError(f"duplicate member name '{k}' (V12)")
        seen.add(k)
    return dict(pairs)


def load(path):
    """The document, normalised; a duplicate member name is a refusal (V12),
    never the last value silently kept."""
    with open(path, encoding='utf-8') as f:
        return normalise(json.load(f, object_pairs_hook=_pairs))


def _current(comp):
    return {name: {"index": name} for name in comp['indices']}


def _selector(comp_name, comp, endpoint, rule_name):
    """The occurrence selector an endpoint denotes: a site of the composition,
    at the current indices unless overridden, or any explicit selector."""
    if 'occurrence' in endpoint:
        return endpoint['occurrence']
    site = endpoint['site']
    if site not in comp['occurrences']:
        raise ModelError(f"composition '{comp_name}', binding '{rule_name}': "
                         f"no site named '{site}'")
    indices = _current(comp)
    for name, e in endpoint.get('indices', {}).items():
        if name not in indices:
            raise ModelError(f"composition '{comp_name}', binding '{rule_name}': "
                             f"'{name}' is not an index of the composition")
        indices[name] = e
    return {"kind": "generated", "composition": comp_name, "occurrence": site, "indices": indices}


def _hoist(comp_name, comp, kind, rule_name, rule):
    top = {"for_each": copy.deepcopy(comp['indices'])}
    if 'when' in rule:
        top['when'] = rule['when']
    qualified = f"{comp_name}.{rule_name}"
    if kind == 'values':
        for side in ('from', 'to'):
            top[side] = {"occurrence": _selector(comp_name, comp, rule[side], rule_name),
                         "port": rule[side]['port']}
        return top
    slot = {'parameters': 'parameter', 'states': 'state', 'constants': 'constant'}[kind]
    top['members'] = [{"occurrence": _selector(comp_name, comp, m, rule_name), slot: m[slot]}
                      for m in rule['members']]
    if 'dtype' in rule:                      # parameter and state identities select a dtype
        top['dtype'] = rule['dtype']
    if kind == 'parameters':
        top['tensor'] = rule.get('tensor', {"name": qualified, "indices": _current(comp)})
    elif kind == 'states':
        top['identity'] = rule.get('identity', {"name": qualified, "indices": _current(comp)})
    else:
        top['constant'] = rule['constant']
    return top


def normalise(model):
    """The document with every composition-scoped binding hoisted to the top
    level under the name `<composition>.<rule>`. Idempotent; the input is not
    modified."""
    model = copy.deepcopy(model)
    for comp_name, comp in model.get('compositions', {}).items():
        scoped = comp.pop('bindings', None)
        if not scoped:
            continue
        for kind, rules in scoped.items():
            for rule_name, rule in rules.items():
                qualified = f"{comp_name}.{rule_name}"
                if qualified in model['bindings'][kind]:
                    raise ModelError(f"binding '{qualified}' is declared both in composition "
                                     f"'{comp_name}' and at the top level")
                model['bindings'][kind][qualified] = _hoist(comp_name, comp, kind, rule_name, rule)
    return model

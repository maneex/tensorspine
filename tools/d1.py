"""`--d1`: emit D1 (§7), the EXPANDED graph of a model.

The model document is not the graph — it is a generator. Read naively, without
unrolling `for_each` or evaluating index expressions, it shows fewer nodes than
reality and apparent cycles. D1 is the form every consumer reads: viewer,
porting, infrastructure matching.

Canonical identities are `<composition>/<site>[<i>=<v>,...]` (§5.2), with
indices in lexicographic order of Unicode code points. A template
contract (§4.6) is expanded in place and its identities are prefixed by the
instance path, so two invocations of one contract share neither state nor
tensor.
"""
import itertools
import json
import os
from collections import defaultdict, deque

import catalog as catalog_mod
import model as model_mod
from expr import (UNRESOLVED, contract_value, index_grid, missing_assignment,
                  model_condition, model_value, resolve_quantities, static_argument)

MAX_DEPTH = 8


def emit(model_path, cat, assignment=None, _prefix="", _depth=0, _stack=()):
    """The D1 document of one model.

    `assignment` supplies the external quantities by their external name; a
    template contract receives its own from the arguments of the occurrence
    that invokes it.
    """
    model = model_mod.load(model_path)
    assignment = assignment or {}
    quantities = resolve_quantities(model, assignment)

    def value(e, env=None):
        return model_value(e, quantities, env)

    def static(v, env=None):
        return static_argument(v, quantities, env)

    def identity(key):
        """Canonical identity of an occurrence key (§5.2)."""
        if key[0] == 'root':
            return key[1]
        _, composition, site, indices = key
        template = ",".join(f"{k}={v}" for k, v in indices)      # already sorted
        return f"{composition}/{site}[{template}]"

    # --- unroll the occurrences -------------------------------------------
    nodes = {}
    keys = {}
    for name, o in model['occurrences'].items():
        keys[('root', name)] = o
    for comp_name, comp in model['compositions'].items():          # document order (§5.2)
        names, ranges = index_grid(comp['indices'], quantities)
        for combo in itertools.product(*ranges):
            env = dict(zip(names, combo))
            for site_name, site in comp['occurrences'].items():
                # `when` is evaluated against the quantities and the current
                # indices; a site that does not fire is not an occurrence.
                if 'when' in site:
                    truth = model_condition(site['when'], quantities, env)
                    if truth is UNRESOLVED:
                        raise ValueError(f"{comp_name}.{site_name}{env}: `when` does not resolve")
                    if not truth:
                        continue
                keys[('gen', comp_name, site_name, tuple(sorted(env.items())))] = site

    edges = []
    instances = {}
    portmap = {}          # (occurrence key, port) -> (real node, real port)
    for key in list(keys):
        contract_name = keys[key]['contract']['name']
        definition = cat['contracts'][contract_name]
        if 'template' not in definition:
            continue                                          # primitive contract
        if contract_name in _stack:
            raise ValueError(f"contract cycle: {' -> '.join(_stack + (contract_name,))}")
        if _depth + 1 > MAX_DEPTH:
            raise ValueError(f"contract nesting deeper than {MAX_DEPTH}")
        occurrence = keys.pop(key)
        instance = _prefix + identity(key)
        sub_assignment = {}
        for arg_name, arg_value in occurrence['arguments'].items():
            v = static(arg_value)
            if v is not UNRESOLVED:
                sub_assignment[arg_name] = v
        sub = emit(catalog_mod.template_path(cat, definition), cat, sub_assignment,
                   instance + "/", _depth + 1, _stack + (contract_name,))
        nodes.update(sub['nodes'])
        edges.extend(sub['edges'])
        instances[instance] = {"contract": occurrence['contract'],
                               "arguments": dict(sub_assignment)}
        instances.update(sub.get('instances') or {})
        for port_name, port in sub['interfaces']['inputs'].items():
            portmap[(key, port_name)] = (port['node'], port['port'])
        for port_name, port in sub['interfaces']['outputs'].items():
            portmap[(key, port_name)] = (port['node'], port['port'])

    for key, occurrence in keys.items():
        contract_name = occurrence['contract']['name']
        definition = cat['contracts'][contract_name]
        args = {a: static(v) for a, v in occurrence['arguments'].items()}
        for arg_name, decl in definition['arguments'].items():
            if arg_name not in args and 'default' in decl:
                args[arg_name] = contract_value(decl['default'], args)
        families = list(occurrence['families'])
        if key[0] == 'gen':
            families = sorted(set(model['compositions'][key[1]]['families']) | set(families))
        nodes[_prefix + identity(key)] = {
            "contract": occurrence['contract'],
            "arguments": {k: v for k, v in args.items() if v is not UNRESOLVED},
            "families": families}

    # --- unroll the edges ---------------------------------------------------
    def loop_envs(binding, label=''):
        envs = [{}]
        if 'for_each' in binding:
            names, ranges = index_grid(binding['for_each'], quantities)
            envs = [dict(zip(names, combo)) for combo in itertools.product(*ranges)]
        if 'when' not in binding:
            return envs
        kept = []
        for env in envs:
            truth = model_condition(binding['when'], quantities, env)
            if truth is UNRESOLVED:
                raise ValueError(f"{label}{env}: `when` does not resolve")
            if truth:
                kept.append(env)
        return kept

    def select(sel, env):
        if sel['kind'] == 'root':
            return ('root', sel['occurrence'])
        return ('gen', sel['composition'], sel['occurrence'],
                tuple(sorted((k, value(v, env)) for k, v in sel['indices'].items())))

    def endpoint(sel, port, env):
        key = select(sel, env)
        if (key, port) in portmap:
            return portmap[(key, port)]
        return _prefix + identity(key), port

    for bid, binding in model['bindings']['values'].items():
        for env in loop_envs(binding, bid):
            src_node, src_port = endpoint(binding['from']['occurrence'],
                                          binding['from']['port'], env)
            dst_node, dst_port = endpoint(binding['to']['occurrence'],
                                          binding['to']['port'], env)
            edges.append({"rule": _prefix + bid,
                          "from": {"node": src_node, "port": src_port},
                          "to": {"node": dst_node, "port": dst_port}})

    interfaces = {"inputs": {}, "outputs": {}}
    for name, decl in model['interfaces']['inputs'].items():
        node, port = endpoint(decl['to']['occurrence'], decl['to']['port'], {})
        interfaces['inputs'][name] = {"node": node, "port": port, "domain": decl['domain']}
    for name, decl in model['interfaces']['outputs'].items():
        node, port = endpoint(decl['from']['occurrence'], decl['from']['port'], {})
        interfaces['outputs'][name] = {"node": node, "port": port, "domain": decl['domain'],
                                       "generative": decl['generative']}

    # --- topological order of one invocation --------------------------------
    adjacency = defaultdict(list)
    indegree = {n: 0 for n in nodes}
    for e in edges:
        adjacency[e['from']['node']].append(e['to']['node'])
        indegree[e['to']['node']] += 1
    queue = deque(sorted(n for n in nodes if indegree[n] == 0))
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in sorted(adjacency[n]):
            indegree[m] -= 1
            if indegree[m] == 0:
                queue.append(m)
    if len(order) != len(nodes):
        raise ValueError(f"{model['model']}: cyclic graph, {len(nodes) - len(order)} node(s) "
                         f"in a cycle — no D1 (V6 rejection)")

    declared = {q['source']['name'] for q in model['quantities'].values()
                if q['source']['kind'] == 'external'}
    out = {"schema": "tensorspine-d1/2.0",
           "model": model['model'],
           "catalog": model['catalog'],
           "assignment": {k: v for k, v in assignment.items() if k in declared},
           "nodes": nodes,
           "edges": edges,
           "interfaces": interfaces,
           "topological_order": order}
    if instances:
        out["instances"] = instances
    return out


def run(model_paths, catalog_bases, output=None, assignment=None,
        models_base=catalog_mod.DEFAULT_MODELS):
    """Emit D1 for each model. Returns (failed, skipped).

    A template with no assignment has no single D1 — it has one per
    admissible assignment — so it is skipped rather than failed.
    """
    failed = 0
    skipped = 0
    for path in model_paths:
        name = os.path.basename(path)
        with open(path, encoding='utf-8') as f:
            document = json.load(f)
        try:
            cat = catalog_mod.load_for(path, document, catalog_bases, models_base=models_base)
        except catalog_mod.CatalogError as e:
            failed += 1
            print(f"  {name:34s} catalog refused: {e}")
            continue
        unset = missing_assignment(document, assignment)
        if unset:
            skipped += 1
            print(f"  {name:34s} skipped: needs --assign for {unset}")
            continue
        try:
            document = emit(path, cat, assignment)
        except (ValueError, KeyError, OSError) as e:
            failed += 1
            print(f"  {name:34s} failed: {e}")
            continue
        if output:
            target = (os.path.join(output, os.path.basename(path).replace('.json', '.d1.json'))
                      if os.path.isdir(output) else output)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
                f.write('\n')
            where = f"  -> {target}"
        else:
            where = ""
        print(f"  {name:34s} {len(document['nodes'])} nodes, "
              f"{len(document['edges'])} edges, acyclic{where}")
    return failed, skipped

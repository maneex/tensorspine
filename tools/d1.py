"""`--d1`: emit D1 (§7), the EXPANDED graph of a model.

The model document is not the graph — it is a generator. Read naively, without
unrolling `for_each` or evaluating index expressions, it shows fewer nodes than
reality and apparent cycles. D1 is the form every consumer reads: viewer,
porting, infrastructure matching.

Identifiers follow §5.2 rule 2: a root occurrence by its name, a generated one
as `<composition>/<site>[<i>=<v>,...]` with indices in name order, an
occurrence of a template prefixed by its instance. The graph is a set (§5.2
rule 4): the listing here is the canonical one — nodes by identifier, edges by
(source, destination) — whatever the order of the document's members. A
binding is emitted only where the occurrences it names are (rule 3).
"""
import itertools
import json
import os
import re
from collections import defaultdict, deque

import catalog as catalog_mod
import model as model_mod
import schema as schema_mod
from expr import (UNRESOLVED, contract_value, index_grid, missing_assignment,
                  model_condition, model_value, resolve_quantities, static_argument)

MAX_DEPTH = 8


def emit(model_path, cat, assignment=None, _prefix="", _depth=0, _stack=()):
    """The D1 document of one model.

    `assignment` supplies the external quantities by name; a template contract
    receives its own from the arguments of the occurrence that invokes it.
    """
    model = model_mod.load(model_path)
    assignment = assignment or {}
    quantities = resolve_quantities(model, assignment)

    def value(e, env=None):
        return model_value(e, quantities, env)

    def static(v, env=None):
        return static_argument(v, quantities, env)

    def identity(key):
        """Canonical identity of an occurrence key (§5.2 rule 2)."""
        if key[0] == 'root':
            return key[1]
        _, composition, site, indices = key
        template = ",".join(f"{k}={v}" for k, v in indices)      # already sorted by name
        return f"{composition}/{site}[{template}]"

    # --- unroll the occurrences; a guarded-out site is remembered as absent -
    keys = {}
    absent = set()
    for name, o in model['occurrences'].items():
        if 'when' in o:
            truth = model_condition(o['when'], quantities, {})
            if truth is UNRESOLVED:
                raise ValueError(f"{name}: `when` does not resolve")
            if not truth:
                absent.add(('root', name))
                continue
        keys[('root', name)] = o
    for comp_name, comp in model['compositions'].items():
        names, ranges = index_grid(comp['indices'], quantities)
        for combo in itertools.product(*ranges):
            env = dict(zip(names, combo))
            for site_name, site in comp['occurrences'].items():
                key = ('gen', comp_name, site_name, tuple(sorted(env.items())))
                if 'when' in site:
                    truth = model_condition(site['when'], quantities, env)
                    if truth is UNRESOLVED:
                        raise ValueError(f"{comp_name}.{site_name}{env}: `when` does not resolve")
                    if not truth:
                        absent.add(key)
                        continue
                keys[key] = site

    edges = []
    instances = {}
    inputs_of = {}        # (instance key, port) -> [(node, port)]: fan-out into a template
    outputs_of = {}       # (instance key, port) -> (node, port)
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
        nodes_sub = sub['d1']['nodes']
        edges.extend(sub['d1']['edges'])
        instances[instance] = {"contract": occurrence['contract'],
                               "arguments": dict(sub_assignment)}
        instances.update(sub['d1'].get('instances') or {})
        for port_name, port in sub['d1']['interfaces']['inputs'].items():
            inputs_of[(key, port_name)] = [(e['node'], e['port']) for e in port['to']]
        for port_name, port in sub['d1']['interfaces']['outputs'].items():
            outputs_of[(key, port_name)] = (port['node'], port['port'])
        keys_sub = nodes_sub
        for n, v in keys_sub.items():
            instances.setdefault('__nodes__', {})[n] = v

    nodes = instances.pop('__nodes__', {}) if instances else {}
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

    def sources(sel, port, env):
        key = select(sel, env)
        if (key, port) in outputs_of:
            return key, [outputs_of[(key, port)]]
        return key, [(_prefix + identity(key), port)]

    def destinations(sel, port, env):
        key = select(sel, env)
        if (key, port) in inputs_of:
            return key, list(inputs_of[(key, port)])
        return key, [(_prefix + identity(key), port)]

    for bid, binding in model['bindings']['values'].items():
        for env in loop_envs(binding, bid):
            src_key, srcs = sources(binding['from']['occurrence'], binding['from']['port'], env)
            dst_key, dsts = destinations(binding['to']['occurrence'], binding['to']['port'], env)
            if src_key in absent or dst_key in absent:
                continue                                      # §5.2 rule 3
            for src_node, src_port in srcs:
                for dst_node, dst_port in dsts:
                    edges.append({"rule": _prefix + bid,
                                  "from": {"node": src_node, "port": src_port},
                                  "to": {"node": dst_node, "port": dst_port}})

    interfaces = {"inputs": {}, "outputs": {}}
    for name, decl in model['interfaces']['inputs'].items():
        to = []
        for endpoint in decl['to']:
            _key, dsts = destinations(endpoint['occurrence'], endpoint['port'], {})
            to.extend({"node": n, "port": p} for n, p in dsts)
        entry = {"to": to, "kind": decl['kind']}
        if 'stream' in decl:
            entry['stream'] = decl['stream']
        if decl.get('fragmented'):
            entry['fragmented'] = True
        interfaces['inputs'][name] = entry
    for name, decl in model['interfaces']['outputs'].items():
        _key, srcs = sources(decl['from']['occurrence'], decl['from']['port'], {})
        node, port = srcs[0]
        interfaces['outputs'][name] = {"node": node, "port": port,
                                       "generative": decl['generative']}

    # --- canonical listing (§5.2 rule 4) and one topological order ---------
    nodes = dict(sorted(nodes.items()))
    edges.sort(key=lambda e: (e['from']['node'], e['from']['port'], e['to']['node'], e['to']['port']))
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

    declared = {name for name, q in model['quantities'].items()
                if q['source']['kind'] == 'external'}
    graph = {"nodes": nodes, "edges": edges, "interfaces": interfaces, "topological_order": order}
    if instances:
        graph["instances"] = instances
    return {"schema": "tensorspine-derived/2.0",
            "model": model['model'],
            "catalog": model['catalog'],
            "assignment": {k: v for k, v in assignment.items() if k in declared},
            "d1": graph}


def output_name(model_path, suffix):
    """`llama3-8b.d1.json`; for a template in its versioned directory,
    `decoder-causal-yarn@1.0.0.d1.json`."""
    base = os.path.basename(model_path)[:-5]
    if re.fullmatch(r'\d+\.\d+\.\d+', base):
        base = f"{os.path.basename(os.path.dirname(model_path))}@{base}"
    return f"{base}.{suffix}.json"


def self_check(document, schema_dir):
    """An emitter validates its own output against the derived schema before
    writing it: a document it cannot vouch for is not written."""
    if schema_dir is None:
        return []
    schema_path = schema_mod.locate(schema_dir, 'derived')
    if schema_path is None:
        return [f"no schema with $id ending in /derived.json under {schema_dir}/"]
    return [schema_mod.format_error(e)
            for e in schema_mod.check_document(schema_path, document, schema_mod.registry(schema_dir))]


def run(model_paths, catalog_bases, output=None, assignment=None, models_base=None,
        schema_dir=None):
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
        except (ValueError, KeyError, OSError, model_mod.ModelError) as e:
            failed += 1
            print(f"  {name:34s} failed: {e}")
            continue
        problems = self_check(document, schema_dir)
        if problems:
            failed += 1
            print(f"  {name:34s} the emitted document is off the derived schema:")
            for line in problems[:5]:
                print(f"      {line}")
            continue
        if output:
            target = (os.path.join(output, output_name(path, 'd1'))
                      if os.path.isdir(output) else output)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
                f.write('\n')
            where = f"  -> {target}"
        else:
            where = ""
        print(f"  {name:34s} {len(document['d1']['nodes'])} nodes, "
              f"{len(document['d1']['edges'])} edges, acyclic{where}")
    return failed, skipped

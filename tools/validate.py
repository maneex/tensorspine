"""`--validate`: the gate. Grammar first, then meaning.

Two stages, in that order, because the second one assumes the first:

  1. structural — the document satisfies its JSON Schema. A missing field, a
     wrong type, a property that does not exist.
  2. semantic — what the grammar cannot see. Does this name designate
     something, do these shapes unify, is this graph acyclic.

Everything reported here is a refusal with its cause, never advice: §8.1 makes
explicit refusal the normative obligation, and I7 forbids silent defaults. What
is legal but questionable belongs to `--lint`, which never blocks.

Coverage of the semantic stage: V1 resolution, V2 arguments, V3p precision
admissibility, V4 shape unification, V5 index domains, V6 acyclicity, V7
totality and uniqueness of bindings, V9 member compatibility, V13 contract
graph, V15 key references. This is not yet the complete validator of §15:
shapes are unified by axis identity and extent only, with no declared views or
permutations.
"""
import itertools
import json
import os
from collections import defaultdict, deque

import catalog as catalog_mod
import schema as schema_mod
from expr import (contract_condition, contract_value, index_grid,
                  missing_assignment, model_condition, model_value,
                  resolve_quantities, static_argument)

MAX_CONTRACT_DEPTH = 8


def structural(model_path, schema_dir, role='model'):
    """Stage 1. Returns the list of error lines, empty when it conforms."""
    schema_path = schema_mod.locate(schema_dir, role)
    if schema_path is None:
        return [f"no schema with $id ending in /{role}.json under {schema_dir}/"]
    reg = schema_mod.registry(schema_dir)
    return [schema_mod.format_error(e) for e in schema_mod.check(schema_path, model_path, reg)]


def semantic(model_path, cat, assignment=None):
    """Stage 2. Returns (errors, stats)."""
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    errors = []
    stats = {}

    def fail(code, message):
        errors.append(f"[{code}] {message}")

    quantities = resolve_quantities(model, assignment)

    def value(e, env=None):
        return model_value(e, quantities, env)

    def static(v, env=None):
        return static_argument(v, quantities, env)

    # --- V13: the contract graph is acyclic and of bounded depth ----------
    def body_of(definition):
        return catalog_mod.body_path(model_path, definition)

    def contract_dependencies(path):
        try:
            with open(path, encoding='utf-8') as f:
                body = json.load(f)
        except OSError:
            return None
        names = set()
        for o in body['occurrences'].values():
            names.add(o['contract']['name'])
        for c in body['compositions'].values():
            for o in c['occurrences'].values():
                names.add(o['contract']['name'])
        return names

    def walk(name, stack, depth):
        definition = cat['contracts'].get(name)
        if definition is None or 'model' not in definition:
            return
        if name in stack:
            fail('V13', f"contract cycle: {' -> '.join(list(stack) + [name])}")
            return
        if depth > MAX_CONTRACT_DEPTH:
            fail('V13', f"contract nesting deeper than {MAX_CONTRACT_DEPTH} at '{name}'")
            return
        deps = contract_dependencies(body_of(definition))
        if deps is None:
            fail('V13', f"body not found for delegated contract '{name}'")
            return
        for d in sorted(deps):
            walk(d, stack + (name,), depth + 1)

    every_occurrence = list(model['occurrences'].values()) + [
        o for c in model['compositions'].values() for o in c['occurrences'].values()]
    seen_contracts = set()
    for o in every_occurrence:
        name = o['contract']['name']
        if name not in seen_contracts:
            seen_contracts.add(name)
            walk(name, (), 1)
    stats['composite_contracts'] = sum(
        1 for c in seen_contracts if 'model' in (cat['contracts'].get(c) or {}))

    # --- expansion: the document is a generator, not the graph ------------
    sites = {}
    for name, o in model['occurrences'].items():
        sites[('root', name)] = o
    for comp_name, comp in model['compositions'].items():
        names, ranges = index_grid(comp['indices'], quantities)
        for combo in itertools.product(*ranges):
            env = dict(zip(names, combo))
            for site_name, site in comp['occurrences'].items():
                # A guarded site that does not fire is not an occurrence. D1
                # applies the same rule, so both commands see one graph.
                if 'when' in site and not model_condition(site['when'], quantities, env):
                    continue
                sites[('gen', comp_name, site_name, tuple(sorted(env.items())))] = site
    stats['occurrences'] = len(sites)

    # --- V1/V2: contracts and arguments -----------------------------------
    resolved = {}
    for key, o in sites.items():
        name = o['contract']['name']
        definition = catalog_mod.contract(cat, o['contract'])
        if definition is None:
            fail('V1', f"contract absent from catalog: {name}")
            continue
        if 'model' in definition:
            # A delegated contract is synthesised from its body's interface.
            # §4.6 requires tensors, state ports, cost and partitions to be
            # DERIVED from the expanded body; this stage does not descend, so
            # it reports none of them. Known shortfall, not a decision.
            with open(body_of(definition), encoding='utf-8') as f:
                body = json.load(f)
            definition = {
                "version": definition['version'],
                "arguments": {q['source']['name']: {"required": True, "affects_template": True}
                              for q in body['quantities'].values()
                              if q['source']['kind'] == 'external'},
                "ports": {
                    "inputs": {k: {"role": "activation.hidden",
                                   "domain": {"kind": v['domain']['kind'], "from": {"self": True}}}
                               for k, v in body['interfaces']['inputs'].items()},
                    "outputs": {k: {"role": "activation.hidden",
                                    "domain": {"kind": v['domain']['kind'], "from": {"self": True}}}
                                for k, v in body['interfaces']['outputs'].items()}},
                "parameters": {}, "constants": {}, "state_ports": {}, "partitions": []}
        if definition['version'] != o['contract']['version']:
            fail('V1', f"{name}: version {o['contract']['version']} "
                       f"!= catalog {definition['version']}")
        args = {}
        for arg_name, arg_value in o['arguments'].items():
            if arg_name not in definition['arguments']:
                fail('V2', f"{name}: unknown argument '{arg_name}'")
            else:
                args[arg_name] = static(arg_value)
        for arg_name, decl in definition['arguments'].items():
            if arg_name in args:
                continue
            if 'default' in decl:
                args[arg_name] = contract_value(decl['default'], args)
            elif decl['required']:
                fail('V2', f"{name} @{key}: required argument missing '{arg_name}'")
        resolved[key] = (name, definition, args)

    def loop_envs(binding):
        if 'for_each' not in binding:
            return [{}]
        names, ranges = index_grid(binding['for_each'], quantities)
        return [dict(zip(names, combo)) for combo in itertools.product(*ranges)]

    def select(sel, env):
        if sel['kind'] == 'root':
            return ('root', sel['occurrence'])
        return ('gen', sel['composition'], sel['occurrence'],
                tuple(sorted((k, value(v, env)) for k, v in sel['indices'].items())))

    # --- V1/V4/V7: value edges --------------------------------------------
    producers = {}
    edges = []
    edge_count = 0
    for bid, binding in model['bindings']['values'].items():
        for env in loop_envs(binding):
            edge_count += 1
            src_key = select(binding['from']['occurrence'], env)
            dst_key = select(binding['to']['occurrence'], env)
            for key, port, side, label in ((src_key, binding['from']['port'], 'outputs', 'from'),
                                           (dst_key, binding['to']['port'], 'inputs', 'to')):
                if key not in resolved:
                    fail('V1', f"{bid}{env}: {label} occurrence does not exist {key}")
                    continue
                name, definition, _ = resolved[key]
                if port not in definition['ports'][side]:
                    fail('V1', f"{bid}: {name} has no {side[:-1]} port '{port}'")
            # V4: shapes unify by axis identity and exact extent
            if src_key in resolved and dst_key in resolved:
                src_name, src_def, src_args = resolved[src_key]
                dst_name, dst_def, dst_args = resolved[dst_key]
                src_port = src_def['ports']['outputs'].get(binding['from']['port'])
                dst_port = dst_def['ports']['inputs'].get(binding['to']['port'])
                if src_port and dst_port and 'shape' in src_port and 'shape' in dst_port:
                    src_shape = tuple((a['axis'], contract_value(a['extent'], src_args))
                                      for a in src_port['shape']['axes'])
                    dst_shape = tuple((a['axis'], contract_value(a['extent'], dst_args))
                                      for a in dst_port['shape']['axes'])
                    if src_shape != dst_shape:
                        fail('V4', f"{bid}: shapes do not unify "
                                   f"{src_name}.{binding['from']['port']}{list(src_shape)} -> "
                                   f"{dst_name}.{binding['to']['port']}{list(dst_shape)}")
            target = (dst_key, binding['to']['port'])
            if target in producers:
                fail('V7', f"input port fed twice: {target[0]}.{target[1]} "
                           f"by {producers[target]} and {bid}")
            producers[target] = bid
            edges.append((src_key, binding['from']['port'],
                          dst_key, binding['to']['port'], bid))
    stats['edges'] = edge_count

    for name, decl in model['interfaces']['inputs'].items():
        key = select(decl['to']['occurrence'], {})
        if key not in resolved:
            fail('V1', f"interface {name}: occurrence does not exist")
            continue
        if decl['to']['port'] not in resolved[key][1]['ports']['inputs']:
            fail('V1', f"interface {name}: port does not exist")
        target = (key, decl['to']['port'])
        if target in producers:
            fail('V7', f"port fed by both an interface and an edge: {target}")
        producers[target] = f"interface:{name}"
    for name, decl in model['interfaces']['outputs'].items():
        key = select(decl['from']['occurrence'], {})
        if key not in resolved:
            fail('V1', f"output {name}: occurrence does not exist")
            continue
        if decl['from']['port'] not in resolved[key][1]['ports']['outputs']:
            fail('V1', f"output {name}: port does not exist")

    # V7: every required input port has a producer
    for key, (name, definition, args) in resolved.items():
        for port_name, port in definition['ports']['inputs'].items():
            if port.get('optional'):
                continue
            if 'present_when' in port and not contract_condition(port['present_when'], args):
                continue
            if (key, port_name) not in producers:
                fail('V7', f"input port with no producer: {name}@{key}.{port_name}")

    # --- V6: acyclicity ----------------------------------------------------
    adjacency = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set(resolved)
    for src_key, _sp, dst_key, _dp, _bid in edges:
        if src_key in nodes and dst_key in nodes:
            adjacency[src_key].append(dst_key)
            indegree[dst_key] += 1
    queue = deque(n for n in nodes if indegree[n] == 0)
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in adjacency[n]:
            indegree[m] -= 1
            if indegree[m] == 0:
                queue.append(m)
    if len(order) != len(nodes):
        fail('V6', f"value cycle: {len(nodes) - len(order)} occurrence(s) in a cycle")
    stats['dag'] = (len(order) == len(nodes))

    # --- V5: domain propagation and declared transforms (§14.4) -----------
    domains = {}
    for name, decl in model['interfaces']['inputs'].items():
        key = select(decl['to']['occurrence'], {})
        domains[(key, decl['to']['port'])] = decl['domain']['kind']
    incoming = {}
    for src_key, src_port, dst_key, dst_port, bid in edges:
        incoming.setdefault(dst_key, []).append((src_key, src_port, dst_port, bid))
    for node in order:
        if node not in resolved:
            continue
        name, definition, args = resolved[node]
        transforms = {t['from_port']: t for t in definition.get('domain_transforms', [])}
        for src_key, src_port, dst_port, bid in incoming.get(node, []):
            source = domains.get((src_key, src_port))
            declared = definition['ports']['inputs'].get(dst_port, {}).get('domain', {}).get('kind')
            if source is None:
                continue
            if declared and declared != 'inherit' and declared != source:
                fail('V5', f"{bid}: {name}.{dst_port} expects {declared}, receives {source}")
            domains[(node, dst_port)] = source
        # The primary domain is that of an input port which is not transformed.
        primary = None
        for port_name in definition['ports']['inputs']:
            if port_name in transforms:
                continue
            if (node, port_name) in domains:
                primary = domains[(node, port_name)]
                break
        for port_name, port in definition['ports']['outputs'].items():
            kind = port['domain']['kind']
            domains[(node, port_name)] = primary if kind == 'inherit' else kind
        # A foreign-domain input is legal only if a transform declares it.
        for src_key, src_port, dst_port, bid in incoming.get(node, []):
            source = domains.get((src_key, src_port))
            if source is None or primary is None or source == primary:
                continue
            if dst_port not in transforms:
                fail('V5', f"{bid}: {name}.{dst_port} receives domain '{source}' while the "
                           f"primitive works in '{primary}', and no domain_transform declares it")
    stats['resolved_domains'] = len(domains)

    # --- V7/V9: parameters -------------------------------------------------
    slots = {}
    ties = 0

    def shape_identity(shape, args):
        return tuple((a['axis'], contract_value(a['extent'], args)) for a in shape['axes'])

    for tid, binding in model['bindings']['parameters'].items():
        for env in loop_envs(binding):
            signatures = []
            for member in binding['members']:
                key = select(member['occurrence'], env)
                if key not in resolved:
                    fail('V1', f"parameter {tid}: occurrence does not exist")
                    continue
                name, definition, args = resolved[key]
                if member['parameter'] not in definition['parameters']:
                    fail('V7', f"parameter {tid}: {name} has no parameter "
                               f"'{member['parameter']}'")
                    continue
                slot = (key, member['parameter'])
                if slot in slots:
                    fail('V7', f"slot {name}.{member['parameter']} bound twice "
                               f"({slots[slot]}, {tid})")
                slots[slot] = tid
                param = definition['parameters'][member['parameter']]
                # The dtype is not carried by the contract: it comes from the
                # binding — hence shared by construction between members of one
                # identity — or from the default of the role.
                signatures.append((name, member['parameter'], param['role'],
                                   shape_identity(param['shape'], args)))
            if len(signatures) > 1:
                ties += 1
                base = signatures[0]
                for other in signatures[1:]:
                    if other[3] != base[3]:
                        fail('V9', f"{tid}: incompatible shapes {base[3]} vs {other[3]}")
                    elif other[2] != base[2]:
                        fail('V9', f"{tid}: incompatible roles {base[2]} vs {other[2]}")
    for key, (name, definition, args) in resolved.items():
        for param_name, param in definition['parameters'].items():
            if 'present_when' in param and not contract_condition(param['present_when'], args):
                continue
            if (key, param_name) not in slots:
                fail('V7', f"unbound parameter slot: {name}@{key}.{param_name}")
    stats['parameter_slots'] = len(slots)
    stats['tensors'] = len(model['bindings']['parameters'])
    stats['shared'] = ties

    # --- V3p: precision admissibility (catalog gives a set, model a value) -
    policy = cat.get('precision', {})

    def dtype_values(d):
        """Possible values of a dtype expression: a literal, or the domain of
        a quantity. Returns a marker when the set cannot be bounded."""
        if d is None:
            return None
        if isinstance(d, str):
            return [d]
        if isinstance(d, dict) and 'quantity' in d:
            q = model['quantities'].get(d['quantity'])
            if q is None:
                return 'UNKNOWN'
            src = q.get('source', {})
            if src.get('kind') == 'literal':
                return [src['value']]
            domain = q.get('domain') or {}
            if domain.get('kind') == 'set':
                return list(domain['values'])
            return 'UNBOUNDED'
        return None

    checked = 0
    for tid, binding in model['bindings']['parameters'].items():
        member = binding['members'][0]
        key = select(member['occurrence'], loop_envs(binding)[0])
        if key not in resolved:
            continue
        name, definition, _ = resolved[key]
        param = definition['parameters'].get(member['parameter'])
        if param is None:
            continue
        rule = policy.get(param['role'])
        if rule is None:
            fail('V3p', f"{tid}: role '{param['role']}' has no precision rule in the catalog")
            continue
        values = dtype_values(binding.get('dtype'))
        if values is None:
            continue                       # catalog default, admissible by construction
        if values in ('UNKNOWN', 'UNBOUNDED'):
            fail('V3p', f"{tid}: precision {values.lower()} — not checkable")
            continue
        bad = [v for v in values if v not in rule['admissible']]
        if bad:
            fail('V3p', f"{tid}: precision {bad} outside the admissible set of role "
                        f"'{param['role']}' {rule['admissible']}")
        checked += 1
    for name, o in model['occurrences'].items():
        contract_name = o['contract']['name']
        for port_name, dtype in (o.get('dtypes') or {}).items():
            definition = cat['contracts'][contract_name]
            ports = {**definition['ports']['inputs'], **definition['ports']['outputs']}
            if port_name not in ports:
                fail('V3p', f"occurrence {name}: unknown port '{port_name}'")
                continue
            rule = policy.get(ports[port_name]['role'])
            values = dtype_values(dtype)
            if rule and isinstance(values, list):
                bad = [v for v in values if v not in rule['admissible']]
                if bad:
                    fail('V3p', f"occurrence {name}.{port_name}: precision {bad} "
                                f"outside admissible {rule['admissible']}")
                checked += 1
    stats['precisions_checked'] = checked

    # --- V7/V9/V15: states -------------------------------------------------
    state_slots = {}
    state_identities = 0
    for sid, binding in model['bindings']['states'].items():
        binding_envs = loop_envs(binding)
        state_identities += len(binding_envs)
        for env in binding_envs:
            for member in binding['members']:
                key = select(member['occurrence'], env)
                if key not in resolved:
                    fail('V1', f"state {sid}: occurrence does not exist")
                    continue
                name, definition, args = resolved[key]
                port = definition['state_ports'].get(member['state'])
                if port is None:
                    fail('V1', f"state {sid}: {name} has no state port '{member['state']}'")
                    continue
                if not contract_condition(port['present_when'], args):
                    fail('V7', f"state {sid}: port '{member['state']}' absent "
                               f"for these arguments")
                    continue
                slot = (key, member['state'])
                if slot in state_slots:
                    fail('V7', f"state port {name}.{member['state']} bound twice")
                state_slots[slot] = sid
                applicable = [r for r in port['rules'] if contract_condition(r['when'], args)]
                if not applicable:
                    fail('V9', f"state {sid}: no applicable rule")
                elif len(applicable) > 1:
                    fail('V9', f"state {sid}: {len(applicable)} applicable rules "
                               f"— the law is ambiguous")
        declared_axes = {a['name'] for a in binding['keys']['axes']}
        for n in binding['keys']['sharing']['equal_on']:
            if n not in declared_axes:
                fail('V15', f"{sid}: equal_on '{n}' is unknown")
        for n in binding['liveness']['classes_by']:
            if n not in declared_axes:
                fail('V15', f"{sid}: classes_by '{n}' is unknown")
    for key, (name, definition, args) in resolved.items():
        for state_name, port in definition['state_ports'].items():
            if not contract_condition(port['present_when'], args):
                continue
            if port.get('internal'):
                continue
            if (key, state_name) not in state_slots:
                fail('V7', f"unbound state port: {name}@{key}.{state_name}")
    stats['state_slots'] = len(state_slots)
    stats['state_identities'] = state_identities

    return errors, stats


def run(model_paths, schema_dir, catalog_bases, assignment=None, max_errors=20):
    """Both stages over several documents. Returns (failed, skipped).

    A parametric body with no assignment is skipped, not failed: it is a family
    of graphs, and refusing it would report a defect where there is none. The
    skip is printed, never silent (I7).
    """
    cat = catalog_mod.load(*catalog_bases)
    failed = 0
    skipped = 0
    for path in model_paths:
        name = os.path.basename(path)
        problems = structural(path, schema_dir)
        if problems:
            failed += 1
            print(f"  {name}")
            print(f"    schema      {len(problems)} error(s)")
            for line in problems[:max_errors]:
                print(f"      {line}")
            if len(problems) > max_errors:
                print(f"      … {len(problems) - max_errors} more")
            continue                       # meaning assumes grammar; stop here
        with open(path, encoding='utf-8') as f:
            unset = missing_assignment(json.load(f), assignment)
        if unset:
            skipped += 1
            print(f"  {name:34s} schema ok; semantic needs --assign for {unset}")
            continue
        errors, stats = semantic(path, cat, assignment)
        summary = " | ".join(f"{k}={v}" for k, v in stats.items())
        if errors:
            failed += 1
            print(f"  {name}")
            print(f"    schema      ok")
            print(f"    semantic    {len(errors)} error(s)   {summary}")
            for line in errors[:max_errors]:
                print(f"      {line}")
            if len(errors) > max_errors:
                print(f"      … {len(errors) - max_errors} more")
        else:
            print(f"  {name:34s} ok   {summary}")
    return failed, skipped

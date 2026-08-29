"""`--validate`: the gate. Grammar first, then meaning.

Two stages, in that order, because the second one assumes the first:

  1. structural — the document satisfies its JSON Schema. A missing field, a
     wrong type, a property that does not exist.
  2. semantic — what the grammar cannot see. Does this name designate
     something, do these shapes unify, is this graph acyclic.

Everything reported here is a refusal with its cause, never advice: §8.1 makes
explicit refusal the normative obligation, and I7 forbids silent defaults. What
is legal but questionable belongs to `--lint`, which never blocks.

Coverage of the semantic stage: V1 resolution, V2 arguments, V3 argument
types (records recursively, defaults applied first), V3p precision
admissibility, V4 shape unification, V5 index domains, V6 acyclicity, V7
totality and uniqueness of bindings, V9 member compatibility and one instance
key per identity, V13 contract graph. This is not yet the complete validator of §15:
shapes are unified by axis identity and extent only, with no declared views or
permutations.
"""
import itertools
import json
import os
from collections import Counter, defaultdict, deque

import catalog as catalog_mod
import schema as schema_mod
from expr import (UNRESOLVED, contract_condition, contract_value, index_grid,
                  missing_assignment, model_condition, model_value,
                  resolve_quantities, static_argument)

MAX_CONTRACT_DEPTH = 8


# --- template contracts: the interface a template exposes (§4.6) -----------

def _to_contract_expression(e, externals, template):
    """A template default, written over quantities, as a contract expression
    over arguments. None when it reads something a caller cannot supply."""
    if 'literal' in e:
        return e
    if 'quantity' in e:
        q = template['quantities'].get(e['quantity'])
        if q is None:
            return None
        if q['source']['kind'] == 'external':
            return {"argument": q['source']['name']}
        if q['source']['kind'] == 'literal':
            return {"literal": q['source']['value']}
        return None
    if 'op' in e:
        args = [_to_contract_expression(x, externals, template) for x in e['args']]
        if any(a is None for a in args):
            return None
        return {"op": e['op'], "args": args}
    return None


def template_interface(definition, template):
    """The contract a template contract presents to a caller: one argument per
    external quantity of the template, with its type, domain and declared
    default; the template's public ports, shapes to be filled by expansion."""
    externals = {k: q for k, q in template['quantities'].items()
                 if q['source']['kind'] == 'external'}
    arguments = {}
    for q in externals.values():
        src = q['source']
        decl = {"type": q['type'], "required": True, "structural": True}
        if 'domain' in q:
            decl['domain'] = q['domain']
        if 'default' in src:
            default = _to_contract_expression(src['default'], externals, template)
            if default is not None:
                decl['default'] = default
                decl['required'] = False
        arguments[src['name']] = decl
    ports = {}
    for side in ('inputs', 'outputs'):
        ports[side] = {k: {"role": "activation.hidden",
                           "domain": {"kind": v['domain']['kind'], "from": {"self": True}}}
                       for k, v in template['interfaces'][side].items()}
    return {"version": definition['version'], "arguments": arguments, "ports": ports,
            "parameters": {}, "constants": {}, "state_ports": {}, "partitions": []}


def instance_ports(exposed):
    """Contract ports of one instance, carrying the shapes the expanded
    template resolved, so that V4 and V5 apply across the boundary."""
    ports = {}
    for side, entries in exposed.items():
        ports[side] = {}
        for pname, entry in entries.items():
            port = {"role": "activation.hidden",
                    "domain": {"kind": entry['domain'], "from": {"self": True}}}
            if entry['shape'] is not None:
                port['shape'] = {"axes": [{"name": axis.split('.')[-1], "axis": axis,
                                           "nature": "feature", "extent": {"literal": extent}}
                                          for axis, extent in entry['shape']]}
            ports[side][pname] = port
    return ports


def _check_domain(v, domain, label, problems):
    """A value against a declared domain (§4.6: admissibility at the call site)."""
    if domain['kind'] == 'set':
        if v not in domain['values']:
            problems.append(('V3', f"argument '{label}' = {v!r} is outside the set {domain['values']}"))
        return
    for edge, op in (('lower', 'below'), ('upper', 'above')):
        bound = domain.get(edge)
        if bound is None:
            continue
        limit = model_value(bound['value'], {})
        if limit is UNRESOLVED:
            continue
        inside = (v >= limit if edge == 'lower' else v <= limit) if bound['inclusive'] \
            else (v > limit if edge == 'lower' else v < limit)
        if not inside:
            problems.append(('V3', f"argument '{label}' = {v!r} is {op} the domain bound "
                                   f"{limit!r} ({'inclusive' if bound['inclusive'] else 'exclusive'})"))


# --- V2/V3: arguments against their declarations --------------------------

def _resolve_record(declared, given, evaluate, root, path, problems, into=None):
    """Resolve one map of values against one map of declarations: the top
    level of an occurrence, or the fields of a record. Unknown names are V2,
    everything about a value's type is V3. Defaults are applied before any
    check (V2), and an inapplicable field is forbidden, not ignored (I2)."""
    values = {} if into is None else into
    if root is None:                      # the top level: the record is the scope
        root = values
    for arg_name, arg_value in given.items():
        if arg_name not in declared:
            problems.append(('V2', f"unknown argument '{path}{arg_name}'"))
        else:
            values[arg_name] = evaluate(arg_value)
    # Defaults may read other arguments, in any order, acyclically (§4.6).
    pending = [n for n, d in declared.items() if n not in values and 'default' in d]
    while pending:
        progress = False
        for arg_name in list(pending):
            # Paths are absolute (`rope.scaling.kind`): the scope is the whole map.
            v = contract_value(declared[arg_name]['default'], root)
            if v is not None and v is not UNRESOLVED:
                values[arg_name] = v
                pending.remove(arg_name)
                progress = True
        if not progress:
            for arg_name in pending:
                problems.append(('V2', f"default of '{path}{arg_name}' does not resolve"))
            break
    for arg_name, decl in declared.items():
        label = f"{path}{arg_name}"
        applicable = True
        if 'present_when' in decl:
            applicable = contract_condition(decl['present_when'], root)
        if arg_name not in values:
            if decl['required'] and applicable:
                problems.append(('V2', f"required argument missing '{label}'"))
            continue
        if not applicable:
            problems.append(('V3', f"argument '{label}' is present but inapplicable "
                                   f"for these arguments"))
            continue
        before = len(problems)
        _check_type(values[arg_name], decl['type'], label, evaluate, root, problems, values)
        if len(problems) == before and 'domain' in decl and values[arg_name] is not UNRESOLVED:
            _check_domain(values[arg_name], decl['domain'], label, problems)
        if len(problems) > before and decl['type']['kind'] != 'record':
            # Refused once, with its reason; nothing downstream reads it as a value.
            values[arg_name] = UNRESOLVED
    return values


def _check_type(v, t, label, evaluate, root, problems, siblings):
    kind = t['kind']
    if v is UNRESOLVED:
        problems.append(('V3', f"argument '{label}' does not resolve to a value"))
        return
    if kind == 'record':
        if not isinstance(v, dict):
            problems.append(('V3', f"argument '{label}' = {v!r} is not a record"))
            return
        # Records are evaluated already: their fields are values, not expressions.
        # `v` is the very dict held under `root`, so it is filled in place and
        # a condition written as an absolute path sees it.
        given = dict(v)
        v.clear()
        _resolve_record(t['fields'], given, lambda x: x, root, label + '.', problems, into=v)
        return
    if isinstance(v, bool):
        if kind != 'boolean':
            problems.append(('V3', f"argument '{label}' = {v!r} is a boolean, not {kind}"))
        return
    if kind == 'boolean':
        problems.append(('V3', f"argument '{label}' = {v!r} is not a boolean"))
    elif kind == 'cardinality':
        if not isinstance(v, int) or v < 0:
            problems.append(('V3', f"argument '{label}' = {v!r} is not a cardinality "
                                   f"(non-negative integer)"))
    elif kind in ('real', 'physical'):
        if not isinstance(v, (int, float)):
            problems.append(('V3', f"argument '{label}' = {v!r} is not a number"
                                   + (f" of {t['unit']}" if kind == 'physical' else "")))
    elif kind == 'enum':
        if v not in t['values']:
            problems.append(('V3', f"argument '{label}' = {v!r} is not among {t['values']}"))
    elif kind == 'port_reference':
        if not isinstance(v, str) or not v:
            problems.append(('V3', f"argument '{label}' = {v!r} is not a port name"))
    else:
        problems.append(('V3', f"argument '{label}': type '{kind}' is unknown to this validator"))


def resolve_arguments(definition, given, evaluate):
    """The complete, typed argument map of one occurrence, and the (code,
    message) problems found on the way. Every declared default is applied,
    every value is checked against its declared type, records recursively."""
    problems = []
    values = _resolve_record(definition['arguments'], given, evaluate, None, '', problems)
    return values, problems


def structural(model_path, schema_dir, role='model'):
    """Stage 1. Returns the list of error lines, empty when it conforms."""
    schema_path = schema_mod.locate(schema_dir, role)
    if schema_path is None:
        return [f"no schema with $id ending in /{role}.json under {schema_dir}/"]
    reg = schema_mod.registry(schema_dir)
    return [schema_mod.format_error(e) for e in schema_mod.check(schema_path, model_path, reg)]


def semantic(model_path, cat, assignment=None):
    """Stage 2. Returns (errors, stats)."""
    result = analyse(model_path, cat, assignment)
    return result['errors'], result['stats']


def analyse(model_path, cat, assignment=None, _depth=0, _cache=None):
    """Stage 2, in full: errors, stats, and the shapes and domains of the
    public interface ports — what a template contract exposes to its caller.

    A template contract is expanded here at every call site (§4.6): the
    template is analysed under the assignment the arguments make, its own
    bindings are checked for totality, and its parameter and state slots are
    counted into the caller's. Two invocations share nothing."""
    _cache = {} if _cache is None else _cache
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    errors = []
    stats = {}
    merged = Counter()

    def fail(code, message):
        errors.append(f"[{code}] {message}")

    quantities = resolve_quantities(model, assignment)

    def value(e, env=None):
        return model_value(e, quantities, env)

    def static(v, env=None):
        return static_argument(v, quantities, env)

    # --- V13: the contract graph is acyclic and of bounded depth ----------
    def template_of(definition):
        return catalog_mod.template_path(cat, definition)

    def contract_dependencies(path):
        try:
            with open(path, encoding='utf-8') as f:
                template = json.load(f)
        except OSError:
            return None
        names = set()
        for o in template['occurrences'].values():
            names.add(o['contract']['name'])
        for c in template['compositions'].values():
            for o in c['occurrences'].values():
                names.add(o['contract']['name'])
        return names

    def walk(name, stack, depth):
        definition = cat['contracts'].get(name)
        if definition is None or 'template' not in definition:
            return
        if name in stack:
            fail('V13', f"contract cycle: {' -> '.join(list(stack) + [name])}")
            return
        if depth > MAX_CONTRACT_DEPTH:
            fail('V13', f"contract nesting deeper than {MAX_CONTRACT_DEPTH} at '{name}'")
            return
        deps = contract_dependencies(template_of(definition))
        if deps is None:
            fail('V13', f"template not found for template contract '{name}'")
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
        1 for c in seen_contracts if 'template' in (cat['contracts'].get(c) or {}))

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
        template_file = None
        if 'template' in definition:
            # A template contract: its arguments are the template's external
            # quantities, with their types, domains and declared defaults.
            template_file = template_of(definition)
            with open(template_file, encoding='utf-8') as f:
                template = json.load(f)
            definition = template_interface(definition, template)
        if definition['version'] != o['contract']['version']:
            fail('V1', f"{name}: version {o['contract']['version']} "
                       f"!= catalog {definition['version']}")
        env = dict(key[3]) if key[0] == 'gen' else {}
        args, problems = resolve_arguments(definition, o['arguments'],
                                           lambda v: static(v, env))
        for code, message in problems:
            fail(code, f"{name} @{key}: {message}")
        if template_file is not None and not problems:
            # Expansion at the call site: the template under this assignment.
            if _depth + 1 > MAX_CONTRACT_DEPTH:
                fail('V13', f"{name} @{key}: contract nesting deeper than {MAX_CONTRACT_DEPTH}")
            else:
                sub_assignment = {k: v for k, v in args.items() if v is not UNRESOLVED}
                cache_key = (template_file, json.dumps(sub_assignment, sort_keys=True, default=str))
                if cache_key not in _cache:
                    _cache[cache_key] = analyse(template_file, cat, sub_assignment,
                                                _depth + 1, _cache)
                sub = _cache[cache_key]
                for line in sub['errors']:
                    errors.append(f"{line}  (in instance {name} @{key})")
                for k, v in sub['stats'].items():
                    if isinstance(v, int) and not isinstance(v, bool):
                        merged[k] += v
                definition = dict(definition)
                definition['ports'] = instance_ports(sub['ports'])
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

    # --- V7/V9: states -------------------------------------------------
    state_slots = {}
    state_identities = 0
    instance_keys = {}
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
            # V9: one identity, one instance key. The key is derived (§4.4):
            # the identity's indices × the contract's key axes of its members,
            # which must therefore agree.
            key_axes = None
            for member in binding['members']:
                mkey = select(member['occurrence'], env)
                if mkey not in resolved:
                    continue
                port = resolved[mkey][1]['state_ports'].get(member['state'])
                if port is None:
                    continue
                if key_axes is None:
                    key_axes = tuple(port['key_axes'])
                elif tuple(port['key_axes']) != key_axes:
                    fail('V9', f"state {sid}: members keyed on {list(key_axes)} and "
                               f"{port['key_axes']} cannot share one allocation")
            if key_axes is not None:
                instance_keys[sid + (f"{env}" if env else "")] = (
                    tuple(binding['identity'].get('indices', {})) + key_axes)
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

    # What this document exposes to a caller: its interface ports, resolved.
    ports = {'inputs': {}, 'outputs': {}}
    for side, sel_key, port_side in (('inputs', 'to', 'inputs'), ('outputs', 'from', 'outputs')):
        for pname, decl in model['interfaces'][side].items():
            key = select(decl[sel_key]['occurrence'], {})
            entry = {'domain': decl['domain']['kind'], 'shape': None}
            if key in resolved:
                _n, definition, args = resolved[key]
                port = definition['ports'][port_side].get(decl[sel_key]['port'])
                if port and 'shape' in port:
                    entry['shape'] = shape_identity(port['shape'], args)
            ports[side][pname] = entry

    # Slots and states of expanded templates count with the caller's (§4.6).
    for k, v in merged.items():
        if k in stats and isinstance(stats[k], int) and not isinstance(stats[k], bool):
            stats[k] += v
    return {'errors': errors, 'stats': stats, 'ports': ports, 'instance_keys': instance_keys}


def check_assignment(model, assignment):
    """An assignment against the external quantities it supplies: types and
    domains, as at any call site (§4.6). Returns error lines."""
    problems = []
    for q in model['quantities'].values():
        src = q['source']
        if src['kind'] != 'external' or src['name'] not in (assignment or {}):
            continue
        v = assignment[src['name']]
        before = len(problems)
        _check_type(v, q['type'], src['name'], lambda x: x, {}, problems, {})
        if len(problems) == before and 'domain' in q:
            _check_domain(v, q['domain'], src['name'], problems)
    return [f"[{code}] assignment: {message}" for code, message in problems]


def run(model_paths, schema_dir, catalog_bases, assignment=None, max_errors=20,
        models_base=catalog_mod.DEFAULT_MODELS):
    """Both stages over several documents. Returns (failed, skipped).

    A template with no assignment is skipped, not failed: it is a family
    of graphs, and refusing it would report a defect where there is none. The
    skip is printed, never silent (I7).
    """
    cat = catalog_mod.load(*catalog_bases, schema_dir=schema_dir, models_base=models_base)
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
            document = json.load(f)
        unset = missing_assignment(document, assignment)
        if unset:
            skipped += 1
            print(f"  {name:34s} schema ok; semantic needs --assign for {unset}")
            continue
        errors = check_assignment(document, assignment)
        if not errors:
            errors, stats = semantic(path, cat, assignment)
        else:
            stats = {}
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

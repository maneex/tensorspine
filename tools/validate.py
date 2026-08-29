"""`--validate`: the gate. Grammar first, then meaning.

Two stages, in that order, because the second one assumes the first:

  1. structural — the document satisfies its JSON Schema. A missing field, a
     wrong type, a property that does not exist. Before that, the JSON layer:
     a duplicate member name is refused (V12), never the last value kept.
  2. semantic — what the grammar cannot see. Does this name designate
     something, do these shapes unify, is this graph acyclic.

Everything reported here is a refusal with its cause, never advice: §8.1 makes
explicit refusal the normative obligation, and I7 forbids silent defaults. What
is legal but questionable belongs to `--lint`, which never blocks.

Coverage of the semantic stage (§6): V1 resolution — catalog bases, contracts,
templates, occurrences, ports, streams; V2 arguments and defaults; V3 argument
types and domains (records recursively, defaults applied first, inapplicable
fields refused); V4 shape unification; V5 indexing domains as (kind, stream)
with the declared transforms; V6 acyclicity; V7 totality and uniqueness of
bindings, state slots included; V9 member compatibility; V10 resolvable
ranges, guards and derivations; V11 a literal quantity against its declared
derivation; V13 no dangling output; V14 precision admissibility on parameter
and state identities; V15 tying compatibility; V16 a carried state on a
fragmented stream. Bindings inherit the presence of the occurrences they name
(§5.2 rule 3). Template contracts are expanded at every call site (§4.6).
"""
import itertools
import json
import os
from collections import Counter, defaultdict, deque

import catalog as catalog_mod
import model as model_mod
import schema as schema_mod
from expr import (UNRESOLVED, contract_condition, contract_value, index_grid,
                  missing_assignment, model_condition, model_value,
                  quantity_references, resolve_quantities, static_argument)

MAX_CONTRACT_DEPTH = 8


# --- template contracts: the interface a template exposes (§4.6) -----------

def _to_contract_expression(e, template):
    """A template default, written over quantities, as a contract expression
    over arguments. None when it reads something a caller cannot supply."""
    if 'literal' in e:
        return e
    if 'quantity' in e:
        q = template['quantities'].get(e['quantity'])
        if q is None:
            return None
        if q['source']['kind'] == 'external':
            return {"argument": e['quantity']}
        if q['source']['kind'] == 'literal':
            return {"literal": q['source']['value']}
        return None
    if 'op' in e:
        args = [_to_contract_expression(x, template) for x in e['args']]
        if any(a is None for a in args):
            return None
        return {"op": e['op'], "args": args}
    return None


def template_interface(definition, template):
    """The contract a template contract presents to a caller: one argument per
    external quantity of the template, with its type, domain and declared
    default; the template's public inputs as input ports with their kinds, its
    public outputs as output ports whose domains expansion resolves (§4.6)."""
    arguments = {}
    for name, q in template['quantities'].items():
        if q['source']['kind'] != 'external':
            continue
        decl = {"type": q['type'], "required": True, "structural": True}
        if 'domain' in q:
            decl['domain'] = q['domain']
        if 'default' in q['source']:
            default = _to_contract_expression(q['source']['default'], template)
            if default is not None:
                decl['default'] = default
                decl['required'] = False
        arguments[name] = decl
    ports = {'inputs': {}, 'outputs': {}}
    for k, v in template['interfaces']['inputs'].items():
        ports['inputs'][k] = {"role": "activation.hidden",
                              "domain": {"kind": v['kind'], "from": {"self": True}}}
    for k in template['interfaces']['outputs']:
        ports['outputs'][k] = {"role": "activation.hidden",
                               "domain": {"kind": "inherit", "from": {"self": True}}}
    return {"version": definition['version'], "arguments": arguments, "ports": ports,
            "parameters": {}, "constants": {}, "state_ports": {}, "partitions": []}


def instance_ports(exposed):
    """Contract ports of one instance, carrying the kinds and shapes the
    expanded template resolved, so that V4 and V5 apply across the boundary."""
    ports = {}
    for side, entries in exposed.items():
        ports[side] = {}
        for pname, entry in entries.items():
            port = {"role": "activation.hidden",
                    "domain": {"kind": entry['kind'] or 'inherit', "from": {"self": True}}}
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
        # A literal is an instance of its type: 32.0 is not a cardinality (V3).
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
    else:
        problems.append(('V3', f"argument '{label}': type '{kind}' is unknown to this validator"))


def resolve_arguments(definition, given, evaluate):
    """The complete, typed argument map of one occurrence, and the (code,
    message) problems found on the way. Every declared default is applied,
    every value is checked against its declared type, records recursively."""
    problems = []
    values = _resolve_record(definition['arguments'], given, evaluate, None, '', problems)
    return values, problems


def duplicate_keys(model_path):
    """The JSON layer (V12): a duplicate member name, or None."""
    try:
        with open(model_path, encoding='utf-8') as f:
            json.load(f, object_pairs_hook=model_mod._pairs)
    except model_mod.ModelError as e:
        return str(e)
    return None


def structural(model_path, schema_dir, role='model'):
    """Stage 1. Returns the list of error lines, empty when it conforms."""
    schema_path = schema_mod.locate(schema_dir, role)
    if schema_path is None:
        return [f"no schema with $id ending in /{role}.json under {schema_dir}/"]
    dup = duplicate_keys(model_path)
    if dup:
        return [f"[V12] {dup}"]
    reg = schema_mod.registry(schema_dir)
    return [schema_mod.format_error(e) for e in schema_mod.check(schema_path, model_path, reg)]


def semantic(model_path, cat, assignment=None):
    """Stage 2. Returns (errors, stats)."""
    result = analyse(model_path, cat, assignment)
    return result['errors'], result['stats']


def _shape_identity(shape, args):
    """A shape as V4 compares it: axis identity and extent, position by position."""
    return tuple((a['axis'], contract_value(a['extent'], args)) for a in shape['axes'])


def _present(element, args):
    return contract_condition(element['present_when'], args) if 'present_when' in element else True


def _dtype_values(model, d):
    """Possible values of a dtype expression: a literal, or the values of an
    enum quantity. A string marker when the set cannot be bounded."""
    if d is None:
        return None
    if isinstance(d, str):
        return [d]
    q = model['quantities'].get(d['quantity'])
    if q is None:
        return 'UNKNOWN'
    if q['type']['kind'] != 'enum':
        return 'NOT_AN_ENUM'
    src = q.get('source', {})
    if src.get('kind') == 'literal':
        return [src['value']]
    return list(q['type']['values'])


def analyse(model_path, cat, assignment=None, _depth=0, _cache=None):
    """Stage 2, in full: errors, stats, the derived domains and shapes of the
    public interface ports — what a template contract exposes to its caller —
    and the carried states.

    A template contract is expanded here at every call site (§4.6): the
    template is analysed under the assignment the arguments make, its own
    bindings are checked for totality, and its parameter and state slots are
    counted into the caller's. Two invocations share nothing."""
    _cache = {} if _cache is None else _cache
    empty = {'errors': [], 'stats': {}, 'ports': {'inputs': {}, 'outputs': {}},
             'instance_keys': {}, 'carried': {}, 'advisories': []}
    try:
        model = model_mod.load(model_path)
    except model_mod.ModelError as e:
        code = 'V12' if ('duplicate' in str(e) or 'declared both' in str(e)) else 'V1'
        return dict(empty, errors=[f"[{code}] {e}"])
    errors = []
    advisories = []
    stats = {}
    merged = Counter()

    def fail(code, message):
        errors.append(f"[{code}] {message}")

    quantities = resolve_quantities(model, assignment)
    for code, message in check_quantities(model, quantities):
        fail(code, message)

    def value(e, env=None):
        return model_value(e, quantities, env)

    def static(v, env=None):
        return static_argument(v, quantities, env)

    # --- the contract citation graph is acyclic and of bounded depth (§4.6) --
    def template_of(definition):
        return catalog_mod.template_path(cat, definition)

    def contract_dependencies(path):
        try:
            template = model_mod.load(path)
        except (OSError, model_mod.ModelError):
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
            fail('V1', f"contract cycle: {' -> '.join(list(stack) + [name])}")
            return
        if depth > MAX_CONTRACT_DEPTH:
            fail('V1', f"contract nesting deeper than {MAX_CONTRACT_DEPTH} at '{name}'")
            return
        deps = contract_dependencies(template_of(definition))
        if deps is None:
            fail('V1', f"template not found for template contract '{name}'")
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
    # A guarded site that does not fire is not an occurrence, and is remembered
    # as absent: a binding naming it is not emitted there (§5.2 rule 3), which
    # is not a reference failure. An unknown name or an index outside the
    # ranges still is (V1). A guard that cannot be decided is a refusal.
    sites = {}
    absent = set()
    for name, o in model['occurrences'].items():
        if 'when' in o:
            truth = model_condition(o['when'], quantities, {})
            if truth is UNRESOLVED:
                fail('V10', f"{name}: `when` does not resolve")
                continue
            if not truth:
                absent.add(('root', name))
                continue
        sites[('root', name)] = o
    for comp_name, comp in model['compositions'].items():
        names, ranges = index_grid(comp['indices'], quantities)
        for combo in itertools.product(*ranges):
            env = dict(zip(names, combo))
            for site_name, site in comp['occurrences'].items():
                key = ('gen', comp_name, site_name, tuple(sorted(env.items())))
                if 'when' in site:
                    truth = model_condition(site['when'], quantities, env)
                    if truth is UNRESOLVED:
                        fail('V10', f"{comp_name}.{site_name}{env}: `when` does not resolve")
                        continue
                    if not truth:
                        absent.add(key)
                        continue
                sites[key] = site
    stats['occurrences'] = len(sites)

    def where(key):
        if key[0] == 'root':
            return key[1]
        return f"{key[1]}/{key[2]}[" + ",".join(f"{k}={v}" for k, v in key[3]) + "]"

    # --- V1/V2: contracts and arguments -----------------------------------
    resolved = {}
    sub_results = {}
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
            template = model_mod.load(template_file)
            definition = template_interface(definition, template)
        if definition['version'] != o['contract']['version']:
            fail('V1', f"{name}: version {o['contract']['version']} "
                       f"!= catalog {definition['version']}")
        env = dict(key[3]) if key[0] == 'gen' else {}
        args, problems = resolve_arguments(definition, o['arguments'],
                                           lambda v: static(v, env))
        for code, message in problems:
            fail(code, f"{name} @{where(key)}: {message}")
        if template_file is not None and not problems:
            # Expansion at the call site: the template under this assignment.
            if _depth + 1 > MAX_CONTRACT_DEPTH:
                fail('V1', f"{name} @{where(key)}: contract nesting deeper than {MAX_CONTRACT_DEPTH}")
            else:
                sub_assignment = {k: v for k, v in args.items() if v is not UNRESOLVED}
                cache_key = (template_file, json.dumps(sub_assignment, sort_keys=True, default=str))
                if cache_key not in _cache:
                    _cache[cache_key] = analyse(template_file, cat, sub_assignment,
                                                _depth + 1, _cache)
                sub = _cache[cache_key]
                for line in sub['errors']:
                    errors.append(f"{line}  (in instance {name} @{where(key)})")
                advisories.extend(f"{line}  (in instance {name})" for line in sub['advisories'])
                for k, v in sub['stats'].items():
                    if isinstance(v, int) and not isinstance(v, bool):
                        merged[k] += v
                definition = dict(definition)
                definition['ports'] = instance_ports(sub['ports'])
                sub_results[key] = sub
        resolved[key] = (name, definition, args)

    def loop_envs(binding, label=''):
        """The index environments a rule fires in: its `for_each` grid, kept
        where its `when` holds. An undecidable guard is a V10 refusal."""
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
                fail('V10', f"{label}{env}: `when` does not resolve")
            elif truth:
                kept.append(env)
        return kept

    def select(sel, env):
        if sel['kind'] == 'root':
            return ('root', sel['occurrence'])
        return ('gen', sel['composition'], sel['occurrence'],
                tuple(sorted((k, value(v, env)) for k, v in sel['indices'].items())))

    def port_shape(port, args):
        return _shape_identity(port['shape'], args) if 'shape' in port else None

    # --- V1/V4/V7: value edges --------------------------------------------
    producers = {}
    consumed = set()
    edges = []
    for bid, binding in model['bindings']['values'].items():
        for env in loop_envs(binding, bid):
            src_key = select(binding['from']['occurrence'], env)
            dst_key = select(binding['to']['occurrence'], env)
            if src_key in absent or dst_key in absent:
                continue                                       # §5.2 rule 3
            ok = True
            for key, port, side, label in ((src_key, binding['from']['port'], 'outputs', 'from'),
                                           (dst_key, binding['to']['port'], 'inputs', 'to')):
                if key not in resolved:
                    fail('V1', f"{bid}{env}: {label} occurrence does not exist {where(key)}")
                    ok = False
                    continue
                name, definition, _ = resolved[key]
                if port not in definition['ports'][side]:
                    fail('V1', f"{bid}: {name} has no {side[:-1]} port '{port}'")
                    ok = False
            if not ok:
                continue
            # V4: shapes unify by axis identity and exact extent
            src_name, src_def, src_args = resolved[src_key]
            dst_name, dst_def, dst_args = resolved[dst_key]
            src_shape = port_shape(src_def['ports']['outputs'][binding['from']['port']], src_args)
            dst_shape = port_shape(dst_def['ports']['inputs'][binding['to']['port']], dst_args)
            if src_shape is not None and dst_shape is not None and src_shape != dst_shape:
                fail('V4', f"{bid}: shapes do not unify "
                           f"{src_name}.{binding['from']['port']}{list(src_shape)} -> "
                           f"{dst_name}.{binding['to']['port']}{list(dst_shape)}")
            target = (dst_key, binding['to']['port'])
            if target in producers:
                fail('V7', f"input port fed twice: {where(dst_key)}.{binding['to']['port']} "
                           f"by {producers[target]} and {bid}")
            producers[target] = bid
            consumed.add((src_key, binding['from']['port']))
            edges.append((src_key, binding['from']['port'],
                          dst_key, binding['to']['port'], bid))
    stats['edges'] = len(edges)

    # --- interfaces: edges like any other, seeds of the streams -----------
    fragmented = {n for n, i in model['interfaces']['inputs'].items() if i.get('fragmented')}
    seeds = {}
    for name, decl in model['interfaces']['inputs'].items():
        stream = decl.get('stream', name)
        if 'stream' in decl and decl['stream'] not in model['interfaces']['inputs']:
            fail('V1', f"input {name}: joins unknown stream '{decl['stream']}'")
        for endpoint in decl['to']:
            key = select(endpoint['occurrence'], {})
            if key not in resolved:
                fail('V1', f"input {name}: occurrence does not exist")
                continue
            if endpoint['port'] not in resolved[key][1]['ports']['inputs']:
                fail('V1', f"input {name}: port '{endpoint['port']}' does not exist")
                continue
            target = (key, endpoint['port'])
            if target in producers:
                fail('V7', f"input {name}: port {where(key)}.{endpoint['port']} also fed "
                           f"by {producers[target]}")
            producers[target] = f"input:{name}"
            seeds[target] = (decl['kind'], stream)
    for name, decl in model['interfaces']['outputs'].items():
        key = select(decl['from']['occurrence'], {})
        if key not in resolved:
            fail('V1', f"output {name}: occurrence does not exist")
            continue
        if decl['from']['port'] not in resolved[key][1]['ports']['outputs']:
            fail('V1', f"output {name}: port '{decl['from']['port']}' does not exist")
            continue
        consumed.add((key, decl['from']['port']))

    # V7: every present input port has a producer; V13: every present output a consumer
    for key, (name, definition, args) in resolved.items():
        for port_name, port in definition['ports']['inputs'].items():
            if _present(port, args) and (key, port_name) not in producers:
                fail('V7', f"input port with no producer: {name}@{where(key)}.{port_name}")
        for port_name, port in definition['ports']['outputs'].items():
            if _present(port, args) and (key, port_name) not in consumed:
                fail('V13', f"output consumed by nothing: {name}@{where(key)}.{port_name}")

    # --- V6: acyclicity ----------------------------------------------------
    adjacency = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set(resolved)
    for src_key, _sp, dst_key, _dp, _bid in edges:
        if src_key in nodes and dst_key in nodes:
            adjacency[src_key].append(dst_key)
            indegree[dst_key] += 1
    queue = deque(sorted(n for n in nodes if indegree[n] == 0))
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

    # --- V5: streams, transforms and the occurrence's own domain (§5.3) ----
    domains = dict(seeds)                 # (key, port) -> (kind, stream)
    own = {}                              # key -> the occurrence's own domain
    incoming = {}
    for src_key, src_port, dst_key, dst_port, bid in edges:
        incoming.setdefault(dst_key, []).append((src_key, src_port, dst_port, bid))
    for node in order:
        name, definition, args = resolved[node]
        transforms = {t['from_port']: t for t in definition.get('domain_transforms', [])}
        for src_key, src_port, dst_port, bid in incoming.get(node, []):
            source = domains.get((src_key, src_port))
            if source is None:
                fail('V5', f"{bid}: the domain of {where(src_key)}.{src_port} is undetermined")
                continue
            domains[(node, dst_port)] = source
        agree = set()
        for port_name, port in definition['ports']['inputs'].items():
            if not _present(port, args):
                continue
            dm = domains.get((node, port_name))
            if dm is None:
                continue
            declared = port['domain']['kind']
            if declared != 'inherit' and declared != dm[0]:
                fail('V5', f"{name}@{where(node)}.{port_name} expects {declared}, receives "
                           f"{dm[0]} (stream '{dm[1]}')")
            if port_name not in transforms:
                agree.add(dm)
        if len(agree) > 1:
            fail('V5', f"{name}@{where(node)}: inputs in different domains {sorted(agree)}, "
                       f"and no domain_transform declares it")
        mine = next(iter(agree)) if agree else None
        own[node] = mine
        merged_to = {t['to_port']: t for t in definition.get('domain_transforms', [])
                     if t['relation'] == 'merge'}
        for port_name, port in definition['ports']['outputs'].items():
            pd = port['domain']
            kind = pd['kind']
            if kind == 'inherit' and 'port' in pd.get('from', {}):
                dm = domains.get((node, pd['from']['port']))
            elif kind == 'inherit':
                dm = mine
            elif port_name in merged_to:
                src = domains.get((node, merged_to[port_name]['from_port']))
                dm = (kind, src[1]) if src else None
            else:
                dm = (kind, mine[1]) if mine else None
            if dm is None:
                if _present(port, args):
                    fail('V5', f"{name}@{where(node)}.{port_name}: domain undetermined")
                continue
            domains[(node, port_name)] = dm
    for name, decl in model['interfaces']['outputs'].items():
        key = select(decl['from']['occurrence'], {})
        dm = domains.get((key, decl['from']['port']))
        if dm is None:
            continue
        if decl['generative'] and dm[0] != 'token':
            fail('V5', f"output {name}: generative, but of kind {dm[0]}")
    stats['resolved_domains'] = len(domains)

    # --- V7/V14/V15: parameters -------------------------------------------
    policy = cat.get('precision', {})
    slots = {}
    ties = 0
    tensor_identities = 0
    checked = 0
    tensor_instances = []               # one per identity instance, for D3
    state_instances = []                # one per identity instance, for D4

    def instance_name(symbol, env):
        indices = symbol.get('indices', {})
        if not indices:
            return symbol['name']
        return symbol['name'] + '[' + ','.join(f"{k}={value(v, env)}" for k, v in sorted(indices.items())) + ']'

    for tid, binding in model['bindings']['parameters'].items():
        values = _dtype_values(model, binding.get('dtype'))
        if isinstance(values, str):
            fail('V14', f"{tid}: dtype selector is {values.lower().replace('_', ' ')}")
            values = None
        for env in loop_envs(binding, tid):
            members = [(select(m['occurrence'], env), m['parameter']) for m in binding['members']]
            if any(key in absent for key, _ in members):
                continue                                       # §5.2 rule 3
            tensor_identities += 1
            signatures = []
            tensor_instances.append({'identity': instance_name(binding['tensor'], env), 'rule': tid,
                                     'members': [(key, pname) for key, pname in members if key in resolved],
                                     'dtype': binding.get('dtype')})
            for key, pname in members:
                if key not in resolved:
                    fail('V1', f"parameter {tid}: occurrence does not exist {where(key)}")
                    continue
                name, definition, args = resolved[key]
                param = definition['parameters'].get(pname)
                if param is None:
                    fail('V7', f"parameter {tid}: {name} has no parameter '{pname}'")
                    continue
                if not _present(param, args):
                    fail('V7', f"parameter {tid}: slot '{pname}' absent for these arguments")
                    continue
                slot = (key, pname)
                if slot in slots:
                    fail('V7', f"slot {name}.{pname} bound twice ({slots[slot]}, {tid})")
                slots[slot] = tid
                signatures.append((name, pname, param['role'],
                                   _shape_identity(param['shape'], args), param['sharing']))
                rule = policy.get(param['role'])
                if rule is not None and values:
                    bad = [v for v in values if v not in rule['admissible']]
                    if bad:
                        fail('V14', f"{tid}: precision {bad} outside the admissible set of "
                                    f"role '{param['role']}' {rule['admissible']}")
                    checked += 1
            if len(signatures) > 1:
                ties += 1
                for s in signatures:
                    if s[4]['kind'] != 'shareable':
                        fail('V15', f"{tid}: {s[0]}.{s[1]} is exclusive, it cannot be tied")
                    for o in signatures:
                        if o is s:
                            continue
                        if o[2] not in s[4].get('roles', []):
                            fail('V15', f"{tid}: {s[0]}.{s[1]} does not share with role '{o[2]}'")
                        if o[3] != s[3]:
                            fail('V15', f"{tid}: incompatible shapes {list(s[3])} vs {list(o[3])}")
    for key, (name, definition, args) in resolved.items():
        for param_name, param in definition['parameters'].items():
            if _present(param, args) and (key, param_name) not in slots:
                fail('V7', f"unbound parameter slot: {name}@{where(key)}.{param_name}")
    stats['parameter_slots'] = len(slots)
    stats['tensors'] = tensor_identities
    stats['shared'] = ties

    # --- D5, first derivation: elements, operations per element (§4.1) ----
    # Two operations per weight element per element of the output domain,
    # scaled by the activated fraction of a sparsity unit (§4.5); a contract
    # adds only the corrections the inventory cannot see, every applying one.
    def elements(shape, args, multiplicity=None):
        n = 1
        for a in shape['axes']:
            extent = contract_value(a['extent'], args)
            if extent is UNRESOLVED or not isinstance(extent, (int, float)):
                return None
            n *= extent
        if multiplicity is not None:
            m = contract_value(multiplicity, args)
            if m is UNRESOLVED or not isinstance(m, (int, float)):
                return None
            n *= m
        return n

    ops = Counter()
    for key, (name, definition, args) in resolved.items():
        fraction = {}
        for unit in definition.get('sparsity', []):
            activated = contract_value(unit['activated_per_element'], args)
            for pname in unit['unit']['parameters']:
                param = definition['parameters'].get(pname)
                if param is None:
                    continue
                extent = None
                for a in param['shape']['axes']:
                    if a['axis'] == unit['unit']['axis']:
                        extent = contract_value(a['extent'], args)
                if activated is not UNRESOLVED and extent:
                    fraction[pname] = activated / extent
        for param_name, param in definition['parameters'].items():
            if not _present(param, args):
                continue
            n = elements(param['shape'], args, param.get('multiplicity'))
            if n is None:
                continue
            ops['element'] += 2 * n * fraction.get(param_name, 1)
        for entry in definition.get('logical_cost', []):
            if 'when' in entry and not contract_condition(entry['when'], args):
                continue
            v = contract_value(entry['expression'], args)
            if v is not UNRESOLVED and v is not None:
                ops[entry['per']] += v
    # Resident elements count each tensor identity once (tied tensors once).
    resident = 0
    for tid, binding in model['bindings']['parameters'].items():
        for env in loop_envs(binding, tid):
            member = binding['members'][0]
            key = select(member['occurrence'], env)
            if key not in resolved:
                continue
            _n, definition, args = resolved[key]
            param = definition['parameters'].get(member['parameter'])
            if param is None:
                continue
            n = elements(param['shape'], args, param.get('multiplicity'))
            if n is not None:
                resident += n
    stats['parameter_elements'] = int(resident)
    stats['ops_per_element'] = int(ops['element'])
    stats['ops_per_cached_position'] = int(ops['cached_position'])
    stats['ops_per_sequence'] = int(ops['sequence'])
    stats['ops_per_invocation'] = int(ops['invocation'])

    # --- V7/V9/V14/V16: states -------------------------------------------
    state_slots = {}
    state_identities = 0
    instance_keys = {}
    carried = {}
    for sid, binding in model['bindings']['states'].items():
        values = _dtype_values(model, binding.get('dtype'))
        if isinstance(values, str):
            fail('V14', f"{sid}: dtype selector is {values.lower().replace('_', ' ')}")
            values = None
        binding_envs = loop_envs(binding, sid)
        for env in binding_envs:
            members = [(select(m['occurrence'], env), m['state']) for m in binding['members']]
            if any(key in absent for key, _ in members):
                continue                                       # §5.2 rule 3
            state_identities += 1
            state_instances.append({'identity': instance_name(binding['identity'], env), 'rule': tid if False else sid,
                                    'members': [(key, sname) for key, sname in members if key in resolved],
                                    'dtype': binding.get('dtype'),
                                    'indices': sorted(binding['identity'].get('indices', {}))})
            key_axes = None
            payload = None
            rule_text = None
            indexing = None
            for key, sname in members:
                if key not in resolved:
                    fail('V1', f"state {sid}: occurrence does not exist {where(key)}")
                    continue
                name, definition, args = resolved[key]
                port = definition['state_ports'].get(sname)
                if port is None:
                    fail('V1', f"state {sid}: {name} has no state port '{sname}'")
                    continue
                if not contract_condition(port['present_when'], args):
                    fail('V7', f"state {sid}: port '{sname}' absent for these arguments")
                    continue
                slot = (key, sname)
                if slot in state_slots:
                    fail('V7', f"state port {name}.{sname} bound twice")
                state_slots[slot] = sid
                applicable = [r for r in port['rules'] if contract_condition(r['when'], args)]
                if not applicable:
                    fail('V9', f"state {sid}: no rule of {name}.{sname} applies to these arguments")
                rule = applicable[0] if applicable else None
                # V9: one identity, one instance key, one payload, one rule, one stream.
                if key_axes is None:
                    key_axes = tuple(port['key_axes'])
                elif tuple(port['key_axes']) != key_axes:
                    fail('V9', f"state {sid}: members keyed on {list(key_axes)} and "
                               f"{port['key_axes']} cannot share one allocation")
                shapes = tuple(sorted((c, comp['role'], _shape_identity(comp['shape'], args))
                                      for c, comp in port['payload'].items()))
                if payload is None:
                    payload = shapes
                elif shapes != payload:
                    fail('V9', f"state {sid}: members with different payloads cannot share "
                               f"one allocation")
                text = json.dumps(rule, sort_keys=True) if rule else None
                if rule_text is None:
                    rule_text = text
                elif text != rule_text:
                    fail('V9', f"state {sid}: members under different derivation rules")
                if rule is not None:
                    stream = own.get(key) if 'self' in rule['indexed_by'] \
                        else domains.get((key, rule['indexed_by']['port']))
                    if indexing is None:
                        indexing = stream
                    elif stream != indexing:
                        fail('V9', f"state {sid}: members indexed by different streams")
                if values:
                    for cname, comp in port['payload'].items():
                        rule_p = policy.get(comp['role'])
                        bad = [v for v in values if rule_p and v not in rule_p['admissible']]
                        if bad:
                            fail('V14', f"{sid}: precision {bad} outside the admissible set of "
                                        f"role '{comp['role']}' {rule_p['admissible']}")
                        checked += 1
            if key_axes is not None:
                instance_keys[sid + (f"{env}" if env else "")] = (
                    tuple(binding['identity'].get('indices', {})) + key_axes)
    stats['precisions_checked'] = checked
    for key, (name, definition, args) in resolved.items():
        for state_name, port in definition['state_ports'].items():
            if not contract_condition(port['present_when'], args):
                continue
            if (key, state_name) not in state_slots:
                fail('V7', f"unbound state port: {name}@{where(key)}.{state_name}")
            ca = port.get('carried_across')
            mine = own.get(key)
            if ca and contract_condition(ca['when'], args):
                if mine is None or mine[1] not in fragmented:
                    fail('V16', f"{name}@{where(key)}.{state_name}: carried across fragments, "
                                f"but its stream {mine} is not a fragmented input")
                carried.setdefault(state_slots.get((key, state_name), where(key)), mine)
            elif mine is not None and mine[1] in fragmented:
                applicable = [r for r in port['rules'] if contract_condition(r['when'], args)]
                if applicable and 'self' in applicable[0]['indexed_by']:
                    advisories.append(f"{name}@{where(key)}.{state_name}: a self-indexed state "
                                      f"on the fragmented stream '{mine[1]}' that is not carried "
                                      f"— reset at every fragment")
    stats['state_slots'] = len(state_slots)
    stats['state_identities'] = state_identities

    # What this document exposes to a caller: its interface ports, resolved.
    ports = {'inputs': {}, 'outputs': {}}
    for pname, decl in model['interfaces']['inputs'].items():
        endpoint = decl['to'][0]
        key = select(endpoint['occurrence'], {})
        entry = {'kind': decl['kind'], 'stream': decl.get('stream', pname), 'shape': None}
        if key in resolved:
            _n, definition, args = resolved[key]
            entry['shape'] = port_shape(definition['ports']['inputs'].get(endpoint['port'], {}), args)
        ports['inputs'][pname] = entry
    for pname, decl in model['interfaces']['outputs'].items():
        key = select(decl['from']['occurrence'], {})
        dm = domains.get((key, decl['from']['port']))
        entry = {'kind': dm[0] if dm else None, 'stream': dm[1] if dm else None, 'shape': None}
        if key in resolved:
            _n, definition, args = resolved[key]
            entry['shape'] = port_shape(definition['ports']['outputs'].get(decl['from']['port'], {}), args)
        ports['outputs'][pname] = entry

    # Slots and states of expanded templates count with the caller's (§4.6).
    for k, v in merged.items():
        if k in stats and isinstance(stats[k], int) and not isinstance(stats[k], bool):
            stats[k] += v
    return {'errors': errors, 'stats': stats, 'ports': ports, 'instance_keys': instance_keys,
            'carried': carried, 'advisories': advisories,
            'graph': {'resolved': resolved, 'edges': edges, 'domains': domains, 'own': own,
                      'slots': slots, 'state_slots': state_slots, 'absent': absent,
                      'model': model, 'quantities': quantities, 'fragmented': fragmented,
                      'sub_results': sub_results, 'tensor_instances': tensor_instances,
                      'state_instances': state_instances, 'order': order}}


def check_quantities(model, quantities):
    """Every quantity resolves (V10), reads only declared quantities (V1),
    conforms to its declared type and domain (V3), and — when a literal
    declares how it follows from the others — agrees with that derivation
    (V11). Returns (code, message) problems."""
    problems = []
    declared = set(model['quantities'])
    for name, q in model['quantities'].items():
        src = q['source']
        expression = src.get('expression') if src['kind'] == 'derived' else src.get('derivation')
        if expression is not None:
            for ref in sorted(quantity_references(expression) - declared):
                problems.append(('V1', f"quantity '{name}': derivation reads undeclared quantity '{ref}'"))
        if name not in quantities:
            if src['kind'] == 'derived':
                problems.append(('V10', f"quantity '{name}': derivation does not resolve "
                                        f"(a cycle, or a reference with no value)"))
            continue
        v = quantities[name]
        before = len(problems)
        _check_type(v, q['type'], name, lambda x: x, {}, problems, {})
        if len(problems) == before and 'domain' in q:
            _check_domain(v, q['domain'], name, problems)
        if src['kind'] == 'literal' and 'derivation' in src:
            d = model_value(src['derivation'], quantities)
            if d is UNRESOLVED:
                problems.append(('V10', f"quantity '{name}': its derivation does not resolve"))
            elif d != v:
                problems.append(('V11', f"quantity '{name}' = {v!r} disagrees with its derivation, "
                                        f"which gives {d!r}"))
    return [(code, m.replace("argument '", "quantity '", 1)) for code, m in problems]


def check_assignment(model, assignment):
    """An assignment against the external quantities it supplies: types and
    domains, as at any call site (§4.6). Returns error lines."""
    problems = []
    for name, q in model['quantities'].items():
        if q['source']['kind'] != 'external' or name not in (assignment or {}):
            continue
        v = assignment[name]
        before = len(problems)
        _check_type(v, q['type'], name, lambda x: x, {}, problems, {})
        if len(problems) == before and 'domain' in q:
            _check_domain(v, q['domain'], name, problems)
    return [f"[{code}] assignment: {message}" for code, message in problems]


def run(model_paths, schema_dir, catalog_bases, assignment=None, max_errors=20,
        models_base=None):
    """Both stages over several documents. Returns (failed, skipped).

    A template with no assignment is skipped, not failed: it is a family
    of graphs, and refusing it would report a defect where there is none. The
    skip is printed, never silent (I7).
    """
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
        try:
            cat = catalog_mod.load_for(path, document, catalog_bases, schema_dir, models_base)
        except catalog_mod.CatalogError as e:
            failed += 1
            print(f"  {name}")
            print(f"    catalog     refused: {e}")
            continue
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

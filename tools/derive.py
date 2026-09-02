"""`--derive`: emit the derived document (§7) — D1, the expanded graph, and
D2–D6, the products a valid document and its contracts make computable
without inference code — as one file, validated against the derived schema.

  D3  parameter tensors: one entry per identity instance — members, role,
      selected dtype, sensitivity, shape, elements, bytes, sparsity unit.
  D4  states: one entry per identity instance — law, access geometry, sharing,
      stream, instance key, carrying, payload per cached position, visits.
  D5  logical costs: resident parameters, operations per element and per
      cached position (inventory plus every applying correction, with the
      status the algebra of §2.2 gives the total), sparsity bounds, state
      bytes, and the payload crossing each legal cut.
  D2  values: every value with its shape, role, dtype and domain, the element
      count of every stream as a combination of the inputs' counts (merges
      divide, inserts add), the payload of every structural cut, and the peak
      of live values along D1's order.
  D6  legal cuts of the expanded graph, the partitions every occurrence's
      contract declares where their condition holds, and the information
      loss of flattened axes without factors (O5.10).

Encodings are outside the specification (§7); this one is the repository's.
Template instances are expanded before anything is derived (§5.1): every
product is computed once, over the expanded graph, and an occurrence, tensor,
state, value or cut inside an instance carries the instance's prefix (§5.2
rule 2) — as D1 names it.
"""
import json
import os
from collections import Counter, defaultdict, deque

import catalog as catalog_mod
import d1 as d1_mod
import validate as validate_mod
from expr import UNRESOLVED, contract_condition, contract_value, missing_assignment

BYTES = {'bool': 1, 'u4': 0.5, 'i4': 0.5, 'u8': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
         'fp4': 0.5, 'f8e4m3': 1, 'f8e4m3fn': 1, 'f8e5m2': 1, 'bf16': 2, 'f16': 2, 'f32': 4, 'f64': 8}


def ident(key):
    """`node` as D1 names it: an occurrence inside a template instance is
    prefixed by the instance (§5.2 rule 2); `_expand` wraps such a key as
    ('sub', prefix, key)."""
    if key[0] == 'sub':
        return key[1] + ident(key[2])
    if key[0] == 'root':
        return key[1]
    return f"{key[1]}/{key[2]}[" + ",".join(f"{k}={v}" for k, v in key[3]) + "]"


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not UNRESOLVED else None


def _shape(shape, args):
    out = []
    for a in shape['axes']:
        entry = {"axis": a['axis'], "extent": _num(contract_value(a['extent'], args))}
        if 'factors' in a:
            entry['factors'] = [{"axis": f['axis'], "extent": _num(contract_value(f['extent'], args))}
                                for f in a['factors']]
        out.append(entry)
    return out


def _elements(shape, args, multiplicity=None):
    n = 1
    for a in shape['axes']:
        e = _num(contract_value(a['extent'], args))
        if e is None:
            return None
        n *= e
    if multiplicity is not None:
        m = _num(contract_value(multiplicity, args))
        if m is None:
            return None
        n *= m
    return n


def _dtype(graph, cat, selector, role):
    """The dtype a binding selects, else the role's default."""
    if selector is None:
        return cat['precision'][role]['default']
    if isinstance(selector, str):
        return selector
    v = graph['quantities'].get(selector['quantity'])
    return v if isinstance(v, str) else cat['precision'][role]['default']


def _status(statuses):
    """The status of a sum of qualified values (§2.2): estimate absorbs,
    opposite bounds cancel into an estimate, one-sided bounds survive."""
    s = set(statuses)
    if 'estimate' in s or {'upper_bound', 'lower_bound'} <= s:
        return 'estimate'
    if 'upper_bound' in s:
        return 'upper_bound'
    if 'lower_bound' in s:
        return 'lower_bound'
    return 'exact'


# --- D3 ---------------------------------------------------------------------

def d3(graph, cat):
    resolved = graph['resolved']
    tensors = []
    for inst in graph['tensor_instances']:
        if not inst['members']:
            continue
        key, pname = inst['members'][0]
        name, definition, args = resolved[key]
        param = definition['parameters'][pname]
        unit = None
        for i, u in enumerate(definition.get('sparsity', [])):
            if pname in u['unit']['parameters']:
                activated = _num(contract_value(u['activated_per_element'], args))
                extent = None
                for a in param['shape']['axes']:
                    if a['axis'] == u['unit']['axis']:
                        extent = _num(contract_value(a['extent'], args))
                unit = {"unit": i, "axis": u['unit']['axis'], "activated_per_element": activated,
                        "units": extent,
                        "activated_fraction": (activated / extent) if activated is not None and extent else None}
        dtype = _dtype(graph, cat, inst['dtype'], param['role'])
        n = _elements(param['shape'], args, param.get('multiplicity'))
        entry = {"identity": inst['identity'],
                 "members": [f"{ident(k)}.{p}" for k, p in inst['members']],
                 "contract": name, "slot": pname, "role": param['role'],
                 "sensitivity": cat['precision'][param['role']]['sensitivity'],
                 "dtype": dtype, "shape": _shape(param['shape'], args),
                 "multiplicity": _num(contract_value(param['multiplicity'], args)) if 'multiplicity' in param else 1,
                 "elements": n, "bytes": (n * BYTES[dtype]) if n is not None else None,
                 "tied": len(inst['members']) > 1}
        if unit:
            entry['sparsity'] = unit
        if inst.get('location') is not None:
            entry['location'] = inst['location']
        tensors.append(entry)
    totals = {"tensors": len(tensors),
              "elements": sum(t['elements'] or 0 for t in tensors),
              "bytes": sum(t['bytes'] or 0 for t in tensors),
              "tied": sum(1 for t in tensors if t['tied'])}
    return {"tensors": tensors, "totals": totals}


# --- D4 ---------------------------------------------------------------------

def _visits(rule, source_indexed, law):
    if source_indexed:
        return {"write": "once per element of the source stream, until the source is complete",
                "read": "once per element produced"}
    if law == 'fixed':
        return {"write": "once per element", "read": "once per element"}
    return {"write": "once per new element of its stream", "read": "once per element produced"}


def d4(graph, cat):
    resolved, domains, own = graph['resolved'], graph['domains'], graph['own']
    states = []
    for inst in graph['state_instances']:
        if not inst['members']:
            continue
        key, sname = inst['members'][0]
        name, definition, args = resolved[key]
        port = definition['state_ports'][sname]
        rules = [r for r in port['rules'] if contract_condition(r['when'], args)]
        rule = rules[0] if rules else None
        source_indexed = rule is not None and 'port' in rule['indexed_by']
        stream = (own.get(key) if not source_indexed else domains.get((key, rule['indexed_by']['port']))) if rule else None
        # carried across fragments (§5.3): the contract's carrying condition holds, or the state is
        # indexed by a port whose stream is a fragmented input, which carries it by definition
        carried = ('carried_across' in port and contract_condition(port['carried_across']['when'], args)) \
            or (source_indexed and stream is not None and stream[1] in _fragmented_streams(graph))
        payload = []
        for cname, comp in port['payload'].items():
            dtype = _dtype(graph, cat, inst['dtype'], comp['role'])
            n = _elements(comp['shape'], args)
            payload.append({"component": cname, "role": comp['role'], "dtype": dtype,
                            "shape": _shape(comp['shape'], args), "elements": n,
                            "bytes": (n * BYTES[dtype]) if n is not None else None})
        per_position = sum(c['bytes'] or 0 for c in payload)
        span = _num(contract_value(rule['span'], args)) if rule and 'span' in rule else None
        entry = {"identity": inst['identity'],
                 "members": [f"{ident(k)}.{s}" for k, s in inst['members']],
                 "contract": name, "state": sname,
                 "law": rule['law'] if rule else None, "access": rule['access'] if rule else None,
                 "sharing": rule['sharing'] if rule else None,
                 "stream": {"kind": stream[0], "stream": stream[1]} if stream else None,
                 "indexed_by_source": source_indexed,
                 "indexed_by_port": rule['indexed_by'].get('port') if rule else None,
                 "instance_key": list(inst['indices']) + list(port['key_axes']),
                 "carried_across_fragments": carried,
                 "span": span,
                 "stride": _num(contract_value(rule['stride'], args)) if rule and 'stride' in rule else None,
                 "payload": payload, "bytes_per_cached_position": per_position,
                 "bytes_bounded": (per_position * span) if span else None,
                 "operations": sorted({o['effect'] for o in port['operations'].values()}),
                 "visits": _visits(rule, source_indexed, rule['law'] if rule else None)}
        states.append(entry)
    totals = {"identities": len(states),
              "by_law": dict(Counter(s['law'] for s in states)),
              "append_bytes_per_cached_position": sum(s['bytes_per_cached_position'] for s in states if s['law'] == 'append'),
              "bounded_bytes": sum(s['bytes_bounded'] or 0 for s in states if s['law'] == 'window'),
              "fixed_bytes": sum(s['bytes_per_cached_position'] for s in states if s['law'] == 'fixed'),
              "carried": [s['identity'] for s in states if s['carried_across_fragments']]}
    return {"states": states, "totals": totals}


def _fragmented_streams(graph):
    """The streams the fragmented public inputs introduce or join (§5.3)."""
    return {decl.get('stream', name) for name, decl in graph['model']['interfaces']['inputs'].items()
            if decl.get('fragmented')}


# --- D2 and the cuts ---------------------------------------------------------

# --- expansion --------------------------------------------------------------

def _wrap(prefix, key):
    return ('sub', prefix, key) if prefix else key


def _expand(graph, prefix=''):
    """The analysis graph with every template instance expanded in place (§5.1)
    — on the analysis side, what D1 does on emission. The instance's occurrences
    carry its prefix (§5.2 rule 2); the edges into and out of it are rewired to
    the template's own endpoints; the template's streams take the names of the
    caller's streams that feed them; its tensor and state identities join the
    caller's under the prefix. The result has no `sub_results`; `meta` gives
    every occurrence its families and, for a generated one, its composition
    (prefixed) and indices; `compositions` lists the prefixed composition names
    in declaration order; `inputs_at` and `outputs_at` resolve the level's
    public interfaces to occurrence ports."""
    model, subs = graph['model'], graph['sub_results']
    inner = {key: _expand(sub['graph'], prefix + ident(key) + '/') for key, sub in subs.items()}
    for key, sub in subs.items():
        # the template's streams are the caller's streams that feed its inputs
        rename = {}
        for iname, decl in sub['graph']['model']['interfaces']['inputs'].items():
            fed = graph['domains'].get((key, iname))
            if fed is not None:
                rename[decl.get('stream', iname)] = fed[1]
        ex = inner[key]
        ex['domains'] = {k: (d[0], rename.get(d[1], d[1])) for k, d in ex['domains'].items()}
        ex['own'] = {k: ((d[0], rename.get(d[1], d[1])) if d else d) for k, d in ex['own'].items()}

    def targets(key, port):
        return inner[key]['inputs_at'][port] if key in inner else [(_wrap(prefix, key), port)]

    def source(key, port):
        return inner[key]['outputs_at'][port] if key in inner else (_wrap(prefix, key), port)

    inputs_at = {iname: [t for e in decl['to'] for t in targets(_select(graph, e['occurrence']), e['port'])]
                 for iname, decl in model['interfaces']['inputs'].items()}
    outputs_at = {oname: source(_select(graph, decl['from']['occurrence']), decl['from']['port'])
                  for oname, decl in model['interfaces']['outputs'].items()}
    resolved, domains, own, order, meta = {}, {}, {}, [], {}
    for key, entry in graph['resolved'].items():
        if key in inner:
            continue
        w = _wrap(prefix, key)
        resolved[w] = entry
        own[w] = graph['own'].get(key)
        if key[0] == 'root':
            fams, comp = set(model['occurrences'][key[1]]['families']), None
        else:
            decl = model['compositions'][key[1]]
            fams = set(decl['occurrences'][key[2]]['families']) | set(decl['families'])
            comp = (prefix + key[1], dict(key[3]))
        meta[w] = {'families': fams, 'composition': comp}
    for (key, port), d in graph['domains'].items():
        if key not in inner:
            domains[(_wrap(prefix, key), port)] = d
    for key in graph['order']:
        if key in inner:
            order.extend(inner[key]['order'])
        elif _wrap(prefix, key) in resolved:
            order.append(_wrap(prefix, key))
    edges = []
    for src, sp, dst, dp, bid in graph['edges']:
        s_key, s_port = source(src, sp)
        for t_key, t_port in targets(dst, dp):
            edges.append((s_key, s_port, t_key, t_port, prefix + bid))
    compositions = [prefix + name for name in model['compositions']]
    for ex in inner.values():
        resolved.update(ex['resolved'])
        domains.update(ex['domains'])
        own.update(ex['own'])
        meta.update(ex['meta'])
        edges.extend(ex['edges'])
        compositions.extend(ex['compositions'])

    def instances(kind):
        out = [dict(inst, identity=prefix + inst['identity'],
                    members=[(_wrap(prefix, k), p) for k, p in inst['members']]) for inst in graph[kind]]
        for ex in inner.values():
            out.extend(ex[kind])
        return out

    return dict(graph, resolved=resolved, edges=edges, domains=domains, own=own, order=order,
                sub_results={}, meta=meta, compositions=compositions,
                inputs_at=inputs_at, outputs_at=outputs_at,
                tensor_instances=instances('tensor_instances'),
                state_instances=instances('state_instances'))


def _counts(graph):
    """Element count of every port, as a combination of the inputs' counts:
    seeded by the public inputs, merges divide by their factor, inserts add."""
    resolved, edges, order = graph['resolved'], graph['edges'], graph['order']
    model = graph['model']
    counts = {}
    for name, decl in model['interfaces']['inputs'].items():
        for key, port in graph['inputs_at'][name]:
            counts[(key, port)] = {decl.get('stream', name): 1.0}
    incoming = defaultdict(list)
    for src, sp, dst, dp, bid in edges:
        incoming[dst].append((src, sp, dp))
    for node in order:
        name, definition, args = resolved[node]
        transforms = {t['from_port']: t for t in definition.get('domain_transforms', [])}
        for src, sp, dp in incoming.get(node, []):
            if (src, sp) in counts:
                counts[(node, dp)] = counts[(src, sp)]
        own = None
        for pname in definition['ports']['inputs']:
            if pname not in transforms and (node, pname) in counts:
                own = counts[(node, pname)]
                break
        for pname, port in definition['ports']['outputs'].items():
            c = own
            for t in definition.get('domain_transforms', []):
                if t['to_port'] != pname:
                    continue
                src = counts.get((node, t['from_port']))
                if t['relation'] == 'merge' and src is not None:
                    f = _num(contract_value(t['factor'], args)) or 1
                    c = {k: v / f for k, v in src.items()}
                elif t['relation'] == 'insert' and src is not None:
                    c = dict(own or {})
                    for k, v in src.items():
                        c[k] = c.get(k, 0) + v
            if c is not None:
                counts[(node, pname)] = c
    return counts


def _present_port(definition, pname, args):
    port = definition['ports']['inputs'][pname]
    return contract_condition(port['present_when'], args) if 'present_when' in port else True


def graph_value(graph, e):
    from expr import model_value
    return model_value(e, graph['quantities'], {})


def _select(graph, sel):
    if sel['kind'] == 'root':
        return ('root', sel['occurrence'])
    return ('gen', sel['composition'], sel['occurrence'],
            tuple(sorted((k, graph_value(graph, v)) for k, v in sel['indices'].items())))


def _value_id(key, port):
    """`node.port` as D1 names it — inside a template instance, under the
    instance's prefix (§5.2 rule 2), the graph being expanded."""
    return f"{ident(key)}.{port}"


def _ancestors(nodes, edges):
    up = defaultdict(set)
    for src, _sp, dst, _dp, _bid in edges:
        up[dst].add(src)
    seen = set(nodes)
    queue = deque(nodes)
    while queue:
        n = queue.popleft()
        for m in up[n]:
            if m not in seen:
                seen.add(m)
                queue.append(m)
    return seen


def _structural_cuts(graph):
    """Legal cuts by construction: the ancestor closure of a layer prefix or of
    a family is downward closed, so every crossing edge points out of it."""
    resolved, edges, meta = graph['resolved'], graph['edges'], graph['meta']
    cuts = []
    layers = defaultdict(dict)                    # composition -> occurrence -> its indices
    for key in resolved:
        comp = meta[key]['composition']
        if comp and len(comp[1]) == 1:
            layers[comp[0]][key] = comp[1]
    for comp_name in graph['compositions']:
        if comp_name not in layers:
            continue
        index = next(iter(next(iter(layers[comp_name].values()))))
        values = sorted({idx[index] for idx in layers[comp_name].values()})
        for v in values[:-1]:
            block = {k for k, idx in layers[comp_name].items() if idx[index] <= v}
            cuts.append((f"{comp_name}[{index}<={v}]", "layer", _ancestors(block, edges)))
    families = defaultdict(set)
    for key in resolved:
        for f in meta[key]['families']:
            families[f].add(key)
    for f in sorted(families):
        block = _ancestors(families[f], edges)
        if len(block) < len(resolved):
            cuts.append((f"family:{f}", "family", block))
    return cuts


def d2(graph, cat):
    resolved, edges, domains = graph['resolved'], graph['edges'], graph['domains']
    counts = _counts(graph)
    values = {}
    for src, sp, dst, dp, bid in edges:
        name, definition, args = resolved[src]
        port = definition['ports']['outputs'][sp]
        vid = _value_id(src, sp)
        if vid not in values:
            dom = domains.get((src, sp))
            n = _elements(port['shape'], args) if 'shape' in port else 0
            dtype = cat['precision'][port['role']]['default']
            values[vid] = {"value": vid, "to": [], "shape": _shape(port['shape'], args) if 'shape' in port else [],
                           "role": port['role'], "dtype": dtype, "elements": n,
                           "bytes_per_element": n * BYTES[dtype] if n is not None else None,
                           "domain": {"kind": dom[0], "stream": dom[1]} if dom else None,
                           "count": counts.get((src, sp))}
        values[vid]['to'].append(_value_id(dst, dp))
    streams = {}
    for (key, pname), c in counts.items():
        dom = domains.get((key, pname))
        if dom and dom[1] not in streams:
            streams[dom[1]] = {"kind": dom[0], "count": c}
    # a public input is an edge like any other (§5.3): the value it delivers, named by the input,
    # with the shape of the port it feeds (V4 makes every fed port agree)
    model = graph['model']
    for iname, decl in model['interfaces']['inputs'].items():
        endpoints = graph['inputs_at'][iname]
        if not endpoints:
            continue
        key, pname = endpoints[0]
        name, definition, args = resolved[key]
        port = definition['ports']['inputs'][pname]
        n = _elements(port['shape'], args) if 'shape' in port else 1
        dtype = cat['precision'][port['role']]['default']
        stream = decl.get('stream', iname)
        values[iname] = {"value": iname, "input": iname,
                         "to": [_value_id(k, p) for k, p in endpoints],
                         "shape": _shape(port['shape'], args) if 'shape' in port else [],
                         "role": port['role'], "dtype": dtype, "elements": n,
                         "bytes_per_element": n * BYTES[dtype], "domain": {"kind": decl['kind'], "stream": stream},
                         "count": streams.get(stream, {}).get('count', {stream: 1.0})}
    # required (§7): an input is required for an output when the output is not evaluated
    # without it — evaluated meaning every input port fed, an insert transform's source excepted
    def evaluated(delivered):
        fed = {}
        for iname in model['interfaces']['inputs']:
            if iname in delivered:
                for key, port in graph['inputs_at'][iname]:
                    fed[(key, port)] = True
        done = set()
        for node in graph['order']:
            name, definition, args = resolved[node]
            inserts = {t['from_port'] for t in definition.get('domain_transforms', []) if t.get('relation') == 'insert'}
            ok = True
            for pname in definition['ports']['inputs']:
                if _present_port(definition, pname, args) and (node, pname) not in fed and pname not in inserts:
                    ok = False
                    break
            if ok:
                done.add(node)
                for src, sp, dst, dp, bid in edges:
                    if src == node:
                        fed[(dst, dp)] = True
        return done
    outputs_at = {oname: key for oname, (key, port) in graph['outputs_at'].items()}
    all_inputs = set(model['interfaces']['inputs'])
    for iname in list(values):
        if 'input' in values[iname]:
            without = evaluated(all_inputs - {iname})
            needed = [oname for oname, key in outputs_at.items() if key not in without]
            values[iname]['required_for'] = needed
            values[iname]['required'] = bool(needed)
    # a public output exposes a value whether or not an edge consumes it
    for oname, (key, oport) in graph['outputs_at'].items():
        vid = _value_id(key, oport)
        if vid not in values and key in resolved:
            name, definition, args = resolved[key]
            port = definition['ports']['outputs'][oport]
            dom = domains.get((key, oport))
            n = _elements(port['shape'], args) if 'shape' in port else 0
            dtype = cat['precision'][port['role']]['default']
            values[vid] = {"value": vid, "to": [], "shape": _shape(port['shape'], args) if 'shape' in port else [],
                           "role": port['role'], "dtype": dtype, "elements": n,
                           "bytes_per_element": n * BYTES[dtype] if n is not None else None,
                           "domain": {"kind": dom[0], "stream": dom[1]} if dom else None,
                           "count": counts.get((key, oport))}
        if vid in values:
            values[vid].setdefault('exposed', []).append(oname)
    # the fragment alignment of a fragmented stream (§5.3): every fragment delivers a multiple of
    # the cumulative merge factors of the values on it, so that every merge sees whole groups
    import math
    for sname in _fragmented_streams(graph):
        if sname not in streams:
            continue
        alignment = 1
        for v in values.values():
            dom, c = v.get('domain'), v.get('count') or {}
            if dom and dom['stream'] == sname and c.get(sname):
                alignment = math.lcm(alignment, max(1, round(1 / c[sname])))
        streams[sname]['fragment_alignment'] = alignment
    cuts = []
    for cid, kind, block in _structural_cuts(graph):
        crossing = {}
        for src, sp, dst, dp, bid in edges:
            if src in block and dst not in block:
                vid = _value_id(src, sp)
                crossing.setdefault(vid, values[vid])
        per_invocation = Counter()
        for v in crossing.values():
            for inp, mult in (v['count'] or {}).items():
                per_invocation[inp] += (v['bytes_per_element'] or 0) * mult
        cuts.append({"cut": cid, "kind": kind, "sizes": [len(block), len(resolved) - len(block)],
                     "payload": [{"value": k, "bytes_per_element": v['bytes_per_element'], "count": v['count']}
                                 for k, v in sorted(crossing.items())],
                     "bytes_per_element": sum(v['bytes_per_element'] or 0 for v in crossing.values()),
                     "bytes_per_invocation": dict(per_invocation)})
    return {"streams": streams, "values": list(values.values()), "cuts": cuts,
            "peak_live": _peak_live(graph, values)}


def _peak_live(graph, values):
    """The peak of live values along D1's topological order: at each node, the values
    produced so far and not yet consumed by every consumer — its own outputs included, its
    inputs still held while it runs — sized per element and per invocation like a cut's
    payload. A value a public output exposes is live to the end. The peak is a property of
    this one order (another legal order may peak lower), stated as such."""
    resolved, edges, order = graph['resolved'], graph['edges'], graph['order']
    remaining = Counter()
    for src, sp, dst, dp, bid in edges:
        remaining[_value_id(src, sp)] += 1
    for oname, (key, oport) in graph['outputs_at'].items():
        remaining[_value_id(key, oport)] += 1              # exposed: consumed at the end
    consumes = defaultdict(list)
    for src, sp, dst, dp, bid in edges:
        consumes[dst].append(_value_id(src, sp))
    produces = defaultdict(list)
    for vid in values:
        if 'input' in values[vid]:
            continue
        node, port = vid.rsplit('.', 1)
        produces[node].append(vid)
    for name in graph['model']['interfaces']['inputs']:            # a public input's value is live from the start
        remaining[name] += len(values[name]['to']) if name in values else 0
    live = {name for name in graph['model']['interfaces']['inputs'] if remaining[name]}
    consumers_of_input = defaultdict(list)
    for name, decl in graph['model']['interfaces']['inputs'].items():
        for key, port in graph['inputs_at'][name]:
            consumers_of_input[key].append(name)

    def size(vids):
        per_element = sum(values[v]['bytes_per_element'] or 0 for v in vids)
        per_invocation = Counter()
        for v in vids:
            for inp, mult in (values[v]['count'] or {}).items():
                per_invocation[inp] += (values[v]['bytes_per_element'] or 0) * mult
        return per_element, dict(per_invocation)

    peak = None
    for key in order:
        node = ident(key)
        live |= set(produces[node])
        per_element, per_invocation = size(sorted(live))
        total = sum(per_invocation.values())
        if peak is None or total > peak[0]:
            peak = (total, node, sorted(live), per_element, per_invocation)
        for vid in consumes[key] + consumers_of_input[key]:
            remaining[vid] -= 1
            if remaining[vid] <= 0:
                live.discard(vid)
    if peak is None:
        return {"node": None, "values": [], "bytes_per_element": 0, "bytes_per_invocation": {}}
    _total, node, vids, per_element, per_invocation = peak
    return {"node": node, "values": vids, "bytes_per_element": per_element, "bytes_per_invocation": per_invocation}


# --- D5 ---------------------------------------------------------------------

def d5(graph, cat, products3, products4, products2, stats):
    resolved = graph['resolved']
    corrections = []
    sparsity = []
    for key, (name, definition, args) in resolved.items():
        for i, entry in enumerate(definition.get('logical_cost', [])):
            if 'when' in entry and not contract_condition(entry['when'], args):
                continue
            v = _num(contract_value(entry['expression'], args))
            corrections.append({"node": ident(key), "contract": name, "entry": i, "value": v,
                                "status": entry['status'], "per": entry['per']})
        for i, u in enumerate(definition.get('sparsity', [])):
            activated = _num(contract_value(u['activated_per_element'], args))
            extent = None
            for pname in u['unit']['parameters']:
                for a in definition['parameters'][pname]['shape']['axes']:
                    if a['axis'] == u['unit']['axis']:
                        extent = _num(contract_value(a['extent'], args))
            bound = u['union_per_invocation']
            sparsity.append({"node": ident(key), "contract": name, "unit": i,
                             "activated_per_element": activated, "units": extent,
                             "activated_fraction": (activated / extent) if activated is not None and extent else None,
                             "union_per_invocation": {"value": _num(contract_value(bound['expression'], args)),
                                                      "status": bound['status']}})
    by_per = defaultdict(list)
    for c in corrections:
        by_per[c['per']].append(c['status'])
    operations = {}
    for per, stat in (('element', 'ops_per_element'), ('cached_position', 'ops_per_cached_position'),
                      ('sequence', 'ops_per_sequence'), ('invocation', 'ops_per_invocation')):
        operations[per] = {"value": stats[stat], "status": _status(['exact'] + by_per.get(per, []))}
    t4 = products4['totals']
    return {"parameters": {"elements": products3['totals']['elements'], "bytes": products3['totals']['bytes'],
                           "status": "exact"},
            "operations": operations,
            "corrections": corrections, "sparsity": sparsity,
            "state": {"append_bytes_per_cached_position": t4['append_bytes_per_cached_position'],
                      "bounded_bytes": t4['bounded_bytes'], "fixed_bytes": t4['fixed_bytes'],
                      "status": "exact"},
            "cuts": [{"cut": c['cut'], "bytes_per_element": c['bytes_per_element'],
                      "bytes_per_invocation": c['bytes_per_invocation']} for c in products2['cuts']]}


# --- D6 ---------------------------------------------------------------------

def d6(graph, cat, products2):
    resolved = graph['resolved']
    partitions = []
    loss = []
    for key, (name, definition, args) in resolved.items():
        for p in definition.get('partitions', []):
            if 'when' in p and not contract_condition(p['when'], args):
                continue
            partitions.append({"node": ident(key), "contract": name, "target": p['target'],
                               "communication": p['communication']})
        for pname, param in definition['parameters'].items():
            if 'present_when' in param and not contract_condition(param['present_when'], args):
                continue
            for a in param['shape']['axes']:
                flattened = isinstance(a['extent'], dict) and a['extent'].get('op') == 'multiply'
                if flattened and 'factors' not in a:
                    loss.append({"node": ident(key), "slot": pname, "axis": a['axis'],
                                 "reason": "flattened axis without declared factors (O5.10): "
                                           "partitionability along its factors is unknown"})
    return {"cuts": [{"cut": c['cut'], "kind": c['kind'], "sizes": c['sizes'],
                      "crossing_values": len(c['payload'])} for c in products2['cuts']],
            "partitions": partitions, "information_loss": loss}


# --- entry points -----------------------------------------------------------

def products(model_path, cat, assignment=None):
    result = validate_mod.analyse(model_path, cat, assignment)
    if result['errors']:
        raise ValueError(f"not valid, no products: {result['errors'][0]}")
    graph = _expand(result['graph'])
    p3 = d3(graph, cat)
    p4 = d4(graph, cat)
    p2 = d2(graph, cat)
    p5 = d5(graph, cat, p3, p4, p2, result['stats'])
    p6 = d6(graph, cat, p2)
    document = d1_mod.emit(model_path, cat, assignment)
    document.update({"d2": p2, "d3": p3, "d4": p4, "d5": p5, "d6": p6})
    return document


def run(model_paths, catalog_bases, output=None, assignment=None, models_base=None,
        schema_dir=None):
    """Emit the derived document of each model, D1 to D6. Returns (failed, skipped)."""
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
            doc = products(path, cat, assignment)
        except (ValueError, KeyError, OSError) as e:
            failed += 1
            print(f"  {name:34s} failed: {e}")
            continue
        problems = d1_mod.self_check(doc, schema_dir)
        if problems:
            failed += 1
            print(f"  {name:34s} the emitted document is off the derived schema:")
            for line in problems[:5]:
                print(f"      {line}")
            continue
        if output:
            target = (os.path.join(output, d1_mod.output_name(path, 'derived'))
                      if os.path.isdir(output) else output)
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=1, ensure_ascii=False)
                f.write('\n')
            where = f"  -> {target}"
        else:
            where = ""
        t3, t4, p5 = doc['d3']['totals'], doc['d4']['totals'], doc['d5']
        print(f"  {name:34s} {t3['tensors']} tensors {t3['bytes'] / 2**30:.2f} GiB, "
              f"{t4['identities']} states {t4['append_bytes_per_cached_position'] / 1024:.0f} KiB/position, "
              f"{p5['operations']['element']['value'] / 1e9:.2f} Gop/element, "
              f"{len(doc['d2']['cuts'])} cuts{where}")
    return failed, skipped

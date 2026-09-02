"""The derived document as Python — and nothing else (R01).

A `Graph` is built from a `.derived.json` or, given a model document, from the
products `tools/derive.py` computes in-process. The backend never reads the
model source or the catalog itself: everything it needs must be in D1–D6.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import catalog as catalog_mod   # noqa: E402
import derive                   # noqa: E402

DTYPES = {'bf16': 'bfloat16', 'f16': 'float16', 'f32': 'float32'}


def derive_document(model_path, assignment=None):
    """D1–D6 of a model document, through the language's own tools."""
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    cat = catalog_mod.load_for(model_path, model)
    return derive.products(model_path, cat, assignment)


def load(path, assignment=None):
    """A `Graph` from a derived document, or from a model document derived here."""
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    if doc.get('schema') == 'tensorspine-derived/2.0':
        return Graph(doc)
    return Graph(derive_document(path, assignment))


def edited(model_path, edits, out_dir, suffix='edited'):
    """A copy of a model document with dotted paths replaced (`quantities.d.source.value=64`,
    `compositions.decoder.indices.layer.stop.literal=3`), its catalog bases made absolute so
    the copy can live anywhere. A test convenience, not a semantics: the document is data
    and this edits it; whether the result is valid is the language's verdict, as always."""
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    notes = []
    for path, value in edits.items():
        node = model
        keys = list(path) if isinstance(path, tuple) else [int(k) if k.isdigit() else k for k in path.split('.')]
        for k in keys[:-1]:
            node = node[k]
        last = keys[-1]
        before = node[last] if isinstance(node, list) else node.get(last)
        notes.append(f"{'.'.join(map(str, keys))}: {before!r} -> {value!r}")
        node[last] = value
    here = os.path.dirname(os.path.abspath(model_path))
    model['catalog'] = [{"base": os.path.normpath(os.path.join(here, e['base']))} for e in model['catalog']]
    model['model'] = f"{model['model']}-{suffix}"
    out = os.path.join(out_dir, f"{os.path.basename(model_path)[:-5]}.{suffix}.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=2)
    return out, notes


def truncated(model_path, spec, out_dir):
    """`decoder.layer=3`: the composition index range shortened, with everything that names its
    old extent — every literal quantity equal to the old stop that an index expression addressing
    the composition cites (`layers`; Whisper's `dec_layers`, never its `enc_layers`), and every
    literal equal to the old last index in the bindings and guards that address it."""
    comp_index, stop = spec.split('=')
    comp, index = comp_index.rsplit('.', 1)
    stop = int(stop)
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    old = model['compositions'][comp]['indices'][index]['stop']['literal']
    edits = {f"compositions.{comp}.indices.{index}.stop.literal": stop}

    def quantities(node, out):          # every quantity an expression tree cites
        if isinstance(node, dict):
            if isinstance(node.get('quantity'), str):
                out.add(node['quantity'])
            for v in node.values():
                quantities(v, out)
        elif isinstance(node, list):
            for v in node:
                quantities(v, out)
        return out

    cited = set()

    def cite(node):                     # the index expressions of the composition's occurrences, wherever written
        if isinstance(node, dict):
            if node.get('kind') == 'generated' and node.get('composition') == comp:
                quantities(node.get('indices', {}), cited)
            elif 'site' in node and 'indices' in node:
                quantities(node['indices'], cited)
            for v in node.values():
                cite(v)
        elif isinstance(node, list):
            for v in node:
                cite(v)
    cite(model.get('bindings', {}))
    cite(model['compositions'][comp].get('bindings', {}))
    for occ in model['compositions'][comp]['occurrences'].values():
        quantities(occ.get('when', {}), cited)
    for q in sorted(cited):
        entry = model.get('quantities', {}).get(q)
        if entry and entry['source'].get('kind') == 'literal' and entry['source'].get('value') == old:
            edits[f"quantities.{q}.source.value"] = stop
    # a document that names its last layer by a literal (`layer < 31`, `mlp_r[layer=31]`):
    # every literal equal to the old last index, in the bindings and in that composition
    def walk(node, path):
        if isinstance(node, dict):
            if set(node) == {'literal'} and node['literal'] == old - 1:
                edits[path + ('literal',)] = stop - 1
            for k, v in node.items():
                walk(v, path + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + (i,))
    walk(model.get('bindings', {}), ('bindings',))
    walk(model['compositions'][comp].get('bindings', {}), ('compositions', comp, 'bindings'))
    for site, occ in model['compositions'][comp]['occurrences'].items():
        walk(occ.get('when', {}), ('compositions', comp, 'occurrences', site, 'when'))
    return edited(model_path, edits, out_dir, suffix=f"{stop}{index}s")


class Graph:
    def __init__(self, doc):
        self.doc = doc
        self.model = doc['model']
        d1 = doc['d1']
        self.nodes = d1['nodes']
        self.order = d1['topological_order']
        self.interfaces = d1['interfaces']
        self.sources = {}          # (node, input port) -> value name it is fed by
        self.consumers = {}        # value name -> [(node, port)]
        for e in d1['edges']:
            vname = f"{e['from']['node']}.{e['from']['port']}"
            self.sources[(e['to']['node'], e['to']['port'])] = vname
            self.consumers.setdefault(vname, []).append((e['to']['node'], e['to']['port']))
        self.values = {v['value']: v for v in doc['d2']['values']}
        self.outputs_of = {}
        self.input_values = {}     # the values public inputs deliver (D2, named by the input)
        for vname, v in self.values.items():
            if 'input' in v:
                self.input_values[v['input']] = v
                continue
            node, port = vname.rsplit('.', 1)
            self.outputs_of.setdefault(node, {})[port] = v
        self.streams = doc['d2']['streams']
        self.cuts = doc['d2']['cuts']
        self.tensors = {t['identity']: t for t in doc['d3']['tensors']}
        self.slots_of = {}
        for t in doc['d3']['tensors']:
            for m in t['members']:
                node, slot = m.rsplit('.', 1)
                self.slots_of.setdefault(node, {})[slot] = t['identity']
        self.states = {s['identity']: s for s in doc['d4']['states']}
        self.states_of = {}
        for s in doc['d4']['states']:
            for m in s['members']:
                node, st = m.rsplit('.', 1)
                self.states_of.setdefault(node, {})[st] = s['identity']
        self.d3_totals = doc['d3']['totals']
        self.d4_totals = doc['d4']['totals']
        # interface inputs: the stream each introduces or joins; the ports it feeds
        self.input_stream = {}
        self.fed_by_input = {}     # (node, port) -> input name
        for name, entry in self.interfaces['inputs'].items():
            self.input_stream[name] = entry.get('stream', name)
            for t in entry['to']:
                self.fed_by_input[(t['node'], t['port'])] = name
        # ports that may receive nothing: the sources of insert transforms (D2 transforms are not
        # emitted; the contract's transform is visible through the value domains: a `source` port
        # of `splice` — the language's only insert today — is recognised by its contract)
        self.insert_sources = {n: ['source'] for n, e in self.nodes.items() if e['contract']['name'] == 'splice'}
        # ports whose delivered elements an `append` state of the occurrence indexed by them holds in
        # full (D4 `indexed_by_port`, §7): such a port may deliver nothing in a later invocation; a
        # window holds a suffix and exempts nothing
        self.state_indexed_by = {}
        for st in doc['d4']['states']:
            if st.get('indexed_by_port') and st['law'] == 'append':
                for m in st['members']:
                    node, sname = m.rsplit('.', 1)
                    self.state_indexed_by.setdefault(node, {})[st['indexed_by_port']] = sname
        self.generative = None     # (output name, stream) — the element fed back at decode
        for name, o in self.interfaces['outputs'].items():
            if o.get('generative'):
                v = self.values[f"{o['node']}.{o['port']}"]       # D2 lists exposed values (finding 2, decided 30 Aug 2026)
                self.generative = (name, v['domain']['stream'])
        self.feedback_input = None
        if self.generative:
            for name, stream in self.input_stream.items():
                if stream == self.generative[1] and 'stream' not in self.interfaces['inputs'][name]:
                    self.feedback_input = name
        # the input ids are delivered to: the feedback input, else (a document without a generative
        # output — an encoder) the public input whose value is a token stream
        self.token_input = self.feedback_input or next(
            (n for n, v in self.input_values.items() if v.get('domain', {}).get('kind') == 'token'), None)

    def layer_cuts(self):
        return [c for c in self.cuts if c['kind'] == 'layer']

    def required_inputs(self):
        """The inputs a first invocation must deliver: those D2 marks `required_for` the
        generative output — every input, for a document without one (§7)."""
        if self.generative is None:
            return set(self.interfaces['inputs'])
        return {n for n, v in self.input_values.items() if self.generative[0] in v.get('required_for', [])}

    def node_domain(self, node):
        """(stream, factor) of the node's own domain (§5.3): the stream its outputs are on — its
        first input's, for a node without outputs — and the D2 `count` on that stream of the
        first input value in it, whose positions are the node's: 1 for a value a public input
        delivers whole, 1/stride behind a merge. A port on another stream (a cross-attention
        source) is transformed and never the node's own."""
        stream = None
        for v in self.outputs_of.get(node, {}).values():
            stream = v['domain']['stream']
            break
        inputs = [self.values[vname] for (n, _port), vname in self.sources.items() if n == node]
        inputs += [self.input_values[name] for (n, _port), name in self.fed_by_input.items() if n == node]
        if stream is None and inputs:
            stream = inputs[0]['domain']['stream']
        for v in inputs:
            if v['domain']['stream'] == stream:
                return stream, float((v.get('count') or {}).get(stream, 1.0))
        return stream, 1.0

    def node_stream(self, node):
        """The stream the node's elements are indexed by."""
        return self.node_domain(node)[0]

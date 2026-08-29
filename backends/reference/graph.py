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
        keys = path.split('.')
        for k in keys[:-1]:
            node = node[k]
        notes.append(f"{path}: {node.get(keys[-1])!r} -> {value!r}")
        node[keys[-1]] = value
    here = os.path.dirname(os.path.abspath(model_path))
    model['catalog'] = [{"base": os.path.normpath(os.path.join(here, e['base']))} for e in model['catalog']]
    model['model'] = f"{model['model']}-{suffix}"
    out = os.path.join(out_dir, f"{os.path.basename(model_path)[:-5]}.{suffix}.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=2)
    return out, notes


def truncated(model_path, spec, out_dir):
    """`decoder.layer=3`: the composition index range shortened, and a literal quantity
    named `layers` equal to the old stop following it."""
    comp_index, stop = spec.split('=')
    comp, index = comp_index.rsplit('.', 1)
    stop = int(stop)
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    old = model['compositions'][comp]['indices'][index]['stop']['literal']
    edits = {f"compositions.{comp}.indices.{index}.stop.literal": stop}
    q = model.get('quantities', {}).get('layers')
    if q and q['source'].get('kind') == 'literal' and q['source'].get('value') == old:
        edits['quantities.layers.source.value'] = stop
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
        for vname, v in self.values.items():
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
        self.generative = None     # (output name, stream) — the element fed back at decode
        for name, o in self.interfaces['outputs'].items():
            if o.get('generative'):
                v = self.values.get(f"{o['node']}.{o['port']}")
                # Finding: D2 lists the values on edges; a public output feeds no edge and
                # has no entry, so its stream is that of the producing node's input.
                stream = (v or {}).get('domain', {}).get('stream') or self.node_stream(o['node'])
                self.generative = (name, stream)
        self.feedback_input = None
        if self.generative:
            for name, stream in self.input_stream.items():
                if stream == self.generative[1] and 'stream' not in self.interfaces['inputs'][name]:
                    self.feedback_input = name

    def layer_cuts(self):
        return [c for c in self.cuts if c['kind'] == 'layer']

    def node_stream(self, node):
        """The stream the node's elements are indexed by: that of its first input."""
        for (n, port), vname in self.sources.items():
            if n == node:
                return self.values[vname]['domain']['stream']
        for (n, port), name in self.fed_by_input.items():
            if n == node:
                return self.input_stream[name]
        return None

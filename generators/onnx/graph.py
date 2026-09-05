"""The derived document as Python — what the emitter needs and nothing else. The generator reads
D1–D6 (`tensorspine --derive`), never the model source or the catalog: a model document is
refused with the command that derives it."""
import json
from fractions import Fraction

DTYPES = {'bf16': 'bfloat16', 'f16': 'float16', 'f32': 'float32'}


def load(path):
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    if doc.get('schema') != 'tensorspine-derived/2.0':
        raise ValueError(f"{path}: not a derived document; derive it first (tensorspine --derive MODEL -o DIR)")
    return Graph(doc)


class Graph:
    def __init__(self, doc):
        self.doc = doc
        self.model = doc['model']
        d1 = doc['d1']
        self.nodes = d1['nodes']
        self.order = d1['topological_order']
        self.interfaces = d1['interfaces']
        self.sources = {}                 # (node, input port) -> the value feeding it
        self.consumers = {}               # value -> [(node, port)]
        for e in d1['edges']:
            vname = f"{e['from']['node']}.{e['from']['port']}"
            self.sources[(e['to']['node'], e['to']['port'])] = vname
            self.consumers.setdefault(vname, []).append((e['to']['node'], e['to']['port']))
        self.values = {v['value']: v for v in doc['d2']['values']}
        self.input_values = {v['input']: v for v in doc['d2']['values'] if 'input' in v}
        self.outputs_of = {}
        for vname, v in self.values.items():
            if 'input' in v:
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
        self.input_stream = {}
        self.fed_by_input = {}            # (node, port) -> input name
        for name, entry in self.interfaces['inputs'].items():
            self.input_stream[name] = entry.get('stream', name)
            for t in entry['to']:
                self.fed_by_input[(t['node'], t['port'])] = name
        # elements of the introducing input per element of each input on its stream (§5.3)
        self.elements_per = {}
        for name, v in self.input_values.items():
            c = float((v.get('count') or {}).get(self.input_stream[name], 1.0))
            self.elements_per[name] = Fraction(c).limit_denominator(1 << 20) ** -1 if c else Fraction(1)
        self.generative = None            # (output name, stream)
        for name, o in self.interfaces['outputs'].items():
            if o.get('generative'):
                self.generative = (name, self.values[f"{o['node']}.{o['port']}"]['domain']['stream'])
        self.feedback_input = None
        if self.generative:
            candidates = [n for n, s in self.input_stream.items()
                          if s == self.generative[1] and self.interfaces['inputs'][n].get('kind') == 'token']
            introducing = [n for n in candidates if 'stream' not in self.interfaces['inputs'][n]]
            self.feedback_input = (introducing or candidates or [None])[0]
        self.token_input = self.feedback_input or next(
            (n for n, v in self.input_values.items() if v.get('domain', {}).get('kind') == 'token'), None)

    def layer_cuts(self):
        return [c for c in self.cuts if c['kind'] == 'layer']

    def required_inputs(self):
        if self.generative is None:
            return set(self.interfaces['inputs'])
        return {n for n, v in self.input_values.items() if self.generative[0] in v.get('required_for', [])}

    def node_domain(self, node):
        """(stream, factor): the stream a node's outputs are on and the D2 count of the first
        input value on that stream — the node's positions are the stream's scaled by it (§5.3)."""
        stream = None
        for v in self.outputs_of.get(node, {}).values():
            stream = v['domain']['stream']
            break
        inputs = [self.values[vname] for (n, _p), vname in self.sources.items() if n == node]
        inputs += [self.input_values[name] for (n, _p), name in self.fed_by_input.items() if n == node]
        if stream is None and inputs:
            stream = inputs[0]['domain']['stream']
        for v in inputs:
            if v['domain']['stream'] == stream:
                return stream, float((v.get('count') or {}).get(stream, 1.0))
        return stream, 1.0

    def counts(self):
        """(occurrences, values, tensors, states, edges, ordered): what the harness compares with the language's own count."""
        return (len(self.nodes), len(self.values), len(self.tensors), len(self.states), len(self.doc['d1']['edges']), len(self.order))

"""The execution plan: D1's topological order as a sequence of blocks at D6's
legal layer cuts (one block by default, R13), and per node its kernel, the
values feeding each port, its parameter and state identities and the D2 shapes
of its outputs. The values crossing each layer cut are the dump points (R07).
"""


class Step:
    __slots__ = ('node', 'kernel', 'contract', 'arguments', 'inputs', 'params', 'states', 'outputs', 'stream')

    def __init__(self, node, kernel, entry, graph):
        self.node = node
        self.kernel = kernel
        self.contract = entry['contract']
        self.arguments = entry['arguments']
        self.inputs = {}
        for (n, port), vname in graph.sources.items():
            if n == node:
                self.inputs[port] = ('value', vname)
        for (n, port), name in graph.fed_by_input.items():
            if n == node:
                self.inputs[port] = ('input', name)
        self.params = graph.slots_of.get(node, {})
        self.states = graph.states_of.get(node, {})
        self.outputs = graph.outputs_of.get(node, {})
        self.stream = graph.node_stream(node)


class Plan:
    def __init__(self, graph, kernels):
        self.graph = graph
        self.steps = []
        for node in graph.order:
            entry = graph.nodes[node]
            key = (entry['contract']['name'], entry['contract']['version'])
            self.steps.append(Step(node, kernels[key], entry, graph))
        self.blocks = [(0, len(self.steps))]
        self.dump_values = {}
        for c in graph.cuts:                       # layer cuts, and the family cuts that close the last layer
            for p in c['payload']:
                self.dump_values.setdefault(p['value'], []).append(c['cut'])
        self.remaining = {v: len(c) for v, c in graph.consumers.items()}

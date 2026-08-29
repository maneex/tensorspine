"""The execution plan: D1's topological order as a sequence of blocks at D6's legal
layer cuts — one block by default (R13) — and per node its kernel, the values feeding
each port, its parameter and state identities and the D2 shapes of its outputs.
The values crossing each cut are the dump points (R07).

A block is the unit of loading under `--max-ram`: consecutive layers between two legal
cuts, the roots in the first and last blocks. Membership comes from D1 alone — the
ancestor closure of a cut's payload producers — so it holds for any document; legality
comes from D6 (every crossing edge forward), so a block needs nothing from a later one.
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


class Block:
    __slots__ = ('name', 'steps', 'identities', 'bytes', 'cut', 'payload_bytes_per_element')

    def __init__(self, name, steps, identities, nbytes, cut, payload):
        self.name = name
        self.steps = steps                       # indices into plan.steps, in order
        self.identities = identities             # D3 identities first used in this block
        self.bytes = nbytes                      # their declared bytes
        self.cut = cut                           # the cut this block opens with (None for the first)
        self.payload_bytes_per_element = payload  # bytes per element crossing that cut


class Plan:
    def __init__(self, graph, kernels, max_bytes=None, elements=1, resident_bytes=0):
        self.graph = graph
        self.steps = []
        index = {}
        for node in graph.order:
            entry = graph.nodes[node]
            key = (entry['contract']['name'], entry['contract']['version'])
            index[node] = len(self.steps)
            self.steps.append(Step(node, kernels[key], entry, graph))
        self.dump_values = {}
        for c in graph.cuts:
            for p in c['payload']:
                self.dump_values.setdefault(p['value'], []).append(c['cut'])
        self.remaining = {v: len(c) for v, c in graph.consumers.items()}
        self.minimal = []                      # the finest legal partition, one block per layer cut
        self.blocks = self._blocks(graph, index, max_bytes, elements, resident_bytes)

    # --- blocks ---------------------------------------------------------------
    def _ancestors(self, graph, nodes):
        seen = set()
        stack = list(nodes)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for (to, port), vname in graph.sources.items():
                if to == n:
                    stack.append(vname.rsplit('.', 1)[0])
        return seen

    def _blocks(self, graph, index, max_bytes, elements, resident_bytes):
        """Minimal blocks: one per layer cut, plus the tail; then merged greedily under
        `max_bytes` (their declared parameter bytes, the payload crossing into them for
        `elements` elements, and `resident_bytes` for states and temporaries)."""
        cuts = graph.layer_cuts()
        groups, prev, payloads = [], set(), []
        for c in cuts:
            producers = [p['value'].rsplit('.', 1)[0] for p in c['payload']]
            closure = self._ancestors(graph, producers)
            groups.append(sorted(closure - prev, key=index.get))
            payloads.append((c['cut'], sum(p['bytes_per_element'] for p in c['payload'])))
            prev |= closure
        groups.append(sorted(set(graph.nodes) - prev, key=index.get))
        payloads.append((None, 0))                # the tail opens with the last cut
        if not cuts:
            groups, payloads = [sorted(graph.nodes, key=index.get)], [(None, 0)]
        # every identity a group's nodes need; a tied identity used by two groups is held —
        # and loaded — by both, and counted in both
        minimal = []
        for k, members in enumerate(groups):
            idents = []
            for node in members:
                for ident in graph.slots_of.get(node, {}).values():
                    if ident not in idents:
                        idents.append(ident)
            nbytes = sum(graph.tensors[i]['bytes'] or 0 for i in idents)
            opening = payloads[k - 1] if k else (None, 0)
            minimal.append(Block(f"block{k}", [index[n] for n in members], idents, nbytes, opening[0], opening[1]))
        self.minimal = minimal
        if max_bytes is None:
            merged = Block('all', [s for b in minimal for s in b.steps], list(graph.tensors),
                           graph.d3_totals['bytes'], None, 0)
            return [merged]
        out, current = [], None
        for b in minimal:
            need = b.bytes + b.payload_bytes_per_element * elements + resident_bytes
            if need > max_bytes:
                raise ValueError(f"{b.name} alone needs {need / 2**30:.2f} GiB — parameters {b.bytes / 2**30:.2f} GiB, "
                                 f"payload {b.payload_bytes_per_element * elements / 2**30:.2f} GiB for {elements} elements, "
                                 f"states and the largest per-operation temporary {resident_bytes / 2**30:.2f} GiB — "
                                 f"which exceeds --max-ram {max_bytes / 2**30:.2f} GiB")
            if current is None:
                current = Block(b.name, list(b.steps), list(b.identities), b.bytes, b.cut, b.payload_bytes_per_element)
                continue
            joined = current.bytes + b.bytes + current.payload_bytes_per_element * elements + resident_bytes
            if joined <= max_bytes:
                current.steps += b.steps
                current.identities += [i for i in b.identities if i not in current.identities]
                current.bytes = sum(graph.tensors[i]['bytes'] or 0 for i in current.identities)
                current.name = f"{current.name}+{b.name}"
            else:
                out.append(current)
                current = Block(b.name, list(b.steps), list(b.identities), b.bytes, b.cut, b.payload_bytes_per_element)
        out.append(current)
        return out

    def summary(self, elements, max_bytes, resident_bytes):
        """The cuts chosen under `--max-ram`: one line per block — the legal cut it opens
        with and the one it closes at (D6's names), its nodes, its parameter bytes and the
        payload crossing into it — then what stays resident and the traffic."""
        gib = lambda n: f"{n / 2**30:.2f} GiB"
        lines = [f"blocks: {len(self.blocks)} at legal cuts under --max-ram {gib(max_bytes)} "
                 f"(states and the largest temporary {gib(resident_bytes)} stay resident)"]
        largest = 0
        for k, b in enumerate(self.blocks):
            opening = b.cut or 'start'
            closing = self.blocks[k + 1].cut if k + 1 < len(self.blocks) else 'end'
            first, last = self.steps[b.steps[0]].node, self.steps[b.steps[-1]].node
            payload = b.payload_bytes_per_element * elements
            largest = max(largest, b.bytes + payload)
            lines.append(f"  {k + 1:2d}. {opening} → {closing}: {len(b.steps)} nodes ({first} … {last}), "
                         f"parameters {gib(b.bytes)}, payload in {payload / 2**20:.1f} MiB for {elements} elements")
        lines.append(f"  resident at most {gib(largest + resident_bytes)}; traffic {gib(self.traffic_bytes())} of parameters per invocation")
        return lines

    def traffic_bytes(self):
        """Parameter bytes moved per invocation: nothing when one block holds the model,
        the whole model when it is streamed block by block."""
        return sum(b.bytes for b in self.blocks) if len(self.blocks) > 1 else 0

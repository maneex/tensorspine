"""A session: the state instances of one (session, branch) key, the positions
consumed per stream, prefill and decode (R08)."""
import torch

import state as state_mod


class Session:
    def __init__(self, model, capacity, device, dtype, decode_model=None):
        self.model = model
        self.decode_model = decode_model          # e.g. the compiled step, used for one-element invocations
        self.graph = model.graph
        self.capacity = capacity
        self.device = device
        self.states = state_mod.allocate(self.graph, capacity, device, dtype)
        self.consumed = {}

    def run(self, inputs, dump=None):
        """One invocation: every declared public input supplied (R08), positions
        continuing per stream."""
        wanted = {self.graph.generative[0]} if self.graph.generative else set(self.graph.interfaces['outputs'])
        for n in self.graph.interfaces['inputs']:
            if n not in inputs:
                needed = set(self.graph.input_values[n].get('required_for', [])) & wanted
                if needed and not self.consumed.get(self.graph.input_stream[n]):
                    raise state_mod.Refusal(f"input {n} delivers nothing, and {sorted(needed)} need it (§7)")
        positions = {}
        for name, t in inputs.items():
            expect = [a['extent'] for a in self.graph.input_values[name]['shape']]
            if list(t.shape[1:]) != expect:
                raise state_mod.Refusal(f"input {name}: D2 says {expect} per element, got {list(t.shape)}")
            stream = self.graph.input_stream[name]
            start = self.consumed.get(stream, 0)
            positions[stream] = torch.arange(start, start + t.shape[0], device=self.device)
        one = all(t.shape[0] == 1 for t in inputs.values())
        runner = self.decode_model if (self.decode_model is not None and one and dump is None) else self.model
        outputs = runner(inputs, positions, self.states, dump)
        for name, t in inputs.items():
            stream = self.graph.input_stream[name]
            self.consumed[stream] = self.consumed.get(stream, 0) + t.shape[0]
        return outputs

    def prefill(self, ids, dump=None):
        name = self.graph.feedback_input
        return self.run({name: torch.as_tensor(ids, device=self.device, dtype=torch.long)}, dump)

    def decode(self, next_id, dump=None):
        name = self.graph.feedback_input
        return self.run({name: torch.tensor([next_id], device=self.device, dtype=torch.long)}, dump)

    def reset(self):
        for s in self.states.values():
            s.reset()
        self.consumed = {}


def greedy(outputs, graph):
    name, _ = graph.generative
    return int(outputs[name][-1].argmax().item())

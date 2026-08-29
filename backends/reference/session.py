"""A session: the state instances of one (session, branch) key, the positions
consumed per stream, prefill and decode (R08)."""
import torch

import state as state_mod


class Session:
    def __init__(self, model, capacity, device, dtype):
        self.model = model
        self.graph = model.graph
        self.capacity = capacity
        self.device = device
        self.states = state_mod.allocate(self.graph, capacity, device, dtype)
        self.consumed = {}

    def run(self, inputs, dump=None):
        """One invocation: every declared public input supplied (R08), positions
        continuing per stream."""
        missing = [n for n in self.graph.interfaces['inputs'] if n not in inputs]
        if missing:
            raise state_mod.Refusal(f"public input(s) not supplied: {missing}")
        positions = {}
        for name, t in inputs.items():
            stream = self.graph.input_stream[name]
            start = self.consumed.get(stream, 0)
            positions[stream] = torch.arange(start, start + t.shape[0], device=self.device)
        outputs = self.model(inputs, positions, self.states, dump)
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

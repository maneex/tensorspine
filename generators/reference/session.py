"""A session: the state instances of one (session, branch) key, the positions
consumed per stream, prefill and decode (R08) — and a fork, by the sharing granularity
each state's contract declares (§4.3)."""
import torch

import state as state_mod

# The sharing granularities `fork` realises across sessions (§4.3), as the manifest declares them:
# by_position, the positions before the fork are copied; within_span and at_fork_point, the state
# is copied whole at the parent's current position and nowhere earlier. by_source is not realised:
# no kernel serves a source-indexed state yet.
SHARING = ('by_position', 'within_span', 'at_fork_point')


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

    def fork(self, at=None):
        """A new session over the same model, forked from this one at position `at` of the
        forked stream (the generative output's; default: the current position), its states
        copied by the granularity each contract declares (§4.3):

          by_position    the entries before `at` are copied: a shared prefix, entry by entry;
          within_span    the ring is copied whole, only when `at` is the current position — the
                         ring holds the last `span` positions and serves no older one;
          at_fork_point  the payload is copied whole, only at the current position, and is
                         shared by nothing afterwards;
          by_source      refused: nothing here serves a source-indexed state.

        A fork the granularity excludes is a refusal naming it, never a stale state."""
        stream = self.graph.generative[1] if self.graph.generative else next(iter(self.consumed), None)
        length = self.consumed.get(stream, 0)
        at = length if at is None else at
        if at > length:
            raise state_mod.Refusal(f"fork at {at}: the session has consumed {length} positions of '{stream}'")
        child = Session.__new__(Session)
        child.model, child.decode_model, child.graph = self.model, self.decode_model, self.graph
        child.capacity, child.device = self.capacity, self.device
        child.consumed = dict(self.consumed)
        child.consumed[stream] = at
        child.states = {}
        for ident, st in self.states.items():
            entry = self.graph.states[ident]
            on_stream = (entry.get('stream') or {}).get('stream') == stream
            sharing = entry.get('sharing')
            if not on_stream or at == length:
                child.states[ident] = st.clone()
            elif sharing == 'by_position':
                child.states[ident] = st.clone()
                child.states[ident].truncate(at)
            elif sharing in ('within_span', 'at_fork_point'):
                raise state_mod.Refusal(f"{ident}: sharing {sharing} — the state is copied whole at the current "
                                        f"position {length} and serves no fork at {at} (§4.3)")
            else:
                raise state_mod.Refusal(f"{ident}: sharing {sharing} is not realised by this generator")
        return child


def greedy(outputs, graph):
    name, _ = graph.generative
    return int(outputs[name][-1].argmax().item())

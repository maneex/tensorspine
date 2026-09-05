"""A session: the state instances of one (session, branch) key, the positions
consumed per stream, prefill and decode (R08) — and a fork, by the sharing granularity
each state's contract declares (§4.3)."""
import torch

import state as state_mod

# The sharing granularities `fork` realises across sessions (§4.3), as the manifest declares them:
# by_position, the positions before the fork are copied; within_span and at_fork_point, the state
# is copied whole at the parent's current position and nowhere earlier; by_source, the state is
# copied whole once its source stream is complete — a cross-attention cache after the audio arrived.
SHARING = ('by_position', 'by_source', 'within_span', 'at_fork_point')

# The sessions one invocation may carry (the manifest's `sessions_per_invocation`, an integer the
# schema bounds: batch-plan finding 1): what `Batch` enforces, on the packed layout (B03).
SESSIONS_PER_INVOCATION = 16


class Session:
    def __init__(self, model, capacity, device, dtype, decode_model=None):
        """`capacity`: the positions every `append` state may hold — one number for every stream,
        or a mapping per stream (`{'tokens': 64, 'audio': 1500}`), each state sized by the stream
        it is indexed by (`state.allocate`)."""
        self.model = model
        self.decode_model = decode_model          # e.g. the compiled step, used for one-element invocations
        self.graph = model.graph
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.states = state_mod.allocate(self.graph, capacity, device, dtype)
        self.consumed = {}

    def prepare(self, inputs):
        """What an invocation delivers, checked (R08, §5.3): the outputs wanted (the generative one,
        else every output), the advance of every stream and its positions, continuing per stream."""
        wanted = {self.graph.generative[0]} if self.graph.generative else set(self.graph.interfaces['outputs'])
        for n in self.graph.interfaces['inputs']:
            if n not in inputs:
                needed = set(self.graph.input_values[n].get('required_for', [])) & wanted
                if needed and not self.consumed.get(self.graph.input_stream[n]):
                    raise state_mod.Refusal(f"input {n} delivers nothing, and {sorted(needed)} need it (§7)")
        # every stream advances by the elements delivered on it, counted in the introducing input's
        # elements (§5.3): an input that joined the stream delivers 1 / count of them per element —
        # eight frames per token on Voxtral's audio — and the inputs on one stream must agree, or
        # the invocation would place their elements on different positions
        advance, by = {}, {}
        for name, t in inputs.items():
            expect = [a['extent'] for a in self.graph.input_values[name]['shape']]
            if list(t.shape[1:]) != expect:
                raise state_mod.Refusal(f"input {name}: D2 says {expect} per element, got {list(t.shape)}")
            stream = self.graph.input_stream[name]
            n = t.shape[0] * self.graph.elements_per[name]
            if n.denominator != 1:
                raise state_mod.Refusal(f"input {name}: {t.shape[0]} elements are {n} of stream '{stream}', not a whole number: "
                                        f"a delivery is aligned to every merge on its stream (§5.3)")
            if stream in advance and advance[stream] != int(n):
                raise state_mod.Refusal(f"inputs {by[stream]} and {name} disagree on the advance of stream '{stream}': "
                                        f"{advance[stream]} and {int(n)} elements (§5.3: one stream, one count per kind)")
            advance[stream], by[stream] = int(n), name
        positions = {}
        for stream, n in advance.items():
            start = self.consumed.get(stream, 0)
            positions[stream] = torch.arange(start, start + n, device=self.device)
        return wanted, advance, positions

    def finish(self, inputs, wanted, advance, outputs):
        """After the invocation: the streams advanced, the wanted outputs there."""
        for stream, n in advance.items():
            self.consumed[stream] = self.consumed.get(stream, 0) + n
        missing = sorted(wanted - set(outputs))
        if missing:
            raise state_mod.Refusal(f"outputs {missing} are not evaluated by the inputs {sorted(inputs)} delivered (§7)")
        return outputs

    def run(self, inputs, dump=None):
        """One invocation: every declared public input supplied (R08), positions
        continuing per stream; the outputs wanted (the generative one, else every output) must
        be evaluated by what was delivered, or the invocation is refused."""
        wanted, advance, positions = self.prepare(inputs)
        one = all(t.shape[0] == 1 for t in inputs.values())
        runner = self.decode_model if (self.decode_model is not None and one and dump is None) else self.model
        outputs = runner(inputs, positions, self.states, dump)
        return self.finish(inputs, wanted, advance, outputs)

    def delivered(self, ids, inputs=None):
        """The prompt, or the element fed back, on the token input, with whatever else `inputs`
        delivers; floating inputs taken to the compute dtype."""
        out = {self.graph.feedback_input: torch.as_tensor(ids, device=self.device, dtype=torch.long)}
        for name, t in (inputs or {}).items():
            out[name] = t.to(self.device, self.dtype) if t.is_floating_point() else t.to(self.device)
        return out

    def prefill(self, ids, dump=None, inputs=None):
        """The supplied elements (§7): the prompt on the token input and, in the same invocation,
        whatever else `inputs` delivers (`{'audio': frames}`, element-major on the input's D2
        shape) — a source stream delivered whole is complete, and the states indexed by it are
        frozen from then on. Floating inputs are taken to the compute dtype."""
        return self.run(self.delivered(ids, inputs), dump)

    def decode(self, next_id, dump=None, inputs=None):
        """The generated element fed back on the token input (§7) and, on a document whose token
        stream joins a fragmented one, the fragment that comes with it: `inputs` delivers it
        (`{'audio': frames}`, the next eight frames of a streaming transcription)."""
        return self.run(self.delivered([next_id], inputs), dump)

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
          by_source      the whole state, copied once its source stream is complete — delivered
                         whole by an input that is not fragmented, as the audio is in one prefill
                         (sessions whose complete source streams are identical share it, §4.3);
                         a fragmented source that has started delivering is refused, since the
                         session cannot tell its completion; one that has delivered nothing copies
                         an empty state, and the child takes its own delivery.

        A fork the granularity excludes is a refusal naming it, never a stale state."""
        stream = self.graph.generative[1] if self.graph.generative else next(iter(self.consumed), None)
        length = self.consumed.get(stream, 0)
        at = length if at is None else at
        if at > length:
            raise state_mod.Refusal(f"fork at {at}: the session has consumed {length} positions of '{stream}'")
        child = Session.__new__(Session)
        child.model, child.decode_model, child.graph = self.model, self.decode_model, self.graph
        child.capacity, child.device, child.dtype = self.capacity, self.device, self.dtype
        child.consumed = dict(self.consumed)
        child.consumed[stream] = at
        child.states = {}
        for ident, st in self.states.items():
            entry = self.graph.states[ident]
            on_stream = (entry.get('stream') or {}).get('stream') == stream
            sharing = entry.get('sharing')
            if sharing == 'by_source':
                source = entry['stream']['stream']
                fragmented = any(self.graph.interfaces['inputs'][n].get('fragmented') for n, s in self.graph.input_stream.items() if s == source)
                if fragmented and self.consumed.get(source):
                    raise state_mod.Refusal(f"{ident}: sharing by_source — the state is shared whole once its source stream "
                                            f"'{source}' is complete, and a fragmented stream still delivering is not (§4.3)")
                child.states[ident] = st.clone()
            elif not on_stream or at == length:
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


class Batch:
    """Several sessions evaluated in one invocation — the packed layout (B03): each session keeps
    its own states and positions (its instance key, §4.4), the model's `step_batch` evaluates the
    occurrences on the sessions' elements together where it may and per session where it must.
    Every session delivers the same inputs (§7: one delivery pattern per invocation), and no more
    than SESSIONS_PER_INVOCATION sessions ride together. Batching is invisible: what a session
    gets is what it would get alone, up to the rounding of a product over more rows (B06, B07)."""

    def __init__(self, sessions):
        if not sessions:
            raise state_mod.Refusal("a batch of no session")
        if len(sessions) > SESSIONS_PER_INVOCATION:
            raise state_mod.Refusal(f"{len(sessions)} sessions in one invocation: this generator carries {SESSIONS_PER_INVOCATION} at most")
        first = sessions[0]
        for s in sessions[1:]:
            if s.model is not first.model:
                raise state_mod.Refusal("a batch's sessions run one model")
        self.sessions = list(sessions)
        self.model = first.model
        self.graph = first.graph

    def run(self, inputs):
        """One invocation for every session: `inputs` holds one delivery per session, all naming
        the same public inputs; returns one output mapping per session."""
        from module import step_batch
        if len(inputs) != len(self.sessions):
            raise state_mod.Refusal(f"{len(inputs)} deliveries for {len(self.sessions)} sessions")
        names = [frozenset(i) for i in inputs]
        for k, n in enumerate(names[1:], 1):
            if n != names[0]:
                raise state_mod.Refusal(f"session {k} delivers {sorted(n)} where session 0 delivers {sorted(names[0])}: "
                                        f"a batch's sessions deliver the same inputs (§7)")
        prepared = [s.prepare(i) for s, i in zip(self.sessions, inputs)]
        outputs = step_batch(self.model, list(inputs), [p[2] for p in prepared], [s.states for s in self.sessions])
        return [s.finish(i, p[0], p[1], o) for s, i, p, o in zip(self.sessions, inputs, prepared, outputs)]

    def prefill(self, ids, inputs=None):
        """One prompt per session (their lengths may differ: the layout is packed), and per session
        whatever else `inputs` delivers."""
        inputs = inputs or [None] * len(self.sessions)
        return self.run([s.delivered(p, i) for s, p, i in zip(self.sessions, ids, inputs)])

    def decode(self, next_ids, inputs=None):
        inputs = inputs or [None] * len(self.sessions)
        return self.run([s.delivered([n], i) for s, n, i in zip(self.sessions, next_ids, inputs)])


def greedy(outputs, graph):
    name, _ = graph.generative
    return int(outputs[name][-1].argmax().item())

"""A session over an emitted graph: onnxruntime runs one invocation per call; the states and the
positions consumed per stream live here between calls (ONNX has no mutable state), prefill and
decode feed the graph's inputs, the state outputs become the next call's state inputs."""
import os
import tempfile

import numpy as np
import onnxruntime as ort

from emit import initializer_bytes, save_model


class Refusal(Exception):
    pass


# The sessions one invocation may carry (the manifest's `sessions_per_invocation`, an integer the
# schema bounds: batch-plan finding 1): what `Batch` enforces, on the aligned layout (B04).
SESSIONS_PER_INVOCATION = 16


def capacity_of(capacity, stream):
    """The positions an `append` state on `stream` may hold: one number for every stream, or a
    mapping per stream; a stream the mapping omits is a refusal, never a default."""
    if isinstance(capacity, dict):
        if stream not in capacity:
            raise Refusal(f"no capacity for stream '{stream}': the mapping names {sorted(capacity)}")
        return int(capacity[stream])
    return int(capacity)


class Session:
    def __init__(self, emitter, graph, dump=False, providers=('CPUExecutionProvider',)):
        """`emitter` emits the graph of an invocation for the inputs it delivers, once per distinct
        delivery (a prefill with the audio and a decode without it are two graphs); the states
        and the positions consumed per stream are this session's, whichever graph runs."""
        self.graph, self.emitter, self.dump, self.providers = graph, emitter, dump, list(providers)
        self.graphs = {}                  # frozenset(delivered inputs) -> (ort session, input names, output names)
        self.models = {}                  # the emitted models, for saving
        self.states = {}
        self.states_after_prefill = {}
        for ident, st in graph.states.items():
            for p in st['payload']:
                shape = [a['extent'] for a in p['shape']]
                self.states[f"{ident}/{p['component']}"] = np.zeros(([0] + shape) if st['law'] != 'fixed' else shape, dtype=np.float32)
        self.consumed = {}

    def run(self, inputs):
        """One invocation: the delivered inputs (element-major numpy arrays) with their streams' positions."""
        feeds = {}
        advance = {}
        for name, t in inputs.items():
            stream = self.graph.input_stream[name]
            n = t.shape[0] * self.graph.elements_per[name]
            if n.denominator != 1:
                raise Refusal(f"input {name}: {t.shape[0]} elements are not aligned on stream '{stream}' (§5.3)")
            if stream in advance and advance[stream] != int(n):
                raise Refusal(f"the inputs disagree on the advance of stream '{stream}'")
            advance[stream] = int(n)
            feeds[name] = t
        for stream, n in advance.items():
            start = self.consumed.get(stream, 0)
            feeds[f"positions/{stream}"] = np.arange(start, start + n, dtype=np.int64)
        for key, buf in self.states.items():
            feeds[f"state/{key}"] = buf
        sess, input_names, output_names = self.session_for(frozenset(inputs))
        missing = [n for n in input_names if n not in feeds]
        if missing:
            raise Refusal(f"the graph expects inputs the invocation does not deliver: {missing[:4]}")
        outs = dict(zip(output_names, sess.run(None, {n: feeds[n] for n in input_names})))
        for key in self.states:
            if f"state_out/{key}" in outs:
                self.states[key] = outs.pop(f"state_out/{key}")
        first = not self.consumed
        for stream, n in advance.items():
            self.consumed[stream] = self.consumed.get(stream, 0) + n
        if first:
            self.states_after_prefill = {k: v.copy() for k, v in self.states.items()}
        return outs

    def session_for(self, delivered):
        if delivered not in self.graphs:
            model = self.emitter.emit(delivered, dump=self.dump)
            self.models[delivered] = model
            if initializer_bytes(model) > 2**30:          # too large for one protobuf message: through a file, external data beside it
                path = os.path.join(tempfile.mkdtemp(prefix='tsonnx-'), f"{self.graph.model}.onnx")
                save_model(model, path)
                sess = ort.InferenceSession(path, providers=self.providers)
            else:
                sess = ort.InferenceSession(model.SerializeToString(), providers=self.providers)
            self.graphs[delivered] = (sess, [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()])
        return self.graphs[delivered]

    def prefill(self, ids, inputs=None):
        delivered = {self.graph.feedback_input or self.graph.token_input: np.asarray(ids, dtype=np.int64)}
        delivered.update(inputs or {})
        return self.run(delivered)

    def decode(self, next_id, inputs=None):
        delivered = {self.graph.feedback_input: np.array([next_id], dtype=np.int64)}
        delivered.update(inputs or {})
        return self.run(delivered)


class Batch:
    """Several sessions in one invocation on the aligned layout (batch-plan B04): every input carries
    a batch axis, every session delivers the same count on each stream per invocation (equal-length
    prompts; one token each at decode — sessions prefilled apart decode together), the `append`
    states are buffers `[b, capacity, …]` and the positions each session holds on a stream go in as
    `held/<stream>`. One graph per delivery pattern, as `Session`; outputs come back per session,
    element-major, so what a session gets reads as a `Session`'s. Batching is invisible (B06, B07)."""

    def __init__(self, emitter, graph, sessions, capacity, dump=False, providers=('CPUExecutionProvider',)):
        if emitter.layout != 'aligned':
            raise Refusal(f"a batch takes an emitter on the aligned layout, not {emitter.layout!r}")
        if not 1 <= sessions <= SESSIONS_PER_INVOCATION:
            raise Refusal(f"{sessions} sessions in one invocation: this generator carries 1 to {SESSIONS_PER_INVOCATION}")
        self.graph, self.emitter, self.dump, self.providers = graph, emitter, dump, list(providers)
        self.b, self.capacity = sessions, capacity
        self.graphs, self.models = {}, {}
        self.states = {}
        for ident, st in graph.states.items():
            if st['law'] != 'append':
                raise Refusal(f"{ident}: a {st['law']} state is not emitted on the aligned layout")
            cap = capacity_of(capacity, st['stream']['stream'])
            for p in st['payload']:
                shape = [a['extent'] for a in p['shape']]
                self.states[f"{ident}/{p['component']}"] = np.zeros([sessions, cap] + shape, dtype=np.float32)
        self.held = {}                    # stream -> [b] positions consumed per session
        self.states_after_prefill = []    # per session: {key: the rows it holds}

    def run(self, inputs):
        """One invocation for every session: `inputs` holds one delivery per session (element-major
        arrays), all naming the same public inputs with the same counts; returns one output mapping
        per session, element-major."""
        if len(inputs) != self.b:
            raise Refusal(f"{len(inputs)} deliveries for {self.b} sessions")
        names = [frozenset(i) for i in inputs]
        if any(n != names[0] for n in names):
            raise Refusal("a batch's sessions deliver the same inputs (§7)")
        feeds, advance = {}, {}
        for name in sorted(names[0]):
            arrays = [np.asarray(i[name]) for i in inputs]
            if any(a.shape != arrays[0].shape for a in arrays):
                raise Refusal(f"input {name}: the sessions deliver {[a.shape[0] for a in arrays]} elements — an aligned batch delivers "
                              f"the same count to every session (batch-plan B04); prefill sessions of unequal prompts apart")
            stream = self.graph.input_stream[name]
            n = arrays[0].shape[0] * self.graph.elements_per[name]
            if n.denominator != 1:
                raise Refusal(f"input {name}: {arrays[0].shape[0]} elements are not aligned on stream '{stream}' (§5.3)")
            if stream in advance and advance[stream] != int(n):
                raise Refusal(f"the inputs disagree on the advance of stream '{stream}'")
            advance[stream] = int(n)
            feeds[name] = np.stack(arrays)
        for stream, n in advance.items():
            held = self.held.setdefault(stream, np.zeros(self.b, dtype=np.int64))
            if int(held.max()) + n > capacity_of(self.capacity, stream):
                raise Refusal(f"stream '{stream}': {int(held.max()) + n} positions exceed the capacity {capacity_of(self.capacity, stream)}")
            feeds[f"positions/{stream}"] = held[:, None] + np.arange(n, dtype=np.int64)[None, :]
            feeds[f"held/{stream}"] = held.copy()
        for key, buf in self.states.items():
            feeds[f"state/{key}"] = buf
        sess, input_names, output_names, model = self.session_for(frozenset(names[0]))
        missing = [n for n in input_names if n not in feeds]
        if missing:
            raise Refusal(f"the graph expects inputs the invocation does not deliver: {missing[:4]}")
        n_new = max(advance.values()) if advance else 0
        if self.b > 1 and n_new > 1 and any(int(h.max()) > 0 for h in self.held.values()) \
                and any(node.op_type == 'GroupQueryAttention' for node in model.graph.node):
            raise Refusal("onnxruntime's GroupQueryAttention takes a fresh prefill of several sessions or a one-token decode of several, "
                          "not a multi-token continuation of several (batch-plan finding 3)")
        outs = dict(zip(output_names, sess.run(None, {n: feeds[n] for n in input_names})))
        for key in self.states:
            if f"state_out/{key}" in outs:
                self.states[key] = outs.pop(f"state_out/{key}")
        first = not self.states_after_prefill
        for stream, n in advance.items():
            self.held[stream] = self.held[stream] + n
        if first:
            self.states_after_prefill = [{k: v[i, :self.length(k, i)].copy() for k, v in self.states.items()} for i in range(self.b)]
        return [{k: v[i] for k, v in outs.items()} for i in range(self.b)]

    def length(self, key, i):
        ident = key.rsplit('/', 1)[0]
        return int(self.held[self.graph.states[ident]['stream']['stream']][i])

    def session_for(self, delivered):
        if delivered not in self.graphs:
            model = self.emitter.emit(delivered, dump=self.dump)
            self.models[delivered] = model
            if initializer_bytes(model) > 2**30:
                path = os.path.join(tempfile.mkdtemp(prefix='tsonnx-'), f"{self.graph.model}.onnx")
                save_model(model, path)
                sess = ort.InferenceSession(path, providers=self.providers)
            else:
                sess = ort.InferenceSession(model.SerializeToString(), providers=self.providers)
            self.graphs[delivered] = (sess, [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()], model)
        return self.graphs[delivered]

    def prefill(self, ids, inputs=None):
        """One prompt per session, of one length (the layout is aligned)."""
        token = self.graph.feedback_input or self.graph.token_input
        inputs = inputs or [None] * self.b
        return self.run([{token: np.asarray(p, dtype=np.int64), **(extra or {})} for p, extra in zip(ids, inputs)])

    def decode(self, next_ids, inputs=None):
        inputs = inputs or [None] * self.b
        return self.run([{self.graph.feedback_input: np.array([n], dtype=np.int64), **(extra or {})} for n, extra in zip(next_ids, inputs)])


def greedy(outputs, graph):
    o = graph.interfaces['outputs'][graph.generative[0]]
    return int(np.argmax(outputs[f"{o['node']}.{o['port']}"][-1]))

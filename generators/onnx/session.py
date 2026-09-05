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


def greedy(outputs, graph):
    o = graph.interfaces['outputs'][graph.generative[0]]
    return int(np.argmax(outputs[f"{o['node']}.{o['port']}"][-1]))

"""The emitter: one ONNX graph for the expanded D1 graph, one invocation per run (R08) — every
public input a graph input with a dynamic element axis, the positions of every stream an input,
every D4 state an input and an output (ONNX has no mutable state: a session carries the tensors
between runs), every parameter an initializer at its D3 dtype cast to the compute dtype where a
primitive uses it (the reference's convention, the cast folded once by the runtime), and every
occurrence a call of its contract's emitter (`primitives/<name>.py`), in D1's order. The exposed
outputs are graph outputs; with `dump`, so is every value crossing a D6 layer cut (R07)."""
import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

OPSET = 17
IR_VERSION = 8
TENSOR = {'bf16': TensorProto.BFLOAT16, 'f16': TensorProto.FLOAT16, 'f32': TensorProto.FLOAT,
          'i32': TensorProto.INT32, 'i64': TensorProto.INT64}
COMPUTE = {'f32': TensorProto.FLOAT}
NUMPY = {'f32': np.float32, 'i32': np.int32, 'i64': np.int64}


class Refusal(Exception):
    pass


class Builder:
    """ONNX nodes, initializers, inputs and outputs, with unique names."""

    def __init__(self, compute='f32'):
        self.compute = compute
        self.ctype = COMPUTE[compute]
        self.nodes, self.initializers, self.inputs, self.outputs = [], [], [], []
        self.taken = set()

    def name(self, hint):
        base = hint
        k = 0
        while hint in self.taken:
            k += 1
            hint = f"{base}#{k}"
        self.taken.add(hint)
        return hint

    def node(self, op, inputs, outputs=1, hint=None, **attrs):
        names = [self.name(f"{hint or op}") for _ in range(outputs)] if isinstance(outputs, int) else list(outputs)
        for n in names:
            self.taken.add(n)
        self.nodes.append(helper.make_node(op, list(inputs), names, **attrs))
        return names[0] if isinstance(outputs, int) and outputs == 1 else names

    def const(self, array, hint='c'):
        array = np.asarray(array)
        name = self.name(hint)
        self.initializers.append(helper.make_tensor(name, helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape), array.flatten().tolist()))
        return name

    def initializer(self, name, words, dtype, shape):
        """A parameter at its D3 dtype from raw words (bf16 as uint16)."""
        name = self.name(name)
        self.initializers.append(helper.make_tensor(name, TENSOR[dtype], list(shape), words.tobytes(), raw=True))
        return name

    def input(self, name, dtype, shape):
        self.taken.add(name)
        self.inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, name, dtype, shape=None):
        self.outputs.append(helper.make_tensor_value_info(name, dtype, shape))

    def model(self, name):
        graph = helper.make_graph(self.nodes, name, self.inputs, self.outputs, self.initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)], producer_name='tensorspine-onnx')
        model.ir_version = IR_VERSION
        return model


class StateRef:
    """A D4 state instance seen by a primitive: `read()` gives the input tensors (the positions held
    so far for `append`, the valid tail for `window`, the payload for `fixed`); `write(values)` names
    the output tensors the session takes as the state after the invocation."""

    def __init__(self, b, entry, compute):
        self.b, self.entry = b, entry
        self.identity, self.law, self.span = entry['identity'], entry['law'], entry.get('span')
        self.components = {p['component']: [a['extent'] for a in p['shape']] for p in entry['payload']}
        self.inputs = {}
        for c, shape in self.components.items():
            dims = shape if self.law == 'fixed' else [f"{self.identity}.len"] + shape
            self.inputs[c] = b.input(f"state/{self.identity}/{c}", b.ctype, dims)
        self.written = {}

    def read(self):
        return dict(self.inputs)

    def write(self, values):
        for c, name in values.items():
            out = f"state_out/{self.identity}/{c}"
            self.b.node('Identity', [name], outputs=[out])
            self.b.output(out, self.b.ctype)
            self.written[c] = out


class Emitter:
    def __init__(self, graph, source, primitives, compute='f32', physical=None):
        self.graph, self.source, self.primitives, self.compute, self.physical = graph, source, primitives, compute, physical

    def refusals(self, nodes=None):
        out = []
        for node, entry in self.graph.nodes.items():
            if nodes is not None and node not in nodes:
                continue
            key = (entry['contract']['name'], entry['contract']['version'])
            p = self.primitives.get(key)
            if p is None:
                out.append(f"{node}: no primitive for {key[0]}@{key[1]}")
                continue
            for r in p.supports(entry['arguments']):
                out.append(f"{node}: {key[0]}@{key[1]} does not implement {r}")
        return out

    def evaluable(self, delivered):
        """The occurrences an invocation evaluates (§7): every input port fed by a delivered input or
        an evaluated occurrence; a port a state indexed by it holds is fed too (a source complete)."""
        g = self.graph
        fed = {(n, p) for (n, p), name in g.fed_by_input.items() if name in delivered}
        held = {}
        for st in g.states.values():
            if st.get('indexed_by_port') and st['law'] == 'append':
                for m in st['members']:
                    node, _s = m.rsplit('.', 1)
                    held[(node, st['indexed_by_port'])] = True
        # the source of an insert transform may deliver nothing (§7): `splice`, the language's only insert
        # today, recognised by its contract, as the reference does — D2 emits no transforms
        inserts = {(n, 'source') for n, e in g.nodes.items() if e['contract']['name'] == 'splice'}
        done = set()
        for node in g.order:
            ok = True
            for (n, port), vname in g.sources.items():
                if n == node and vname.rsplit('.', 1)[0] not in done and (n, port) not in held and (n, port) not in inserts:
                    ok = False
            for (n, port), name in g.fed_by_input.items():
                if n == node and (n, port) not in fed and (n, port) not in held and (n, port) not in inserts:
                    ok = False
            if ok:
                done.add(node)
        return done

    def emit(self, delivered=None, dump=False):
        g, b = self.graph, Builder(self.compute)
        delivered = set(g.interfaces['inputs']) if delivered is None else set(delivered)
        refused = self.refusals(self.evaluable(delivered))
        if refused:
            raise Refusal('; '.join(refused[:5]))
        values = {}
        for name in sorted(delivered):
            v = g.input_values[name]
            # an integer input (token identifiers, a count) is int64 on the wire; a floating one is in the compute dtype
            dtype = TensorProto.INT64 if v['dtype'] in ('i32', 'i64') else b.ctype
            values[name] = b.input(name, dtype, ['n'] + [a['extent'] for a in v['shape']])
        positions = {}
        for name in sorted(delivered):
            stream = g.input_stream[name]
            if stream not in positions:
                positions[stream] = b.input(f"positions/{stream}", TensorProto.INT64, [f"n.{stream}"])
        params = {}                       # identity -> (raw initializer, cast to the compute dtype)
        states = {ident: StateRef(b, entry, self.compute) for ident, entry in g.states.items()}
        needed = {f"{o['node']}.{o['port']}" for o in g.interfaces['outputs'].values()}
        dumped = {p['value'] for c in g.layer_cuts() for p in c['payload']} if dump else set()
        active = self.evaluable(delivered)
        for node in g.order:
            if node not in active:
                continue
            entry = g.nodes[node]
            key = (entry['contract']['name'], entry['contract']['version'])
            prim = self.primitives[key]
            ins = {}
            for (n, port), vname in g.sources.items():
                if n == node and vname in values:
                    ins[port] = values[vname]          # a port fed by nothing this invocation is absent: the primitive decides
            for (n, port), name in g.fed_by_input.items():
                if n == node and name in values:
                    ins[port] = values[name]
            stream, factor = g.node_domain(node)
            ctx = Context(b, self, params, node, entry, positions.get(stream), factor)
            outs = prim.emit(ctx, entry['arguments'], ins, {s: i for s, i in g.slots_of.get(node, {}).items()},
                             {s: states[i] for s, i in g.states_of.get(node, {}).items()})
            for port, name in outs.items():
                vname = f"{node}.{port}"
                out_name = b.name(vname) if vname in b.taken else vname
                b.taken.add(out_name)
                b.node('Identity', [name], outputs=[out_name])
                values[vname] = out_name
                if vname in needed or vname in dumped:
                    b.output(out_name, b.ctype)
        for ident, st in states.items():
            if not st.written:            # a state the invocation did not visit passes through unchanged
                st.write(st.read())
        return b.model(g.model)


class Context:
    """What a primitive's emitter sees: the builder, the parameters as initializers, the node's positions."""

    def __init__(self, b, emitter, params, node, entry, positions, factor):
        self.b, self.emitter, self.params, self.node, self.entry, self.factor = b, emitter, params, node, entry, factor
        self.compute = b.compute
        self._positions = positions

    @property
    def positions(self):
        """The node's positions: the stream's, scaled by the D2 count of the value it works on (§5.3)."""
        if self._positions is None:
            return None
        if self.factor == 1:
            return self._positions
        raise Refusal(f"{self.node}: positions on a merged domain (count {self.factor}) are not emitted yet")

    def param(self, identity, cast=True):
        """The identity's initializer — the raw one at its D3 dtype, or its cast to the compute dtype (once)."""
        if identity not in self.params:
            t = self.emitter.graph.tensors[identity]
            from loader import raw
            words = raw(self.emitter.source.fetch(identity))
            name = self.b.initializer(f"param/{identity}", words, t['dtype'], [a['extent'] for a in t['shape']])
            self.params[identity] = [name, None]
        entry = self.params[identity]
        if not cast:
            return entry[0]
        if entry[1] is None:
            entry[1] = self.b.node('Cast', [entry[0]], hint=f"param/{identity}.f32", to=self.b.ctype)
        return entry[1]


def initializer_bytes(model):
    return sum(len(t.raw_data) for t in model.graph.initializer)


def save_model(model, path, threshold=int(1.5 * 2**30)):
    """, the initializers in an external file beside the model once they exceed the
    threshold — protobuf holds two gigabytes at most, and a three-layer Llama already carries more."""
    if initializer_bytes(model) > threshold:
        onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=os.path.basename(path) + '.data', size_threshold=1024)
    else:
        onnx.save_model(model, path)

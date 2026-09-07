"""The emitter: one ONNX graph for the expanded D1 graph, one invocation per run (R08) — every
public input a graph input with a dynamic element axis (under the `aligned` layout, a batch axis
before it: several sessions in one invocation, batch-plan B04), the positions of every stream an input,
every D4 state an input and an output (ONNX has no mutable state: a session carries the tensors
between runs), every parameter an initializer at its D3 dtype cast to the compute dtype where a
primitive uses it (the reference's convention, the cast folded once by the runtime), and every
instance a call of its primitive's emitter (`primitives/<name>.py`), in D1's order. The exposed
outputs are graph outputs; with `dump`, so is every value crossing a D6 layer graph_split (R07)."""
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
LAYOUTS = ('none', 'aligned')     # one session on the element axis; a batch axis, every session delivering the same count


class Refusal(Exception):
    pass


class Builder:
    """ONNX nodes, initializers, inputs and outputs, with unique names."""

    def __init__(self, compute='f32'):
        self.compute = compute
        self.ctype = COMPUTE[compute]
        self.nodes, self.initializers, self.inputs, self.outputs = [], [], [], []
        self.taken = set()
        self.by_output = {}               # value name -> the node that produces it
        self.transposed = {}              # weight name -> its transpose, once (MatMul against a stored [out, in] weight)

    def name(self, hint):
        base = hint
        k = 0
        while hint in self.taken:
            k += 1
            hint = f"{base}#{k}"
        self.taken.add(hint)
        return hint

    def node(self, op, inputs, outputs=1, hint=None, domain=None, **attrs):
        names = [self.name(f"{hint or op}") for _ in range(outputs)] if isinstance(outputs, int) else list(outputs)
        for n in names:
            self.taken.add(n)
        node = helper.make_node(op, list(inputs), names, domain=domain, **attrs) if domain else helper.make_node(op, list(inputs), names, **attrs)
        self.nodes.append(node)
        for n in names:
            if n:
                self.by_output[n] = node
        return names[0] if isinstance(outputs, int) and outputs == 1 else names

    def remove(self, node):
        """Drop a node whose outputs another one now provides (a fusion across instances)."""
        self.nodes.remove(node)
        for n in node.output:
            self.by_output.pop(n, None)

    def unsqueeze0(self, x, hint='u'):
        return self.node('Unsqueeze', [x, self.const(np.array([0], dtype=np.int64), 'ax0')], hint=hint)

    def squeeze0(self, x, hint='s'):
        return self.node('Squeeze', [x, self.const(np.array([0], dtype=np.int64), 'ax0')], hint=hint)

    def i64(self, *v):
        return self.const(np.array(v, dtype=np.int64), 'i')

    def shape_of(self, t, hint='shape'):
        return self.node('Shape', [t], hint=hint)

    def dim(self, t, k, hint='dim'):
        """Dimension k of t as an int64 scalar."""
        return self.node('Gather', [self.shape_of(t), self.const(np.array(k, dtype=np.int64), 'k')], hint=hint)

    def reshape_tail(self, t, keep, tail, hint='rs'):
        """t reshaped to its first `keep` dimensions followed by `tail`: the batch and element axes kept, the rest laid out."""
        head = self.node('Slice', [self.shape_of(t), self.i64(0), self.i64(keep)], hint=f"{hint}.head")
        return self.node('Reshape', [t, self.node('Concat', [head, self.i64(*tail)], hint=f"{hint}.shape", axis=0)], hint=hint)

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
        opsets = [helper.make_opsetid('', OPSET)]
        if any(n.domain == 'com.microsoft' for n in self.nodes):
            opsets.append(helper.make_opsetid('com.microsoft', 1))    # onnxruntime's fused operators, when a backend asked for them
        model = helper.make_model(graph, opset_imports=opsets, producer_name='tensorspine-onnx')
        model.ir_version = IR_VERSION
        return model


class StateRef:
    """A D4 state instance seen by a primitive: `read()` gives the input tensors (the positions held
    so far for `append`, the valid tail for `window`, the payload for `fixed`); `write(values)` names
    the output tensors the session takes as the state after the invocation."""

    def __init__(self, b, entry, compute, layout='none', held=None):
        self.b, self.entry, self.layout = b, entry, layout
        self.identity, self.evolution, self.span = entry['identity'], entry['evolution'], entry.get('span')
        self.stream = (entry.get('stream') or {}).get('stream')
        self.held = held                  # aligned: the positions each session holds on the state's stream, [b] (the input's name)
        self.components = {p['component']: [a['extent'] for a in p['shape']] for p in entry['payload']}
        self.inputs = {}
        for c, shape in self.components.items():
            if layout == 'aligned':       # a buffer of the capacity per session; a fixed state per session
                dims = (['b'] if self.evolution == 'fixed' else ['b', f"{self.identity}.cap"]) + shape
            else:
                dims = shape if self.evolution == 'fixed' else [f"{self.identity}.len"] + shape
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
    def __init__(self, graph, source, primitives, compute='f32', physical=None, target='onnx', layout='none'):
        """`primitives`: the emitters per target (`registry.load_all()`), or one target's set. `target`:
        the runtime the graph is emitted for — `onnx`, standard operators only, runs anywhere;
        `onnxruntime`, its fused operators wherever the target has a primitive of its own. The physical
        parameters may name a `backend` per instance, which picks that instance's target. `layout`:
        how sessions share an invocation (batch-plan B02, B04) — `none`, one session on the element
        axis; `aligned`, a batch axis before it, every session delivering the same count on each
        stream, the `append` states buffers of a capacity per session with the positions each holds
        (`held/<stream>`) an input. One layout per graph, reaching every primitive as `ctx.layout`."""
        if not primitives or not isinstance(next(iter(primitives.values())), dict):
            primitives = {target: primitives}
        if target not in primitives:
            raise Refusal(f"target {target!r}: one of {sorted(primitives)}")
        if layout not in LAYOUTS:
            raise Refusal(f"layout {layout!r}: one of {LAYOUTS}")
        self.graph, self.source, self.primitives, self.compute, self.physical, self.target = graph, source, primitives, compute, physical, target
        self.layout = layout

    def primitive(self, node, entry):
        """The instance's emitter: the target's, or the one the physical parameters' `backend` names."""
        target = (physical_for(self.physical, node, entry['primitive']) or {}).get('backend') or self.target
        table = self.primitives.get(target)
        if table is None:
            raise Refusal(f"{node}: backend {target!r} names no target of this generator ({sorted(self.primitives)})")
        return table.get((entry['primitive']['name'], entry['primitive']['version']))

    def refusals(self, nodes=None):
        out = []
        for node, entry in self.graph.nodes.items():
            if nodes is not None and node not in nodes:
                continue
            key = (entry['primitive']['name'], entry['primitive']['version'])
            p = self.primitive(node, entry)
            if p is None:
                out.append(f"{node}: no primitive for {key[0]}@{key[1]}")
                continue
            for r in p.supports(entry['arguments']):
                out.append(f"{node}: {key[0]}@{key[1]} does not implement {r}")
        return out

    def evaluable(self, delivered):
        """The instances an invocation evaluates (§7): every input port fed by a delivered input or
        an evaluated instance; a port a state indexed by it holds is fed too (a source complete)."""
        g = self.graph
        fed = {(n, p) for (n, p), name in g.fed_by_input.items() if name in delivered}
        held = {}
        for st in g.states.values():
            if st.get('indexed_by_port') and st['evolution'] == 'append':
                for m in st['members']:
                    node, _s = m.rsplit('.', 1)
                    held[(node, st['indexed_by_port'])] = True
        # the source of an insert transform may deliver nothing (§7): `splice`, the language's only insert
        # today, recognised by its primitive, as the reference does — D2 emits no transforms
        inserts = {(n, 'source') for n, e in g.nodes.items() if e['primitive']['name'] == 'splice'}
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
        aligned = self.layout == 'aligned'
        lead = ['b'] if aligned else []
        values = {}
        for name in sorted(delivered):
            v = g.input_values[name]
            # an integer input (token identifiers, a count) is int64 on the wire; a floating one is in the compute dtype
            dtype = TensorProto.INT64 if v['dtype'] in ('i32', 'i64') else b.ctype
            values[name] = b.input(name, dtype, lead + ['n'] + [a['extent'] for a in v['shape']])
        positions, held = {}, {}
        for name in sorted(delivered):
            stream = g.input_stream[name]
            if stream not in positions:
                positions[stream] = b.input(f"positions/{stream}", TensorProto.INT64, lead + [f"n.{stream}"])
                if aligned:               # the positions each session holds on the stream before this invocation
                    held[stream] = b.input(f"held/{stream}", TensorProto.INT64, ['b'])
        self.held = held
        params = {}                       # identity -> (raw initializer, cast to the compute dtype)
        states = {ident: StateRef(b, entry, self.compute, self.layout, held.get((entry.get('stream') or {}).get('stream')))
                  for ident, entry in g.states.items()}
        needed = {f"{o['node']}.{o['port']}" for o in g.interfaces['outputs'].values()}
        dumped = {p['value'] for c in g.layer_graph_splits() for p in c['payload']} if dump else set()
        active = self.evaluable(delivered)
        self.origin = {}                  # value name -> (primitive name, the primitive's raw output, its Identity node)
        for node in g.order:
            if node not in active:
                continue
            entry = g.nodes[node]
            prim = self.primitive(node, entry)
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
                self.origin[out_name] = (entry['primitive']['name'], name, b.by_output[out_name])
                values[vname] = out_name
                if vname in needed or vname in dumped:
                    b.output(out_name, b.ctype)
        for ident, st in states.items():
            if not st.written:            # a state the invocation did not visit passes through unchanged
                st.write(st.read())
        return b.model(g.model)


def physical_for(physical, node, primitive):
    """The opaque parameters addressed to an instance (generators/CAPABILITIES.md): by its exact
    identifier, by a site pattern where `*` alone is a wildcard, or by its primitive version; more
    specific entries override more general ones (primitive < pattern < exact)."""
    import re
    if not physical:
        return None
    out = {}
    cid = f"{primitive['name']}@{primitive['version']}"
    for key, value in physical.items():
        if key == cid:
            out.update(value)
    for key, value in physical.items():
        if key != cid and key != node and '*' in key and re.fullmatch(re.escape(key).replace(r'\*', '.*'), node):
            out.update(value)
    if node in physical:
        out.update(physical[node])
    return out or None


class Context:
    """What a primitive's emitter sees: the builder, the parameters as initializers, the node's
    positions, and the opaque physical parameters addressed to the instance — the channel a
    backend-specific realisation (a fused operator of one runtime) is selected through."""

    def __init__(self, b, emitter, params, node, entry, positions, factor):
        self.b, self.emitter, self.params, self.node, self.entry, self.factor = b, emitter, params, node, entry, factor
        self.compute = b.compute
        self.physical = physical_for(emitter.physical, node, entry['primitive'])
        self.layout = emitter.layout      # the batch layout of the whole graph (B02): `none` or `aligned`
        self._positions = positions

    def held(self, stream=None):
        """Aligned layout: the positions each session holds on `stream` (the node's own by default)
        before this invocation, an int64 `[b]` input; None under the `none` layout."""
        if self.layout != 'aligned':
            return None
        stream = stream or self.emitter.graph.node_domain(self.node)[0]
        name = self.emitter.held.get(stream)
        if name is None:
            raise Refusal(f"{self.node}: no held positions for stream '{stream}': the delivery introduces none")
        return name

    def target(self):
        """The target this instance is emitted for: the physical parameters' `backend` when they name
        one, else the generator's target — the registry chose the primitive by it already."""
        return (self.physical or {}).get('backend') or self.emitter.target

    def origin(self, value):
        """(primitive name, the primitive's raw output name, the Identity node naming the value) of a value
        an earlier instance produced — what a fusion across instances reasons on."""
        return self.emitter.origin.get(value)

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

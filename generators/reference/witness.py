"""The witness at work (Specification §4.1, O1.3): one unit fixture per case a kernel declares
(docs/TENSORSPINE-FIXTURE.md).

A case is a primitive version with arguments at small quantities, a seed and a list of
invocations. It becomes a one-instance model definition — the instance under its arguments,
one public input per input port, one public output per output port, every slot bound to an
identity named after it and located at the fixture's own `param/<identity>` key — derived and run
by the language's tools and this generator's own machinery, exactly as a whole model is. What the
run produced is the fixture: the parameters, and per invocation the inputs, the positions, the
outputs and every state. A conformer runs the same document from the same file and must agree
within the tolerance the kernel declares for its compute dtype.

The union (harness guide §8): an instance that does not read across positions (D1's
`across_positions`), holds no state and reads one stream may be evaluated on several sessions'
elements at once. On every unit fixture the rule admits, the fixture's invocations run as sessions
of one packed invocation — each with its recorded positions, one invocation twice when there is
only one — must give every session its recorded outputs: a kernel that secretly reads across
positions, or a primitive silent about it, fails here the day its fixture exists.

    ref.py witness NAME@VERSION|all             regenerate every case and compare with the committed fixture
    ref.py witness NAME@VERSION|all --record    write the fixtures
    ref.py witness NAME@VERSION/CASE [--record]  one case of a primitive version

The regeneration is provenance, apart from conformance: the parameters and inputs regenerated
from the seed are compared exactly with the recorded ones and the line says `regenerated exactly`,
`within the f32 tolerance` or `DIFFER`; the verdict is the run against the stored tensors, which
are the evidence, and every recorded position, output and state is required of it. A drift of the
regeneration fails only under `--strict-provenance` — the rule on the box the fixtures are recorded
on, where a silent change of the witness is what it catches: refused unless the primitive version
changes or the correction is a declared patch (§8.2), whose re-recorded fixtures say so. The same
run at every other dtype the kernel declares a tolerance for is the check that the tolerance table
holds.
"""
import json
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if os.path.join(ROOT, 'tools') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import primitive_library as primitive_library_mod          # noqa: E402
import validate as validate_mod        # noqa: E402
from expr import primitive_condition, static_argument   # noqa: E402
import graph as graph_mod              # noqa: E402
import loader                          # noqa: E402
import registry                        # noqa: E402
from compare import compare, read_fixture, tolerance_for, write_fixture   # noqa: E402
from module import TensorspineModel, step_batch    # noqa: E402
from plan import Plan                  # noqa: E402
from session import Session            # noqa: E402

FIXTURES = os.path.join(HERE, 'fixtures', 'primitives')
PRIMITIVE_LIBRARY = os.path.join(ROOT, 'data', 'primitive-library')
COMPUTE = {'f32': torch.float32, 'bf16': torch.bfloat16, 'f16': torch.float16}
CAPACITY = 64


# --- the one-instance document -----------------------------------------------

def _expression(v):
    """An argument value as the model grammar writes it: literals, records of literals."""
    if isinstance(v, dict):
        return {"record": {k: _expression(x) for k, x in v.items()}}
    return {"literal": v}


def document(name, version, arguments, cat, base):
    """The one-instance model definition of a case; `base` is the primitive library base as the document
    will write it, relative to where it will be read from."""
    definition = cat['by_id'][(name, version)]
    given = {k: _expression(v) for k, v in arguments.items()}
    args, problems = validate_mod.resolve_arguments(definition, given, lambda v: static_argument(v, {}, {}))
    if problems:
        raise ValueError(f"{name}@{version}: {problems[0][1]}")

    def present(element):
        return primitive_condition(element['present_when'], args) if 'present_when' in element else True

    root = {"kind": "root", "instance": "unit"}
    inputs, outputs = {}, {}
    transformed = {t['from_port'] for t in definition.get('domain_transforms', [])}
    # a port a state rule indexes (`indexed_by.port`) carries the stream its state grows along: its own,
    # as a transformed port's — the cross-attention source, the front end's frames
    transformed |= {r['indexed_by']['port'] for sp in definition['state_ports'].values() for r in sp.get('rules', [])
                    if 'port' in (r.get('indexed_by') or {})}
    own = None                      # the instance's own domain is one stream (§5.3, V5): the first
    for pname, port in definition['ports']['inputs'].items():
        if present(port):
            kind = port['domain']['kind']
            inputs[pname] = {"to": [{"instance": root, "port": pname}], "kind": kind if kind != 'inherit' else 'token'}
            if pname in transformed:
                continue                # a transformed port carries its own stream
            if own is None:
                own = pname
            else:
                inputs[pname]['stream'] = own
    for pname, port in definition['ports']['outputs'].items():
        if present(port):
            outputs[pname] = {"from": {"instance": root, "port": pname}, "generative": False}
    parameters = {}
    for slot, param in definition['parameters'].items():
        if present(param):
            ident = f"unit.{slot}"
            parameters[ident] = {"members": [{"instance": root, "parameter": slot}], "tensor": {"name": ident},
                                 "location": {"tensor": [f"param/{ident}"]}}
            if 'f32' in cat['precision'][param['role']]['admissible']:
                parameters[ident]['dtype'] = 'f32'          # stored exactly: the fixture is its own checkpoint
    states = {}
    for sname, port in definition['state_ports'].items():
        if primitive_condition(port['present_when'], args):
            ident = f"unit.{sname}"
            states[ident] = {"members": [{"instance": root, "state": sname}], "identity": {"name": ident}}
            # a state carried across fragments sits on a fragmented stream (V16): the input carrying
            # that stream — the port the state is indexed by, else the instance's own, its first
            # untransformed input — is delivered in fragments, as the streaming cases run it
            ca = port.get('carried_across')
            if ca and primitive_condition(ca['when'], args):
                rule = next((r for r in port.get('rules', []) if primitive_condition(r['when'], args)), None)
                carrier = (rule.get('indexed_by') or {}).get('port') if rule else None
                carrier = carrier or own
                if carrier in inputs:
                    inputs[carrier]['fragmented'] = True
    return {"schema": "tensorspine/2.0", "model": f"unit-{name.replace('.', '_')}-{version.replace('.', '_')}",
            "primitive_libraries": [{"base": base}], "quantities": {}, "constants": {},
            "instances": {"unit": {"primitive": {"name": name, "version": version}, "arguments": given, "families": ["unit"]}},
            "compositions": {},
            "bindings": {"values": {}, "parameters": parameters, "constants": {}, "states": states},
            "interfaces": {"inputs": inputs, "outputs": outputs}}


def _materialise(doc, directory):
    """The document written where the primitive library base resolves: beside the fixtures, the base as
    written; anywhere else, the base made absolute."""
    doc = json.loads(json.dumps(doc))
    doc['primitive_libraries'] = [{"base": PRIMITIVE_LIBRARY + os.sep}]
    path = os.path.join(directory, doc['model'] + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    return path


def fixture_dir(name, version):
    return os.path.join(FIXTURES, f"{name}@{version}")


def primitive_library_base_from(directory):
    """The primitive library base, relative to a directory: what the embedded document writes."""
    return os.path.relpath(PRIMITIVE_LIBRARY, directory) + '/'


# --- random parameters and inputs -------------------------------------------------

def parameters(g, seed):
    """Every D3 identity drawn from the seed, in identity order: matrices at 1/sqrt(fan-in) so
    that outputs stay O(1), biases and scalars small, normalisation scales around one."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for ident in sorted(g.tensors):
        t = g.tensors[ident]
        shape = [a['extent'] for a in t['shape']]
        role = t['role']
        if role == 'norm.scale':
            x = 1 + 0.1 * torch.randn(shape, generator=gen)
        elif role.endswith('bias') or role.endswith('.scalar') or role == 'norm.bias':
            x = 0.1 * torch.randn(shape, generator=gen)
        elif len(shape) >= 2:
            x = torch.randn(shape, generator=gen) / (shape[-1] ** 0.5)
        else:
            x = torch.randn(shape, generator=gen)
        out[ident] = x.to(getattr(torch, graph_mod.DTYPES[t['dtype']]))
    return out


def inputs_for(g, delivered, gen, compute):
    """The tensors one invocation delivers: identifiers for a token-index input, small integers
    for a count, else values drawn from the seed on the input's D2 shape."""
    out = {}
    for name in sorted(delivered):
        n = delivered[name]
        v = g.input_values[name]
        shape = [a['extent'] for a in v['shape']]
        if v['role'] == 'activation.token_index':
            vocabulary = g.nodes['unit']['arguments'].get('vocabulary', 256)
            out[name] = torch.randint(0, vocabulary, (n,), generator=gen)
        elif v['role'] == 'activation.count':          # an integer per element: a delay of 1 to 31 tokens
            out[name] = torch.randint(1, 32, (n,), generator=gen, dtype=torch.int32)
        else:
            out[name] = torch.randn([n] + shape, generator=gen).to(compute)
    return out


# --- producing and checking ------------------------------------------------------------

def run(g, kernels, params, invocations, compute, seed=None, given=None):
    """The invocations of a case — inputs, positions, outputs and states per invocation, every
    tensor an f32 copy. The inputs are drawn from `seed` (the witness recording) or taken from
    `given`, a recorded fixture's tensors (a conformer repeating it)."""
    model = TensorspineModel(g, Plan(g, kernels), params, compute, 'cpu')
    session = Session(model, CAPACITY, 'cpu', compute)
    gen = torch.Generator().manual_seed(seed + 1) if seed is not None else None
    tensors = {}
    for k, delivered in enumerate(invocations):
        if given is None:
            ins = inputs_for(g, delivered, gen, compute)
        else:
            ins = {name: (given[f"in/{k}/{name}"].to(compute) if given[f"in/{k}/{name}"].is_floating_point() else given[f"in/{k}/{name}"])
                   for name in sorted(delivered)}
        positions = {}
        for name, t in ins.items():
            stream = g.input_stream[name]
            start = session.consumed.get(stream, 0)
            positions[stream] = torch.arange(start, start + t.shape[0])
        outs = session.run(ins)
        for name, t in ins.items():
            tensors[f"in/{k}/{name}"] = t.detach().to('cpu', torch.float32 if t.is_floating_point() else t.dtype).clone()
        for stream, p in positions.items():
            tensors[f"positions/{k}/{stream}"] = p.clone()
        for name, t in outs.items():
            tensors[f"out/{k}/{name}"] = t.detach().to('cpu', torch.float32).clone()
        for ident, st in session.states.items():
            bufs, length = st.read()
            for c, buf in bufs.items():
                tensors[f"state/{k}/{ident}/{c}"] = (buf[:length] if st.evolution == 'append' else buf).detach().to('cpu', torch.float32).clone()
    return tensors


def cases(kernels, only=None):
    """(name, version, kernel, case) for every case the kernels declare, those of one primitive
    version, or one case (`NAME@VERSION/CASE`)."""
    primitive, _, case_name = (only or '').partition('/')
    out = []
    for (name, version), k in sorted(kernels.items()):
        if only and f"{name}@{version}" != primitive:
            continue
        for case in getattr(k, 'FIXTURES', []):
            if case_name and case['case'] != case_name:
                continue
            out.append((name, version, k, case))
    return out


def _version():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def produce(name, version, kernel, case, cat, kernels):
    """(tensors, metadata) of one case, as the witness computes it in f32."""
    directory = fixture_dir(name, version)
    doc = document(name, version, case['arguments'], cat, primitive_library_base_from(directory))
    tmp = tempfile.mkdtemp(prefix='tensorspine-witness-')
    g = graph_mod.load(_materialise(doc, tmp))
    refused = registry.refusals(g, kernels)
    if refused:
        raise ValueError(f"{name}@{version}/{case['case']}: {refused[0]}")
    params = parameters(g, case['seed'])
    tensors = {f"param/{ident}": t.detach().to('cpu', torch.float32).clone() for ident, t in params.items()}
    tensors.update(run(g, kernels, params, case['invocations'], torch.float32, seed=case['seed']))
    metadata = {'schema': 'tensorspine-fixture/1', 'kind': 'unit', 'id': f"{name}@{version}/{case['case']}",
                'primitive': {'name': name, 'version': version}, 'arguments': g.nodes['unit']['arguments'],
                'document': doc, 'invocations': case['invocations'], 'seed': case['seed'],
                'witness': {'generator': 'reference', 'version': _version(),
                            'kernel': os.path.relpath(kernel.__file__, HERE), 'versions': {'torch': torch.__version__}},
                'compute': 'f32', 'tolerance': {d: dict(t) for d, t in kernel.TOLERANCE.items()}}
    return tensors, metadata


def fixture_path(fid):
    name_version, case = fid.split('/')
    return os.path.join(FIXTURES, name_version, case + '.safetensors')


def record(name, version, kernel, case, cat, kernels):
    tensors, metadata = produce(name, version, kernel, case, cat, kernels)
    path = fixture_path(metadata['id'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_fixture(path, tensors, metadata)
    return path, len(tensors)


def verify(fid, kernels, strict_provenance=False):
    """The committed fixture against the witness now: the parameters and inputs regenerated from
    the seed (provenance), the run repeated with the parameters loaded from the fixture as a
    conformer would load them, at f32 and at every other dtype the kernel states a tolerance for
    (conformance), and the union. Returns (ok, lines)."""
    path = fixture_path(fid)
    tensors, meta = read_fixture(path)
    name, version = meta['primitive']['name'], meta['primitive']['version']
    kernel = kernels.get((name, version))
    lines = []
    if kernel is None:
        return False, [f"{fid}: no kernel for {name}@{version}"]
    tmp = tempfile.mkdtemp(prefix='tensorspine-witness-')
    g = graph_mod.load(_materialise(meta['document'], tmp))
    errors, _, _ = loader.verify(g, path)
    if errors:
        return False, [f"{fid}: the fixture is not its own checkpoint: {errors[0]}"]
    ok = True
    # provenance (docs/TENSORSPINE-FIXTURE.md §5): the parameters and inputs regenerated from the seed
    # against the recorded ones — exactly, within the f32 tolerance, or differing. The stored tensors
    # are the evidence a conformer is checked against; regeneration says whether the fixture is what
    # its seed claims, and a drift fails only under strict provenance (the recording box)
    atol, rtol = tolerance_for(meta, 'f32')
    fresh = {f"param/{i}": t.to(torch.float32) for i, t in parameters(g, meta['seed']).items()}
    gen = torch.Generator().manual_seed(meta['seed'] + 1)
    for k, delivered in enumerate(meta['invocations']):
        for name, t in inputs_for(g, delivered, gen, torch.float32).items():
            fresh[f"in/{k}/{name}"] = t.to(torch.float32) if t.is_floating_point() else t
    seeded = {k: v for k, v in tensors.items() if k.startswith(('param/', 'in/'))}
    seed = compare(fresh, seeded, atol, rtol, required=set(seeded))
    exact = seed.ok and seed.worst == 0.0
    provenance = ('regenerated exactly' if exact else f"within the f32 tolerance (max |d| {seed.worst:.1e}), not exactly" if seed.ok
                  else f"DIFFER (max |d| {seed.worst:.1e}) {seed.detail()}")
    ok &= exact or not strict_provenance
    lines.append(f"{fid}: {seed.compared} parameters and inputs from seed {meta['seed']}: {provenance}"
                 + ('' if exact else '  — provenance drift, ' + ('refused under --strict-provenance' if strict_provenance else 'the verdict is conformance against the stored tensors')))
    # conformance, as a conformer repeats it: parameters and inputs from the fixture, at each dtype;
    # every recorded position, output and state is required of the run
    params = loader.load_parameters(g, path, 'cpu')
    recorded = {k: v for k, v in tensors.items() if not k.startswith(('param/', 'in/'))}
    for dtype, tol in sorted(meta['tolerance'].items()):
        compute = COMPUTE[dtype]
        got = run(g, kernels, {i: t.to(compute) if t.is_floating_point() else t for i, t in params.items()},
                  meta['invocations'], compute, given=tensors)
        got = {k: v for k, v in got.items() if not k.startswith('in/')}
        verdict = compare(got, recorded, tol['atol'], tol['rtol'])
        ok &= verdict.ok
        lines.append(f"{fid}: {verdict.compared} positions, outputs and states at {dtype} within atol {tol['atol']:g} rtol {tol['rtol']:g} "
                     f"(max |d| {verdict.worst:.1e})" + (f"  {verdict.detail()}" if not verdict.ok else ''))
    # §5.3's invariance (R17): a fixture whose stream is fragmented and delivered in two or more
    # invocations must give, delivered whole in one, the concatenated recorded outputs and the
    # recorded final states — the sufficient condition the primitive's invariant now guarantees. A
    # non-fragmented multi-invocation fixture (a bidirectional pass, cross with a frozen source) is
    # not invariant to concatenation and is not checked; a fixture under a declared conditions entry
    # (finding 26, a shared window reader once its ring has wrapped) is skipped and says so.
    fragmented = [n for n, e in g.interfaces['inputs'].items() if e.get('fragmented')]
    if fragmented and len(meta['invocations']) > 1:
        K = len(meta['invocations'])
        names = sorted({n for inv in meta['invocations'] for n in inv})
        whole = {}
        for name in names:
            parts = [tensors[f"in/{k}/{name}"] for k in range(K) if f"in/{k}/{name}" in tensors]
            whole[name] = torch.cat(parts, 0)
        given_whole = {f"in/0/{name}": t for name, t in whole.items()}
        one = [{name: whole[name].shape[0] for name in whole}]
        got = run(g, kernels, {i: t.to(torch.float32) if t.is_floating_point() else t for i, t in params.items()},
                  one, torch.float32, given=given_whole)
        out_names = sorted({key.split('/', 3)[2] for key in tensors if key.startswith('out/0/')})
        want = {f"out/0/{name}": torch.cat([tensors[f"out/{k}/{name}"] for k in range(K) if f"out/{k}/{name}" in tensors])
                for name in out_names}
        want.update({f"state/0/{key.split('/', 2)[2]}": t for key, t in tensors.items() if key.startswith(f"state/{K - 1}/")})
        got = {k: v for k, v in got.items() if not k.startswith('in/')}
        verdict = compare(got, want, atol, rtol)
        ok &= verdict.ok
        total = whole[fragmented[0]].shape[0]
        lines.append(f"{fid}: the {K} fragments delivered whole ({total} elements on {fragmented[0]}) give the concatenated "
                     f"outputs and the final states (§5.3, max |d| {verdict.worst:.1e})" + ('' if verdict.ok else f"  DIFFER {verdict.detail()}"))

    # the union (harness guide §8): the invocations as sessions of one packed invocation, at f32;
    # every recorded output of every invocation is required of the batch
    model = TensorspineModel(g, Plan(g, kernels), {i: t.to(torch.float32) for i, t in params.items()}, torch.float32, 'cpu')
    if model.per_session['unit']:
        lines.append(f"{fid}: evaluated per session under a batch — reads across positions, holds a state or reads several streams")
        return ok, lines
    recorded_k = list(range(len(meta['invocations']))) if len(meta['invocations']) > 1 else [0, 0]
    invocations = [meta['invocations'][r] for r in recorded_k]
    ins = [{name: tensors[f"in/{r}/{name}"] for name in sorted(delivered)} for r, delivered in zip(recorded_k, invocations)]
    pos = [{g.input_stream[name]: tensors[f"positions/{r}/{g.input_stream[name]}"] for name in delivered}
           for r, delivered in zip(recorded_k, invocations)]
    outs = step_batch(model, ins, pos, [{} for _ in invocations])
    got = {f"out/{k}/{name}": t.detach().to(torch.float32) for k, o in enumerate(outs) for name, t in o.items()}
    want = {f"out/{k}/{key.split('/', 2)[2]}": t for k, r in enumerate(recorded_k) for key, t in tensors.items() if key.startswith(f"out/{r}/")}
    verdict = compare(got, want, atol, rtol)
    ok &= verdict.ok
    lines.append(f"{fid}: {len(meta['invocations'])} invocation(s) as {len(invocations)} sessions of one packed invocation give each "
                 f"session its recorded outputs on the union (max |d| {verdict.worst:.1e})" + ('' if verdict.ok else f"  DIFFER {verdict.detail()}"))
    return ok, lines


def committed(only=None):
    """The ids of the committed unit fixtures, those of one primitive version, or one case."""
    primitive, _, case_name = (only or '').partition('/')
    out = []
    if not os.path.isdir(FIXTURES):
        return out
    for d in sorted(os.listdir(FIXTURES)):
        if only and d != primitive:
            continue
        for f in sorted(os.listdir(os.path.join(FIXTURES, d))):
            if f.endswith('.safetensors') and not (case_name and f != case_name + '.safetensors'):
                out.append(f"{d}/{f[:-len('.safetensors')]}")
    return out


def main(target, do_record, strict_provenance=False):
    kernels = registry.load_kernels()
    cat = primitive_library_mod.load(PRIMITIVE_LIBRARY)
    only = None if target == 'all' else target
    if do_record:
        for name, version, kernel, case in cases(kernels, only):
            path, n = record(name, version, kernel, case, cat, kernels)
            print(f"  {name}@{version}/{case['case']}: {n} tensors -> {os.path.relpath(path, ROOT)}")
        return 0
    ok = True
    for fid in committed(only):
        good, lines = verify(fid, kernels, strict_provenance)
        ok &= good
        for line in lines:
            print(f"  {'ok  ' if good else 'FAIL'} {line}")
    declared = {f"{n}@{v}/{c['case']}" for n, v, _k, c in cases(kernels, only)}
    missing = sorted(declared - set(committed(only)))
    if missing:
        ok = False
        print(f"  FAIL not recorded: {missing} (ref.py witness ... --record)")
    print("witness: all good" if ok else "witness: FAILED")
    return 0 if ok else 1

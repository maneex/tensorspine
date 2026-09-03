#!/usr/bin/env python3
"""The reference generator's command line.

    ref.py info    MODEL [--capacity N]                      bytes from D3/D4, free memory, refusals
    ref.py verify  MODEL --checkpoint DIR                    V17 against the file headers; nothing is read
    ref.py run     MODEL --checkpoint DIR --ids 1,2,3 [--steps N] [--dump F] [--compile]
    ref.py run     MODEL --random [--seed N] …               parameters drawn from the D3 shapes
    ref.py run     MODEL … --input audio=FIXTURE[:in/audio]  a safetensors tensor delivered to a non-token input with the prompt
    ref.py run     MODEL --checkpoint DIR --audio WAV --ids … [--stop]   a WAV through the checkpoint's own feature extractor, into the
                                                             document's audio input; --stop ends decoding at the artifact's end-of-text;
                                                             on a document whose token stream joins the audio stream (Voxtral Realtime)
                                                             the artifact's processor builds the prompt and the delay, the prompt takes
                                                             its tokens' frames, every step a token's, and the end of the audio ends the run
    ref.py chat    MODEL --checkpoint DIR [--max-new-tokens N] [--temperature T --top-p P --seed N]
    ref.py compare OURS FIXTURE [--atol A --rtol R]          a dump against a fixture, at every cut and state
    ref.py witness NAME@VERSION|all [--record]               the unit fixtures of a contract version: regenerated and compared, or written

Common options: --device cpu|cuda[:i], --compute f32|bf16, --capacity N|STREAM=N,… (positions per append state,
for every stream or per stream: tokens=64,audio=1500), --truncate decoder.layer=N, --set path=value. MODEL is a model document (derived here) or a derived document. In a chat, an
empty line quits; the session persists across turns. See generators/reference/README.md.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import graph as graph_mod      # noqa: E402
import loader                  # noqa: E402
import registry                # noqa: E402
from chat import load_tokenizer, stop_ids  # noqa: E402
from compare import read_fixture, read_dump, tolerance_for, write_dump  # noqa: E402
from module import TensorspineModel  # noqa: E402
from plan import Plan          # noqa: E402
from session import Session, greedy  # noqa: E402

COMPUTE = {'f32': torch.float32, 'bf16': torch.bfloat16, 'f16': torch.float16}


def parse_capacity(text):
    """`N`: every stream; `STREAM=N,…`: per stream — a cross-attention cache holds the source
    stream's positions, and one number cannot size it and the token caches alike."""
    if '=' not in text:
        return int(text)
    return {k.strip(): int(v) for k, v in (item.split('=', 1) for item in text.split(','))}


def load_inputs(specs, device, dtype):
    """`NAME=FILE[:KEY]`: a tensor of a safetensors file (a fixture's `in/NAME`, by default, else
    the file's only tensor), delivered to the public input NAME with the prompt."""
    from safetensors.torch import load_file
    out = {}
    for spec in specs:
        name, rest = spec.split('=', 1)
        file, key = rest.rsplit(':', 1) if ':' in rest else (rest, None)
        tensors = load_file(file)
        if key is None:
            key = f"in/{name}" if f"in/{name}" in tensors else (next(iter(tensors)) if len(tensors) == 1 else None)
            if key is None:
                raise SystemExit(f"--input {spec}: name the tensor, {file} holds {sorted(tensors)[:5]}")
        t = tensors[key]
        out[name] = t.to(device, dtype) if t.is_floating_point() else t.to(device)
    return out


def audio_frames(path, checkpoint, g, device, dtype):
    """`--audio WAV`: the checkpoint's own feature extractor turns a mono 16-bit WAV at its rate into
    the frames the document's audio input takes (the public input whose value has the role
    `activation.audio_frames`), element-major — the delivery's preprocessing, taken from the
    artifact as the tokenizer is, the model class never instantiated. The extractor's window
    bounds the signal: what lies beyond it is cut, and silence pads a shorter one."""
    import wave
    import numpy as np
    from transformers import AutoFeatureExtractor
    name = next((n for n, v in g.input_values.items() if v.get('role') == 'activation.audio_frames'), None)
    if name is None:
        raise SystemExit(f"--audio: {g.model} has no public input of audio frames")
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
    with wave.open(path) as wav:
        channels, width, rate, n = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if (channels, width, rate) != (1, 2, extractor.sampling_rate):
            raise SystemExit(f"--audio {path}: {channels} channel(s), {8 * width}-bit, {rate} Hz — the extractor takes mono 16-bit at {extractor.sampling_rate} Hz")
        pcm = np.frombuffer(wav.readframes(n), dtype='<i2').astype(np.float32) / 32768.0
    x = extractor(pcm, sampling_rate=rate, return_tensors='pt').input_features[0].T.contiguous()
    window = getattr(extractor, 'n_samples', None)
    cut = f"; cut to the extractor's {window // rate} s window" if window and n > window else ''
    print(f"  audio: {os.path.basename(path)}, {n / rate:.1f} s -> {x.shape[0]} frames of {x.shape[1]} on input {name} ({type(extractor).__name__}{cut})")
    return {name: x.to(device, dtype)}


def streaming_delivery(path, checkpoint, g, device, dtype):
    """`--audio WAV` on a document whose token stream joins the audio stream (Voxtral Realtime): the
    artifact's processor builds the streaming prefill — the prompt, the left-padded audio turned into
    frames by the checkpoint's extractor, and the delay — and the schedule follows D2's count: the
    prompt's tokens take as many frames as their count says (eight each), every step one token's
    worth. Returns (ids, the inputs of the prefill, the frames left for the steps)."""
    import wave
    import numpy as np
    from transformers import AutoProcessor
    audio = next((n for n, v in g.input_values.items() if v.get('role') == 'activation.audio_frames'), None)
    count = next((n for n, v in g.input_values.items() if v.get('role') == 'activation.count'), None)
    try:
        processor = AutoProcessor.from_pretrained(checkpoint)
    except ImportError as e:
        raise SystemExit(f"--audio: the artifact's processor does not import ({e}); it needs mistral_common and soundfile")
    with wave.open(path) as wav:
        channels, width, rate, n = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if (channels, width, rate) != (1, 2, processor.feature_extractor.sampling_rate):
            raise SystemExit(f"--audio {path}: {channels} channel(s), {8 * width}-bit, {rate} Hz — the extractor takes mono 16-bit at {processor.feature_extractor.sampling_rate} Hz")
        pcm = np.frombuffer(wav.readframes(n), dtype='<i2').astype(np.float32) / 32768.0
    enc = processor(pcm, is_streaming=True, is_first_audio_chunk=True, sampling_rate=rate, return_tensors='pt')
    ids = enc['input_ids'][0].tolist()
    frames = enc['input_features'][0].T.contiguous().to(device, dtype)
    per_token = int(g.elements_per[g.feedback_input])
    first = len(ids) * per_token
    if frames.shape[0] < first:
        raise SystemExit(f"--audio {path}: {frames.shape[0]} frames, and the prompt of {len(ids)} tokens takes {first}")
    prefill = {audio: frames[:first]}
    if count is not None:
        prefill[count] = torch.tensor([int(enc['num_delay_tokens'])], dtype=torch.int32, device=device)
    print(f"  audio: {os.path.basename(path)}, {n / rate:.1f} s -> {frames.shape[0]} frames of {frames.shape[1]} on input {audio} "
          f"({type(processor).__name__}: a prompt of {len(ids)} tokens with the first {first}, then {per_token} per step"
          + (f", {count} = {int(enc['num_delay_tokens'])}" if count else '') + ")")
    return ids, prefill, frames[first:]


def artifact_tokenizer(checkpoint):
    """The artifact's tokenizer, for the text of the tokens and its end-of-text ids; None when it has none."""
    try:
        return load_tokenizer(checkpoint)
    except Exception:  # noqa: BLE001  a checkpoint without a tokenizer: the ids are printed alone
        return None


def common(p):
    p.add_argument('model')
    p.add_argument('--truncate', help='shorten one composition index range, e.g. decoder.layer=3')
    p.add_argument('--set', action='append', default=[], metavar='PATH=VALUE',
                   help='edit the document before deriving, e.g. quantities.d.source.value=64 (JSON value)')
    p.add_argument('--capacity', type=parse_capacity, default=1024, metavar='N|STREAM=N,…',
                   help='positions every append state may hold: one number for every stream, or per stream (tokens=64,audio=1500)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--compute', default=None, help='f32 (CPU default) | bf16 (CUDA default)')
    p.add_argument('--physical', metavar='FILE', help='opaque parameters for the primitives (generators/CAPABILITIES.md): '
                   'a JSON object keyed by occurrence, by site pattern with * as the wildcard (decoder/attn[layer=*]) or by contract version')
    p.add_argument('--max-ram', type=float, default=None, metavar='GIB',
                   help='run in blocks of layers at legal cuts so that the parameters held at once, the '
                        'payload crossing into a block, the states and the largest temporary stay under this bound')


def open_graph(args):
    path = args.model
    if args.truncate or args.set:
        tmp = tempfile.mkdtemp(prefix='tensorspine-ref-')
        if args.truncate:
            path, notes = graph_mod.truncated(path, args.truncate, tmp)
            for n in notes:
                print(f"  edited: {n}")
        if args.set:
            edits = {}
            for item in args.set:
                k, v = item.split('=', 1)
                edits[k] = json.loads(v)
            path, notes = graph_mod.edited(path, edits, tmp, suffix='set')
            for n in notes:
                print(f"  edited: {n}")
    return graph_mod.load(path)


def physical_of(args):
    if not getattr(args, 'physical', None):
        return None
    with open(args.physical, encoding='utf-8') as f:
        return json.load(f)


def compute_dtype(args):
    if args.compute:
        return COMPUTE[args.compute]
    return torch.bfloat16 if str(args.device).startswith('cuda') else torch.float32


def make_plan(g, kernels, args, dtype):
    resident = loader.state_bytes(g, args.capacity, dtype) + loader.largest_temporary(g, dtype)
    max_bytes = int(args.max_ram * 2**30) if args.max_ram else None
    elements = loader.largest_capacity(args.capacity)
    plan = Plan(g, kernels, max_bytes=max_bytes, elements=elements, resident_bytes=resident)
    if max_bytes is not None:
        for line in plan.summary(elements, max_bytes, resident):
            print("  " + line)
    return plan


def delivery(g, extra=()):
    """The inputs a first invocation delivers: the token input and whatever `--input` names, so
    that refusals are computed over the occurrences that delivery evaluates (§7)."""
    return ({g.feedback_input} if g.feedback_input else set(g.interfaces['inputs'])) | set(extra)


def report(g, plan, i, args, dtype):
    capacity = args.capacity if isinstance(args.capacity, int) else ', '.join(f"{k}={v}" for k, v in args.capacity.items())
    print(f"  parameters {loader.gib(i['parameter_bytes'])} at the declared dtypes; states {loader.gib(i['state_bytes'])} "
          f"at {dtype} for a capacity of {capacity}; largest per-operation temporary {loader.gib(i['temporary_bytes'])}")
    print(f"  resident   {loader.gib(i['resident_bytes'])} — {i['mode']}; free on {args.device}: "
          f"{loader.gib(i['free_bytes']) if i['free_bytes'] is not None else 'unknown'}; activations not budgeted")



def cmd_info(args):
    g = open_graph(args)
    dtype = compute_dtype(args)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels, Plan(g, kernels).evaluable(g.required_inputs()))
    print(f"{g.model}: {len(g.nodes)} nodes, {len(g.tensors)} tensors, {len(g.states)} states")
    if r:
        print(f"  refusals: {len(r)}")
        for line in r[:20]:
            print("    " + line)
        return 1
    try:
        plan = make_plan(g, kernels, args, dtype)
    except ValueError as e:
        print(f"  refused: {e}")
        return 1
    report(g, plan, loader.info(g, args.capacity, dtype, args.device, plan, args.capacity), args, dtype)
    return 0


def cmd_compare(args):
    """A dump of this generator against a fixture (docs/TENSORSPINE-FIXTURE.md): the tolerance is
    the fixture's, for the dump's compute dtype, unless both --atol and --rtol override it."""
    from compare import compare
    ours, ho = read_dump(args.ours)
    theirs, ht = read_fixture(args.theirs)
    compute = {'torch.float32': 'f32', 'torch.bfloat16': 'bf16', 'torch.float16': 'f16'}.get(ho.get('compute'), 'f32')
    atol, rtol = tolerance_for(ht, compute)
    if args.atol is not None or args.rtol is not None:
        atol = args.atol if args.atol is not None else atol
        rtol = args.rtol if args.rtol is not None else rtol
    rows, failures, only = compare(ours, theirs, atol, rtol)
    print(f"{len(rows)} keys compared (atol {atol}, rtol {rtol} for {compute}); tokens ours {ho.get('tokens')} theirs {ht.get('tokens')}")
    for key, mabs, mrel, note in rows:
        print(f"  {key:60s} " + (f"max|d| {mabs:.3e}  max rel {mrel:.3e}  {note}" if mabs is not None else note))
    for k in only[:10]:
        print(f"  only on one side: {k}")
    print("compare: within tolerance" if not failures else f"compare: {failures} key(s) exceed")
    return 1 if failures else 0


def compiled(model, args):
    if not getattr(args, 'compile', False):
        return None
    model.static = True                        # masked attention over the whole capacity
    t0 = time.time()
    c = torch.compile(model, dynamic=False)
    print(f"  decode step compiled lazily (torch.compile; first decode pays for it)")
    return c


def build(args, g):
    dtype = compute_dtype(args)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels, Plan(g, kernels).evaluable(g.required_inputs()))
    if r:
        print(f"refused: {len(r)} reason(s)")
        for line in r[:20]:
            print("  " + line)
        return None
    try:
        plan = make_plan(g, kernels, args, dtype)
    except ValueError as e:
        print(f"refused: {e}")
        return None
    i = loader.info(g, args.capacity, dtype, args.device, plan, args.capacity)
    if i['free_bytes'] is not None and i['resident_bytes'] > i['free_bytes']:
        print(f"refused: {loader.gib(i['resident_bytes'])} resident ({i['mode']}), {loader.gib(i['free_bytes'])} free on {args.device}")
        return None
    errors, advisories, stats = loader.verify(g, args.checkpoint)
    if errors:
        print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
        for line in errors[:10]:
            print("  " + line)
        return None
    print(f"  verified {stats['located']} located tensors against {stats['physical']} physical ({stats['unnamed']} unnamed)")
    if len(plan.blocks) > 1:
        return TensorspineModel(g, plan, None, dtype, args.device, source=loader.Source(g, args.checkpoint, args.device).materialise, physical=physical_of(args))
    return TensorspineModel(g, plan, loader.load_parameters(g, args.checkpoint, args.device), dtype, args.device, physical=physical_of(args))


def cmd_chat(args):
    from chat import chat
    g = open_graph(args)
    model = build(args, g)
    if model is None:
        return 1
    dtype = compute_dtype(args)
    return chat(model, g, args.checkpoint, args.capacity, args.device, dtype, args.max_new_tokens,
                args.temperature, args.top_p, args.seed, decode_model=compiled(model, args))


def manifest():
    """The reference generator's capabilities, from its code (generators/CAPABILITIES.md): the
    witness manifest, each entry binding the contract version to its kernel, the tolerances the
    kernel declares and the unit fixtures its cases name (docs/TENSORSPINE-FIXTURE.md)."""
    import datetime
    import subprocess
    import state as state_mod
    import session as session_mod
    from graph import DTYPES
    kernels = registry.load_kernels()
    try:
        version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        version = 'unknown'
    contracts = {}
    for (name, ver), k in sorted(kernels.items()):
        cap = dict(k.CAPABILITIES)
        entry = {'arguments': cap['arguments'], 'states': list(cap.get('states', []))}
        for key in ('excluding', 'transforms', 'notes'):
            if cap.get(key):
                entry[key] = list(cap[key])
        entry['witness'] = {'kernel': os.path.relpath(k.__file__, HERE),
                            'tolerance': {d: dict(t) for d, t in k.TOLERANCE.items()},
                            'fixtures': [f"{name}@{ver}/{case['case']}" for case in getattr(k, 'FIXTURES', [])]}
        contracts[f"{name}@{ver}"] = entry
    return {'schema': 'tensorspine-capabilities/1',
            'role': 'witness',
            'generator': {'name': 'reference', 'version': version, 'generator': 'generators/reference/ref.py capabilities',
                          'generated': datetime.date.today().isoformat()},
            'compute_dtypes': ['f32', 'bf16'],
            'parameter_dtypes': sorted(DTYPES),
            'state_laws': list(state_mod.LAWS), 'access': list(state_mod.ACCESS),
            'sharing': list(session_mod.SHARING), 'partitions': [],
            'domains': {'kinds': ['sequence', 'token', 'position', 'patch'],
                        'transforms': sorted({t for k in kernels.values() for t in k.CAPABILITIES.get('transforms', [])}),
                        'fragmented': True},
            'sessions_per_invocation': 1,
            'locations': list(loader.FORMS),
            'contracts': contracts}


def cmd_capabilities(args):
    m = manifest()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2)
        f.write('\n')
    print(f"{len(m['contracts'])} contracts -> {args.out}")
    if args.check:
        import capabilities
        return capabilities.run(args.out, [])
    return 0


def cmd_witness(args):
    import witness
    return witness.main(args.contract, args.record)


def cmd_verify(args):
    g = open_graph(args)
    errors, advisories, stats = loader.verify(g, args.checkpoint)
    print(f"{g.model}: {stats['located']} located, {stats['physical']} physical, {stats['unnamed']} named by no location")
    for line in errors[:20]:
        print("  [V17] " + line)
    for line in advisories[:10]:
        print("  advice: " + line)
    return 1 if errors else 0


def cmd_run(args):
    g = open_graph(args)
    dtype = compute_dtype(args)
    kernels = registry.load_kernels()
    extra = load_inputs(args.input, args.device, dtype)
    ids = [int(x) for x in args.ids.split(',')] if args.ids else None
    fragments = None                              # the frames each decode step delivers, on a joined token stream
    if args.audio:
        if not args.checkpoint:
            print("refused: --audio takes the feature extractor from the artifact, so it needs --checkpoint DIR")
            return 1
        audio = next((n for n, v in g.input_values.items() if v.get('role') == 'activation.audio_frames'), None)
        if audio is not None and g.feedback_input and g.input_stream[g.feedback_input] == g.input_stream[audio]:
            processor_ids, prefill_inputs, rest = streaming_delivery(args.audio, args.checkpoint, g, args.device, dtype)
            if ids is not None and ids != processor_ids:
                print(f"refused: --ids differ from the processor's streaming prefill of {len(processor_ids)} tokens; the prompt is the processor's")
                return 1
            ids, fragments = processor_ids, rest
            extra.update(prefill_inputs)
        else:
            extra.update(audio_frames(args.audio, args.checkpoint, g, args.device, dtype))
    ids = ids or [1]
    r = registry.refusals(g, kernels, Plan(g, kernels).evaluable(delivery(g, extra)))
    if r:
        print(f"refused: {len(r)} reason(s)")
        for line in r[:20]:
            print("  " + line)
        return 1
    if not args.random and not args.checkpoint:
        print("refused: give --checkpoint DIR (the document locates its weights) or --random")
        return 1
    try:
        plan = make_plan(g, kernels, args, dtype)
    except ValueError as e:
        print(f"refused: {e}")
        return 1
    i = loader.info(g, args.capacity, dtype, args.device, plan, args.capacity)
    if i['free_bytes'] is not None and i['resident_bytes'] > i['free_bytes']:
        print(f"refused: {loader.gib(i['resident_bytes'])} resident ({i['mode']}), {loader.gib(i['free_bytes'])} free on {args.device}")
        return 1
    t0 = time.time()
    blocks = len(plan.blocks) > 1
    if args.random:
        params = loader.random_parameters(g, args.device, args.seed)
        model = TensorspineModel(g, plan, None if blocks else params, dtype, args.device,
                                 source=loader.RandomSource(params).materialise if blocks else None, physical=physical_of(args))
    else:
        errors, advisories, stats = loader.verify(g, args.checkpoint)
        if errors:
            print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
            for line in errors[:10]:
                print("  " + line)
            return 1
        print(f"  verified {stats['located']} located tensors against {stats['physical']} physical "
              f"({stats['unnamed']} unnamed)")
        if blocks:
            model = TensorspineModel(g, plan, None, dtype, args.device, source=loader.Source(g, args.checkpoint, args.device).materialise, physical=physical_of(args))
        else:
            model = TensorspineModel(g, plan, loader.load_parameters(g, args.checkpoint, args.device), dtype, args.device, physical=physical_of(args))
    session = Session(model, args.capacity, args.device, dtype, decode_model=compiled(model, args))
    print(f"{g.model}: {len(plan.steps)} steps, {len(g.tensors)} tensors, {len(session.states)} states, "
          f"{'random parameters' if args.random else 'loaded from ' + args.checkpoint}, "
          f"{loader.gib(i['parameter_bytes'])}, {i['mode']} ({time.time() - t0:.1f}s)")

    dump = {} if args.dump else None
    t0 = time.time()
    if g.generative is None:                      # an encoder: one invocation, the exposed outputs, nothing to decode
        out = session.run({g.token_input: torch.as_tensor(ids, device=args.device, dtype=torch.long), **extra}, dump)
        for oname, t in out.items():
            print(f"  {oname}: {list(t.shape)}, mean norm {float(t.float().norm(dim=-1).mean()):.6f} ({time.time() - t0:.1f}s)")
        if args.dump:
            for oname, o in g.interfaces['outputs'].items():
                dump[f"value/{o['node']}.{o['port']}"] = out[oname].detach().to('cpu', torch.float32).clone()
            write_dump(args.dump, dump, {'model': g.model, 'ids': ids, 'tokens': [], 'capacity': args.capacity,
                                         'compute': str(dtype), 'random_seed': args.seed if args.random else None,
                                         'cuts': [c['cut'] for c in g.layer_cuts()]})
            print(f"  dumped {len(dump)} tensors -> {args.dump}")
        return 0
    out = session.prefill(ids, dump, inputs=extra)
    first_logits = out[g.generative[0]]
    nxt = greedy(out, g)
    print(f"  prefill {len(ids)} elements -> next {nxt} ({time.time() - t0:.1f}s)")
    if dump is not None:                          # the states as prefill left them (R07): a window's valid tail
        for ident, st in session.states.items():
            bufs, length = st.tail() if st.law == 'window' else st.read()
            for c, buf in bufs.items():
                dump[f"state/{ident}/{c}"] = (buf[:length] if length is not None else buf).detach().to('cpu', torch.float32).clone()
    tokenizer = artifact_tokenizer(args.checkpoint) if args.checkpoint else None
    stops = stop_ids(args.checkpoint, tokenizer) if args.stop else set()
    tokens = [nxt]
    per_token = int(g.elements_per[g.feedback_input]) if fragments is not None else 0
    for k in range(args.steps):
        if nxt in stops:
            print(f"  stop: {nxt} is an end-of-text id of the artifact")
            break
        step_inputs = None
        if fragments is not None:                 # a token and its fragment; the end of the audio ends the run
            if fragments.shape[0] < (k + 1) * per_token:
                print(f"  stop: the audio is consumed ({k} steps of {per_token} frames after the prompt)")
                break
            step_inputs = {audio: fragments[k * per_token:(k + 1) * per_token]}
        t0 = time.time()
        out = session.decode(nxt, inputs=step_inputs)
        nxt = greedy(out, g)
        tokens.append(nxt)
        print(f"  decode -> {nxt} ({time.time() - t0:.2f}s)")
    if fragments is not None:                     # the frames the run consumed, as a fixture records them
        extra[audio] = torch.cat([extra[audio], fragments[:(len(tokens) - 1) * per_token]])
    print("tokens:", tokens)
    if tokenizer is not None:
        print("text:", repr(tokenizer.decode(tokens, skip_special_tokens=True)))
    if args.dump:
        for name, t in extra.items():                # the non-token inputs delivered, as a fixture records them
            dump[f"in/{name}"] = t.detach().to('cpu', torch.float32).clone()
        dump['logits/last'] = first_logits[-1].detach().to('cpu', torch.float32).clone()
        dump['logits/argmax'] = first_logits.argmax(-1).detach().cpu().clone()
        write_dump(args.dump, dump, {'model': g.model, 'ids': ids, 'tokens': tokens, 'capacity': args.capacity,
                                     'compute': str(dtype), 'random_seed': args.seed if args.random else None,
                                     'cuts': [c['cut'] for c in g.layer_cuts()]})
        print(f"  dumped {len(dump)} tensors -> {args.dump}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('info'); common(p); p.set_defaults(fn=cmd_info)
    p = sub.add_parser('compare')
    p.add_argument('ours'); p.add_argument('theirs')
    p.add_argument('--atol', type=float, help='override the fixture\'s tolerance'); p.add_argument('--rtol', type=float)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser('capabilities')
    p.add_argument('--out', default=os.path.join(HERE, 'capabilities.json'))
    p.add_argument('--check', action='store_true', help='also validate the manifest against its schema and the catalog')
    p.set_defaults(fn=cmd_capabilities)
    p = sub.add_parser('witness')
    p.add_argument('contract', help='NAME@VERSION, NAME@VERSION/CASE, or all')
    p.add_argument('--record', action='store_true', help='write the fixtures under fixtures/contracts/ (else regenerate and compare)')
    p.set_defaults(fn=cmd_witness)
    p = sub.add_parser('verify'); common(p)
    p.add_argument('--checkpoint', metavar='DIR', required=True)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser('run'); common(p)
    p.add_argument('--random', action='store_true', help='parameters drawn from the D3 shapes, no checkpoint')
    p.add_argument('--checkpoint', metavar='DIR', help='the safetensors checkpoint the document locates its weights in')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ids', help='comma-separated token ids of the prompt')
    p.add_argument('--input', action='append', default=[], metavar='NAME=FILE[:KEY]',
                   help='a safetensors tensor delivered to the public input NAME with the prompt (audio=FIXTURE:in/audio); '
                        'KEY defaults to in/NAME, else the only tensor of the file')
    p.add_argument('--audio', metavar='WAV', help='a mono 16-bit WAV at the extractor\'s rate, turned into frames by the checkpoint\'s own '
                                                  'feature extractor and delivered to the document\'s audio input with the prompt (needs --checkpoint); '
                                                  'on a streaming document the processor\'s prompt and delay come with it, and the frames follow the tokens')
    p.add_argument('--stop', action='store_true', help='end decoding at an end-of-text id of the artifact (generation_config.json, the tokenizer), before --steps')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--dump', help='write the values at every layer cut and the states to this safetensors file')
    p.add_argument('--compile', action='store_true', help='torch.compile the decode step (prefill stays eager)')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('chat'); common(p)
    p.add_argument('--checkpoint', metavar='DIR', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compile', action='store_true')
    p.set_defaults(fn=cmd_chat)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())

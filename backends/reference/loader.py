"""verify / refuse / info / load (R05, R06). M0: random parameters from D3 shapes
and the feasibility report; loading by location follows the location plan."""
import os
import sys
import torch

from graph import DTYPES, ROOT

sys.path.insert(0, os.path.join(ROOT, 'tools'))
import artifact                 # noqa: E402  the language's own header reader and V17 check

WIDTH = {'bf16': 2, 'f16': 2, 'f32': 4, 'f8e4m3': 1, 'fp4': 0.5}


def random_parameters(graph, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for ident, t in graph.tensors.items():
        if t['dtype'] not in DTYPES:
            raise ValueError(f"{ident}: dtype {t['dtype']} not representable here")
        shape = [a['extent'] for a in t['shape']]
        dtype = getattr(torch, DTYPES[t['dtype']])
        if t['role'] == 'norm.scale':
            x = torch.ones(shape)
        else:
            x = torch.randn(shape, generator=g) * 0.02
        out[ident] = x.to(dtype).to(device)
    return out


def state_bytes(graph, capacity, compute_dtype):
    width = torch.tensor([], dtype=compute_dtype).element_size()
    total = 0
    for s in graph.states.values():
        per = sum(p['elements'] for p in s['payload']) * width
        if s['law'] == 'append':
            total += per * capacity
        elif s['law'] == 'window':
            total += per * (s['span'] or 0)
        else:
            total += per
    return total


def free_memory(device):
    if str(device).startswith('cuda'):
        free, _ = torch.cuda.mem_get_info(device)
        return free
    try:
        with open('/proc/meminfo', encoding='utf-8') as f:      # MemAvailable counts reclaimable cache
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (ValueError, OSError):
        return None


def largest_temporary(graph, compute_dtype):
    """The per-operation upcast of the largest weight: the biggest anonymous allocation
    of a run whose parameters stay at their declared dtype. An embedding table is never
    upcast whole — it is gathered, or projected in chunks bounded by UPCAST_CHUNK_BYTES."""
    from kernels._common import UPCAST_CHUNK_BYTES
    width = torch.tensor([], dtype=compute_dtype).element_size()
    biggest = 0
    for t in graph.tensors.values():
        n = (t['elements'] or 0) * width
        if t['role'] == 'embedding.table':
            n = min(n, UPCAST_CHUNK_BYTES)
        biggest = max(biggest, n)
    return biggest


def info(graph, capacity, compute_dtype, device, plan=None, elements=None):
    """Declared bytes (D3, D4) and the bytes a run holds: on CPU the weights are
    memory-mapped and the kernel pages them, so what is allocated is the states, the
    largest per-operation temporary and — under blocks — the largest block; on CUDA
    the weights live on the device."""
    p = graph.d3_totals['bytes']
    s = state_bytes(graph, capacity, compute_dtype)
    temp = largest_temporary(graph, compute_dtype)
    cuda = str(device).startswith('cuda')
    elements = elements or capacity
    if plan is not None and len(plan.blocks) > 1:
        largest = max(b.bytes + b.payload_bytes_per_element * elements for b in plan.blocks)
        resident = largest + s + temp
        mode = f"{len(plan.blocks)} blocks, the largest {largest / 2**30:.2f} GiB with its payload"
    elif cuda:
        resident = p + s + temp
        mode = "one block on the device"
    else:
        resident = s + temp
        mode = "one block, weights memory-mapped and paged by the kernel"
    return {'parameter_bytes': p, 'state_bytes': s, 'temporary_bytes': temp, 'resident_bytes': resident,
            'mode': mode, 'traffic_bytes': plan.traffic_bytes() if plan is not None else 0,
            'append_bytes_per_position': graph.d4_totals.get('append_bytes_per_cached_position', 0),
            'free_bytes': free_memory(device)}


def gib(n):
    return f"{n / 2**30:.2f} GiB"


def verify(graph, checkpoint):
    """V17 against the checkpoint headers, through the language's own check."""
    headers = artifact.read_headers(checkpoint)
    return artifact.check(graph.doc['d3'], headers)


FORMS = ('tensor', 'stack', 'concat', 'slice')       # the location forms assemble() handles


class Source:
    """One tensor per D3 identity, assembled from its evaluated location: read, stack,
    concatenate, slice, unit axes dropped; shard by shard, straight to the device (R05).
    `fetch(identity)` returns a memory-mapped view where the checkpoint allows it;
    `materialise(identity)` returns an owned copy — what a block holds under `--max-ram`."""

    def __init__(self, graph, checkpoint, device):
        from safetensors import safe_open
        self.graph, self.checkpoint, self.device = graph, checkpoint, device
        self.headers = artifact.read_headers(checkpoint)
        self.root = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
        self.handles = {}
        self._safe_open = safe_open

    def get(self, name):
        h = self.headers[name]
        f = self.handles.get(h['file'])
        if f is None:
            path = os.path.join(self.root, h['file']) if os.path.isdir(self.checkpoint) else self.checkpoint
            f = self.handles[h['file']] = self._safe_open(path, framework='pt', device=str(self.device))
        return f.get_tensor(name)

    def assemble(self, ev, logical):
        get, headers = self.get, self.headers
        if 'tensor' in ev:
            t = get(ev['tensor'])
            return t.reshape(logical) if list(t.shape) != logical else t
        if 'stack' in ev:
            dim = ev['stack']['dim']
            inner = logical[:dim] + logical[dim + 1:]
            return torch.stack([self.assemble(p, inner) for p in ev['stack']['parts']], dim=dim)
        if 'concat' in ev:
            dim = ev['concat']['dim']
            parts = []
            for p in ev['concat']['parts']:
                names, _ = artifact._names(p)
                extent = artifact.squeeze(headers[names[0]]['shape'])[[i for i, d in enumerate(logical) if d != 1].index(dim)]
                own = list(logical)
                own[dim] = extent
                parts.append(self.assemble(p, own))
            return torch.cat(parts, dim=dim)
        if 'slice' in ev:
            sl = ev['slice']
            t = get(sl['tensor'])
            t = t.reshape(artifact.squeeze(list(t.shape)))
            pos = [i for i, d in enumerate(logical) if d != 1].index(sl['dim'])
            return t.narrow(pos, sl['offset'], sl['extent']).reshape(logical)
        raise ValueError(f"unknown location form {list(ev)}")

    def fetch(self, ident):
        t = self.graph.tensors[ident]
        ev = t.get('location')
        if ev is None:
            raise ValueError(f"{ident}: no location — the document does not locate its weights")
        logical = [a['extent'] for a in t['shape']]
        return self.assemble(ev, logical).to(getattr(torch, DTYPES[t['dtype']]))

    def materialise(self, ident):
        return self.fetch(ident).clone()


class RandomSource:
    """The random parameters, materialised per identity like a checkpoint's."""

    def __init__(self, params):
        self.params = params

    def materialise(self, ident):
        return self.params[ident].clone()


def load_parameters(graph, checkpoint, device):
    """Every identity at once, memory-mapped where the checkpoint allows it (one block)."""
    src = Source(graph, checkpoint, device)
    return {ident: src.fetch(ident) for ident in graph.tensors}



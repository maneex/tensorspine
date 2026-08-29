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


def info(graph, capacity, compute_dtype, device):
    p = graph.d3_totals['bytes']
    s = state_bytes(graph, capacity, compute_dtype)
    return {'parameter_bytes': p, 'state_bytes': s, 'total_bytes': p + s,
            'append_bytes_per_position': graph.d4_totals.get('append_bytes_per_cached_position', 0),
            'free_bytes': free_memory(device)}


def gib(n):
    return f"{n / 2**30:.2f} GiB"


def verify(graph, checkpoint):
    """V17 against the checkpoint headers, through the language's own check."""
    headers = artifact.read_headers(checkpoint)
    return artifact.check(graph.doc['d3'], headers)


def load_parameters(graph, checkpoint, device):
    """One tensor per D3 identity, assembled from its evaluated location: read, stack,
    concatenate, slice, unit axes dropped; shard by shard, straight to the device (R05)."""
    from safetensors import safe_open
    headers = artifact.read_headers(checkpoint)
    root = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
    handles = {}

    def get(name):
        h = headers[name]
        f = handles.get(h['file'])
        if f is None:
            path = os.path.join(root, h['file']) if os.path.isdir(checkpoint) else checkpoint
            f = handles[h['file']] = safe_open(path, framework='pt', device=str(device))
        return f.get_tensor(name)

    def fetch(ev, logical):
        if 'tensor' in ev:
            t = get(ev['tensor'])
            return t.reshape(logical) if list(t.shape) != logical else t
        if 'stack' in ev:
            dim = ev['stack']['dim']
            inner = logical[:dim] + logical[dim + 1:]
            return torch.stack([fetch(p, inner) for p in ev['stack']['parts']], dim=dim)
        if 'concat' in ev:
            dim = ev['concat']['dim']
            parts = []
            for p in ev['concat']['parts']:
                names, _ = artifact._names(p)
                extent = artifact.squeeze(headers[names[0]]['shape'])[[i for i, d in enumerate(logical) if d != 1].index(dim)]
                own = list(logical)
                own[dim] = extent
                parts.append(fetch(p, own))
            return torch.cat(parts, dim=dim)
        if 'slice' in ev:
            s = ev['slice']
            t = get(s['tensor'])
            squeezed = artifact.squeeze(list(t.shape))
            t = t.reshape(squeezed)
            pos = [i for i, d in enumerate(logical) if d != 1].index(s['dim'])
            return t.narrow(pos, s['offset'], s['extent']).reshape(logical)
        raise ValueError(f"unknown location form {list(ev)}")

    out = {}
    for ident, t in graph.tensors.items():
        ev = t.get('location')
        if ev is None:
            raise ValueError(f"{ident}: no location — the document does not locate its weights")
        logical = [a['extent'] for a in t['shape']]
        out[ident] = fetch(ev, logical).to(getattr(torch, DTYPES[t['dtype']]))
    for f in handles.values():
        del f
    return out

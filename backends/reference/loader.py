"""verify / refuse / info / load (R05, R06). M0: random parameters from D3 shapes
and the feasibility report; loading by location follows the location plan."""
import os
import torch

from graph import DTYPES

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

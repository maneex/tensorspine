"""Parameters from the checkpoint by their D3 locations — `tensor`, `stack`, `concat`, `slice`,
unit axes dropped — read through the language's own header reader and V17 check
(`tools/artifact.py`), as raw arrays at the D3 dtype: what an initializer holds."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if os.path.join(ROOT, 'tools') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
import artifact                 # noqa: E402

FORMS = ('tensor', 'stack', 'concat', 'slice')
TORCH = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}


def verify(graph, checkpoint):
    """V17 against the checkpoint's headers: (errors, advisories, stats); nothing is read."""
    return artifact.check(graph.doc['d3'], artifact.read_headers(checkpoint))


class Source:
    def __init__(self, graph, checkpoint):
        from safetensors import safe_open
        self.graph, self.checkpoint = graph, checkpoint
        self.headers = artifact.read_headers(checkpoint)
        self.root = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
        self.handles = {}
        self._safe_open = safe_open

    def get(self, name):
        h = self.headers[name]
        f = self.handles.get(h['file'])
        if f is None:
            path = os.path.join(self.root, h['file']) if os.path.isdir(self.checkpoint) else self.checkpoint
            f = self.handles[h['file']] = self._safe_open(path, framework='pt')
        return f.get_tensor(name)

    def assemble(self, ev, logical):
        if 'tensor' in ev:
            t = self.get(ev['tensor'])
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
                extent = artifact.squeeze(self.headers[names[0]]['shape'])[[i for i, d in enumerate(logical) if d != 1].index(dim)]
                own = list(logical)
                own[dim] = extent
                parts.append(self.assemble(p, own))
            return torch.cat(parts, dim=dim)
        if 'slice' in ev:
            sl = ev['slice']
            t = self.get(sl['tensor']).reshape(artifact.squeeze(list(self.get(sl['tensor']).shape)))
            pos = [i for i, d in enumerate(logical) if d != 1].index(sl['dim'])
            return t.narrow(pos, sl['offset'], sl['extent']).reshape(logical)
        raise ValueError(f"unknown location form {list(ev)}")

    def fetch(self, ident):
        """The identity's tensor at its D3 dtype (torch), assembled from its location."""
        t = self.graph.tensors[ident]
        ev = t.get('location')
        if ev is None:
            raise ValueError(f"{ident}: no location — the document does not locate its weights")
        logical = [a['extent'] for a in t['shape']]
        return self.assemble(ev, logical).to(TORCH[t['dtype']]).contiguous()


class RandomSource:
    """Parameters drawn from the D3 shapes, for a run without a checkpoint (smoke tests)."""

    def __init__(self, graph, seed=0):
        self.graph, self.gen = graph, torch.Generator().manual_seed(seed)

    def fetch(self, ident):
        t = self.graph.tensors[ident]
        shape = [a['extent'] for a in t['shape']]
        x = torch.ones(shape) if t['role'] == 'norm.scale' else torch.randn(shape, generator=self.gen) * 0.02
        return x.to(TORCH[t['dtype']])


def raw(t):
    """A torch tensor as the bytes and numpy view an ONNX initializer takes: bf16 as uint16 words."""
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()

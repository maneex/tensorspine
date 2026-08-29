"""State instances: one per D4 identity for one (session, branch), each law
implemented once under a declared capacity (R04, R07).

    append   buffer [capacity, *shape], a cursor; read() -> (buffers, length)
    window   buffer [span, *shape] as a ring; read() -> chronological, zero-padded
    fixed    one tensor per component; read()/write()

Kernels read a whole buffer plus a length and mask beyond it — the same code
eager or compiled.
"""
import torch


class Refusal(Exception):
    pass


class StateInstance:
    def __init__(self, entry, capacity, device, dtype):
        self.identity = entry['identity']
        self.law = entry['law']
        self.access = entry['access']
        self.span = entry.get('span')
        self.capacity = capacity if self.law == 'append' else (self.span if self.law == 'window' else None)
        if self.law not in ('append', 'window', 'fixed'):
            raise Refusal(f"{self.identity}: unknown state law '{self.law}'")
        self.components = {}
        for p in entry['payload']:
            shape = [a['extent'] for a in p['shape']]
            full = ([self.capacity] + shape) if self.capacity is not None else shape
            self.components[p['component']] = torch.zeros(full, device=device, dtype=dtype)
        self.length = 0

    def read(self):
        if self.law == 'fixed':
            return self.components, None
        if self.law == 'append':
            return self.components, self.length
        n = min(self.length, self.span)
        out = {}
        for c, buf in self.components.items():
            if self.length <= self.span:
                chrono = buf[:n]
            else:
                cursor = self.length % self.span
                chrono = torch.cat([buf[cursor:], buf[:cursor]], dim=0)
            if n < self.span:
                pad = torch.zeros((self.span - n,) + tuple(buf.shape[1:]), device=buf.device, dtype=buf.dtype)
                chrono = torch.cat([pad, chrono], dim=0)
            out[c] = chrono
        return out, n

    def append(self, values):
        m = next(iter(values.values())).shape[0]
        if self.law == 'append':
            if self.length + m > self.capacity:
                raise Refusal(f"{self.identity}: {self.length + m} positions exceed the capacity {self.capacity}")
            for c, x in values.items():
                self.components[c][self.length:self.length + m] = x.to(self.components[c].dtype)
        elif self.law == 'window':
            idx = (self.length + torch.arange(m, device=next(iter(values.values())).device)) % self.span
            for c, x in values.items():
                if m >= self.span:
                    self.components[c].copy_(x[-self.span:].to(self.components[c].dtype))
                else:
                    self.components[c][idx] = x.to(self.components[c].dtype)
            if m >= self.span:
                self.length = 0   # the buffer is now chronological from index 0
        else:
            raise Refusal(f"{self.identity}: append on a fixed state")
        self.length += m

    def write(self, values):
        if self.law != 'fixed':
            raise Refusal(f"{self.identity}: write on a {self.law} state")
        for c, x in values.items():
            self.components[c] = x.to(self.components[c].dtype)
        self.length = 1

    def reset(self):
        for c in self.components:
            self.components[c].zero_()
        self.length = 0


def allocate(graph, capacity, device, dtype):
    return {ident: StateInstance(entry, capacity, device, dtype) for ident, entry in graph.states.items()}


def bytes_per_position(graph):
    return graph.d4_totals.get('append_bytes_per_cached_position', 0)

"""{name, version} -> kernel module, and the refusal report (R02).

A kernel module declares `PRIMITIVE = (name, version)`, `supports(arguments)`
returning the 'name=value' pairs it does not implement (empty when it runs),
and `run(ctx, arguments, inputs, params, states)`.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load_kernels():
    kernels = {}
    for f in sorted(os.listdir(os.path.join(HERE, 'kernels'))):
        if f.endswith('.py') and not f.startswith('_'):
            m = importlib.import_module(f"kernels.{f[:-3]}")
            kernels[tuple(m.PRIMITIVE)] = m
    return kernels


def refusals(graph, kernels, nodes=None):
    """Every reason the model cannot run, collected before any weight is read; over the
    instances an invocation evaluates when `nodes` is given (§7)."""
    out = []
    for node, entry in graph.nodes.items():
        if nodes is not None and node not in nodes:
            continue
        key = (entry['primitive']['name'], entry['primitive']['version'])
        k = kernels.get(key)
        if k is None:
            out.append(f"{node}: no kernel for {key[0]}@{key[1]}")
            continue
        for r in k.supports(entry['arguments']):
            out.append(f"{node}: {key[0]}@{key[1]} does not implement {r}")
    return out

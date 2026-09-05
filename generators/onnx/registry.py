"""{name, version} -> emitter module (the ONNX generator's primitives, one file each)."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load_primitives():
    out = {}
    for f in sorted(os.listdir(os.path.join(HERE, 'primitives'))):
        if f.endswith('.py') and not f.startswith('_'):
            m = importlib.import_module(f"primitives.{f[:-3]}")
            out[tuple(m.CONTRACT)] = m
    return out

"""{name, version} -> emitter module, per target: `primitives/` holds the portable emitters (standard
operators, every target's fallback), `primitives/<target>/` a target's own — onnxruntime's fused
forms — overriding the portable one primitive by primitive."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _modules(package, directory):
    out = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith('.py') and not f.startswith('_'):
            m = importlib.import_module(f"{package}.{f[:-3]}")
            out[tuple(m.PRIMITIVE)] = m
    return out


def targets():
    """The targets the tree knows: `onnx` (the portable forms) and one per directory of fused forms."""
    base = os.path.join(HERE, 'primitives')
    return ['onnx'] + sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith('_'))


def load_primitives(target='onnx'):
    """The emitters of a target: the portable set, the target's own laid over it."""
    base = os.path.join(HERE, 'primitives')
    prims = _modules('primitives', base)
    if target != 'onnx':
        if target not in targets():
            raise ValueError(f"target {target!r}: one of {targets()}")
        prims.update(_modules(f"primitives.{target}", os.path.join(base, target)))
    return prims


def load_all():
    return {t: load_primitives(t) for t in targets()}

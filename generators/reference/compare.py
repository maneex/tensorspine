"""Fixtures and dumps. A fixture is a safetensors file whose metadata is a document on the
language's fixture schema (`schemas/tensorspine-fixture.schema.json`,
`docs/TENSORSPINE-FIXTURE.md`): what a conformer is checked against, whether it was produced by
a contract version's witness (`unit`) or dumped from the delivery implementation of a whole model
(`integration`). A dump is what a run of this generator leaves behind for a comparison: the same
container, a header of plain facts, no schema. The comparison is one tolerance test per key."""
import json
import os
import struct
import sys

from safetensors.torch import save_file, load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCHEMAS = os.path.join(ROOT, 'schemas')
if os.path.join(ROOT, 'tools') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'tools'))

SCHEMA_TAG = 'tensorspine-fixture/1'


def write_dump(path, tensors, header):
    save_file({k: v.detach().contiguous().clone() for k, v in tensors.items()}, path,
              metadata={k: json.dumps(v) for k, v in header.items()})


def read_dump(path):
    return load_file(path), read_metadata(path)


def read_metadata(path):
    """The JSON-valued metadata of a safetensors file, from its header alone: no tensor is read."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    out = {}
    for k, v in (header.get('__metadata__') or {}).items():
        try:
            out[k] = json.loads(v)
        except (TypeError, json.JSONDecodeError):
            out[k] = v
    return out


def check_metadata(metadata):
    """Errors of a fixture's metadata against the fixture schema; empty when it conforms."""
    import schema as schema_mod
    path = schema_mod.locate(SCHEMAS, 'fixture')
    if path is None:
        return [f"no schema with $id ending in /fixture.json under {SCHEMAS}/"]
    errors = schema_mod.check_document(path, metadata, schema_mod.registry(SCHEMAS))
    return [schema_mod.format_error(e) for e in schema_mod.deepest(errors)]


def write_fixture(path, tensors, metadata):
    """A fixture is validated before it is written: a file off the schema is not a fixture."""
    problems = check_metadata(metadata)
    if problems:
        raise ValueError(f"{path}: the fixture metadata is off the schema: {problems[0]}")
    write_dump(path, tensors, metadata)


def read_fixture(path):
    """(tensors, metadata) of a fixture, refused when its metadata is off the schema."""
    metadata = read_metadata(path)
    problems = check_metadata(metadata)
    if problems:
        raise ValueError(f"{path}: not a fixture on {SCHEMA_TAG}: {problems[0]}")
    return load_file(path), metadata


def tolerance_for(metadata, compute):
    """(atol, rtol) a conformer computing in `compute` must meet against this fixture."""
    entry = metadata['tolerance'].get(compute)
    if entry is None:
        raise KeyError(f"the fixture states no tolerance for compute dtype {compute}: "
                       f"{sorted(metadata['tolerance'])}")
    return entry['atol'], entry['rtol']


def compare(ours, theirs, atol=1e-3, rtol=1e-2):
    """One line per key present in both: max |a-b|, max |a-b| / (|b| + 1e-6), the worst
    element. Returns (rows, failures, only-on-one-side)."""
    import torch
    rows, failures = [], 0
    for key in sorted(set(ours) & set(theirs)):
        a, b = ours[key].to(torch.float32), theirs[key].to(torch.float32)
        if a.shape != b.shape:
            rows.append((key, None, None, f"shape {list(a.shape)} vs {list(b.shape)}"))
            failures += 1
            continue
        if a.dtype in (torch.int64, torch.int32) or key.endswith('argmax'):
            same = int((a == b).sum())
            rows.append((key, None, None, f"{same}/{a.numel()} equal"))
            failures += same != a.numel()
            continue
        d = (a - b).abs()
        rel = d / (b.abs() + 1e-6)
        worst = int(d.argmax())
        ok = bool((d <= atol + rtol * b.abs()).all())
        rows.append((key, float(d.max()), float(rel.max()), f"worst at {list(torch.unravel_index(torch.tensor(worst), a.shape))}" + ("" if ok else "  EXCEEDS")))
        failures += not ok
    only = sorted(set(ours) ^ set(theirs))
    return rows, failures, only

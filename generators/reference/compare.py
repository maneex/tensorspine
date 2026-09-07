"""Fixtures and dumps. A fixture is a safetensors file whose metadata is a document on the
language's fixture schema (`schemas/tensorspine-fixture.schema.json`,
`docs/TENSORSPINE-FIXTURE.md`): what a conformer is checked against, whether it was produced by
a primitive version's witness (`unit`) or dumped from the delivery implementation of a whole model
(`integration`). A dump is what a run of this generator leaves behind for a comparison: the same
container, a header of plain facts, no schema. The comparison is a verdict: one test per key
present on both sides — exact for integers and booleans, a tolerance for floating values — and a
failure for every required key absent on either side, or when nothing was compared at all."""
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


INTEGER = ('torch.bool', 'torch.uint8', 'torch.int8', 'torch.int16', 'torch.int32', 'torch.int64')


class Verdict:
    """What a comparison found (docs/TENSORSPINE-FIXTURE.md §4): `rows`, one per key compared —
    `(key, max |a−b|, max relative, note)`, the numbers None when the key failed before any
    difference was measured (a shape or a dtype); `failures`, how many rows failed; `missing`, the
    required keys absent on either side; `unexpected`, the keys of ours the other side lacks
    (reported, never failed); `compared`, how many keys were. `ok` is the verdict: no failure,
    nothing required missing, and at least one key compared — an empty comparison proves nothing."""

    def __init__(self, rows, failures, missing, unexpected):
        self.rows, self.failures, self.missing, self.unexpected = rows, failures, missing, unexpected
        self.compared = len(rows)

    @property
    def ok(self):
        return not self.failures and not self.missing and self.compared > 0

    @property
    def worst(self):
        """The largest absolute difference measured, 0.0 when none was."""
        return max((r[1] for r in self.rows if r[1] is not None), default=0.0)

    @property
    def bad(self):
        """The keys that failed: exceeding the tolerance, unequal, or refused on shape or dtype."""
        return [r[0] for r in self.rows if 'EXCEEDS' in r[3] or 'unequal' in r[3] or r[1] is None]

    def summary(self):
        if self.compared == 0:
            return "0 keys compared"
        parts = []
        if self.failures:
            parts.append(f"{self.failures} key(s) exceed")
        if self.missing:
            parts.append(f"{len(self.missing)} required key(s) missing")
        return ", ".join(parts) if parts else "within tolerance"

    def detail(self, n=3):
        """The first failing and missing keys, for a test's one-line explanation."""
        return (f"EXCEEDS: {self.bad[:n]}" if self.bad else '') + (f"  MISSING: {self.missing[:n]}" if self.missing else '')


def compare(ours, theirs, atol=1e-3, rtol=1e-2, required=None):
    """The verdict of `ours` against `theirs`, the recorded side. Every key present on both is
    tested: the dtypes are read before any cast — an integer or boolean tensor on either side
    must be one on both, and the two compare exactly in int64 whatever their widths, a
    disagreement (int64 against float32) being a failure; floating tensors compare in float32
    element by element, |a − b| ≤ atol + rtol·|b|. `required` names the keys the verdict needs on
    both sides; by default every key of `theirs` not under `in/` or `param/` — the inputs a fixture
    records and the parameters a unit fixture is the checkpoint of, which a dump does not repeat."""
    import torch
    rows, failures = [], 0
    for key in sorted(set(ours) & set(theirs)):
        a, b = ours[key], theirs[key]
        if a.shape != b.shape:
            rows.append((key, None, None, f"shape {list(a.shape)} vs {list(b.shape)}"))
            failures += 1
            continue
        integer = (str(a.dtype) in INTEGER, str(b.dtype) in INTEGER)
        if any(integer):
            if not all(integer):
                rows.append((key, None, None, f"dtype {str(a.dtype)[6:]} vs {str(b.dtype)[6:]}"))
                failures += 1
                continue
            a, b = a.to(torch.int64), b.to(torch.int64)
            same = int((a == b).sum())
            rows.append((key, None, None, f"{same}/{a.numel()} equal" + ("" if same == a.numel() else "  unequal")))
            failures += same != a.numel()
            continue
        a, b = a.to(torch.float32), b.to(torch.float32)
        d = (a - b).abs()
        rel = d / (b.abs() + 1e-6)
        worst = int(d.argmax()) if d.numel() else 0
        ok = bool((d <= atol + rtol * b.abs()).all())
        where = f"worst at {list(torch.unravel_index(torch.tensor(worst), a.shape))}" if d.numel() else "empty"
        rows.append((key, float(d.max()) if d.numel() else 0.0, float(rel.max()) if d.numel() else 0.0, where + ("" if ok else "  EXCEEDS")))
        failures += not ok
    if required is None:
        required = {k for k in theirs if not k.startswith(('in/', 'param/'))}
    missing = sorted(k for k in required if k not in ours or k not in theirs)
    unexpected = sorted(set(ours) - set(theirs))
    return Verdict(rows, failures, missing, unexpected)

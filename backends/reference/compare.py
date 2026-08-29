"""Dumps at legal cuts and their comparison (R07). M0: the writer and reader."""
import json
from safetensors.torch import save_file, load_file


def write_dump(path, tensors, header):
    save_file({k: v.detach().contiguous().clone() for k, v in tensors.items()}, path,
              metadata={k: json.dumps(v) for k, v in header.items()})


def read_dump(path):
    from safetensors import safe_open
    with safe_open(path, framework='pt') as f:
        meta = {k: json.loads(v) for k, v in (f.metadata() or {}).items()}
    return load_file(path), meta


def compare(ours, theirs, atol=1e-3, rtol=1e-2):
    """One line per key present in both: max |a-b|, max |a-b| / (|b| + 1e-6), the worst
    element. Returns (rows, failures)."""
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

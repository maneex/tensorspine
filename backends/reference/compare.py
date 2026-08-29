"""Dumps at legal cuts and their comparison (R07). M0: the writer and reader."""
import json
from safetensors.torch import save_file, load_file


def write_dump(path, tensors, header):
    save_file({k: v.contiguous() for k, v in tensors.items()}, path,
              metadata={k: json.dumps(v) for k, v in header.items()})


def read_dump(path):
    from safetensors import safe_open
    with safe_open(path, framework='pt') as f:
        meta = {k: json.loads(v) for k, v in (f.metadata() or {}).items()}
    return load_file(path), meta

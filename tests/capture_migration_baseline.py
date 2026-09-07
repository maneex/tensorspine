#!/usr/bin/env python3
"""Reproducibly capture the migration's immutable reference without requiring Git in CI.

This is a historical fixture producer, not a numerical fixture regenerator.
"""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = '9da0897'


def capture():
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', REVISION], cwd=ROOT, text=True).splitlines()
    documents, fixtures = {}, {}
    for path in files:
        if path.endswith('.json') and (path.startswith(('data/', 'schemas/')) or path.endswith('/capabilities.json')):
            documents[path] = json.loads(subprocess.check_output(['git', 'show', f'{REVISION}:{path}'], cwd=ROOT))
        elif path.endswith('.safetensors'):
            data = subprocess.check_output(['git', 'show', f'{REVISION}:{path}'], cwd=ROOT)
            size = struct.unpack('<Q', data[:8])[0]
            header = json.loads(data[8:8 + size])
            fixtures[path] = {'header': header, 'payload_sha256': hashlib.sha256(data[8 + size:]).hexdigest()}
    encoded = json.dumps({'revision': REVISION, 'documents': documents, 'fixtures': fixtures},
                         sort_keys=True, separators=(',', ':')).encode()
    (ROOT / 'tests/migration-baseline.json.gz').write_bytes(gzip.compress(encoded, mtime=0))
    print(f'captured {len(documents)} legacy documents and {len(fixtures)} fixture headers and payload hashes')


if __name__ == '__main__':
    capture()

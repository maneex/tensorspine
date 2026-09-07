#!/usr/bin/env python3
"""Migration boundary and semantic equivalence against the immutable Git baseline."""
import hashlib
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import migrate

BASELINE = json.loads(gzip.decompress((ROOT / 'tests/migration-baseline.json.gz').read_bytes()))


def previous(path):
    return json.dumps(BASELINE['documents'][str(path)]).encode()


def tensors(data):
    size = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8 + size])
    header.pop('__metadata__')
    return header, data[8 + size:]


class Migration(unittest.TestCase):
    def test_revisions_and_collisions(self):
        for old, new in migrate.VERSIONS.items():
            self.assertEqual(migrate.convert({'schema': old}), {'schema': new})
            with self.assertRaises(ValueError):
                migrate.convert({'schema': new})
        for revision in ['tensorspine/1.0', 'tensorspine/99', None]:
            with self.assertRaises(ValueError):
                migrate.convert({'schema': revision})
        for fields in [{'occurrences': {}, 'instances': {}}, {'instances': {}},
                       {'contract': {}, 'primitive': {}}, {'law': 'fixed', 'evolution': 'append'},
                       {'catalog': [], 'primitive_libraries': []}]:
            with self.assertRaises(ValueError):
                migrate.convert({'schema': 'tensorspine/2.0', **fields})
        with self.assertRaises(ValueError):
            json.loads('{"schema": "a", "schema": "b"}', object_pairs_hook=migrate.pairs)

    def test_authored_names(self):
        old = {'schema': 'tensorspine/2.0', 'occurrences': {'contract': {
            'contract': {'name': 'contract', 'version': '1.0.0'},
            'arguments': {'law': {'record': {'contract': {'literal': 'occurrence'}}}}}},
            'quantities': {'law': {'type': {'kind': 'cardinality'}}}}
        result = migrate.convert(old)
        self.assertEqual(result['instances']['contract']['primitive']['name'], 'contract')
        self.assertEqual(result['instances']['contract']['arguments'], old['occurrences']['contract']['arguments'])
        self.assertEqual(result['quantities'], old['quantities'])

    def test_corpus_conversion(self):
        for path in sorted((ROOT / 'data/models').rglob('*.json')):
            with self.subTest(path=path.name):
                old = json.loads(previous(str(path.relative_to(ROOT)).replace('/2.0.0.json', '/1.0.0.json')))
                self.assertEqual(migrate.convert(old), json.loads(path.read_text()))

    def test_fixture_payloads(self):
        paths = sorted((ROOT / 'generators/reference/fixtures').rglob('*.safetensors'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                old_path = str(path.relative_to(ROOT)).replace('/primitives/', '/contracts/').replace('@2.0.0/', '@1.0.0/')
                current = path.read_bytes()
                record = BASELINE['fixtures'][old_path]
                header = dict(record['header'])
                header.pop('__metadata__')
                current_header, payload = tensors(current)
                self.assertEqual(header, current_header)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record['payload_sha256'])
                encoded = json.dumps(record['header'], separators=(',', ':')).encode()
                old = struct.pack('<Q', len(encoded)) + encoded + payload
                converted = migrate.fixture_bytes(old)
                converted_size = struct.unpack('<Q', converted[:8])[0]
                current_size = struct.unpack('<Q', current[:8])[0]
                self.assertEqual(json.loads(converted[8:8 + converted_size]), json.loads(current[8:8 + current_size]))

    def test_cli_relative_bases_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old.json'
            old.write_bytes(previous('data/models/llama3-8b.json'))
            new = root / 'new.json'
            command = [sys.executable, str(ROOT / 'tools/migrate.py'), str(old), '-o', str(new),
                       '--library-base', str(ROOT / 'data/primitive-library')]
            subprocess.run(command, check=True, capture_output=True)
            result = json.loads(new.read_text())
            self.assertEqual((new.parent / result['primitive_libraries'][0]['base']).resolve(),
                             ROOT / 'data/primitive-library')
            before = new.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(new.read_bytes(), before)

    def test_full_derived_facts(self):
        import derive
        import primitive_library
        expected = json.loads((ROOT / 'tests/migration-derived-sha256.json').read_text())
        library = primitive_library.load(str(ROOT / 'data/primitive-library'))
        for name, digest in expected.items():
            with self.subTest(model=name):
                doc = derive.products(str(ROOT / 'data/models' / (name + '.json')), library)
                encoded = json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
                import capabilities
                for generator in ('reference', 'onnx', 'zml'):
                    path = f'generators/{generator}/capabilities.json'
                    before = migrate.convert(BASELINE['documents'][path])
                    after = json.loads((ROOT / path).read_text())
                    self.assertEqual(capabilities.can_run(before, doc, library),
                                     capabilities.can_run(after, doc, library))


if __name__ == '__main__':
    unittest.main()

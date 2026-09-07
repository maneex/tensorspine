#!/usr/bin/env python3
"""Explicit conversion of the earlier untagged TensorSpine field layout.

This module is the compatibility boundary; current readers never alias old keys.
"""
import argparse
import json
from pathlib import Path
import struct

VERSIONS = {
    'tensorspine/2.0': 'tensorspine/2.0',
    'tensorspine-catalog-unit/2.0': 'tensorspine-primitive-library-unit/2.0',
    'tensorspine-derived/2.1': 'tensorspine-derived/2.1',
    'tensorspine-capabilities/1': 'tensorspine-capabilities/1',
    'tensorspine-fixture/1': 'tensorspine-fixture/1',
    'tensorspine-primitive-request/1': 'tensorspine-primitive-request/1',
    'tensorspine-primitive-response/1': 'tensorspine-primitive-response/1',
}
KEYS = {
    'contract': 'primitive', 'contracts': 'primitives',
    'occurrence': 'instance', 'occurrences': 'instances',
    'law': 'evolution', 'state_laws': 'state_evolutions', 'by_law': 'by_evolution',
    'cut': 'graph_split', 'cuts': 'graph_splits', 'partitions': 'partition_options',
    'per_contract': 'per_primitive',
}
# These keys introduce maps keyed by authored identities, not format field names.
NAMED_MAPS = {
    'occurrences', 'instances', 'compositions', 'quantities', 'constants', 'nodes',
    'contracts', 'primitives', 'axes', 'precision', 'inputs', 'outputs',
    'state_ports', 'parameter_slots', 'constant_slots', 'arguments', 'indices',
    'assignment', 'roles', 'tolerance', 'hook_map', 'per_contract', 'per_primitive',
}
# Values of these fields are user data; never rewrite keys inside records.
OPAQUE = {'arguments', 'assignment', 'indices', 'hook_map', 'tolerance', 'delivery', 'artifact'}


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate member name {key!r}')
        result[key] = value
    return result


def rename(value, parent=None):
    """Apply the explicit structural bijection, preserving authored map keys."""
    if parent in OPAQUE:
        return value
    if isinstance(value, list):
        return [rename(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        current_key = key in set(KEYS.values()) | {'primitive_libraries', 'primitive_library'}
        if key == 'instances' and parent == 'd1':
            current_key = False  # Composite expansion provenance already used this field.
        if parent not in NAMED_MAPS and current_key:
            raise ValueError(f'mixed legacy/current vocabulary: {key!r}')
        if parent in NAMED_MAPS:
            new = key
        elif key == 'catalog':
            new = 'primitive_libraries' if isinstance(item, list) else 'primitive_library'
        else:
            new = KEYS.get(key, key)
        if (new != key and new in value) or new in out:
            raise ValueError(f'mixed keys or collision: {key!r} and {new!r}')
        if key == 'schema' and isinstance(item, str):
            if item not in VERSIONS:
                raise ValueError(f'unsupported revision for this field layout: {item!r}')
            out[new] = VERSIONS[item]
        elif key == 'kind' and item == 'contract':
            out[new] = 'primitive'
        elif key == 'base' and isinstance(item, str):
            out[new] = '/'.join('primitive-library' if p == 'catalog' else p for p in item.split('/'))
        else:
            out[new] = rename(item, key)
    return out


def vocabulary(value, parent=None):
    """Classify structural keys, excluding authored identities and data records."""
    if parent in OPAQUE:
        return set()
    if isinstance(value, list):
        return set().union(*(vocabulary(item) for item in value))
    if not isinstance(value, dict):
        return set()
    found = set()
    for key, item in value.items():
        if parent not in NAMED_MAPS:
            if key in KEYS or key == 'catalog':
                found.add('legacy')
            if key in set(KEYS.values()) | {'primitive_libraries', 'primitive_library'}:
                if not (key == 'instances' and parent == 'd1'):
                    found.add('current')
        found.update(vocabulary(item, key))
    return found


def legacy_layout(document):
    return 'legacy' in vocabulary(document)


def convert(document):
    if not isinstance(document, dict) or document.get('schema') not in set(VERSIONS) | set(VERSIONS.values()):
        raise ValueError(f'unsupported revision: {document.get("schema") if isinstance(document, dict) else None!r}')
    layout = vocabulary(document)
    if layout == {'legacy', 'current'}:
        raise ValueError('mixed legacy/current vocabulary')
    if layout == {'current'} or (not layout and document['schema'] not in VERSIONS):
        import copy
        return copy.deepcopy(document)
    return rename(document)


def fixture_bytes(data):
    """Replace metadata alone, retaining every tensor name, descriptor and byte."""
    size = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8 + size], object_pairs_hook=pairs)
    metadata = {k: json.loads(v, object_pairs_hook=pairs) for k, v in header['__metadata__'].items()}
    header['__metadata__'] = {k: json.dumps(v) for k, v in convert(metadata).items()}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    return struct.pack('<Q', len(encoded)) + encoded + data[8 + size:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--library-base', type=Path, action='append')
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError('choose a separate output path to preserve the original')
        if args.input.suffix == '.safetensors':
            if args.library_base:
                raise ValueError('--library-base applies to JSON model documents')
            result = fixture_bytes(args.input.read_bytes())
        else:
            document = json.loads(args.input.read_text(), object_pairs_hook=pairs)
            migrated = convert(document)
            if 'primitive_libraries' in migrated:
                import os
                bases = args.library_base or [args.input.parent / e['base'] for e in migrated['primitive_libraries']]
                if len(bases) != len(migrated['primitive_libraries']):
                    raise ValueError('supply one --library-base for each library reference')
                for entry, base in zip(migrated['primitive_libraries'], bases):
                    entry['base'] = os.path.relpath(base.resolve(), args.output.parent.resolve())
            result = (json.dumps(migrated, indent=2) + '\n').encode()
        # Exclusive creation also prevents accidental replacement of other input files.
        with args.output.open('xb') as stream:
            stream.write(result)
    except (ValueError, OSError, KeyError, struct.error) as error:
        parser.exit(1, f'migration: {error}\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Explicit conversion of supported pre-v3 TensorSpine representations.

This module is the compatibility boundary; current readers never alias old keys.
"""
import argparse
import json
from pathlib import Path
import struct

VERSIONS = {
    'tensorspine/2.0': 'tensorspine/3.0',
    'tensorspine-catalog-unit/2.0': 'tensorspine-primitive-library-unit/3.0',
    'tensorspine-derived/2.1': 'tensorspine-derived/3.0',
    'tensorspine-capabilities/1': 'tensorspine-capabilities/2',
    'tensorspine-fixture/1': 'tensorspine-fixture/2',
    'tensorspine-primitive-request/1': 'tensorspine-primitive-request/2',
    'tensorspine-primitive-response/1': 'tensorspine-primitive-response/2',
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
SUCCESSORS = {name: ('1.0.0', '2.0.0') for name in (
    'attention.dense', 'attention.latent_compressed', 'conditioning.layer_select',
    'conditioning.multiplicative', 'conditioning.scale', 'conv_frontend', 'decoder.causal_yarn',
    'embed', 'embedding.time', 'embedding.token_auxiliary', 'embedding.token_position',
    'embedding.token_position_type', 'ffn.dense', 'ffn.gated', 'lm_head', 'mix.collapse',
    'mix.doubly_stochastic', 'moe', 'mtp.merge', 'norm.layer', 'norm.rms', 'patch_embed',
    'pooler', 'projector.patch_merge_bottleneck', 'projector.patch_merge_mlp',
    'projector.temporal_stack', 'residual.add', 'residual.altup_correct',
    'residual.altup_predict', 'residual.combine', 'residual.laurel',
    'residual.stream_collapse', 'residual.stream_expand', 'residual.stream_inject',
    'sequence.gated_delta', 'splice', 'decoder-causal-yarn',
)}


def successor(identity):
    """Map a published name@version (optionally followed by a fixture case)."""
    if not isinstance(identity, str):
        return identity
    for name, (old, new) in SUCCESSORS.items():
        prefix = f'{name}@{old}'
        if identity == prefix or identity.startswith(prefix + '/'):
            return f'{name}@{new}' + identity[len(prefix):]
    return identity


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
        return [successor(item) if parent == 'fixtures' else rename(item) for item in value]
    if not isinstance(value, dict):
        return successor(value) if parent in {'id', 'fixtures'} else value
    out = {}
    for key, item in value.items():
        current_key = key in set(KEYS.values()) | {'primitive_libraries', 'primitive_library'}
        if key == 'instances' and parent == 'd1':
            current_key = False  # Composite expansion provenance already used this field.
        if parent not in NAMED_MAPS and current_key:
            raise ValueError(f'mixed legacy/current vocabulary: {key!r}')
        if parent in NAMED_MAPS:
            new = successor(key) if parent in {'contracts', 'primitives'} else key
        elif key == 'catalog':
            new = 'primitive_libraries' if isinstance(item, list) else 'primitive_library'
        else:
            new = KEYS.get(key, key)
        if (new != key and new in value) or new in out:
            raise ValueError(f'mixed keys or collision: {key!r} and {new!r}')
        if key == 'schema' and isinstance(item, str):
            if item not in VERSIONS:
                raise ValueError(f'unsupported or current revision: {item!r}')
            out[new] = VERSIONS[item]
        elif key == 'version' and item == '1.0.0' and value.get('name') in SUCCESSORS and parent in {'contract', 'primitive', 'template'}:
            out[new] = '2.0.0'
        elif key == 'version' and value.get('model') == 'decoder_causal_yarn' and item == '1.0.0':
            out[new] = '2.0.0'
        elif key == 'kind' and item == 'contract':
            out[new] = 'primitive'
        elif key == 'base' and isinstance(item, str):
            out[new] = '/'.join('primitive-library' if p == 'catalog' else p for p in item.split('/'))
        else:
            out[new] = rename(item, key)
    if value.get('kind') == 'contract' and value.get('name') in SUCCESSORS:
        if out.get('definition', {}).get('version') == '1.0.0':
            out['definition']['version'] = '2.0.0'
    return out


def convert(document):
    if not isinstance(document, dict) or document.get('schema') not in VERSIONS:
        raise ValueError(f'unsupported or current revision: {document.get("schema") if isinstance(document, dict) else None!r}')
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

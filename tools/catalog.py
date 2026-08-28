"""Loading and resolution of an Armature catalog.

A base is either an exploded directory — with `contracts/`, `axes/` and
`precision/` — or a single monolithic file, still accepted. Bases are consulted
in order: the first one that carries an identity provides it, later ones do not
override it.

A unit's path reproduces its identity, dot by dot; a contract carries its
version as the file name, so two versions of one identity coexist instead of
replacing each other (§8.2).

    contracts/attention/dense/1.0.0.json     contract attention.dense 1.0.0
    axes/model/width.json                    axis model.width
    precision/norm/scale.json                precision role norm.scale

`load` returns contracts, axes and precision, plus `by_id` keyed by
(name, version). A disagreement between the path and the identity written
inside the file is a refusal, not a preference.
"""
import glob
import json
import os

UNIT_SCHEMA = "armature-catalog-unit/2.0"
SECTIONS = (('contracts', 'contract'), ('axes', 'axis'), ('precision', 'precision_role'))


def _semver(v):
    try:
        return tuple(int(x) for x in v.split('.'))
    except ValueError:
        return (0, 0, 0)


def _units(base):
    """Yield (section, name, version, definition) for every unit of a base."""
    for section, kind in SECTIONS:
        root = os.path.join(base, section)
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, '**', '*.json'), recursive=True)):
            with open(path, encoding='utf-8') as f:
                unit = json.load(f)
            parts = os.path.relpath(path, root)[:-len('.json')].split(os.sep)
            if section == 'contracts':
                name, version = '.'.join(parts[:-1]), parts[-1]
            else:
                name, version = '.'.join(parts), None
            if unit.get('schema') != UNIT_SCHEMA:
                raise ValueError(f"{path}: schema {unit.get('schema')!r}, expected {UNIT_SCHEMA!r}")
            if unit.get('kind') != kind:
                raise ValueError(f"{path}: kind {unit.get('kind')!r}, expected {kind!r}")
            if unit.get('name') != name:
                raise ValueError(f"{path}: name {unit.get('name')!r}, path says {name!r}")
            definition = unit['definition']
            if version is not None and definition.get('version') != version:
                raise ValueError(f"{path}: version {definition.get('version')!r}, "
                                 f"path says {version!r}")
            yield section, name, version, definition


def load(*bases):
    """Catalog gathered from the given bases, in order; first to answer wins."""
    axes, precision, by_id = {}, {}, {}
    for base in bases:
        if os.path.isdir(base):
            for section, name, version, definition in _units(base):
                if section == 'contracts':
                    by_id.setdefault((name, version), definition)
                elif section == 'axes':
                    axes.setdefault(name, definition)
                else:
                    precision.setdefault(name, definition)
        else:
            with open(base, encoding='utf-8') as f:
                mono = json.load(f)
            for name, d in mono.get('contracts', {}).items():
                by_id.setdefault((name, d['version']), d)
            for name, d in mono.get('axes', {}).items():
                axes.setdefault(name, d)
            for name, d in mono.get('precision', {}).items():
                precision.setdefault(name, d)

    # Index by name alone: the highest version, for readings that do not pin one.
    contracts = {}
    for (name, version), d in by_id.items():
        current = contracts.get(name)
        if current is None or _semver(version) > _semver(current['version']):
            contracts[name] = d
    return {"contracts": contracts, "axes": axes, "precision": precision, "by_id": by_id}


def contract(cat, ref):
    """The contract a {name, version} reference designates — the pinned version
    first, falling back to the name so that a version mismatch is reported by
    the caller rather than looking like an absent contract."""
    found = cat.get('by_id', {}).get((ref['name'], ref['version']))
    return found if found is not None else cat['contracts'].get(ref['name'])


def delegated_bodies(cat):
    """Names of the contracts whose body is a model document (§4.6)."""
    return {name for name, d in cat['contracts'].items() if 'model' in d}


def body_path(model_path, contract_definition):
    """Where the body of a delegated contract is looked for.

    Only the last dotted segment of the URI is used, as a file name, in the
    directory of the model that invokes it. The leading segments are ignored,
    so the body must sit beside its caller.
    """
    base_dir = os.path.dirname(os.path.abspath(model_path))
    uri = contract_definition['model']['uri']
    return os.path.join(base_dir, uri.split('.')[-1] + '.json')

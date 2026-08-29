"""Loading and resolution of an Tensorspine catalog.

A base is either an exploded directory — with `contracts/`, `axes/` and
`precision/` — or a single monolithic file, still accepted. Bases are consulted
in order: the first one that carries an identity provides it, later ones do not
override it.

A unit's path reproduces its identity, dot by dot; a contract carries its
version as the file name, so an identity `{name, version}` is one file and
never changes meaning (§8.2).

    contracts/attention/dense/1.0.0.json     contract attention.dense 1.0.0
    axes/model/width.json                    axis model.width
    precision/norm/scale.json                precision role norm.scale

`load` returns contracts, axes and precision, plus `by_id` keyed by
(name, version). A disagreement between the path and the identity written
inside the file is a refusal, not a preference. `load_for` resolves the bases a
model document declares in its `catalog` field, relative to the document, so
that field is what a reading resolves from (§2) unless the command line says
otherwise.

Every unit is read against the catalog-unit JSON Schema, and the references
one unit makes to another — an axis, a precision role, an argument named by a
condition — are resolved once the base is gathered. The catalog is the closed
vocabulary a consumer implements in advance (§4.3, O0.6); a unit outside it is
a load error naming the file, never a document read with a guess (I7).
"""


class CatalogError(ValueError):
    """A catalog unit that cannot be read: off-schema, misplaced, or citing
    something the catalog does not hold. Carries the file path."""

import glob
import json
import os

import schema as schema_mod

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMAS = os.path.join(os.path.dirname(HERE), 'schemas')
DEFAULT_MODELS = os.path.join(os.path.dirname(HERE), 'data', 'models')


DEFAULT_CATALOG = os.path.join(os.path.dirname(HERE), 'data', 'catalog')
_LOADED = {}


def bases_of(model_path, model, override=None):
    """The catalog bases a document resolves from: the command line when it
    names some, else the document's own `catalog` entries, taken relative to
    the document's directory."""
    if override:
        return [os.path.abspath(b) for b in override]
    here = os.path.dirname(os.path.abspath(model_path))
    return [os.path.normpath(os.path.join(here, entry['base'])) for entry in model['catalog']]


def load_for(model_path, model, override=None, schema_dir=None, models_base=None):
    """The catalog a document resolves from (see `bases_of`), loaded once per
    distinct set of bases. A declared base that does not exist is a rejection
    (V1), not a fallback to some other catalog."""
    schema_dir = DEFAULT_SCHEMAS if schema_dir is None else schema_dir
    models_base = DEFAULT_MODELS if models_base is None else models_base
    bases = bases_of(model_path, model, override)
    for base in bases:
        if not os.path.exists(base):
            raise CatalogError(f"{os.path.basename(model_path)}: catalog base '{base}' "
                               f"does not exist (V1)")
    key = (tuple(bases), schema_dir, models_base)
    if key not in _LOADED:
        _LOADED[key] = load(*bases, schema_dir=schema_dir, models_base=models_base)
    return _LOADED[key]

UNIT_SCHEMA = "tensorspine-catalog-unit/2.0"
SECTIONS = (('contracts', 'contract'), ('axes', 'axis'), ('precision', 'precision_role'))


def _semver(v):
    try:
        return tuple(int(x) for x in v.split('.'))
    except ValueError:
        return (0, 0, 0)


def _units(base, unit_schema=None, registry=None):
    """Yield (section, name, version, definition, path) for every unit of a base."""
    for section, kind in SECTIONS:
        root = os.path.join(base, section)
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, '**', '*.json'), recursive=True)):
            if unit_schema is not None:
                problems = schema_mod.deepest(schema_mod.check(unit_schema, path, registry))
                if problems:
                    lines = "\n".join("    " + schema_mod.format_error(e) for e in problems[:8])
                    raise CatalogError(f"{path}: off the catalog-unit schema\n{lines}")
            with open(path, encoding='utf-8') as f:
                unit = json.load(f)
            parts = os.path.relpath(path, root)[:-len('.json')].split(os.sep)
            if section == 'contracts':
                name, version = '.'.join(parts[:-1]), parts[-1]
            else:
                name, version = '.'.join(parts), None
            if unit.get('schema') != UNIT_SCHEMA:
                raise CatalogError(f"{path}: schema {unit.get('schema')!r}, expected {UNIT_SCHEMA!r}")
            if unit.get('kind') != kind:
                raise CatalogError(f"{path}: kind {unit.get('kind')!r}, expected {kind!r}")
            if unit.get('name') != name:
                raise CatalogError(f"{path}: name {unit.get('name')!r}, path says {name!r}")
            definition = unit['definition']
            if version is not None and definition.get('version') != version:
                raise CatalogError(f"{path}: version {definition.get('version')!r}, "
                                   f"path says {version!r}")
            yield section, name, version, definition, path


def load(*bases, schema_dir=DEFAULT_SCHEMAS, models_base=DEFAULT_MODELS):
    """Catalog gathered from the given bases, in order; first to answer wins.

    `schema_dir` holds the catalog-unit schema every exploded unit is read
    against; None skips the structural stage (a monolithic base is never
    checked that way, its units carry no file of their own). `models_base` is
    where the templates of template contracts are resolved (§4.6): a template
    is a model document and lives alongside the models."""
    axes, precision, by_id, origin = {}, {}, {}, {}
    unit_schema = registry = None
    if schema_dir is not None:
        unit_schema = schema_mod.locate(schema_dir, 'catalog-unit')
        if unit_schema is None:
            raise CatalogError(f"no schema with $id ending in /catalog-unit.json under {schema_dir}/")
        registry = schema_mod.registry(schema_dir)
    for base in bases:
        if os.path.isdir(base):
            for section, name, version, definition, path in _units(base, unit_schema, registry):
                origin.setdefault((section, name, version), path)
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

    # Index by name alone, for readings that address a contract by name (the
    # linter's inventory, D1's template walk); an occurrence always pins.
    contracts = {}
    for (name, version), d in by_id.items():
        current = contracts.get(name)
        if current is None or _semver(version) > _semver(current['version']):
            contracts[name] = d
    cat = {"contracts": contracts, "axes": axes, "precision": precision, "by_id": by_id,
           "models_base": models_base, "templates": {}}
    model_schema = schema_mod.locate(schema_dir, 'model') if schema_dir is not None else None
    for (name, version), d in sorted(by_id.items()):
        where = origin.get(('contracts', name, version), f"{name}@{version}")
        problems = contract_references(d, cat)
        if problems:
            raise CatalogError(f"{where}: unresolved reference(s)\n"
                               + "\n".join("    " + m for m in problems))
        if 'template' in d:
            cat['templates'][(name, version)] = _pinned_template(d, where, models_base,
                                                                  model_schema, registry)
    return cat


def _pinned_template(d, where, models_base, model_schema, registry):
    """The template file a template contract pins, once it is known to exist,
    to be a model document, and to carry the pinned version (§4.6)."""
    ref = d['template']
    path = os.path.join(models_base, ref['name'] + '.json')
    if not os.path.isfile(path):
        raise CatalogError(f"{where}: template '{ref['name']}' is not in {models_base}/")
    if model_schema is not None:
        problems = schema_mod.deepest(schema_mod.check(model_schema, path, registry))
        if problems:
            lines = "\n".join("    " + schema_mod.format_error(e) for e in problems[:8])
            raise CatalogError(f"{where}: template {path} is off the model schema\n{lines}")
    with open(path, encoding='utf-8') as f:
        template = json.load(f)
    if template.get('version') != ref['version']:
        raise CatalogError(f"{where}: pins template '{ref['name']}' at {ref['version']}, "
                           f"but {path} carries version {template.get('version')!r}")
    if template.get('model') != ref['id']:
        raise CatalogError(f"{where}: template '{ref['name']}' declares model id "
                           f"{template.get('model')!r}, the contract says {ref['id']!r}")
    return path


# --- cross-references: what the schema cannot see -------------------------

def _expression_paths(e):
    """Argument paths an expression reads."""
    if not isinstance(e, dict):
        return
    if 'argument' in e:
        yield e['argument']
    for x in e.get('args', []):
        yield from _expression_paths(x)


def _condition_paths(c):
    """Argument paths a condition tests."""
    if 'not' in c:
        yield from _condition_paths(c['not'])
    for x in c.get('all', []) + c.get('any', []):
        yield from _condition_paths(x)
    if 'present' in c:
        yield c['present']
    if 'compare' in c:
        yield from _expression_paths(c['compare']['left'])
        yield from _expression_paths(c['compare']['right'])


def _declared_paths(arguments, prefix=''):
    """Every argument path a set of declarations makes addressable, records
    flattened: `window`, `window.span`."""
    for name, decl in arguments.items():
        yield prefix + name
        t = decl['type']
        if t['kind'] == 'record':
            yield from _declared_paths(t['fields'], prefix + name + '.')


def _shape_problems(shape, label, cat, paths):
    for a in shape['axes']:
        if a['axis'] not in cat['axes']:
            yield f"{label}: axis '{a['axis']}' is not in the catalog"
        for f in a.get('factors', []):
            if f['axis'] not in cat['axes']:
                yield f"{label}: factor axis '{f['axis']}' is not in the catalog"
            for pth in _expression_paths(f['extent']):
                if pth not in paths:
                    yield f"{label}: extent reads undeclared argument '{pth}'"
        for pth in _expression_paths(a['extent']):
            if pth not in paths:
                yield f"{label}: extent reads undeclared argument '{pth}'"


def contract_references(d, cat):
    """Problems of one contract definition against the catalog it lives in:
    every axis, precision role, port and argument path it cites must exist,
    every default and enum description must name a declared value."""
    if 'template' in d:
        return []
    out = []
    paths = set(_declared_paths(d['arguments']))

    def enum_checks(decl, label):
        t = decl['type']
        if t['kind'] == 'enum':
            if 'default' in decl and 'literal' in decl['default'] \
                    and decl['default']['literal'] not in t['values']:
                out.append(f"{label}: default {decl['default']['literal']!r} is not among {t['values']}")
            for v in decl.get('value_descriptions', {}):
                if v not in t['values']:
                    out.append(f"{label}: value_descriptions names '{v}', not an enum value")
        elif 'value_descriptions' in decl:
            out.append(f"{label}: value_descriptions on a non-enum type")
        for pth in _expression_paths(decl.get('default', {})):
            if pth not in paths:
                out.append(f"{label}: default reads undeclared argument '{pth}'")
        if 'present_when' in decl:
            for pth in _condition_paths(decl['present_when']):
                if pth not in paths:
                    out.append(f"{label}: present_when tests undeclared argument '{pth}'")
        if t['kind'] == 'record':
            for fname, fdecl in t['fields'].items():
                enum_checks(fdecl, f"{label}.{fname}")

    for name, decl in d['arguments'].items():
        enum_checks(decl, f"argument '{name}'")

    ports = {}
    for side in ('inputs', 'outputs'):
        for pname, port in d['ports'][side].items():
            ports[pname] = side
            label = f"port '{pname}'"
            if port['role'] not in cat['precision']:
                out.append(f"{label}: role '{port['role']}' has no precision rule")
            if 'shape' in port:
                out.extend(_shape_problems(port['shape'], label, cat, paths))
            if 'present_when' in port:
                for pth in _condition_paths(port['present_when']):
                    if pth not in paths:
                        out.append(f"{label}: present_when tests undeclared argument '{pth}'")
            frm = port['domain'].get('from', {})
            if 'argument' in frm and frm['argument'] not in paths:
                out.append(f"{label}: domain from undeclared argument '{frm['argument']}'")
    for section in ('parameters', 'constants'):
        for pname, param in d[section].items():
            label = f"{section[:-1]} '{pname}'"
            if param['role'] not in cat['precision']:
                out.append(f"{label}: role '{param['role']}' has no precision rule")
            out.extend(_shape_problems(param['shape'], label, cat, paths))
            if 'present_when' in param:
                for pth in _condition_paths(param['present_when']):
                    if pth not in paths:
                        out.append(f"{label}: present_when tests undeclared argument '{pth}'")
            for pth in _expression_paths(param.get('multiplicity', {})):
                if pth not in paths:
                    out.append(f"{label}: multiplicity reads undeclared argument '{pth}'")
    for sname, port in d['state_ports'].items():
        label = f"state '{sname}'"
        for pth in _condition_paths(port['present_when']):
            if pth not in paths:
                out.append(f"{label}: present_when tests undeclared argument '{pth}'")
        for cname, comp in port['payload'].items():
            if comp['role'] not in cat['precision']:
                out.append(f"{label}.{cname}: role '{comp['role']}' has no precision rule")
            out.extend(_shape_problems(comp['shape'], f"{label}.{cname}", cat, paths))
        for ax in port['key_axes']:
            if ax not in cat['axes']:
                out.append(f"{label}: key axis '{ax}' is not in the catalog")
            elif cat['axes'][ax]['space'] != 'instance':
                out.append(f"{label}: key axis '{ax}' is a value axis, not an instance axis")
        for i, rule in enumerate(port['rules']):
            for pth in _condition_paths(rule['when']):
                if pth not in paths:
                    out.append(f"{label} rule {i}: when tests undeclared argument '{pth}'")
            for field in ('span', 'stride'):
                for pth in _expression_paths(rule.get(field, {})):
                    if pth not in paths:
                        out.append(f"{label} rule {i}: {field} reads undeclared argument '{pth}'")
            ib = rule['indexed_by']
            if 'argument' in ib and ib['argument'] not in paths:
                out.append(f"{label} rule {i}: indexed_by undeclared argument '{ib['argument']}'")
    for side, key in (('inputs', 'reads'), ('outputs', 'writes')):
        for pname in d['effects'][key]:
            if ports.get(pname) != side:
                out.append(f"effects.{key}: '{pname}' is not an {side[:-1]} port")
    for i, part in enumerate(d['partitions']):
        t = part['target']
        if 'argument_axis' in t and t['argument_axis'] not in cat['axes']:
            out.append(f"partition {i}: axis '{t['argument_axis']}' is not in the catalog")
        if 'instance_key_axis' in t and t['instance_key_axis'] not in cat['axes']:
            out.append(f"partition {i}: axis '{t['instance_key_axis']}' is not in the catalog")
        if 'payload_axis' in t:
            pa = t['payload_axis']
            comp = d['state_ports'].get(pa['state'], {}).get('payload', {}).get(pa['component'])
            if comp is None:
                out.append(f"partition {i}: no payload '{pa['state']}.{pa['component']}'")
            elif pa['axis'] not in {a['axis'] for a in comp['shape']['axes']}:
                out.append(f"partition {i}: '{pa['axis']}' is not an axis of "
                           f"'{pa['state']}.{pa['component']}'")
        if 'when' in part:
            for pth in _condition_paths(part['when']):
                if pth not in paths:
                    out.append(f"partition {i}: when tests undeclared argument '{pth}'")
    for i, tr in enumerate(d.get('domain_transforms', [])):
        if ports.get(tr['from_port']) != 'inputs':
            out.append(f"domain_transform {i}: from_port '{tr['from_port']}' is not an input port")
        if ports.get(tr['to_port']) != 'outputs':
            out.append(f"domain_transform {i}: to_port '{tr['to_port']}' is not an output port")
        for pth in _expression_paths(tr.get('factor', {})):
            if pth not in paths:
                out.append(f"domain_transform {i}: factor reads undeclared argument '{pth}'")
    for cname, cost in d.get('logical_cost', {}).items():
        for pth in _expression_paths(cost['expression']):
            if pth not in paths:
                out.append(f"logical_cost.{cname}: reads undeclared argument '{pth}'")
    return out


def contract(cat, ref):
    """The contract a {name, version} reference designates — the pinned version
    first, falling back to the name so that a version mismatch is reported by
    the caller rather than looking like an absent contract."""
    found = cat.get('by_id', {}).get((ref['name'], ref['version']))
    return found if found is not None else cat['contracts'].get(ref['name'])


def template_contracts(cat):
    """Names of the contracts whose template is a model document (§4.6)."""
    return {name for name, d in cat['contracts'].items() if 'template' in d}


def template_path(cat, contract_definition):
    """The template file of a template contract, as pinned and checked at load."""
    ref = contract_definition['template']
    for (name, version), path in cat['templates'].items():
        if cat['by_id'][(name, version)] is contract_definition:
            return path
    return os.path.join(cat['models_base'], ref['name'] + '.json')

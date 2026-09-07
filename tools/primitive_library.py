"""Loading and resolution of a TensorSpine primitive_library.

A base is either an exploded directory — with `primitives/`, `axes/`,
`precision/` and a `primitive-library.json` manifest — or a single monolithic file, still
accepted. Bases form a set: an identity is provided by whichever base carries
it, and two bases carrying one identity with different contents are a conflict
(V1), never a choice.

A unit's path reproduces its identity, dot by dot; a primitive carries its
version as the file name, so an identity `{name, version}` is one file and
never changes meaning (§8.2).

    primitives/attention/dense/1.0.0.json     primitive attention.dense 1.0.0
    axes/model/width.json                    axis model.width
    precision/norm/scale.json                precision role norm.scale

`load` returns primitives, axes and precision, plus `by_id` keyed by
(name, version). A disagreement between the path and the identity written
inside the file is a refusal, not a preference. The templates of a base's
template primitives live where its manifest says (`templates`, relative to the
base), one document `<name>/<version>.json` per template version (§4.6).
`load_for` resolves the bases a model definition declares in its `primitive library`
field, relative to the document, so that field is what a reading resolves from
(§2) unless the command line says otherwise.

Every unit is read against the primitive-library-unit JSON Schema, and the references
one unit makes to another — an axis, a precision role, an argument named by a
condition — are resolved once the base is gathered. The primitive library is the closed
vocabulary a consumer implements in advance (§4.3, O0.6); a unit outside it is
a load error naming the file, never a document read with a guess (I7).
"""


class PrimitiveLibraryError(ValueError):
    """A primitive library unit that cannot be read: off-schema, misplaced, or citing
    something the primitive library does not hold. Carries the file path."""

import glob
import json
import os

import schema as schema_mod
from migrate import legacy_layout


def _pairs(pairs):
    """Object hook refusing duplicate member names: a parser that keeps the
    last value would drop a declaration silently (V12, I7)."""
    seen = set()
    for k, _v in pairs:
        if k in seen:
            raise ValueError(f"duplicate member name '{k}'")
        seen.add(k)
    return dict(pairs)


def read_json(path):
    """A JSON file, duplicate member names refused."""
    with open(path, encoding='utf-8') as f:
        try:
            document = json.load(f, object_pairs_hook=_pairs)
            if legacy_layout(document):
                raise ValueError('legacy field layout; convert it with python3 tools/migrate.py INPUT -o OUTPUT')
            revision = document.get('schema') if isinstance(document, dict) else None
            if isinstance(revision, str) and revision.startswith('tensorspine') and revision not in {
                'tensorspine/2.0', 'tensorspine-primitive-library-unit/2.0',
                'tensorspine-derived/2.1', 'tensorspine-capabilities/1', 'tensorspine-fixture/1',
            }:
                raise ValueError(f'unsupported revision {revision!r}; convert supported legacy input with '
                                 'python3 tools/migrate.py INPUT -o OUTPUT')
            return document
        except ValueError as e:
            raise PrimitiveLibraryError(f"{path}: {e} (V12)") from None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMAS = os.path.join(os.path.dirname(HERE), 'schemas')
DEFAULT_MODELS = os.path.join(os.path.dirname(HERE), 'data', 'models')


DEFAULT_PRIMITIVE_LIBRARY = os.path.join(os.path.dirname(HERE), 'data', 'primitive-library')
_LOADED = {}


def bases_of(model_path, model, override=None):
    """The primitive library bases a document resolves from: the command line when it
    names some, else the document's own `primitive library` entries, taken relative to
    the document's directory."""
    if model.get('schema') != 'tensorspine/2.0' or legacy_layout(model):
        raise PrimitiveLibraryError(f"{model_path}: expected tensorspine/2.0; convert supported legacy input "
                                    "with python3 tools/migrate.py INPUT -o OUTPUT")
    if override:
        return [os.path.abspath(b) for b in override]
    here = os.path.dirname(os.path.abspath(model_path))
    return [os.path.normpath(os.path.join(here, entry['base'])) for entry in model['primitive_libraries']]


def load_for(model_path, model, override=None, schema_dir=None, models_base=None):
    """The primitive library a document resolves from (see `bases_of`), loaded once per
    distinct set of bases. A declared base that does not exist is a rejection
    (V1), not a fallback to some other primitive_library. `models_base` overrides the
    templates location every base declares (tests only)."""
    schema_dir = DEFAULT_SCHEMAS if schema_dir is None else schema_dir
    bases = bases_of(model_path, model, override)
    for base in bases:
        if not os.path.exists(base):
            raise PrimitiveLibraryError(f"{os.path.basename(model_path)}: primitive library base '{base}' "
                               f"does not exist (V1)")
    key = (tuple(bases), schema_dir, models_base)
    if key not in _LOADED:
        _LOADED[key] = load(*bases, schema_dir=schema_dir, models_base=models_base)
    return _LOADED[key]

UNIT_SCHEMA = "tensorspine-primitive-library-unit/2.0"
SECTIONS = (('primitives', 'primitive'), ('axes', 'axis'), ('precision', 'precision_role'))


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
                    raise PrimitiveLibraryError(f"{path}: off the primitive-library-unit schema\n{lines}")
            unit = read_json(path)
            parts = os.path.relpath(path, root)[:-len('.json')].split(os.sep)
            if section == 'primitives':
                name, version = '.'.join(parts[:-1]), parts[-1]
            else:
                name, version = '.'.join(parts), None
            if unit.get('schema') != UNIT_SCHEMA:
                raise PrimitiveLibraryError(f"{path}: schema {unit.get('schema')!r}, expected {UNIT_SCHEMA!r}")
            if unit.get('kind') != kind:
                raise PrimitiveLibraryError(f"{path}: kind {unit.get('kind')!r}, expected {kind!r}")
            if unit.get('name') != name:
                raise PrimitiveLibraryError(f"{path}: name {unit.get('name')!r}, path says {name!r}")
            definition = unit['definition']
            if version is not None and definition.get('version') != version:
                raise PrimitiveLibraryError(f"{path}: version {definition.get('version')!r}, "
                                   f"path says {version!r}")
            yield section, name, version, definition, path


def _manifest(base, unit_schema, registry):
    """The base manifest (`primitive-library.json`, kind `base`) of an exploded base,
    or None when it carries none."""
    path = os.path.join(base, 'primitive-library.json')
    if not os.path.isfile(path):
        return None
    if unit_schema is not None:
        problems = schema_mod.deepest(schema_mod.check(unit_schema, path, registry))
        if problems:
            lines = "\n".join("    " + schema_mod.format_error(e) for e in problems[:8])
            raise PrimitiveLibraryError(f"{path}: off the primitive-library-unit schema\n{lines}")
    unit = read_json(path)
    if unit.get('kind') != 'base':
        raise PrimitiveLibraryError(f"{path}: kind {unit.get('kind')!r}, expected 'base'")
    return unit['definition']


def _provide(store, key, definition, where, origin_key, origin):
    """One identity, one content: a second base carrying the same identity
    must carry the same definition (V1)."""
    current = store.get(key)
    if current is None:
        store[key] = definition
        origin[origin_key] = where
    elif current != definition:
        raise PrimitiveLibraryError(f"{where}: identity {key} is also carried by "
                           f"{origin.get(origin_key)} with different contents (V1)")


def load(*bases, schema_dir=DEFAULT_SCHEMAS, models_base=None):
    """Primitive Library gathered from the given bases, a set: one identity, one content.

    `schema_dir` holds the primitive-library-unit schema every exploded unit is read
    against; None skips the structural stage (a monolithic base is never
    checked that way, its units carry no file of their own). The templates of
    a base's template primitives are resolved where its manifest says
    (`templates`); `models_base` overrides that location for every base."""
    axes, precision, by_id, origin = {}, {}, {}, {}
    templates_dir = {}                      # base -> where its templates live
    unit_schema = registry = None
    if schema_dir is not None:
        unit_schema = schema_mod.locate(schema_dir, 'primitive-library-unit')
        if unit_schema is None:
            raise PrimitiveLibraryError(f"no schema with $id ending in /primitive-library-unit.json under {schema_dir}/")
        registry = schema_mod.registry(schema_dir)
    for base in bases:
        if os.path.isdir(base):
            manifest = _manifest(base, unit_schema, registry)
            if manifest and 'templates' in manifest:
                templates_dir[base] = os.path.normpath(os.path.join(base, manifest['templates']))
            if models_base is not None:
                templates_dir[base] = models_base
            for section, name, version, definition, path in _units(base, unit_schema, registry):
                if section == 'primitives':
                    _provide(by_id, (name, version), definition, path, ('primitives', name, version), origin)
                    origin.setdefault(('base-of', name, version), base)
                elif section == 'axes':
                    _provide(axes, name, definition, path, ('axes', name, None), origin)
                else:
                    _provide(precision, name, definition, path, ('precision', name, None), origin)
        else:
            mono = read_json(base)
            if models_base is not None:
                templates_dir[base] = models_base
            for name, d in mono.get('primitives', {}).items():
                _provide(by_id, (name, d['version']), d, base, ('primitives', name, d['version']), origin)
                origin.setdefault(('base-of', name, d['version']), base)
            for name, d in mono.get('axes', {}).items():
                _provide(axes, name, d, base, ('axes', name, None), origin)
            for name, d in mono.get('precision', {}).items():
                _provide(precision, name, d, base, ('precision', name, None), origin)
    for name, role in precision.items():
        if role['default'] not in role['admissible']:
            raise PrimitiveLibraryError(f"{origin.get(('precision', name, None), name)}: default "
                               f"'{role['default']}' is not in the admissible set {role['admissible']}")

    # Index by name alone, for readings that address a primitive by name (the
    # linter's inventory, D1's template walk); an instance always pins.
    primitives = {}
    for (name, version), d in by_id.items():
        current = primitives.get(name)
        if current is None or _semver(version) > _semver(current['version']):
            primitives[name] = d
    cat = {"primitives": primitives, "axes": axes, "precision": precision, "by_id": by_id,
           "templates": {}}
    model_schema = schema_mod.locate(schema_dir, 'model') if schema_dir is not None else None
    for (name, version), d in sorted(by_id.items()):
        where = origin.get(('primitives', name, version), f"{name}@{version}")
        problems = primitive_references(d, cat)
        if problems:
            raise PrimitiveLibraryError(f"{where}: unresolved reference(s)\n"
                               + "\n".join("    " + m for m in problems))
        if 'template' in d:
            base = origin.get(('base-of', name, version))
            location = templates_dir.get(base)
            if location is None:
                raise PrimitiveLibraryError(f"{where}: a template primitive, but its base declares no "
                                   f"`templates` location (§4.6)")
            cat['templates'][(name, version)] = _pinned_template(d, where, location,
                                                                  model_schema, registry)
    return cat


def _pinned_template(d, where, location, model_schema, registry):
    """The template file a template primitive pins — `<name>/<version>.json`
    in the declared location — once it is known to exist, to be a model
    document, and to carry the pinned version and id (§4.6)."""
    ref = d['template']
    path = os.path.join(location, ref['name'], ref['version'] + '.json')
    if not os.path.isfile(path):
        raise PrimitiveLibraryError(f"{where}: template '{ref['name']}' {ref['version']} is not at {path}")
    if model_schema is not None:
        problems = schema_mod.deepest(schema_mod.check(model_schema, path, registry))
        if problems:
            lines = "\n".join("    " + schema_mod.format_error(e) for e in problems[:8])
            raise PrimitiveLibraryError(f"{where}: template {path} is off the model schema\n{lines}")
    template = read_json(path)
    if template.get('version') != ref['version']:
        raise PrimitiveLibraryError(f"{where}: pins template '{ref['name']}' at {ref['version']}, "
                           f"but {path} carries version {template.get('version')!r}")
    if template.get('model') != ref['id']:
        raise PrimitiveLibraryError(f"{where}: template '{ref['name']}' declares model id "
                           f"{template.get('model')!r}, the primitive says {ref['id']!r}")
    return path


# --- cross-references: what the schema cannot see -------------------------

def _expression_paths(e):
    """Argument paths an expression reads, conditionals included."""
    if not isinstance(e, dict):
        return
    if 'argument' in e:
        yield e['argument']
    for x in e.get('args', []):
        yield from _expression_paths(x)
    for k in ('then', 'else'):
        if k in e:
            yield from _expression_paths(e[k])
    if 'if' in e:
        yield from _condition_paths(e['if'])


def _absent_comparisons(c, guarded, optional):
    """Paths of optional arguments without a default that a condition compares
    or computes with outside a `present` test of them (§4.3): undecidable, so
    refused at load rather than read as false."""
    if not isinstance(c, dict):
        return
    if 'present' in c:
        return
    if 'all' in c:
        g = set(guarded) | {s['present'] for s in c['all'] if isinstance(s, dict) and 'present' in s}
        for s in c['all']:
            yield from _absent_comparisons(s, g, optional)
        return
    if 'any' in c:
        # the dual of `all` + `present`: `any[not present X, P(X)]` reads X only when it is present,
        # so a `not present` disjunct guards X for the others — an invariant that holds vacuously
        # when a conditional field is absent (a yarn-only bound under llama3 scaling)
        g = set(guarded) | {s['not']['present'] for s in c['any']
                            if isinstance(s, dict) and isinstance(s.get('not'), dict) and 'present' in s['not']}
        for s in c['any']:
            yield from _absent_comparisons(s, g, optional)
        return
    if 'not' in c:
        yield from _absent_comparisons(c['not'], guarded, optional)
        return
    if 'compare' in c:
        for r in _expression_paths(c['compare']['left']):
            yield from _unguarded(r, guarded, optional)
        for r in _expression_paths(c['compare']['right']):
            yield from _unguarded(r, guarded, optional)


def _unguarded(path, guarded, optional):
    if any(path == o or path.startswith(o + '.') for o in optional)\
            and not any(path == g or path.startswith(g + '.') for g in guarded):
        yield path


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


def _optional_paths(arguments, prefix=''):
    """Paths that may be absent at an instance: optional, without a default."""
    for name, decl in arguments.items():
        if not decl['required'] and 'default' not in decl:
            yield prefix + name
        t = decl['type']
        if t['kind'] == 'record':
            yield from _optional_paths(t['fields'], prefix + name + '.')


def _shape_problems(shape, label, cat, paths):
    for a in shape['axes']:
        if a['axis'] not in cat['axes']:
            yield f"{label}: axis '{a['axis']}' is not in the primitive_library"
        for f in a.get('factors', []):
            if f['axis'] not in cat['axes']:
                yield f"{label}: factor axis '{f['axis']}' is not in the primitive_library"
            for pth in _expression_paths(f['extent']):
                if pth not in paths:
                    yield f"{label}: extent reads undeclared argument '{pth}'"
        for pth in _expression_paths(a['extent']):
            if pth not in paths:
                yield f"{label}: extent reads undeclared argument '{pth}'"


def primitive_references(d, cat):
    """Problems of one primitive definition against the primitive library it lives in:
    every axis, precision role, port and argument path it cites must exist,
    every default and enum description must name a declared value."""
    if 'template' in d:
        return []
    out = []
    paths = set(_declared_paths(d['arguments']))
    optional = set(_optional_paths(d['arguments']))
    conditions = []                     # (label, condition) — every one the primitive writes

    def enum_checks(decl, label, path):
        t = decl['type']
        # Inside a record, the record itself is present by construction: a
        # field's condition is only ever evaluated when its record is.
        enclosing = {path.rsplit('.', n)[0] for n in range(1, path.count('.') + 1)}
        if t['kind'] == 'enum':
            if 'default' in decl and 'literal' in decl['default']\
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
        dom = decl.get('domain')
        if dom and dom['kind'] == 'interval':
            # a bound may name another argument; it must be declared and always resolve, so a
            # maybe-absent argument (optional, no default) is refused rather than read as no bound
            for edge in ('lower', 'upper'):
                b = dom.get(edge)
                if b and isinstance(b['value'], dict) and 'argument' in b['value']:
                    pth = b['value']['argument']
                    if pth not in paths:
                        out.append(f"{label}: domain {edge} bound reads undeclared argument '{pth}'")
                    elif pth in optional:
                        out.append(f"{label}: domain {edge} bound reads '{pth}', which may be absent — a bound must always resolve (§4.6)")
        if 'present_when' in decl:
            conditions.append((f"{label} present_when", decl['present_when'], enclosing))
        if t['kind'] == 'record':
            for fname, fdecl in t['fields'].items():
                enum_checks(fdecl, f"{label}.{fname}", f"{path}.{fname}")

    for name, decl in d['arguments'].items():
        enum_checks(decl, f"argument '{name}'", name)

    for i, inv in enumerate(d.get('invariants', [])):
        # an invariant's condition is a primitive condition: every argument it reads must be
        # declared, and one that may be absent must be guarded by a `present` test of it (§4.3),
        # exactly as every other condition of the primitive is checked below
        conditions.append((f"invariant {i} ('{inv['description']}')", inv['holds'], set()))

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
                conditions.append((f"{label} present_when", port['present_when'], set()))
            frm = port['domain'].get('from', {})
            if 'port' in frm:
                target = d['ports']['inputs'].get(frm['port'])
                if target is None:
                    out.append(f"{label}: domain inherited from unknown input port '{frm['port']}'")
                elif 'port' in target['domain'].get('from', {}):
                    out.append(f"{label}: domain inherited from '{frm['port']}', which itself "
                               f"inherits from a port")
    for section in ('parameters', 'constants'):
        for pname, param in d[section].items():
            label = f"{section[:-1]} '{pname}'"
            if param['role'] not in cat['precision']:
                out.append(f"{label}: role '{param['role']}' has no precision rule")
            out.extend(_shape_problems(param['shape'], label, cat, paths))
            out.extend(f"{label}: axis name 'multiplicity' is reserved — the storage axis of a slot that declares a multiplicity (§3.4)"
                       for a in param['shape']['axes'] if a['name'] == 'multiplicity')
            if 'present_when' in param:
                conditions.append((f"{label} present_when", param['present_when'], set()))
            if param.get('sharing', {}).get('kind') == 'shareable':
                for role in param['sharing'].get('roles', []):
                    if role not in cat['precision']:
                        out.append(f"{label}: sharing role '{role}' has no precision rule")
            for pth in _expression_paths(param.get('multiplicity', {})):
                if pth not in paths:
                    out.append(f"{label}: multiplicity reads undeclared argument '{pth}'")
    for sname, port in d['state_ports'].items():
        label = f"state '{sname}'"
        conditions.append((f"{label} present_when", port['present_when'], set()))
        if 'written_when' in port:
            conditions.append((f"{label} written_when", port['written_when'], set()))
        if 'carried_across' in port:
            conditions.append((f"{label} carried_across", port['carried_across']['when'], set()))
        growing = sorted({rule['evolution'] for rule in port['rules']} - {'fixed'})
        for cname, comp in port['payload'].items():
            if comp['role'] not in cat['precision']:
                out.append(f"{label}.{cname}: role '{comp['role']}' has no precision rule")
            out.extend(_shape_problems(comp['shape'], f"{label}.{cname}", cat, paths))
            if growing and any(a['axis'] == 'sequence.position' for a in comp['shape']['axes']):
                out.append(f"{label}.{cname}: a sequence.position axis under a {'/'.join(growing)} rule "
                           f"— a payload is declared per position (§4.3)")
        for ax in port['key_axes']:
            if ax not in cat['axes']:
                out.append(f"{label}: key axis '{ax}' is not in the primitive_library")
            elif cat['axes'][ax]['space'] != 'instance':
                out.append(f"{label}: key axis '{ax}' is a value axis, not an instance axis")
        for i, rule in enumerate(port['rules']):
            conditions.append((f"{label} rule {i}", rule['when'], set()))
            for field in ('span', 'stride'):
                for pth in _expression_paths(rule.get(field, {})):
                    if pth not in paths:
                        out.append(f"{label} rule {i}: {field} reads undeclared argument '{pth}'")
            ib = rule['indexed_by']
            if 'port' in ib and ports.get(ib['port']) != 'inputs':
                out.append(f"{label} rule {i}: indexed_by '{ib['port']}', not an input port")
    for side, key in (('inputs', 'reads'), ('outputs', 'writes')):
        for pname in d['effects'][key]:
            if ports.get(pname) != side:
                out.append(f"effects.{key}: '{pname}' is not an {side[:-1]} port")
    for i, part in enumerate(d['partition_options']):
        t = part['target']
        if 'argument_axis' in t and t['argument_axis'] not in cat['axes']:
            out.append(f"partition {i}: axis '{t['argument_axis']}' is not in the primitive_library")
        if 'instance_key_axis' in t and t['instance_key_axis'] not in cat['axes']:
            out.append(f"partition {i}: axis '{t['instance_key_axis']}' is not in the primitive_library")
        if 'payload_axis' in t:
            pa = t['payload_axis']
            comp = d['state_ports'].get(pa['state'], {}).get('payload', {}).get(pa['component'])
            if comp is None:
                out.append(f"partition {i}: no payload '{pa['state']}.{pa['component']}'")
            elif pa['axis'] not in {a['axis'] for a in comp['shape']['axes']}:
                out.append(f"partition {i}: '{pa['axis']}' is not an axis of "
                           f"'{pa['state']}.{pa['component']}'")
        if 'when' in part:
            conditions.append((f"partition {i}", part['when'], set()))
        for pth in _expression_paths(part.get('granularity', {})):
            if pth not in paths:
                out.append(f"partition {i}: granularity reads undeclared argument '{pth}'")
    for i, tr in enumerate(d.get('domain_transforms', [])):
        if ports.get(tr['from_port']) != 'inputs':
            out.append(f"domain_transform {i}: from_port '{tr['from_port']}' is not an input port")
        if ports.get(tr['to_port']) != 'outputs':
            out.append(f"domain_transform {i}: to_port '{tr['to_port']}' is not an output port")
        if tr['relation'] == 'merge' and 'factor' not in tr:
            out.append(f"domain_transform {i}: a merge declares its factor")
        for pth in _expression_paths(tr.get('factor', {})):
            if pth not in paths:
                out.append(f"domain_transform {i}: factor reads undeclared argument '{pth}'")
    for i, cost in enumerate(d.get('logical_cost', [])):
        if 'when' in cost:
            conditions.append((f"logical_cost {i}", cost['when'], set()))
        for pth in _expression_paths(cost['expression']):
            if pth not in paths:
                out.append(f"logical_cost {i}: reads undeclared argument '{pth}'")
    for i, sp in enumerate(d.get('sparsity', [])):
        label = f"sparsity {i}"
        for pname in sp['unit']['parameters']:
            param = d['parameters'].get(pname)
            if param is None:
                out.append(f"{label}: unit parameter '{pname}' is not a slot of this primitive")
            elif sp['unit']['axis'] not in {a['axis'] for a in param['shape']['axes']}:
                out.append(f"{label}: unit axis '{sp['unit']['axis']}' is not an axis of slot '{pname}'")
        if sp['unit']['axis'] not in cat['axes']:
            out.append(f"{label}: axis '{sp['unit']['axis']}' is not in the primitive_library")
        policy = sp['policy']
        if 'argument' in policy and policy['argument'] not in paths:
            out.append(f"{label}: policy names undeclared argument '{policy['argument']}'")
        if 'port' in policy and ports.get(policy['port']) != 'inputs':
            out.append(f"{label}: policy names '{policy['port']}', not an input port")
        for field, e in (('activated_per_element', sp['activated_per_element']),
                         ('union_per_invocation', sp['union_per_invocation']['expression'])):
            for pth in _expression_paths(e):
                if pth not in paths:
                    out.append(f"{label}.{field}: reads undeclared argument '{pth}'")
    for label, c, guarded in conditions:
        for pth in _condition_paths(c):
            if pth not in paths:
                out.append(f"{label}: tests undeclared argument '{pth}'")
        for pth in _absent_comparisons(c, guarded, optional):
            out.append(f"{label}: compares '{pth}', which may be absent, outside a present test "
                       f"of it (§4.3)")
    for i, part in enumerate(d['partition_options']):
        communications = part['communication'] if isinstance(part['communication'], list) else [part['communication']]
        if 'none' in part['target'] and communications != ['none']:
            out.append(f"partition {i}: a `none` target implies no communication")
        if 'none' in part['target'] and len(d['partition_options']) > 1:
            out.append(f"partition {i}: `none` cannot stand beside other partition_options")
    return out


def primitive(cat, ref):
    """The primitive a {name, version} reference designates — the pinned version
    first, falling back to the name so that a version mismatch is reported by
    the caller rather than looking like an absent primitive."""
    found = cat.get('by_id', {}).get((ref['name'], ref['version']))
    return found if found is not None else cat['primitives'].get(ref['name'])


def template_primitives(cat):
    """Names of the primitives whose template is a model definition (§4.6)."""
    return {name for name, d in cat['primitives'].items() if 'template' in d}


def template_path(cat, primitive_definition):
    """The template file of a template primitive, as pinned and checked at load."""
    for (name, version), path in cat['templates'].items():
        if cat['by_id'][(name, version)] == primitive_definition:
            return path
    ref = primitive_definition['template']
    raise PrimitiveLibraryError(f"template '{ref['name']}' {ref['version']} was not resolved at load")

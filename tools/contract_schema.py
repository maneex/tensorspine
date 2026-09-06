"""`--document contract-schema`: one JSON Schema per contract version, generated from its argument
declarations (§4.1, §4.6). Non-normative, and not a derived product (§7): the validator stays the
authority for whether an occurrence's arguments satisfy a contract. This is a rendering, like
`--document catalog` — a convenience for a tool that wants to check a bag of arguments with a plain
JSON Schema validator, and for the editing forms a workbench builds.

What JSON Schema can carry, it carries: the argument types, enums, records
(`additionalProperties: false`), a literal interval or set domain as `minimum`/`maximum`/`enum`, a
whole-number physical argument as `integer`. What it cannot — a bound that names another argument, a
relation between arguments (an invariant), conditional presence, the unit of a physical value — is
kept in `x-tensorspine-*` annotations so nothing is lost and nothing is misread as enforceable. The
declaration remains the source; the generated schema is a shadow of it.
"""
import json
import os

import catalog as catalog_mod


def _numeric(t):
    """The JSON-Schema base type of a scalar argument type, or None for a record."""
    kind = t['kind']
    if kind == 'cardinality':
        return {'type': 'integer', 'minimum': 0}
    if kind == 'real':
        return {'type': 'number'}
    if kind == 'physical':
        base = {'type': 'integer'} if t['unit'] != 'seconds' else {'type': 'number'}
        base['x-tensorspine-unit'] = t['unit']
        return base
    if kind == 'boolean':
        return {'type': 'boolean'}
    if kind == 'enum':
        return {'enum': list(t['values'])}
    return None


def _domain(schema, domain):
    """Fold a literal interval or set domain into a scalar schema; a bound naming another argument
    stays an annotation (JSON Schema cannot reference a sibling's value)."""
    if domain['kind'] == 'set':
        schema['enum'] = list(domain['values'])
        return
    for edge, incl_key, excl_key in (('lower', 'minimum', 'exclusiveMinimum'),
                                     ('upper', 'maximum', 'exclusiveMaximum')):
        bound = domain.get(edge)
        if bound is None:
            continue
        value = bound['value']
        if 'literal' in value:
            schema[incl_key if bound['inclusive'] else excl_key] = value['literal']
        else:
            schema.setdefault('x-tensorspine-domain', {})[edge] = {
                'argument': value['argument'], 'inclusive': bound['inclusive']}


def _property(decl):
    """One argument declaration as a JSON-Schema property, records recursively."""
    t = decl['type']
    if t['kind'] == 'record':
        schema = _object(t['fields'])
    else:
        schema = _numeric(t) or {}
    if 'domain' in decl:
        _domain(schema, decl['domain'])
    if 'present_when' in decl:
        schema['x-tensorspine-present-when'] = decl['present_when']
    if 'default' in decl:
        schema.setdefault('x-tensorspine-default', decl['default'])
    return schema


def _object(arguments):
    """A record (or the whole argument map) as an object schema: every field a property, the
    unconditionally required ones in `required`; a present_when'd field is conditionally required
    and kept out of `required` (its condition is annotated on it)."""
    props = {name: _property(decl) for name, decl in arguments.items()}
    required = [name for name, decl in arguments.items()
                if decl.get('required') and 'present_when' not in decl]
    schema = {'type': 'object', 'additionalProperties': False, 'properties': props}
    if required:
        schema['required'] = required
    return schema


def schema_for(name, version, definition):
    """The JSON Schema of one contract version's arguments."""
    schema = {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        '$id': f'https://tensorspine.dev/contract/{name}/{version}.json',
        'title': f'{name}@{version} arguments',
        'description': f'Generated from the {name} {version} contract (non-normative; the validator '
                       f'is the authority). Relations JSON Schema cannot express are in x-tensorspine-* keys.',
    }
    schema.update(_object(definition.get('arguments', {})))
    if definition.get('invariants'):
        schema['x-tensorspine-invariants'] = [{'holds': i['holds'], 'description': i['description']}
                                              for i in definition['invariants']]
    return schema


def render(cat):
    """{`name@version`: schema} for every non-template contract version of the catalog."""
    out = {}
    for (name, version), d in sorted(cat['by_id'].items()):
        if 'template' in d:
            continue
        out[f'{name}@{version}'] = schema_for(name, version, d)
    return out


def run(catalog_bases, schema_dir, output=None):
    """Write one schema per contract to a directory (`-o DIR`), or the whole map as one JSON object
    to `-o FILE` or stdout. Returns a CLI exit status."""
    cat = catalog_mod.load(*catalog_bases, schema_dir=schema_dir)
    schemas = render(cat)
    if output and os.path.isdir(output):
        for cid, schema in schemas.items():
            with open(os.path.join(output, cid.replace('@', '_') + '.json'), 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2)
                f.write('\n')
        print(f"  {len(schemas)} contract schema(s) -> {output}/", file=os.sys.stderr)
    else:
        text = json.dumps(schemas, indent=2) + '\n'
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(text)
        else:
            print(text, end='')
    return 0

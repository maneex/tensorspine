"""JSON Schema validation, cross-file $ref included.

The stock `jsonschema` CLI cannot follow a $ref from one schema to another: it
has no idea where https://armature.dev/schema/2.0/catalog.json lives. This
module builds the registry that maps the published namespace onto the files of
the repository.

The mapping is discovered, not declared: every schema in the directory is read
and indexed under its own `$id`. Renaming a schema file, or moving the whole
directory, therefore needs no change here — which is the point, since the
previous table of file names broke on the first restructuring.
"""
import glob
import json
import os

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


def discover(schema_dir):
    """Map every `$id` found in the directory to the file that declares it."""
    found = {}
    for path in sorted(glob.glob(os.path.join(schema_dir, '*.json'))):
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        identity = doc.get('$id')
        if identity:
            found[identity] = (path, doc)
    return found


def registry(schema_dir):
    """A referencing registry holding every schema of the directory."""
    resources = [(identity, Resource.from_contents(doc, default_specification=DRAFT202012))
                 for identity, (_path, doc) in discover(schema_dir).items()]
    return Registry().with_resources(resources)


def locate(schema_dir, role):
    """Path of the schema whose `$id` ends with `<role>.json`.

    Roles are the last segment of the published namespace: `model`, `catalog`,
    `catalog-unit`, `d1`. The file may be named anything.
    """
    suffix = '/' + role + '.json'
    for identity, (path, _doc) in discover(schema_dir).items():
        if identity.endswith(suffix):
            return path
    return None


def check(schema_path, doc_path, reg):
    """Errors of one document against one schema; empty when it conforms."""
    with open(schema_path, encoding='utf-8') as f:
        schema = json.load(f)
    with open(doc_path, encoding='utf-8') as f:
        doc = json.load(f)
    validator = Draft202012Validator(schema, registry=reg)
    return sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))


def format_error(e):
    """One error as a single line: where, then what."""
    where = '/'.join(str(p) for p in e.absolute_path) or '<root>'
    return f"{where}: {e.message}"

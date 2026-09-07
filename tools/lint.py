"""`--lint`: hygiene. What is valid and worth knowing anyway.

Nothing here is a refusal. §8.1 makes explicit refusal the obligation of the
primitive, so anything the primitive can judge is already handled by
`--validate`. What is left over is outside the document's jurisdiction: the
consistency of the repository around it, and the curation of an open primitive_library.

A lint finding is an opinion a reasonable author may decline. That is why this
command always exits 0.
"""
import json
import os

import primitive_library as primitive_library_mod
import validate as validate_mod
from expr import missing_assignment


def _called_primitives(model_path, cat, seen=None):
    """PrimitiveReference names a model calls, following templates."""
    seen = seen if seen is not None else set()
    try:
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
    except OSError:
        return seen
    names = set()
    for o in model['instances'].values():
        names.add(o['primitive']['name'])
    for c in model['compositions'].values():
        for o in c['instances'].values():
            names.add(o['primitive']['name'])
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        definition = cat['primitives'].get(name)
        if definition is not None and 'template' in definition:
            _called_primitives(primitive_library_mod.template_path(cat, definition), cat, seen)
    return seen


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def uncalled_primitives(cat, model_paths):
    """Primitives the primitive library carries that none of the linted models calls.

    Not dead code: the primitive library is an open vocabulary and the corpus is a
    falsification sample, not the definition. A curation question.
    """
    called = set()
    for path in model_paths:
        called |= _called_primitives(path, cat)
    return [("primitive_library", f"primitive '{name}' {definition['version']} is in the primitive library, "
                        f"called by none of the {len(model_paths)} model(s) linted")
            for name, definition in sorted(cat['primitives'].items())
            if name not in called]


def unreferenced_vocabulary(cat):
    """Axes and precision roles no primitive cites."""
    cited = set(_strings(cat['primitives']))
    findings = []
    for name in sorted(cat['axes']):
        if cat['axes'][name]['space'] == 'storage':
            continue                    # cited by the derivation (D3's storage axis, §3.4), never by a primitive shape
        if name not in cited:
            findings.append(("primitive_library", f"axis '{name}' is cited by no primitive"))
    for name in sorted(cat['precision']):
        if name not in cited:
            findings.append(("primitive_library", f"precision role '{name}' is cited by no primitive"))
    return findings


def model_advisories(cat, model_paths, schema_dir=None):
    """What the validator noticed but does not refuse: a self-indexed state on
    a fragmented stream that is not carried across fragments — reset at every
    fragment, which is valid and worth a second look. The analysis assumes the
    grammar, so a document off the schema is reported as such and not analysed:
    `--validate` is where its refusal belongs."""
    findings = []
    schema_dir = primitive_library_mod.DEFAULT_SCHEMAS if schema_dir is None else schema_dir
    for path in model_paths:
        problems = validate_mod.structural(path, schema_dir)
        if problems:
            findings.append((os.path.basename(path), f"off the schema, not analysed (--validate refuses it): {problems[0]}"))
            continue
        with open(path, encoding='utf-8') as f:
            document = json.load(f)
        if missing_assignment(document):
            continue
        result = validate_mod.analyse(path, cat)
        for line in result.get('advisories', []):
            findings.append((os.path.basename(path), line))
    return findings


def run(model_paths, primitive_library_bases, relative_to=None, models_base=None, schema_dir=None):
    """Every rule, over the given models. Always returns 0 — advisory only.
    The primitive library is the command line's, else the bases the first model declares."""
    schema_dir = primitive_library_mod.DEFAULT_SCHEMAS if schema_dir is None else schema_dir
    if not primitive_library_bases:
        with open(model_paths[0], encoding='utf-8') as f:
            primitive_library_bases = primitive_library_mod.bases_of(model_paths[0], json.load(f))
    cat = primitive_library_mod.load(*primitive_library_bases, schema_dir=schema_dir, models_base=models_base)
    findings = []
    findings += uncalled_primitives(cat, model_paths)
    findings += unreferenced_vocabulary(cat)
    findings += model_advisories(cat, model_paths, schema_dir)

    seen = set()
    for scope, message in findings:
        if (scope, message) in seen:
            continue
        seen.add((scope, message))
        print(f"  W  {scope}: {message}")
    if not seen:
        print("  nothing to report")
    else:
        print(f"  {len(seen)} advisory finding(s) — nothing blocking")
    return 0

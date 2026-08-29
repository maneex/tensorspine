"""`--lint`: hygiene. What is legal and worth knowing anyway.

Nothing here is a refusal. §8.1 makes explicit refusal the obligation of the
contract, so anything the contract can judge is already handled by
`--validate`. What is left over is outside the document's jurisdiction: the
consistency of the repository around it, and the curation of an open catalog.

A lint finding is an opinion a reasonable author may decline. That is why this
command always exits 0.
"""
import json
import os

import catalog as catalog_mod
import validate as validate_mod
from expr import missing_assignment


def _called_contracts(model_path, cat, seen=None):
    """Contract names a model calls, following templates."""
    seen = seen if seen is not None else set()
    try:
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
    except OSError:
        return seen
    names = set()
    for o in model['occurrences'].values():
        names.add(o['contract']['name'])
    for c in model['compositions'].values():
        for o in c['occurrences'].values():
            names.add(o['contract']['name'])
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        definition = cat['contracts'].get(name)
        if definition is not None and 'template' in definition:
            _called_contracts(catalog_mod.template_path(cat, definition), cat, seen)
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


def uncalled_contracts(cat, model_paths):
    """Contracts the catalog carries that none of the linted models calls.

    Not dead code: the catalog is an open vocabulary and the corpus is a
    falsification sample, not the definition. A curation question.
    """
    called = set()
    for path in model_paths:
        called |= _called_contracts(path, cat)
    return [("catalog", f"contract '{name}' {definition['version']} is in the catalog, "
                        f"called by none of the {len(model_paths)} model(s) linted")
            for name, definition in sorted(cat['contracts'].items())
            if name not in called]


def unreferenced_vocabulary(cat):
    """Axes and precision roles no contract cites."""
    cited = set(_strings(cat['contracts']))
    findings = []
    for name in sorted(cat['axes']):
        if name not in cited:
            findings.append(("catalog", f"axis '{name}' is cited by no contract"))
    for name in sorted(cat['precision']):
        if name not in cited:
            findings.append(("catalog", f"precision role '{name}' is cited by no contract"))
    return findings


def model_advisories(cat, model_paths):
    """What the validator noticed but does not refuse: a self-indexed state on
    a fragmented stream that is not carried across fragments — reset at every
    fragment, which is legal and worth a second look."""
    findings = []
    for path in model_paths:
        with open(path, encoding='utf-8') as f:
            document = json.load(f)
        if missing_assignment(document):
            continue
        result = validate_mod.analyse(path, cat)
        for line in result.get('advisories', []):
            findings.append((os.path.basename(path), line))
    return findings


def run(model_paths, catalog_bases, relative_to=None, models_base=None):
    """Every rule, over the given models. Always returns 0 — advisory only.
    The catalog is the command line's, else the bases the first model declares."""
    if not catalog_bases:
        with open(model_paths[0], encoding='utf-8') as f:
            catalog_bases = catalog_mod.bases_of(model_paths[0], json.load(f))
    cat = catalog_mod.load(*catalog_bases, models_base=models_base)
    findings = []
    findings += uncalled_contracts(cat, model_paths)
    findings += unreferenced_vocabulary(cat)
    findings += model_advisories(cat, model_paths)

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

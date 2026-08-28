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


def _called_contracts(model_path, cat, seen=None):
    """Contract names a model calls, following delegated bodies."""
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
        if definition is not None and 'model' in definition:
            _called_contracts(catalog_mod.body_path(model_path, definition), cat, seen)
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


def declared_base_mismatch(model_paths, catalog_bases, relative_to=None):
    """The `catalog` field of a model declares where its contracts live. No
    tool resolves from it — resolution comes from the command line — so the
    declaration can drift from reality without anything objecting.

    One cause, one finding: when every model drifts the same way, saying it
    twelve times hides the fact that it is a single thing to fix.
    """
    def show(p):
        p = os.path.normpath(p).rstrip('/')
        return os.path.relpath(p, relative_to) if relative_to and os.path.isabs(p) else p

    used = {show(b) for b in catalog_bases}
    drifting = {}
    for path in model_paths:
        with open(path, encoding='utf-8') as f:
            model = json.load(f)
        declared = model.get('catalog')
        if not isinstance(declared, list):
            continue
        bases = frozenset(show(b['base']) for b in declared if 'base' in b)
        if bases and not (bases & used):
            drifting.setdefault(bases, []).append(os.path.basename(path))

    findings = []
    for bases, models in sorted(drifting.items(), key=lambda kv: sorted(kv[0])):
        who = models[0] if len(models) == 1 else f"{len(models)} models"
        findings.append(("models", f"{who} declare catalog base {sorted(bases)} but "
                                   f"resolution used {sorted(used)}; nothing checks "
                                   f"this field"))
    return findings


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


def body_uri_prefix(cat, model_paths):
    """A delegated contract names its body by a dotted URI, but resolution
    keeps only the last segment and looks beside the calling model. When the
    leading segments do not match where the body actually is, the URI reads
    like a path and is not one."""
    findings = []
    for path in model_paths:
        for name in sorted(_called_contracts(path, cat)):
            definition = cat['contracts'].get(name)
            if definition is None or 'model' not in definition:
                continue
            uri = definition['model']['uri']
            prefix = '.'.join(uri.split('.')[:-1])
            if not prefix:
                continue
            actual = os.path.basename(os.path.dirname(
                os.path.abspath(catalog_mod.body_path(path, definition))))
            if prefix != actual:
                findings.append(
                    ("catalog", f"contract '{name}' points at '{uri}' but its body is "
                                f"resolved in '{actual}/'; the prefix is ignored"))
    return findings


def run(model_paths, catalog_bases, relative_to=None):
    """Every rule, over the given models. Always returns 0 — advisory only."""
    cat = catalog_mod.load(*catalog_bases)
    findings = []
    findings += declared_base_mismatch(model_paths, catalog_bases, relative_to)
    findings += uncalled_contracts(cat, model_paths)
    findings += unreferenced_vocabulary(cat)
    findings += body_uri_prefix(cat, model_paths)

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

#!/usr/bin/env python3
"""The harness document names the fields of the derived document that answer each serving
decision (docs/HARNESS.md); every field it names must exist in the derived schema, so that
the argument an engine maintainer reads cannot name a fact the products do not carry.

A field is a backticked path rooted at a product (`d4.states[].law`, `d1.nodes.*.contract`,
`d3.totals.bytes`); a backticked path starting with `.` continues the last rooted path's
container (`.access` after `d4.states[].law` is `d4.states[].access`) or, when that container
lacks it, the one container named anywhere in the document that carries it — `.visits` is
`d4.states[].visits` wherever it is written, and a name no container carries fails; ` == value`
and the like after a path are dropped. Paths are resolved through the schema's `$ref`s: `[]` steps into
`items`, `*` into `additionalProperties`.

    python3 tests/run_harness.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import schema as schema_mod            # noqa: E402

DOCUMENT = os.path.join(ROOT, 'docs', 'HARNESS.md')
SCHEMAS = os.path.join(ROOT, 'schemas')
TOKEN = re.compile(r'`([^`]+)`')
PATH = re.compile(r'^(d[1-6](\.[A-Za-z_][A-Za-z0-9_]*(\[\])?|\.\*)*|(\.[A-Za-z_][A-Za-z0-9_]*(\[\])?)+)')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def paths(text, schema):
    """(line, shown, path) for every field the document names, relative ones made absolute:
    against the last rooted path's container when it carries the field, else against the one
    container of the document that does; `path` is None when no container carries it."""
    out = []
    container = None
    containers = []
    for number, line in enumerate(text.splitlines(), 1):
        for token in TOKEN.findall(line):
            m = PATH.match(token)
            if not m:
                continue
            p = m.group(0)
            if p.startswith('.'):
                candidates = [c for c in ([container] if container else []) if resolve(schema, c + p) is not None]
                if not candidates:
                    candidates = sorted({c for c in containers if resolve(schema, c + p) is not None})
                out.append((number, p, container + p if container and resolve(schema, container + p) is not None
                            else (candidates[0] if len(candidates) == 1 else None)))
            else:
                container = p.rsplit('.', 1)[0] if '.' in p else p
                if container not in containers:
                    containers.append(container)
                out.append((number, p, p))
    return out


def resolve(schema, path):
    """The subschema a path designates, or None."""
    root = schema

    def deref(node):
        while isinstance(node, dict) and '$ref' in node:
            ref = node['$ref']
            if not ref.startswith('#/'):
                return {'$external': ref}
            node = root
            for part in ref[2:].split('/'):
                node = node[part]
        return node

    node = deref(schema)
    for step in path.split('.'):
        items = step.endswith('[]')
        name = step[:-2] if items else step
        node = deref(node)
        if name == '*':
            node = node.get('additionalProperties')
        else:
            props = node.get('properties', {})
            if name not in props:
                return None
            node = props[name]
        node = deref(node)
        if node is None:
            return None
        if items:
            node = deref(node.get('items'))
            if node is None:
                return None
    return node


def main():
    ok = True
    if not os.path.isfile(DOCUMENT):
        print(f"  skip: {os.path.relpath(DOCUMENT, ROOT)} is not in the tree yet")
        print("harness: nothing to check")
        return 0
    with open(schema_mod.locate(SCHEMAS, 'derived'), encoding='utf-8') as f:
        derived = json.load(f)
    with open(DOCUMENT, encoding='utf-8') as f:
        text = f.read()
    found = paths(text, derived)
    ok &= check(f"{len(found)} field references found in the harness document", bool(found))
    seen = set()
    for number, shown, path in found:
        if path is None:
            ok &= check(f"line {number}: `{shown}` continues a rooted path of the document", False,
                        "no container named in the document carries it, or several do")
            continue
        if path in seen:
            continue
        seen.add(path)
        ok &= check(f"{path} exists in the derived schema", resolve(derived, path) is not None, f"line {number}")
    ok &= check("the batching section reads d2.peak_live", 'd2.peak_live' in seen)
    print("harness: all good" if ok else "harness: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

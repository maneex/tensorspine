#!/usr/bin/env python3
"""Rejection tests: every case of §10.2 the tools cover must be refused, with
the code and the reason the manifest names. A case that is accepted, or
refused for another reason, fails the run.

    python3 tests/run_rejections.py            # exit 0 when every case is refused as expected
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import validate                        # noqa: E402

SCHEMAS = os.path.join(ROOT, 'schemas')
REFERENCE = os.path.join(ROOT, 'data', 'catalog')
CASES = os.path.join(HERE, 'rejections')


def model_cases():
    with open(os.path.join(CASES, 'models.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    cat = catalog_mod.load(REFERENCE)
    for case in manifest['cases']:
        path = os.path.join(CASES, case['document'])
        if case['expect'] == 'schema':
            lines = validate.structural(path, SCHEMAS)
        else:
            lines = validate.structural(path, SCHEMAS)
            if not lines:
                lines, _stats = validate.semantic(path, cat)
                lines = [l for l in lines if l.startswith(f"[{case['expect']}]")]
        hit = [l for l in lines if case['match'] in l]
        yield case['document'], bool(hit), lines[:3]


def catalog_cases():
    with open(os.path.join(CASES, 'catalog.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    for case in manifest['cases']:
        base = os.path.join(CASES, case['base'])
        try:
            catalog_mod.load(base, REFERENCE)
            yield case['base'], False, ["accepted"]
        except catalog_mod.CatalogError as e:
            text = str(e)
            yield case['base'], case['match'] in text, text.splitlines()[:3]


def main():
    failed = 0
    total = 0
    for name, ok, lines in list(model_cases()) + list(catalog_cases()):
        total += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failed += 1
            for l in lines:
                print(f"         {l}")
    print(f"{total - failed}/{total} rejected as expected" + (f", {failed} not" if failed else ""))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""The status page (`tensorspine --document status`, generated at site build and never tracked)
states nothing the tools do not compute: every number on it equals what the suites compute from
the same tools — the catalog's counts from `catalog.py`, the corpus's from the models directory
and the validator, a generator's coverage from `capabilities.coverage()` and its unwitnessed
contracts from `capabilities.unwitnessed()` (Specification §10.2).

Skips, and says so, when `tools/status.py` is not in the tree.

    python3 tests/run_status.py
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

SCHEMAS = os.path.join(ROOT, 'schemas')
MODELS = os.path.join(ROOT, 'data', 'models')
MANIFESTS = [os.path.join(ROOT, 'generators', 'reference', 'capabilities.json'),
             os.path.join(ROOT, 'generators', 'zml', 'capabilities.json')]


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def tables(text):
    """Every Markdown table of the page: (headers, rows) with cells stripped of backticks."""
    out, lines = [], text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('|') and i + 1 < len(lines) and re.match(r'^\|(\s*-+\s*\|)+\s*$', lines[i + 1]):
            headers = [c.strip().strip('`') for c in lines[i].strip().strip('|').split('|')]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip().strip('`') for c in lines[i].strip().strip('|').split('|')])
                i += 1
            out.append((headers, rows))
        else:
            i += 1
    return out


def table_with(found, *headers):
    for hs, rows in found:
        if all(h in hs for h in headers):
            return hs, rows
    return None, None


def section(text, title):
    """The text of one `### \\`name\\`` section."""
    m = re.search(r'^### `' + re.escape(title) + r'`\n(.*?)(?=^### |^## |\Z)', text, re.M | re.S)
    return m.group(1) if m else ''


def main():
    if not os.path.isfile(os.path.join(ROOT, 'tools', 'status.py')):
        print("  skip: tools/status.py is not in the tree")
        print("status: nothing to check")
        return 0
    import capabilities
    import catalog as catalog_mod
    import derive
    import status
    import validate
    ok = True
    corpus = sorted(glob.glob(os.path.join(MODELS, '*.json')))
    manifests = [m for m in MANIFESTS if os.path.isfile(m)]
    state = status.facts(corpus, None, SCHEMAS, manifests)
    text = status.render_status(state)
    found = tables(text)
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))

    # the catalog and the corpus, counted here
    valid = located = 0
    for path in corpus:
        errors, _ = validate.semantic(path, cat)
        valid += not errors
        if not errors:
            located += all('location' in t for t in derive.products(path, cat)['d3']['tensors'])
    hs, rows = table_with(found, 'Catalog contracts', 'Concrete documents')
    ok &= check("the page has the catalog-and-corpus table", rows is not None and len(rows) == 1)
    if rows:
        row = dict(zip(hs, rows[0]))
        want = {'Catalog contracts': len(cat['contracts']), 'Axes': len(cat['axes']), 'Precision roles': len(cat['precision']),
                'Concrete documents': len(corpus), 'Valid as written': valid, 'Fully located': located,
                'Templates': len(cat['templates'])}
        for k, v in want.items():
            ok &= check(f"catalog and corpus: {k} = {v}", row.get(k) == str(v), f"page says {row.get(k)!r}")

    # every generator's coverage, computed here from the same reader
    for path in manifests:
        manifest, errors = capabilities.load(path)
        errors += capabilities.names(manifest, cat)
        if errors:
            ok &= check(f"{os.path.relpath(path, ROOT)}: readable", False, errors[0])
            continue
        name = manifest['generator']['name']
        missing, branches, verdicts = capabilities.coverage(manifest, cat, corpus)
        unwitnessed = capabilities.unwitnessed(manifest, cat)
        hs, rows = table_with(tables(section(text, name)), 'Contract entries')
        ok &= check(f"{name}: the page has its coverage table", rows is not None and len(rows) == 1)
        if not rows:
            continue
        row = dict(zip(hs, rows[0]))
        can_run = sum(1 for good, _ in verdicts.values() if good)
        want = {'Contract entries': str(len(manifest['contracts'])),
                'Contracts without an entry': str(len(missing)),
                'Entries with branch gaps': str(len([c for c in branches if c not in missing])),
                'Corpus documents runnable': f"{can_run}/{len(verdicts)}"}
        for k, v in want.items():
            ok &= check(f"{name}: {k} = {v}", row.get(k) == v, f"page says {row.get(k)!r}")
        shown = row.get('Contracts without a witness')
        if unwitnessed is None:
            ok &= check(f"{name}: a conformer's manifest witnesses nothing, and the page states no count",
                        shown is not None and not shown.isdigit(), f"page says {shown!r}")
        else:
            ok &= check(f"{name}: Contracts without a witness = {len(unwitnessed)}", shown == str(len(unwitnessed)), f"page says {shown!r}")
        hs2, rows2 = table_with(tables(section(text, name)), 'Document', 'Verdict')
        ok &= check(f"{name}: the admission table has one row per corpus document with the reader's verdict",
                    rows2 is not None and len(rows2) == len(verdicts)
                    and all(dict(zip(hs2, r)).get('Verdict') == ('can run' if verdicts[dict(zip(hs2, r))['Document']][0] else 'cannot run')
                            for r in rows2 if dict(zip(hs2, r)).get('Document') in verdicts))
    print("status: all good" if ok else "status: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

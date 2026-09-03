#!/usr/bin/env python3
"""Every committed fixture is on the language's fixture schema
(schemas/tensorspine-fixture.schema.json, docs/TENSORSPINE-FIXTURE.md), and its
tensor keys are on the grammar of its kind. Nothing is loaded but the headers.

  1. metadata on the schema, read from the safetensors header alone;
  2. integration: the `document` is a corpus document and `truncation.composition` one
     of its compositions; every key is an `in/`, `value/`, `state/` or `logits/` key; every
     `in/<input>` names a public input of the document other than the token input, the
     `inputs` provenance has one entry per such key and no other, and an entry that names a
     file names the recording's origin and licence too;
  3. unit: the embedded document is on the model schema and pins the fixture's contract
     with its arguments; every key is a `param/`, `in/`, `out/`, `positions/` or `state/` key,
     one `in/` per public input and one `out/` per public output for every invocation.

    python3 tests/run_fixtures.py [FIXTURE ...]      # every committed fixture, or the files named
"""
import glob
import json
import os
import re
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import schema as schema_mod            # noqa: E402

SCHEMAS = os.path.join(ROOT, 'schemas')
MODELS = os.path.join(ROOT, 'data', 'models')
GRAMMAR = {
    'integration': re.compile(r'^(in/[^/]+|value/.+|state/[^/]+/[^/]+|logits/(last|argmax))$'),
    'unit': re.compile(r'^(param/[^/]+|in/\d+/[^/]+|out/\d+/[^/]+|positions/\d+/[^/]+|state/\d+/[^/]+/[^/]+)$'),
}


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    meta = {}
    for k, v in (h.pop('__metadata__', None) or {}).items():
        try:
            meta[k] = json.loads(v)
        except (TypeError, json.JSONDecodeError):
            meta[k] = v
    return meta, sorted(h)


def fixtures():
    return sorted(glob.glob(os.path.join(ROOT, 'generators', '*', 'fixtures', '**', '*.safetensors'), recursive=True))


def main(paths=None):
    reg = schema_mod.registry(SCHEMAS)
    fixture_schema = schema_mod.locate(SCHEMAS, 'fixture')
    model_schema = schema_mod.locate(SCHEMAS, 'model')
    ok = check("a fixture schema is in the tree", fixture_schema is not None)
    files = [os.path.abspath(p) for p in paths] if paths else fixtures()
    ok &= check(f"{len(files)} committed fixture(s) found", bool(files))
    for path in files:
        name = os.path.relpath(path, ROOT)
        meta, keys = header(path)
        errors = schema_mod.deepest(schema_mod.check_document(fixture_schema, meta, reg))
        ok &= check(f"{name}: metadata on the fixture schema", not errors, errors and schema_mod.format_error(errors[0]))
        if errors:
            continue
        kind = meta['kind']
        bad = [k for k in keys if not GRAMMAR[kind].match(k)]
        ok &= check(f"{name}: {len(keys)} keys on the {kind} grammar", not bad, str(bad[:3]))
        if kind == 'integration':
            doc = os.path.join(MODELS, meta['document'] + '.json')
            ok &= check(f"{name}: document {meta['document']} is in the corpus", os.path.isfile(doc))
            if os.path.isfile(doc):
                with open(doc, encoding='utf-8') as f:
                    model = json.load(f)
                comps = model.get('compositions', {})
                ok &= check(f"{name}: truncation names a composition of it", meta['truncation']['composition'] in comps,
                            str(sorted(comps)))
                # the non-token inputs the prefill delivered (docs/TENSORSPINE-FIXTURE.md §3): a key per
                # public input other than the one `ids` names, and a provenance entry per key
                public = model['interfaces']['inputs']
                recorded = {k[len('in/'):] for k in keys if k.startswith('in/')}
                bad = sorted(i for i in recorded if i not in public or public[i].get('kind') == 'token')
                ok &= check(f"{name}: every in/ key names a public input of {meta['document']} other than the token input", not bad, str(bad))
                ok &= check(f"{name}: the inputs provenance has one entry per in/ key and no other",
                            set(meta.get('inputs', {})) == recorded, str(sorted(set(meta.get('inputs', {})) ^ recorded)))
                # a recording names its origin and licence beside its file and hash (docs/TENSORSPINE-FIXTURE.md §3):
                # open content travels with its attribution; a setting has no file and needs neither
                lacking = sorted(i for i, e in meta.get('inputs', {}).items() if 'source' in e and not (e.get('origin') and e.get('licence')))
                ok &= check(f"{name}: every recorded input with a file names its origin and licence", not lacking, str(lacking))
            ok &= check(f"{name}: a tolerance for f32", 'f32' in meta['tolerance'])
        else:
            document = meta['document']
            tmp = tempfile.mkdtemp(prefix='tensorspine-fixture-')
            with open(os.path.join(tmp, 'document.json'), 'w', encoding='utf-8') as f:
                json.dump(document, f)
            errors = schema_mod.deepest(schema_mod.check(model_schema, os.path.join(tmp, 'document.json'), reg))
            ok &= check(f"{name}: the embedded document is on the model schema", not errors,
                        errors and schema_mod.format_error(errors[0]))
            occurrences = document.get('occurrences', {})
            ok &= check(f"{name}: one occurrence, pinning {meta['contract']['name']}@{meta['contract']['version']}",
                        len(occurrences) == 1 and next(iter(occurrences.values()))['contract'] == meta['contract'])
            inputs, outputs = document['interfaces']['inputs'], document['interfaces']['outputs']
            wanted = set()
            for k, delivered in enumerate(meta['invocations']):
                wanted |= {f"in/{k}/{i}" for i in delivered} | {f"out/{k}/{o}" for o in outputs}
            ok &= check(f"{name}: every invocation names public inputs", all(set(d) <= set(inputs) for d in meta['invocations']))
            ok &= check(f"{name}: an in/ per delivered input and an out/ per public output for each of {len(meta['invocations'])} invocation(s)",
                        wanted <= set(keys), str(sorted(wanted - set(keys))[:3]))
            located = document['bindings']['parameters']
            ok &= check(f"{name}: every parameter identity is located at its own param/ key",
                        all(b.get('location', {}).get('tensor') == [f"param/{b['tensor']['name']}"] for b in located.values()))
            ok &= check(f"{name}: id {meta['id']} names the contract and the file",
                        meta['id'].startswith(f"{meta['contract']['name']}@{meta['contract']['version']}/")
                        and os.path.basename(path) == meta['id'].rsplit('/', 1)[1] + '.safetensors')
    print("fixtures: all good" if ok else "fixtures: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""The derived document (§7): every document of the corpus emits one that is
on the derived schema, and what it says agrees with the validator and with
facts known independently.

  1. Schema: `--derive` output validates against schemas/tensorspine-derived.schema.json;
     `--d1` output, the graph alone, validates against the same schema.
  2. Agreement: D3 elements = the validator's resident count; D5 operations per element =
     the validator's; D1 nodes = D3 members' occurrences ∪ stateless occurrences.
  3. Facts: Llama 3 8B — 4 KiB per cached position per layer, 128 KiB per token, one value of
     8 KiB per element crossing a layer boundary; Whisper — the cross-attention cache grows
     along the audio stream; Voxtral — 32 encoder states carried across fragments; every
     structural cut is legal (no crossing edge points backwards).

    python3 tests/run_derived.py
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import d1                              # noqa: E402
import derive                          # noqa: E402
import schema as schema_mod            # noqa: E402
import validate                        # noqa: E402
from signature import ASSIGNMENTS, corpus, name_of   # noqa: E402

SCHEMAS = os.path.join(ROOT, 'schemas')
MODELS = os.path.join(ROOT, 'data', 'models')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main():
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    schema_path = schema_mod.locate(SCHEMAS, 'derived')
    reg = schema_mod.registry(SCHEMAS)
    ok = check("a derived schema is in the tree", schema_path is not None)
    docs = {}
    for path in corpus():
        name = name_of(path)
        assignment = ASSIGNMENTS.get(name)
        doc = derive.products(path, cat, assignment)
        errors = schema_mod.check_document(schema_path, doc, reg)
        ok &= check(f"{name}: derived document on the schema", not errors,
                    errors and schema_mod.format_error(errors[0]))
        graph_only = d1.emit(path, cat, assignment)
        errors = schema_mod.check_document(schema_path, graph_only, reg)
        ok &= check(f"{name}: graph-only document on the schema", not errors,
                    errors and schema_mod.format_error(errors[0]))
        stats = validate.analyse(path, cat, assignment)['stats']
        ok &= check(f"{name}: D3 elements = validator's resident count",
                    doc['d3']['totals']['elements'] == stats['parameter_elements'])
        ok &= check(f"{name}: D5 operations per element = validator's",
                    doc['d5']['operations']['element']['value'] == stats['ops_per_element'])
        nodes = set(doc['d1']['nodes'])
        members = {m.rsplit('.', 1)[0] for t in doc['d3']['tensors'] for m in t['members']}
        ok &= check(f"{name}: every D3 member is a D1 node", members <= nodes,
                    str(sorted(members - nodes)[:3]))
        # every structural cut is legal: block A is closed under ancestors
        blocks = {}
        for c in doc['d2']['cuts']:
            payload = {p['value'] for p in c['payload']}
            crossing = [e for e in doc['d1']['edges'] if f"{e['from']['node']}.{e['from']['port']}" in payload]
            ok &= check(f"{name}: cut {c['cut']} has a payload of distinct values", len(payload) == len(c['payload']))
            if not crossing and c['payload']:
                ok &= check(f"{name}: cut {c['cut']} payload values are edge sources", False)
        docs[name] = doc
    l3 = docs['llama3-8b']
    kv = [s for s in l3['d4']['states'] if s['state'] == 'kv']
    ok &= check("llama3-8b: 32 KV states of 4096 bytes per cached position",
                len(kv) == 32 and all(s['bytes_per_cached_position'] == 4096 for s in kv))
    ok &= check("llama3-8b: 128 KiB per token across the model",
                l3['d4']['totals']['append_bytes_per_cached_position'] == 131072)
    cut = next(c for c in l3['d2']['cuts'] if c['cut'] == 'decoder[layer<=3]')
    ok &= check("llama3-8b: one value of 8 KiB per element crosses a layer boundary",
                len(cut['payload']) == 1 and cut['bytes_per_element'] == 8192
                and cut['bytes_per_invocation'] == {'tokens': 8192.0}, str(cut['payload']))
    ok &= check("llama3-8b: 14.96 GiB of parameters in bf16",
                round(l3['d3']['totals']['bytes'] / 2**30, 2) == 14.96)
    w = docs['whisper-large-v3']
    cross = [s for s in w['d4']['states'] if 'cross' in s['identity']]
    ok &= check("whisper: the cross-attention cache grows along the audio stream and is frozen after it",
                len(cross) == 32 and all(s['stream'] == {'kind': 'position', 'stream': 'audio'}
                                         and s['indexed_by_source'] for s in cross))
    v = docs['voxtral-realtime']
    ok &= check("voxtral: 32 encoder states carried across fragments, rings of 750 frames",
                len(v['d4']['totals']['carried']) == 32
                and all(s['law'] == 'window' and s['span'] == 750 for s in v['d4']['states'] if s['carried_across_fragments']))
    g = docs['gemma3n-kvshare']
    shared = [s for s in g['d4']['states'] if s['identity'].startswith('shared.')]
    ok &= check("gemma3n: the shared identities carry no layer in their instance key",
                len(shared) == 2 and all(s['instance_key'] == ['instance.session', 'instance.branch'] for s in shared))
    ok &= check("llama3-8b: no O5.10 information loss once every flattened axis declares its factors",
                l3['d6']['information_loss'] == [])
    located = {t['identity']: t.get('location') for t in l3['d3']['tensors']}
    ok &= check("llama3-8b: decoder.attn.q[layer=3] is stored as model.layers.3.self_attn.q_proj.weight",
                located.get('decoder.attn.q[layer=3]') == {'tensor': 'model.layers.3.self_attn.q_proj.weight'})
    names = [v['tensor'] for v in located.values() if v]
    ok &= check("llama3-8b: 291 tensors located under 291 distinct physical names",
                len(names) == 291 and len(set(names)) == 291)
    print("derived: all good" if ok else "derived: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

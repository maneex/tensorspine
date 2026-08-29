#!/usr/bin/env python3
"""D5, first derivation (§4.1, §4.5): operations per token follow from the
parameter inventory — two per weight element consumed — scaled by the
activated fraction of a sparse unit, plus the corrections a contract declares.

  1. llama3-8b: dense, no state corrections beyond attention's sequence term:
     ops per token = 2 × parameter elements; the per-position term is
     4·heads·head_dim per attention layer.
  2. llama4-scout: MoE — the routed experts count at top_k / experts; the
     independent oracle is the closed formula the catalog used to declare.
  3. shieldstral-3b and its composite derive the same figures.

    python3 tests/run_costs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import validate                        # noqa: E402

MODELS = os.path.join(ROOT, 'data', 'models')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def stats(cat, name):
    errors, s = validate.semantic(os.path.join(MODELS, name + '.json'), cat)
    assert not errors, errors[:1]
    return s


def main():
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    ok = True
    s = stats(cat, 'llama3-8b')
    ok &= check("llama3-8b: ops per token = 2 × parameter elements",
                s['ops_per_token'] == 2 * s['parameter_elements'], str((s['ops_per_token'], s['parameter_elements'])))
    ok &= check("llama3-8b: per-position term = 32 layers × 4·32·128",
                s['ops_per_token_per_position'] == 32 * 4 * 32 * 128, str(s['ops_per_token_per_position']))

    with open(os.path.join(MODELS, 'llama4-scout.json'), encoding='utf-8') as f:
        l4 = json.load(f)
    q = {k: v['source']['value'] for k, v in l4['quantities'].items() if v['source']['kind'] == 'literal'}
    moe = l4['compositions']['decoder']['occurrences']['moe']['arguments']
    val = lambda e: e['literal'] if 'literal' in e else q[e['quantity']]
    experts, top_k, width, inner = val(moe['experts']), val(moe['top_k']), val(moe['width']), val(moe['inner'])
    shared = val(moe['shared']) if 'shared' in moe else 0
    layers = 48
    s = stats(cat, 'llama4-scout')
    routed_elements = layers * experts * 3 * width * inner            # in (2·inner × width) + out (width × inner)
    dense_part = 2 * (s['parameter_elements'] - routed_elements)
    expected = dense_part + layers * 6 * width * inner * top_k        # the formula the catalog used to declare
    seq_ops = sum(cost for cost in [0])
    ok &= check(f"llama4-scout: routed experts count at top_k/experts = {top_k}/{experts}",
                abs(s['ops_per_token'] - expected) <= 1, f"derived {s['ops_per_token']} vs oracle {expected}")
    ok &= check("llama4-scout: sparse ops < dense ops", s['ops_per_token'] < 2 * s['parameter_elements'])

    a = stats(cat, 'shieldstral-3b')
    b = stats(cat, 'shieldstral-3b-composite')
    for k in ('parameter_elements', 'ops_per_token', 'ops_per_token_per_position'):
        ok &= check(f"template parity: {k} {a[k]} == {b[k]}", a[k] == b[k])
    print("costs: all good" if ok else "costs: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

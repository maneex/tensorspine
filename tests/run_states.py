#!/usr/bin/env python3
"""State identities (§3.4, §4.4): what the validator derives from a document.

  1. llama3-8b: 32 identities, each keyed (layer, session, branch).
  2. gemma3n-kvshare: 20 identities from 30 slots; the two shared identities carry
     no layer index — sharing is several members under one identity, nothing else.
  3. voxtral-realtime: the encoder state keeps its invocation boundary.

    python3 tests/run_states.py
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


def main():
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    ok = True
    r = validate.analyse(os.path.join(MODELS, 'llama3-8b.json'), cat)
    keys = r['instance_keys']
    ok &= check("llama3-8b: 32 instance keys", len(keys) == 32, str(len(keys)))
    ok &= check("llama3-8b: every key is (layer, session, branch)",
                all(v == ('layer', 'instance.session', 'instance.branch') for v in keys.values()),
                str(set(keys.values())))
    r = validate.analyse(os.path.join(MODELS, 'gemma3n-kvshare.json'), cat)
    keys = r['instance_keys']
    shared = [k for k, v in keys.items() if v == ('instance.session', 'instance.branch')]
    ok &= check("gemma3n: 30 slots, 20 identities", r['stats']['state_slots'] == 30
                and r['stats']['state_identities'] == 20, str(r['stats']))
    ok &= check("gemma3n: exactly the two shared identities have no layer index",
                sorted(shared) == ['shared.full.kv', 'shared.sliding.kv'], str(shared))
    with open(os.path.join(MODELS, 'voxtral-realtime.json'), encoding='utf-8') as f:
        vox = json.load(f)
    boundaries = {sid: b['invocation_boundary'] for sid, b in vox['bindings']['states'].items()
                  if 'invocation_boundary' in b}
    ok &= check("voxtral: one state keeps an invocation boundary over the audio fragment domain",
                len(boundaries) == 1 and list(boundaries.values())[0]['domain'] == {'kind': 'fragment', 'source': 'audio'},
                str(boundaries))
    print("states: all good" if ok else "states: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

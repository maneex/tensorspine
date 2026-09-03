#!/usr/bin/env python3
"""State identities (§3.4, §4.4): what the validator derives from a document.

  1. llama3-8b: 32 identities, each keyed (layer, session, branch).
  2. gemma3n-kvshare: 20 identities from 30 slots; the two shared identities carry
     no layer index — sharing is several members under one identity, nothing else.
  3. voxtral-realtime: the encoder state is carried across the fragments of `audio` — derived
     from the contract's carrying condition and the input's fragmentation, declared nowhere —
     and so are the front end's two convolution histories, indexed by the frames' stream (V18),
     and the decoder's rings, since the token input joins the fragmented stream (§2.3).

    python3 tests/run_states.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import model as model_mod              # noqa: E402
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
    r = validate.analyse(os.path.join(MODELS, 'voxtral-realtime.json'), cat)
    ok &= check("voxtral: the encoder attention state, the front end's two histories and the decoder attention state are carried "
                "across the fragments of `audio` — the decoder's, at kind token, since the token input joins that stream (V18)",
                r['carried'] == {'encoder.attn.kv': ('position', 'audio'), 'conv_frontend.conv1_history': ('position', 'audio'),
                                 'conv_frontend.conv2_history': ('position', 'audio'), 'decoder.attn.kv': ('token', 'audio')}, str(r['carried']))
    ok &= check("voxtral: nothing else is carried, and no advisory: every self-indexed state on the fragmented stream is carried",
                not r['advisories'], str(r['advisories']))
    print("states: all good" if ok else "states: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

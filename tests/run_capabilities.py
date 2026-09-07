#!/usr/bin/env python3
"""The capabilities reader (generators/CAPABILITIES.md) on the committed manifests: the witness
binding (Specification §4.1, O1.3), the release rule (§10.2) and the branch ledger.

  1. The manifests are every `generators/*/capabilities.json` in the tree (`capabilities.manifests()`),
     exactly one of them the witness's, first; a tree with two witnesses, or none, is refused.
  2. The witness manifest (the reference's): every entry carries a witness block naming a kernel
     that exists and fixtures that exist; the primitives without a witness are exactly the
     primitive_library's primitive versions without an entry, and `--coverage --strict` exits 1 while there
     is one.
  3. Every other manifest is a conformer's: no role, no witness block; a witness block added to
     it is refused by the reader.
  3. The branch ledger lists, for a primitive without an entry, every branch of its arguments —
     the enum values, both values of a boolean, a record's presence and its fields — so the to-do
     list per model-and-generator pair is complete; for an entry, the branches it does not admit.

    python3 tests/run_capabilities.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import capabilities                    # noqa: E402
import primitive_library as primitive_library_mod          # noqa: E402
import derive                          # noqa: E402



def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main():
    cat = primitive_library_mod.load(os.path.join(ROOT, 'data', 'primitive-library'))
    ok = True
    paths = capabilities.manifests(ROOT)
    REFERENCE, conformers = paths[0], paths[1:]
    ok &= check(f"the manifests come from the tree, the witness first: {[os.path.relpath(p, ROOT) for p in paths]}",
                REFERENCE.endswith(os.path.join('reference', 'capabilities.json')) and len(conformers) >= 2)
    tmp = tempfile.mkdtemp(prefix='tensorspine-capabilities-')
    for name in ('a', 'b'):
        os.makedirs(os.path.join(tmp, 'generators', name))
        with open(os.path.join(tmp, 'generators', name, 'capabilities.json'), 'w', encoding='utf-8') as f:
            json.dump({'role': 'witness'}, f)
    for forged_root, label in ((tmp, 'two witness manifests'), (os.path.join(tmp, 'generators'), 'no manifest at all')):
        try:
            capabilities.manifests(forged_root)
            ok &= check(f"a tree with {label} is refused", False, "accepted")
        except ValueError as e:
            ok &= check(f"a tree with {label} is refused, naming the rule", '§4.1' in str(e), str(e)[:120])
    manifest, errors = capabilities.load(REFERENCE)
    errors += capabilities.names(manifest, cat) + capabilities.witness_problems(manifest, os.path.dirname(REFERENCE))
    ok &= check("reference: the manifest validates, names resolve, witness kernels and fixtures exist", not errors, errors[:2])
    ok &= check("reference: a witness manifest", manifest.get('role') == 'witness')
    ok &= check("reference: every entry carries a witness block with a tolerance for f32",
                all('f32' in e.get('witness', {}).get('tolerance', {}) for e in manifest['primitives'].values()))
    missing, branches, _ = capabilities.coverage(manifest, cat, [])
    without = capabilities.unwitnessed(manifest, cat)
    ok &= check(f"reference: the {len(without)} primitives without a witness are the {len(missing)} without an entry",
                without == missing, str(sorted(set(without) ^ set(missing))[:3]))
    primitives = sorted(f"{n}@{v}" for (n, v), d in cat['by_id'].items() if 'template' not in d)
    ok &= check("reference: witnessed plus unwitnessed is every primitive definition of the primitive_library",
                sorted(list(manifest['primitives']) + without) == primitives)
    ok &= check("ledger: a primitive without an entry lists every branch of its arguments — patch_embed's bias, both ways",
                'patch_embed@1.0.0' in branches and {'bias=True', 'bias=False'} <= set(branches['patch_embed@1.0.0']),
                str(branches.get('patch_embed@1.0.0')))
    ok &= check("ledger: a record argument of a missing primitive lists its presence and its fields' branches",
                any(g.endswith('=present') for g in branches.get('attention.latent_compressed@1.0.0', [])),
                str(branches.get('attention.latent_compressed@1.0.0', [])[:6]))
    ok &= check("ledger: an entry lists only the branches it does not admit — attention.dense's mask=chunked, not mask=causal",
                'mask=chunked' in branches.get('attention.dense@1.0.0', []) and 'mask=causal' not in branches.get('attention.dense@1.0.0', []))
    # the three forms of a combination limit (generators/CAPABILITIES.md), on hand-built entries so
    # the mechanism is tested apart from any one manifest
    flat = {'arguments': {'cross': [True, False], 'mask': ['causal', 'none']}, 'excluding': [{'cross': True, 'mask': 'causal'}]}
    ok &= check("excluding, flat: a matching combination is refused, a non-matching one is not",
                capabilities.supports(flat, {'cross': True, 'mask': 'causal'}) and not capabilities.supports(flat, {'cross': True, 'mask': 'none'}))
    rope_pred = {'when': {'all': [{'compare': {'operator': 'equal', 'left': {'argument': 'cross'}, 'right': {'literal': True}}},
                                  {'present': 'rope'}]}, 'reason': 'cross attention with rope: the source positions are not delivered'}
    pred = {'arguments': {'cross': [True, False], 'rope': {'absent': True, 'fields': {'theta': 'any'}}}, 'excluding': [rope_pred]}
    ok &= check("excluding, predicate: cross with rope is refused with its reason, cross without rope is not",
                capabilities.supports(pred, {'cross': True, 'rope': {'theta': 1.0}}) == ['cross attention with rope: the source positions are not delivered']
                and not capabilities.supports(pred, {'cross': True}))
    # conditions: a {when, note} entry reports the note for an admitted instance whose `when` holds
    gemma = os.path.join(ROOT, 'data', 'models', 'gemma3n-kvshare.json')
    gc = primitive_library_mod.load_for(gemma, json.load(open(gemma, encoding='utf-8')))
    gdoc = derive.products(gemma, gc)
    probe = json.loads(json.dumps(manifest))
    probe['primitives']['attention.dense@1.0.0']['conditions'] = [{
        'when': {'all': [{'compare': {'operator': 'equal', 'left': {'argument': 'kv_source'}, 'right': {'literal': 'shared'}}}, {'present': 'window'}]},
        'note': 'one position per invocation once the ring has wrapped (finding 26)'}]
    reported = capabilities.conditions(probe, gdoc, gc)
    ok &= check("conditions: the shared-window note is reported for every reader that runs under it, and for no other instance",
                reported and all('finding 26' in n for _n, _c, n in reported)
                and all(gdoc['d1']['nodes'][node]['arguments'].get('kv_source') == 'shared' for node, _c, _n in reported),
                str(reported[:1]))
    # names(): a predicate reading an argument the primitive does not declare is refused at load
    bad = json.loads(json.dumps(manifest))
    bad['primitives']['norm.rms@1.0.0']['excluding'] = [{'when': {'compare': {'operator': 'equal', 'left': {'argument': 'nonexistent'}, 'right': {'literal': 1}}}, 'reason': 'x'}]
    errs = capabilities.names(bad, cat)
    ok &= check("names: an excluding predicate on an undeclared argument is refused, naming it",
                any("undeclared argument 'nonexistent'" in e for e in errs), str(errs[:2]))
    cli = [os.path.join(ROOT, 'tools', 'tensorspine'), '--capabilities', REFERENCE, '--coverage', '--strict',
           os.path.join(ROOT, 'data', 'models', 'llama3-8b.json')]
    run = subprocess.run(cli, capture_output=True, text=True)
    ok &= check("reference: --coverage --strict exits 1 while a primitive version has no witness (§10.2)",
                run.returncode == 1 and '--strict: refused' in run.stdout, run.stdout[-300:])
    for path in conformers:
        name = os.path.basename(os.path.dirname(path))
        conformer, errors = capabilities.load(path)
        errors += capabilities.names(conformer, cat) + capabilities.witness_problems(conformer, os.path.dirname(path))
        ok &= check(f"{name}: a conformer's manifest — validates, names resolve, no role, no witness block",
                    not errors and 'role' not in conformer and not any('witness' in e for e in conformer['primitives'].values()), errors[:1])
        ok &= check(f"{name}: a conformer witnesses nothing", capabilities.unwitnessed(conformer, cat) is None)
        forged = json.loads(json.dumps(conformer))
        first = next(iter(forged['primitives']))
        forged['primitives'][first]['witness'] = {'kernel': 'primitives/x', 'tolerance': {'f32': {'atol': 0, 'rtol': 0}}, 'fixtures': []}
        problems = capabilities.witness_problems(forged, os.path.dirname(path))
        ok &= check(f"{name}: a witness block in a conformer's manifest is refused", len(problems) == 1 and 'role is conformer' in problems[0], problems[:1])
    forged = json.loads(json.dumps(manifest))
    forged['primitives']['norm.rms@1.0.0']['witness']['fixtures'] = ['norm.rms@1.0.0/nowhere']
    problems = capabilities.witness_problems(forged, os.path.dirname(REFERENCE))
    ok &= check("reference: a fixture the manifest names and the tree lacks is refused", len(problems) == 1 and 'nowhere' in problems[0], problems[:1])
    print("capabilities: all good" if ok else "capabilities: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

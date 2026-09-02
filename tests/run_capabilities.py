#!/usr/bin/env python3
"""The capabilities reader (generators/CAPABILITIES.md) on the committed manifests: the witness
binding (Specification §4.1, O1.3), the release rule (§10.2) and the branch ledger.

  1. The reference manifest is a witness manifest: every entry carries a witness block naming a
     kernel that exists and fixtures that exist; the contracts without a witness are exactly the
     catalog's contract versions without an entry, and `--coverage --strict` exits 1 while there
     is one.
  2. The ZML manifest is a conformer's: no role, no witness block; a witness block added to it is
     refused by the reader.
  3. The branch ledger lists, for a contract without an entry, every branch of its arguments —
     the enum values, both values of a boolean, a record's presence and its fields — so the to-do
     list per model-and-generator pair is complete; for an entry, the branches it does not admit.

    python3 tests/run_capabilities.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import capabilities                    # noqa: E402
import catalog as catalog_mod          # noqa: E402

REFERENCE = os.path.join(ROOT, 'generators', 'reference', 'capabilities.json')
ZML = os.path.join(ROOT, 'generators', 'zml', 'capabilities.json')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main():
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    ok = True
    manifest, errors = capabilities.load(REFERENCE)
    errors += capabilities.names(manifest, cat) + capabilities.witness_problems(manifest, os.path.dirname(REFERENCE))
    ok &= check("reference: the manifest validates, names resolve, witness kernels and fixtures exist", not errors, errors[:2])
    ok &= check("reference: a witness manifest", manifest.get('role') == 'witness')
    ok &= check("reference: every entry carries a witness block with a tolerance for f32",
                all('f32' in e.get('witness', {}).get('tolerance', {}) for e in manifest['contracts'].values()))
    missing, branches, _ = capabilities.coverage(manifest, cat, [])
    without = capabilities.unwitnessed(manifest, cat)
    ok &= check(f"reference: the {len(without)} contracts without a witness are the {len(missing)} without an entry",
                without == missing, str(sorted(set(without) ^ set(missing))[:3]))
    primitives = sorted(f"{n}@{v}" for (n, v), d in cat['by_id'].items() if 'template' not in d)
    ok &= check("reference: witnessed plus unwitnessed is every primitive contract of the catalog",
                sorted(list(manifest['contracts']) + without) == primitives)
    ok &= check("ledger: a contract without an entry lists every branch of its arguments — patch_embed's bias, both ways",
                'patch_embed@1.0.0' in branches and {'bias=True', 'bias=False'} <= set(branches['patch_embed@1.0.0']),
                str(branches.get('patch_embed@1.0.0')))
    ok &= check("ledger: a record argument of a missing contract lists its presence and its fields' branches",
                any(g.endswith('=present') for g in branches.get('attention.latent_compressed@1.0.0', [])),
                str(branches.get('attention.latent_compressed@1.0.0', [])[:6]))
    ok &= check("ledger: an entry lists only the branches it does not admit — attention.dense's mask=chunked, not mask=causal",
                'mask=chunked' in branches.get('attention.dense@1.0.0', []) and 'mask=causal' not in branches.get('attention.dense@1.0.0', []))
    cli = [os.path.join(ROOT, 'tools', 'tensorspine'), '--capabilities', REFERENCE, '--coverage', '--strict',
           os.path.join(ROOT, 'data', 'models', 'llama3-8b.json')]
    run = subprocess.run(cli, capture_output=True, text=True)
    ok &= check("reference: --coverage --strict exits 1 while a contract version has no witness (§10.2)",
                run.returncode == 1 and '--strict: refused' in run.stdout, run.stdout[-300:])
    zml, errors = capabilities.load(ZML)
    errors += capabilities.witness_problems(zml, os.path.dirname(ZML))
    ok &= check("zml: a conformer's manifest — no role, no witness block, accepted",
                not errors and 'role' not in zml and not any('witness' in e for e in zml['contracts'].values()), errors[:1])
    ok &= check("zml: a conformer witnesses nothing", capabilities.unwitnessed(zml, cat) is None)
    forged = json.loads(json.dumps(zml))
    first = next(iter(forged['contracts']))
    forged['contracts'][first]['witness'] = {'kernel': 'primitives/x.zig', 'tolerance': {'f32': {'atol': 0, 'rtol': 0}}, 'fixtures': []}
    problems = capabilities.witness_problems(forged, os.path.dirname(ZML))
    ok &= check("zml: a witness block in a conformer's manifest is refused", len(problems) == 1 and 'role is conformer' in problems[0], problems[:1])
    forged = json.loads(json.dumps(manifest))
    forged['contracts']['norm.rms@1.0.0']['witness']['fixtures'] = ['norm.rms@1.0.0/nowhere']
    problems = capabilities.witness_problems(forged, os.path.dirname(REFERENCE))
    ok &= check("reference: a fixture the manifest names and the tree lacks is refused", len(problems) == 1 and 'nowhere' in problems[0], problems[:1])
    print("capabilities: all good" if ok else "capabilities: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

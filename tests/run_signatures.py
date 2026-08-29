#!/usr/bin/env python3
"""Every model of the corpus must still denote the graph its recorded
signature describes (tests/signatures/<model>.json). Rewriting a document —
merging sites under `when`, scoping bindings — must leave its signature
intact; a changed signature is either a regression or a deliberate change of
the model, to be re-recorded with --record.

    python3 tests/run_signatures.py            # check
    python3 tests/run_signatures.py --record   # re-record every signature
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
from signature import signature        # noqa: E402

STORE = os.path.join(HERE, 'signatures')


def main(record=False):
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    failed = 0
    for path in sorted(glob.glob(os.path.join(ROOT, 'data', 'models', '*.json'))):
        name = os.path.basename(path)[:-5]
        target = os.path.join(STORE, name + '.json')
        current = signature(path, cat)
        if record:
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(current, f, indent=1)
                f.write('\n')
            print(f"  recorded {name}: {current['nodes']} nodes, wl {current['wl']}")
            continue
        if not os.path.isfile(target):
            failed += 1
            print(f"  FAIL {name}: no recorded signature")
            continue
        with open(target, encoding='utf-8') as f:
            recorded = json.load(f)
        diff = {k: (recorded.get(k), current.get(k)) for k in set(recorded) | set(current)
                if recorded.get(k) != current.get(k)}
        if diff:
            failed += 1
            print(f"  FAIL {name}")
            for k, (a, b) in sorted(diff.items()):
                print(f"         {k}: recorded {json.dumps(a)[:80]} → now {json.dumps(b)[:80]}")
        else:
            print(f"  ok   {name}")
    if not record:
        print("signatures: all intact" if not failed else f"signatures: {failed} changed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(record='--record' in sys.argv))

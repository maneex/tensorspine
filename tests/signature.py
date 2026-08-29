"""Structural signature of a model document: what must not change when the
document is rewritten — sugar rearranged, sites merged under `when`, bindings
scoped — while the denoted graph stays the same (§1.1: functional denotation
is about the graph, not the text).

The signature is computed from D1 and the validator's derivations, never from
identifiers: node names change with the writing, the graph does not.

  * nodes and edges, per contract;
  * a Weisfeiler-Lehman hash of the value graph, nodes labelled by contract and
    resolved arguments, edges by their ports (4 refinement rounds);
  * parameter slots, tensor identities and ties; state slots, identities and
    the multiset of derived instance keys.
"""
import glob
import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import d1                              # noqa: E402
import validate                        # noqa: E402

ASSIGNMENTS = {
    'decoder-causal-yarn@1.0.0': {"width": 3072, "layers": 26, "heads": 32, "kv_heads": 8,
                                  "head_dim": 128, "inner": 9216, "eps": 0.00001,
                                  "precision": "bf16"},
}
MODELS = os.path.join(ROOT, 'data', 'models')


def corpus():
    """Every document of the corpus: the models at the top level, the templates
    one directory down, one file per version (§4.6)."""
    return (sorted(glob.glob(os.path.join(MODELS, '*.json')))
            + sorted(glob.glob(os.path.join(MODELS, '*', '*.json'))))


def name_of(path):
    """`llama3-8b` for a model, `decoder-causal-yarn@1.0.0` for a template."""
    rel = os.path.relpath(path, MODELS)
    if os.sep in rel:
        directory, version = rel.split(os.sep)
        return f"{directory}@{version[:-5]}"
    return rel[:-5]


def _h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def wl_hash(document, rounds=4):
    nodes = document['nodes']
    label = {n: _h([v['contract']['name'], v['arguments']]) for n, v in nodes.items()}
    out_e, in_e = {n: [] for n in nodes}, {n: [] for n in nodes}
    for e in document['edges']:
        out_e[e['from']['node']].append((e['from']['port'], e['to']['port'], e['to']['node']))
        in_e[e['to']['node']].append((e['from']['port'], e['to']['port'], e['from']['node']))
    for _ in range(rounds):
        label = {n: _h([label[n],
                        sorted((p, q, label[m]) for p, q, m in out_e[n]),
                        sorted((p, q, label[m]) for p, q, m in in_e[n])])
                 for n in nodes}
    return _h(sorted(label.values()))


def signature(model_path, cat=None):
    cat = cat or catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    name = name_of(model_path)
    assignment = ASSIGNMENTS.get(name)
    document = d1.emit(model_path, cat, assignment)
    result = validate.analyse(model_path, cat, assignment)
    if result['errors']:
        raise ValueError(f"{name}: not valid, no signature: {result['errors'][0]}")
    stats = result['stats']
    per_contract = Counter(v['contract']['name'] for v in document['nodes'].values())
    return {
        "nodes": len(document['nodes']),
        "edges": len(document['edges']),
        "per_contract": dict(sorted(per_contract.items())),
        "wl": wl_hash(document),
        "parameter_slots": stats['parameter_slots'],
        "tensors": stats['tensors'],
        "shared": stats['shared'],
        "state_slots": stats['state_slots'],
        "state_identities": stats['state_identities'],
        "instance_keys": dict(sorted(Counter(
            "×".join(k) for k in result['instance_keys'].values()).items())),
    }


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps({os.path.basename(p): signature(p)}, indent=1))

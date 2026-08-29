#!/usr/bin/env python3
"""The expression and condition language, on the model side (§2.2, §5.2):
conditionals evaluate, an undecidable guard is UNRESOLVED (never false), and a
composition-scoped binding expands to exactly its top-level form.

    python3 tests/run_expressions.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import model as model_mod                                        # noqa: E402
from expr import UNRESOLVED, model_condition, model_value        # noqa: E402


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main():
    q = {"layers": 30}
    cond = {"compare": {"operator": "equal", "left": {"op": "modulo", "args": [{"index": "i"}, {"literal": 5}]},
                        "right": {"literal": 4}}}
    ok = check("condition: i mod 5 = 4 at i=9", model_condition(cond, q, {"i": 9}) is True)
    ok &= check("condition: i mod 5 = 4 at i=8", model_condition(cond, q, {"i": 8}) is False)
    ok &= check("condition over an unbound index is UNRESOLVED, not false",
                model_condition(cond, q, {}) is UNRESOLVED)
    ok &= check("`not` of an undecidable condition stays undecidable",
                model_condition({"not": cond}, q, {}) is UNRESOLVED)
    ok &= check("`all` with one undecidable part is undecidable",
                model_condition({"all": [{"boolean": True}, cond]}, q, {}) is UNRESOLVED)
    ite = {"if": {"compare": {"operator": "less", "left": {"index": "i"}, "right": {"literal": 10}}},
           "then": {"literal": 0.95}, "else": {"literal": 0}}
    ok &= check("if/then/else: then-branch", model_value(ite, q, {"i": 3}) == 0.95)
    ok &= check("if/then/else: else-branch", model_value(ite, q, {"i": 12}) == 0)
    ok &= check("if/then/else over an unbound index is UNRESOLVED", model_value(ite, q, {}) is UNRESOLVED)
    ok &= check("comparison across types is undecidable, not an exception",
                model_condition({"compare": {"operator": "less", "left": {"literal": "a"}, "right": {"literal": 1}}}, q) is UNRESOLVED)

    doc = {"compositions": {"C": {"indices": {"i": {"start": {"literal": 0}, "stop": {"literal": 4}, "step": {"literal": 1}}},
                                  "families": ["c"], "occurrences": {"a": {}, "b": {}},
                                  "bindings": {"values": {"e": {"from": {"site": "a", "port": "o"},
                                                                "to": {"site": "b", "port": "x", "indices": {"i": {"op": "subtract", "args": [{"index": "i"}, {"literal": 1}]}}},
                                                                "when": {"compare": {"operator": "greater_or_equal", "left": {"index": "i"}, "right": {"literal": 1}}}}},
                                               "parameters": {"a.w": {"members": [{"site": "a", "parameter": "w"}]}}}}},
           "bindings": {"values": {}, "parameters": {}, "constants": {}, "states": {}}}
    n = model_mod.normalise(doc)
    e = n['bindings']['values'].get('C.e', {})
    ok &= check("scoped value rule hoisted as C.e with the composition's for_each",
                e.get('for_each') == doc['compositions']['C']['indices'] and 'when' in e)
    ok &= check("site endpoint becomes the generated selector at the current index",
                e.get('from', {}).get('occurrence') == {"kind": "generated", "composition": "C", "occurrence": "a", "indices": {"i": {"index": "i"}}})
    ok &= check("an index override is kept",
                e.get('to', {}).get('occurrence', {}).get('indices', {}).get('i') == {"op": "subtract", "args": [{"index": "i"}, {"literal": 1}]})
    p = n['bindings']['parameters'].get('C.a.w', {})
    ok &= check("scoped parameter rule names its tensor C.a.w indexed by i",
                p.get('tensor') == {"name": "C.a.w", "indices": {"i": {"index": "i"}}})
    ok &= check("composition no longer carries bindings after normalisation",
                'bindings' not in n['compositions']['C'])
    ok &= check("normalisation is idempotent", model_mod.normalise(n) == n)
    print("expressions: all good" if ok else "expressions: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Template contracts (§4.6): what must be accepted, and what must come out equal.

  1. Parity: a model written flat and the same model written through a template
     contract derive the same parameter slots, tensors and states.
  2. Declared defaults: an external quantity with a default makes the argument
     optional at the call site, and the default is applied.
  3. Assignment: the template validated alone under an admissible assignment
     passes; under an inadmissible one it is refused with its reason.

    python3 tests/run_templates.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import validate                        # noqa: E402

SCHEMAS = os.path.join(ROOT, 'schemas')
REFERENCE = os.path.join(ROOT, 'data', 'catalog')
MODELS = os.path.join(ROOT, 'data', 'models')
ASSIGNMENT = {"width": 3072, "layers": 26, "heads": 32, "kv_heads": 8, "head_dim": 128,
              "inner": 9216, "eps": 0.00001, "precision": "bf16"}


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def parity(cat):
    flat = validate.semantic(os.path.join(MODELS, 'shieldstral-3b.json'), cat)
    comp = validate.semantic(os.path.join(MODELS, 'shieldstral-3b-composite.json'), cat)
    ok = not flat[0] and not comp[0]
    for k in ('parameter_slots', 'tensors', 'state_slots', 'state_identities', 'shared'):
        ok &= check(f"parity {k}: flat {flat[1].get(k)} == composite {comp[1].get(k)}",
                    flat[1].get(k) == comp[1].get(k))
    return ok


def defaults():
    """A models base where the template gives `eps` a default; the composite
    then omits `eps` at the call site and must pass."""
    tmp = tempfile.mkdtemp(prefix='tensorspine-templates-')
    try:
        os.makedirs(os.path.join(tmp, 'decoder-causal-yarn'))
        shutil.copy(os.path.join(MODELS, 'decoder-causal-yarn', '1.0.0.json'),
                    os.path.join(tmp, 'decoder-causal-yarn', '1.0.0.json'))
        shutil.copy(os.path.join(MODELS, 'shieldstral-3b-composite.json'), tmp)
        path = os.path.join(tmp, 'decoder-causal-yarn', '1.0.0.json')
        with open(path, encoding='utf-8') as f:
            template = json.load(f)
        template['quantities']['eps']['source']['default'] = {"literal": 0.00001}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        path = os.path.join(tmp, 'shieldstral-3b-composite.json')
        with open(path, encoding='utf-8') as f:
            composite = json.load(f)
        del composite['occurrences']['text']['arguments']['eps']
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(composite, f, indent=2)
        cat = catalog_mod.load(REFERENCE, models_base=tmp)
        errors, stats = validate.semantic(path, cat)
        ok = check("declared default: `eps` omitted at the call site, template default applied",
                   not errors, errors[:1] and errors[0])
        ok &= check("declared default: slots still complete", stats.get('parameter_slots') == 309,
                    str(stats.get('parameter_slots')))
        return ok
    finally:
        shutil.rmtree(tmp)


def assignment(cat):
    path = os.path.join(MODELS, 'decoder-causal-yarn', '1.0.0.json')
    with open(path, encoding='utf-8') as f:
        template = json.load(f)
    errors = validate.check_assignment(template, ASSIGNMENT)
    ok = check("admissible assignment accepted", not errors, errors[:1] and errors[0])
    if not errors:
        errors, _ = validate.semantic(path, cat, ASSIGNMENT)
        ok &= check("template alone under that assignment passes", not errors, errors[:1] and errors[0])
    bad = dict(ASSIGNMENT, layers=0, precision='fp4')
    errors = validate.check_assignment(template, bad)
    ok &= check("inadmissible assignment refused (layers=0, precision=fp4)",
                len(errors) == 2 and 'below the domain bound' in errors[0]
                and 'not among' in errors[1], '\n         '.join(errors))
    return ok


def main():
    cat = catalog_mod.load(REFERENCE)
    ok = parity(cat)
    ok &= defaults()
    ok &= assignment(cat)
    print("templates: all good" if ok else "templates: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

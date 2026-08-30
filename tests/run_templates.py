#!/usr/bin/env python3
"""Template contracts (§4.6): what must be accepted, and what must come out equal.

  1. Parity: a model written flat and the same model written through a template
     contract derive the same parameter slots, tensors and states — and the same
     derived products D2–D6, value for value and cut for cut, up to the instance
     prefix (§5.1: the expanded graph is authoritative).
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
import derive                          # noqa: E402
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


def derived_parity(cat):
    """D2–D6 of the composite equal the flat document's once the instance prefix
    `text/` is dropped: template instances are expanded before derivation."""
    flat = derive.products(os.path.join(MODELS, 'shieldstral-3b.json'), cat)
    comp = derive.products(os.path.join(MODELS, 'shieldstral-3b-composite.json'), cat)

    def strip(name):
        return name.replace('text/', '')

    def canon(x):
        return json.dumps(x, sort_keys=True)

    def values(doc):
        return sorted((strip(v['value']), tuple(sorted(strip(t) for t in v['to'])), canon(v['shape']),
                       v['elements'], canon(v['domain']), canon(v['count']),
                       tuple(v.get('required_for', [])), tuple(v.get('exposed', [])))
                      for v in doc['d2']['values'])

    def cuts(doc):
        return sorted((strip(c['cut']), c['kind'], tuple(c['sizes']), c['bytes_per_element'],
                       canon(c['bytes_per_invocation']), tuple(sorted(strip(p['value']) for p in c['payload'])))
                      for c in doc['d2']['cuts'])

    def tensors(doc):
        return sorted((strip(t['identity']), t['bytes'], t['dtype'], tuple(strip(m) for m in t['members']))
                      for t in doc['d3']['tensors'])

    def states(doc):
        return sorted((strip(s['identity']), s['law'], s['access'], canon(s['stream']),
                       s['bytes_per_cached_position'], s['bytes_bounded'], tuple(strip(m) for m in s['members']))
                      for s in doc['d4']['states'])

    def costs(doc):
        d = doc['d5']
        return (canon(d['parameters']), canon(d['operations']), canon(d['state']),
                sorted((strip(c['node']), c['entry'], c['value']) for c in d['corrections']),
                sorted((strip(c['cut']), c['bytes_per_element']) for c in d['cuts']))

    def placement(doc):
        d = doc['d6']
        return (sorted((strip(c['cut']), c['kind'], c['crossing_values']) for c in d['cuts']),
                sorted((strip(p['node']), canon(p['target']), p['communication']) for p in d['partitions']),
                sorted((strip(l['node']), l['slot'], l['axis']) for l in d['information_loss']))

    ok = True
    for label, f in (('D2 values', values), ('D2 cuts', cuts), ('D3 tensors', tensors), ('D4 states', states),
                     ('D5 costs, corrections and cuts', costs), ('D6 cuts, partitions and information loss', placement)):
        a, b = f(flat), f(comp)
        size = f"{len(a)} == {len(b)}" if isinstance(a, list) else ''
        ok &= check(f"derived parity, {label}: composite equals flat up to the instance prefix {size}".rstrip(),
                    a == b, f"first difference: {next((x for x in (a if isinstance(a, list) else [a]) if x not in (b if isinstance(b, list) else [b])), None)}")
    ok &= check("derived parity: D2 and D6 list the composite's 59 cuts", len(comp['d2']['cuts']) == 59 == len(comp['d6']['cuts']))
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
        ok &= check("declared default: slots still complete", stats.get('parameter_slots') == 459,
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
    ok &= derived_parity(cat)
    ok &= defaults()
    ok &= assignment(cat)
    print("templates: all good" if ok else "templates: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

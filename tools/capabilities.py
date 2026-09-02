"""Capabilities of a generator (generators/CAPABILITIES.md): can a generator run a document, and
what does it still not cover.

    tensorspine --capabilities MANIFEST [MODEL ...] [--inputs a,b] [--coverage [--strict]]

A manifest is validated against `generators/capabilities.schema.json` and its names against the
catalog before anything is inferred: a claim outside the vocabulary is an error, not a capability.
A witness manifest (Specification §4.1, O1.3) binds each contract version to the kernel that is
the authority for it and to the unit fixtures that kernel produced; the reader checks that both
exist, `--coverage` lists the contract versions still without a witness — the release rule of
§10.2 — and `--strict` exits 1 on any, for a tag workflow.
"""
import json
import os

import artifact as artifact_mod
import catalog as catalog_mod
import derive
from expr import contract_condition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, 'generators', 'capabilities.schema.json')


# --- the argument rules ------------------------------------------------------

def _check(rule, name, value):
    if rule == 'any':
        return []
    if rule == 'absent':
        return [] if value is None else [f"{name}={value}"]
    if isinstance(rule, list):
        return [] if (value is None or value in rule) else [f"{name}={value}"]
    if isinstance(rule, dict):
        if value is None:
            return [] if rule.get('absent', True) else [f"{name}=absent"]
        if 'values' in rule:
            return [] if value in rule['values'] else [f"{name}={value}"]
        if not isinstance(value, dict):
            return [f"{name}={value}"]
        out = []
        for k, v in value.items():
            if k not in rule.get('fields', {}):
                out.append(f"{name}.{k}={v!r} (field unknown to the manifest)")
            else:
                out += _check(rule['fields'][k], f"{name}.{k}", v)
        for k, r in rule.get('fields', {}).items():
            if k not in value and isinstance(r, dict) and not r.get('absent', True):
                out.append(f"{name}.{k}=absent")
        return out
    raise ValueError(f"bad rule for {name}: {rule!r}")


def supports(entry, arguments):
    """The 'name=value' pairs of resolved D1 arguments an entry does not implement."""
    table = entry['arguments']
    reasons = []
    for name, value in arguments.items():
        if name not in table:
            reasons.append(f"{name}={value!r} (argument unknown to the manifest)")
        else:
            reasons += _check(table[name], name, value)
    for name, rule in table.items():
        if name not in arguments and isinstance(rule, dict) and not rule.get('absent', True):
            reasons.append(f"{name}=absent")
    for combo in entry.get('excluding', []):
        if all(arguments.get(k) == v for k, v in combo.items()):
            reasons.append("combination " + ", ".join(f"{k}={v}" for k, v in combo.items()))
    return reasons


# --- loading and checking a manifest -------------------------------------------

def load(path):
    with open(path, encoding='utf-8') as f:
        manifest = json.load(f)
    import jsonschema
    with open(SCHEMA, encoding='utf-8') as f:
        schema = json.load(f)
    errors = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(manifest)]
    return manifest, errors


def _enum_values(t):
    k = t.get('kind')
    if k == 'enum':
        return list(t['values'])
    if k == 'boolean':
        return [True, False]
    return None


def _rule_names(rule, decl, where, errors):
    """Every name a rule uses exists in the contract's argument declaration."""
    t = decl.get('type', {})
    if isinstance(rule, dict) and 'fields' in rule:
        if t.get('kind') != 'record':
            errors.append(f"{where}: rule has fields but the argument is not a record")
            return
        for k, r in rule['fields'].items():
            if k not in t['fields']:
                errors.append(f"{where}.{k}: no such field")
            else:
                _rule_names(r, t['fields'][k], f"{where}.{k}", errors)
    literals = rule if isinstance(rule, list) else rule.get('values') if isinstance(rule, dict) else None
    if literals is not None:
        allowed = _enum_values(t)
        if allowed is not None:
            for v in literals:
                if v not in allowed:
                    errors.append(f"{where}: value {v!r} is not one of {allowed}")


def witness_problems(manifest, base_dir):
    """Errors of the witness blocks (§4.1): only a witness manifest carries them; the kernel each
    names exists beside the manifest; each fixture it names exists there too, under
    `fixtures/contracts/<id>.safetensors`, and is a unit fixture of that contract version."""
    errors = []
    role = manifest.get('role', 'conformer')
    for cid, entry in manifest['contracts'].items():
        w = entry.get('witness')
        if w is None:
            continue
        if role != 'witness':
            errors.append(f"{cid}: a witness block in a manifest whose role is {role}")
            continue
        if not os.path.isfile(os.path.join(base_dir, w['kernel'])):
            errors.append(f"{cid}: witness kernel '{w['kernel']}' is not beside the manifest")
        for fid in w['fixtures']:
            if not fid.startswith(cid + '/'):
                errors.append(f"{cid}: fixture '{fid}' is another contract's")
                continue
            path = os.path.join(base_dir, 'fixtures', 'contracts', fid + '.safetensors')
            if not os.path.isfile(path):
                errors.append(f"{cid}: fixture '{fid}' is not at fixtures/contracts/ beside the manifest")
                continue
            try:
                meta = artifact_mod.read_metadata(path)
            except (OSError, ValueError) as e:
                errors.append(f"{cid}: fixture '{fid}' is not readable: {e}")
                continue
            name, version = cid.rsplit('@', 1)
            if meta.get('kind') != 'unit' or meta.get('id') != fid or meta.get('contract') != {'name': name, 'version': version}:
                errors.append(f"{cid}: '{fid}' is not a unit fixture of this contract version")
    return errors


def names(manifest, cat):
    """Errors: contract versions, arguments, fields and values the catalog does not know."""
    errors = []
    for cid, entry in manifest['contracts'].items():
        name, version = cid.rsplit('@', 1)
        definition = cat['by_id'].get((name, version))
        if definition is None:
            errors.append(f"{cid}: not in the catalog")
            continue
        for arg, rule in entry['arguments'].items():
            if arg not in definition['arguments']:
                errors.append(f"{cid}: no argument '{arg}'")
            else:
                _rule_names(rule, definition['arguments'][arg], f"{cid}.{arg}", errors)
        for law in entry.get('states', []):
            if law not in manifest['state_laws']:
                errors.append(f"{cid}: state law {law} not in the manifest's state_laws")
    return errors


# --- can it run? -----------------------------------------------------------------

def evaluated(doc, cat, delivered):
    """The occurrences a delivery evaluates (§7), over the derived document."""
    d1 = doc['d1']
    fed = set()
    for name, entry in d1['interfaces']['inputs'].items():
        if name in delivered:
            for t in entry['to']:
                fed.add((t['node'], t['port']))
    incoming = {}
    for e in d1['edges']:
        incoming.setdefault(e['to']['node'], []).append((e['from']['node'], e['to']['port']))
    done = set()
    for node in d1['topological_order']:
        entry = d1['nodes'][node]
        definition = catalog_mod.contract(cat, entry['contract'])
        inserts = {t['from_port'] for t in definition.get('domain_transforms', []) if t.get('relation') == 'insert'}
        source_held = {t for t in ('source_values',) if any(s.get('indexed_by_source') and any(m.rsplit('.', 1)[0] == node for m in s['members']) for s in doc['d4']['states'])}
        ok = True
        for pname, port in definition['ports']['inputs'].items():
            present = contract_condition(port['present_when'], entry['arguments']) if 'present_when' in port else True
            if not present or pname in inserts or pname in source_held:
                continue
            if (node, pname) in fed or any(src in done and dp == pname for src, dp in incoming.get(node, [])):
                continue
            ok = False
            break
        if ok:
            done.add(node)
    return done


def can_run(manifest, doc, cat, delivered=None):
    """(ok, reasons) — the first reasons a generator cannot evaluate this document for this delivery."""
    d1, d2, d3, d4 = doc['d1'], doc['d2'], doc['d3'], doc['d4']
    generative = [n for n, o in d1['interfaces']['outputs'].items() if o.get('generative')]
    if delivered is None:
        delivered = {v['input'] for v in d2['values'] if 'input' in v and (set(v.get('required_for', [])) & set(generative) or not generative)}
    reasons = []
    active = evaluated(doc, cat, delivered)
    outputs = {n: f"{o['node']}" for n, o in d1['interfaces']['outputs'].items()}
    for oname in generative or outputs:
        if outputs[oname] not in active:
            reasons.append(f"output {oname} is not evaluated by the inputs {sorted(delivered)}")
    for node in d1['topological_order']:
        if node not in active:
            continue
        entry = d1['nodes'][node]
        cid = f"{entry['contract']['name']}@{entry['contract']['version']}"
        cap = manifest['contracts'].get(cid)
        if cap is None:
            reasons.append(f"{node}: no entry for {cid}")
            continue
        for r in supports(cap, entry['arguments']):
            reasons.append(f"{node}: {cid} does not implement {r}")
    for t in d3['tensors']:
        if t['dtype'] not in manifest['parameter_dtypes']:
            reasons.append(f"{t['identity']}: parameter dtype {t['dtype']} not loadable")
            break
    for t in d3['tensors']:
        loc = t.get('location')
        if loc is None:
            reasons.append("the document does not locate its weights")
            break
        form = next(iter(loc))
        if form not in manifest['locations']:
            reasons.append(f"{t['identity']}: location form {form} not assembled")
            break
    for s in d4['states']:
        node = s['members'][0].rsplit('.', 1)[0]
        if node in active and s['law'] not in manifest['state_laws']:
            reasons.append(f"{s['identity']}: state law {s['law']} not implemented")
        if node in active and s['access'] not in manifest['access']:
            reasons.append(f"{s['identity']}: access {s['access']} not implemented")
    for name, entry in d1['interfaces']['inputs'].items():
        if name in delivered and entry.get('fragmented') and not manifest['domains']['fragmented']:
            reasons.append(f"input {name} is fragmented, which the candidate does not handle")
    for sname, st in d2['streams'].items():
        if st['kind'] not in manifest['domains']['kinds']:
            reasons.append(f"stream {sname}: kind {st['kind']} not handled")
    return (not reasons), reasons


# --- coverage --------------------------------------------------------------------

def _missing_values(rule, decl):
    """Enum values / booleans of an argument declaration the rule does not admit."""
    t = decl.get('type', {})
    allowed = _enum_values(t)
    out = []
    if allowed is not None:
        if rule == 'any':
            return []
        literals = rule if isinstance(rule, list) else rule.get('values', []) if isinstance(rule, dict) else []
        out = [v for v in allowed if v not in literals]
    if t.get('kind') == 'record':
        if rule == 'absent':
            return ['present']
        if isinstance(rule, dict) and 'fields' in rule:
            for k, sub in t['fields'].items():
                if k in rule['fields']:
                    out += [f"{k}={v}" for v in _missing_values(rule['fields'][k], sub)]
                else:
                    out.append(f"{k}")
    return out


def _branches(decl, prefix=''):
    """Every branch of an argument declaration: the values of an enum or boolean, a record's
    presence and its fields' branches; an argument that takes any number has none."""
    t = decl.get('type', {})
    allowed = _enum_values(t)
    if allowed is not None:
        return [f"{prefix}={v}" for v in allowed]
    if t.get('kind') == 'record':
        out = [f"{prefix}=present"]
        for k, sub in t['fields'].items():
            out += _branches(sub, f"{prefix}.{k}")
        return out
    return []


def unwitnessed(manifest, cat):
    """For a witness manifest, the contract versions of the catalog without a witness — no entry,
    or an entry without a witness block; None for a conformer, which witnesses nothing."""
    if manifest.get('role', 'conformer') != 'witness':
        return None
    return sorted(f"{n}@{v}" for (n, v), d in cat['by_id'].items() if 'template' not in d
                  and not manifest['contracts'].get(f"{n}@{v}", {}).get('witness'))


def coverage(manifest, cat, documents):
    """(contracts without an entry, per contract the branches not admitted — every branch, for
    a contract without an entry — per document the verdict): the branch ledger, the to-do list
    of the generator over the catalog and the corpus."""
    missing = sorted(f"{n}@{v}" for (n, v), d in cat['by_id'].items()
                     if f"{n}@{v}" not in manifest['contracts'] and 'template' not in d)
    branches = {}
    for cid in missing:
        name, version = cid.rsplit('@', 1)
        gaps = []
        for arg, decl in cat['by_id'][(name, version)]['arguments'].items():
            gaps += _branches(decl, arg) or [arg]
        branches[cid] = gaps
    for cid, entry in manifest['contracts'].items():
        name, version = cid.rsplit('@', 1)
        definition = cat['by_id'][(name, version)]
        gaps = []
        for arg, decl in definition['arguments'].items():
            rule = entry['arguments'].get(arg)
            if rule is None:
                gaps.append(arg)
            else:
                gaps += [f"{arg}={v}" for v in _missing_values(rule, decl)]
        if gaps:
            branches[cid] = gaps
    verdicts = {}
    for path in documents:
        with open(path, encoding='utf-8') as f:
            model = json.load(f)
        c = catalog_mod.load_for(path, model)
        try:
            doc = derive.products(path, c)
        except Exception as e:  # noqa: BLE001
            verdicts[os.path.basename(path)] = (False, [f"not derivable: {e}"])
            continue
        verdicts[os.path.basename(path)] = can_run(manifest, doc, c)
    return missing, branches, verdicts


def condensed(reasons):
    """Reasons without their occurrence prefix, each once, with how many occurrences share it."""
    from collections import Counter
    counts = Counter(r.split(': ', 1)[1] if ': ' in r and not r.startswith('output ') else r for r in reasons)
    return [f"{msg} (x{n})" if n > 1 else msg for msg, n in counts.items()]


def run(manifest_path, documents, catalog_bases=None, inputs=None, report_coverage=False, corpus=None, strict=False):
    manifest, errors = load(manifest_path)
    cat = catalog_mod.load(*(catalog_bases or [os.path.join(ROOT, 'data', 'catalog')]))
    errors += names(manifest, cat)
    errors += witness_problems(manifest, os.path.dirname(os.path.abspath(manifest_path)))
    role = manifest.get('role', 'conformer')
    print(f"capabilities  {manifest['generator']['name']} ({manifest['generator']['version']}), {len(manifest['contracts'])} contracts, {role}")
    for e in errors[:20]:
        print(f"  [manifest] {e}")
    if errors:
        return 1
    failed = 0
    for path in documents:
        with open(path, encoding='utf-8') as f:
            model = json.load(f)
        c = catalog_mod.load_for(path, model, catalog_bases)
        doc = derive.products(path, c)
        ok, reasons = can_run(manifest, doc, c, set(inputs) if inputs else None)
        print(f"  {os.path.basename(path):34s} {'can run' if ok else 'cannot'}")
        for r in condensed(reasons)[:8]:
            print(f"      {r}")
        failed += not ok
    if report_coverage:
        missing, branches, verdicts = coverage(manifest, cat, corpus or [])
        print(f"coverage  {len(missing)} contract(s) without an entry:")
        for cid in missing:
            print(f"    {cid}")
        print(f"  branch ledger: branches not admitted in {len(branches)} contract(s), {len(missing)} of them without an entry:")
        for cid, gaps in branches.items():
            print(f"    {cid}{' (no entry)' if cid in missing else ''}: {', '.join(gaps)}")
        if verdicts:
            print(f"  corpus: {sum(1 for ok, _ in verdicts.values() if ok)}/{len(verdicts)} documents can run")
            for name, (ok, reasons) in verdicts.items():
                if not ok:
                    print(f"    {name:34s} {condensed(reasons)[0]}")
        without = unwitnessed(manifest, cat)
        if without is None:
            print("  witness: a conformer's manifest witnesses nothing")
        else:
            print(f"  witness: {len(without)} contract version(s) without a witness (§10.2: a catalog is released fully witnessed)")
            for cid in without:
                print(f"    {cid}")
            if without and strict:
                print("  --strict: refused")
                return 1
    return 1 if failed else 0

"""The checkpoint side of V17: every located tensor of a document exists in the
safetensors checkpoint with the shape and dtype D3 declares. Headers only — no
weight is read.

    tensorspine --validate --checkpoint DIR MODEL

A physical tensor may carry unit axes the logical shape lacks (`torch.nn.Conv1d`
stores `[C, 1, K]`); they are dropped before shapes are compared. Physical
tensors no location names are listed as advice.
"""
import glob
import json
import os
import struct

DTYPES = {'BF16': 'bf16', 'F16': 'f16', 'F32': 'f32', 'F64': 'f64', 'F8_E4M3': 'f8e4m3', 'F8_E5M2': 'f8e5m2',
          'I8': 'i8', 'I16': 'i16', 'I32': 'i32', 'I64': 'i64', 'U8': 'u8', 'BOOL': 'bool'}


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    return header


def read_headers(checkpoint):
    """name -> {'dtype', 'shape', 'file'} over the index's shards, or every
    `*.safetensors` in the directory, or one file."""
    if os.path.isfile(checkpoint):
        files = [checkpoint]
    else:
        index = os.path.join(checkpoint, 'model.safetensors.index.json')
        if os.path.exists(index):
            with open(index, encoding='utf-8') as f:
                files = sorted({os.path.join(checkpoint, v) for v in json.load(f)['weight_map'].values()})
        else:
            files = sorted(glob.glob(os.path.join(checkpoint, '*.safetensors')))
    if not files:
        raise FileNotFoundError(f"no safetensors file under {checkpoint}")
    out = {}
    for path in files:
        for name, entry in read_header(path).items():
            out[name] = {'dtype': DTYPES.get(entry['dtype'], entry['dtype'].lower()),
                         'shape': list(entry['shape']), 'file': os.path.basename(path)}
    return out


def squeeze(shape):
    return [d for d in shape if d != 1]


def _check_part(ev, logical, dtype, headers, identity, errors):
    """Check one evaluated location against `logical` (the shape it must fill);
    returns the physical extent it contributes along a concat axis, when asked."""
    if 'tensor' in ev:
        name = ev['tensor']
        h = headers.get(name)
        if h is None:
            errors.append(f"{identity}: physical tensor '{name}' is absent from the checkpoint")
            return None
        if squeeze(h['shape']) != squeeze(logical) or len(h['shape']) < len(squeeze(logical)):
            errors.append(f"{identity}: '{name}' has shape {h['shape']}, the document says {logical}")
        if h['dtype'] != dtype:
            errors.append(f"{identity}: '{name}' is {h['dtype']}, the document says {dtype}")
        return h['shape']
    if 'stack' in ev:
        dim = ev['stack']['dim']
        inner = logical[:dim] + logical[dim + 1:]
        if len(ev['stack']['parts']) != logical[dim]:
            errors.append(f"{identity}: stack of {len(ev['stack']['parts'])} along '{ev['stack']['axis']}' "
                          f"of extent {logical[dim]}")
        for part in ev['stack']['parts']:
            _check_part(part, inner, dtype, headers, identity, errors)
        return logical
    if 'concat' in ev:
        dim = ev['concat']['dim']
        total = 0
        for part in ev['concat']['parts']:
            # the part's extent along the axis is its own: take it from the checkpoint
            names, _ = _names(part)
            h = headers.get(names[0]) if names else None
            if h is None:
                _check_part(part, logical, dtype, headers, identity, errors)   # reports the absence
                return None
            phys = squeeze(h['shape'])
            want = squeeze(logical)
            if len(phys) != len(want):
                errors.append(f"{identity}: '{names[0]}' has shape {h['shape']}, a part of {logical}")
                return None
            extent = phys[[i for i, d in enumerate(logical) if d != 1].index(dim)] if logical[dim] != 1 else 1
            own = list(logical)
            own[dim] = extent
            _check_part(part, own, dtype, headers, identity, errors)
            total += extent
        if total != logical[dim]:
            errors.append(f"{identity}: concat parts sum to {total} along '{ev['concat']['axis']}', "
                          f"the document says {logical[dim]}")
        return logical
    if 'slice' in ev:
        s = ev['slice']
        h = headers.get(s['tensor'])
        if h is None:
            errors.append(f"{identity}: physical tensor '{s['tensor']}' is absent from the checkpoint")
            return None
        phys = squeeze(h['shape'])
        want = squeeze(logical)
        if len(phys) != len(want):
            errors.append(f"{identity}: '{s['tensor']}' has shape {h['shape']}, sliced for {logical}")
            return None
        pos = [i for i, d in enumerate(logical) if d != 1].index(s['dim'])
        other = [d for i, d in enumerate(phys) if i != pos] == [d for i, d in enumerate(want) if i != pos]
        if not other or phys[pos] < s['offset'] + s['extent']:
            errors.append(f"{identity}: '{s['tensor']}' has shape {h['shape']}; the slice [{s['offset']}, "
                          f"{s['offset'] + s['extent']}) along '{s['axis']}' does not fit {logical}")
        if h['dtype'] != dtype:
            errors.append(f"{identity}: '{s['tensor']}' is {h['dtype']}, the document says {dtype}")
        return h['shape']
    errors.append(f"{identity}: unknown location form {list(ev)}")
    return None


def _names(ev):
    whole, slices = [], []
    if 'tensor' in ev:
        whole.append(ev['tensor'])
    elif 'slice' in ev:
        slices.append(ev['slice']['tensor'])
    else:
        key = 'stack' if 'stack' in ev else 'concat'
        for part in ev[key]['parts']:
            w, s = _names(part)
            whole.extend(w)
            slices.extend(s)
    return whole, slices


def check(d3, headers):
    """(errors, advisories, stats) of a derived document's D3 against the headers."""
    errors, used = [], set()
    located = 0
    for t in d3['tensors']:
        ev = t.get('location')
        if ev is None:
            continue
        located += 1
        logical = [a['extent'] for a in t['shape']]
        _check_part(ev, logical, t['dtype'], headers, t['identity'], errors)
        w, s = _names(ev)
        used.update(w)
        used.update(s)
    unnamed = sorted(set(headers) - used)
    advisories = [f"physical tensor '{n}' ({headers[n]['dtype']} {headers[n]['shape']}) is named by no location"
                  for n in unnamed]
    return errors, advisories, {'located': located, 'physical': len(headers), 'unnamed': len(unnamed)}


def run(model_paths, catalog_bases, checkpoint, assignment=None):
    """The check over documents; prints a report; returns the number that fail."""
    import catalog as catalog_mod
    import derive
    headers = read_headers(checkpoint)
    failed = 0
    for path in model_paths:
        with open(path, encoding='utf-8') as f:
            model = json.load(f)
        cat = catalog_mod.load_for(path, model, catalog_bases)
        doc = derive.products(path, cat, assignment)
        errors, advisories, stats = check(doc['d3'], headers)
        name = os.path.basename(path)
        if stats['located'] == 0:
            print(f"  {name:34s} no location: nothing to check against {checkpoint}")
            continue
        verdict = 'ok  ' if not errors else 'FAIL'
        print(f"  {name:34s} {verdict} {stats['located']} located, {stats['physical']} physical, "
              f"{stats['unnamed']} named by no location")
        for e in errors[:20]:
            print(f"      [V17] {e}")
        for a in advisories[:10]:
            print(f"      advice: {a}")
        if len(advisories) > 10:
            print(f"      … {len(advisories) - 10} more")
        failed += bool(errors)
    return failed

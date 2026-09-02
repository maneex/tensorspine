"""Generated project state for ``tensorspine --document``.

The status page and branch ledger contain only facts read from the catalog,
model corpus, generator manifests, validation/derivation tools, and verification
records.  They are site-build products, not tracked documentation.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import runpy
import struct
import subprocess
import sys

import capabilities as capabilities_mod
import catalog as catalog_mod
import derive
import validate
from expr import missing_assignment


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cell(value):
    text = str(value if value is not None else '').strip().replace('|', '\\|')
    return text.replace('\r\n', '\n').replace('\n', '<br>')


def _table(headers, rows):
    lines = [
        '| ' + ' | '.join(_cell(value) for value in headers) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|',
    ]
    lines.extend('| ' + ' | '.join(_cell(value) for value in row) + ' |' for row in rows)
    return lines + ['']


def _git_commit():
    result = subprocess.run(
        ['git', '-C', ROOT, 'rev-parse', 'HEAD'],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() or 'unknown'


def _build_date():
    epoch = os.environ.get('SOURCE_DATE_EPOCH')
    instant = (dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc)
               if epoch else dt.datetime.now(dt.timezone.utc))
    return instant.date().isoformat()


def _metadata(path):
    """The JSON-valued metadata of a safetensors fixture, without loading tensors."""
    try:
        with open(path, 'rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
    except (OSError, ValueError, struct.error, json.JSONDecodeError):
        return {}
    out = {}
    for key, value in header.get('__metadata__', {}).items():
        try:
            out[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            out[key] = value
    return out


def _catalog(catalog_bases):
    return catalog_mod.load(*(catalog_bases or [os.path.join(ROOT, 'data', 'catalog')]))


def corpus(model_paths, catalog_bases, schema_dir):
    """Validation and location state, computed by the same readers as the CLI."""
    rows = []
    for path in model_paths:
        name = os.path.basename(path)
        structural = validate.structural(path, schema_dir)
        if structural:
            rows.append({'name': name, 'validation': 'refused by schema',
                         'located': 'not derived', 'detail': structural[0]})
            continue
        with open(path, encoding='utf-8') as stream:
            model = json.load(stream)
        unset = missing_assignment(model)
        if unset:
            rows.append({'name': name, 'validation': 'assignment required',
                         'located': 'not derived', 'detail': ', '.join(unset)})
            continue
        try:
            cat = catalog_mod.load_for(path, model, catalog_bases, schema_dir)
            errors, _stats = validate.semantic(path, cat)
        except (OSError, ValueError, KeyError, catalog_mod.CatalogError) as error:
            rows.append({'name': name, 'validation': 'refused',
                         'located': 'not derived', 'detail': str(error)})
            continue
        if errors:
            rows.append({'name': name, 'validation': 'refused by semantics',
                         'located': 'not derived', 'detail': errors[0]})
            continue
        try:
            products = derive.products(path, cat)
            tensors = products['d3']['tensors']
            located = 'all tensors' if all('location' in tensor for tensor in tensors) else 'partial'
        except Exception as error:  # The status page must report a derivation refusal.
            rows.append({'name': name, 'validation': 'valid',
                         'located': 'derivation refused', 'detail': str(error)})
            continue
        rows.append({'name': name, 'validation': 'valid', 'located': located, 'detail': ''})
    return rows


def generator(manifest_path, cat, model_paths):
    manifest, errors = capabilities_mod.load(manifest_path)
    errors += capabilities_mod.names(manifest, cat)
    if errors:
        raise ValueError(f"{manifest_path}: " + '; '.join(errors))
    missing, branches, verdicts = capabilities_mod.coverage(manifest, cat, model_paths)
    witnessed = {identity for identity, entry in manifest['contracts'].items()
                 if entry.get('witness')}
    catalog_contracts = {f'{name}@{version}' for name, version in cat['by_id']}
    return {
        'path': manifest_path,
        'manifest': manifest,
        'missing': missing,
        'branches': branches,
        'verdicts': verdicts,
        'unwitnessed': sorted(catalog_contracts - witnessed),
    }


def facts(model_paths, catalog_bases, schema_dir, manifest_paths):
    cat = _catalog(catalog_bases)
    return {
        'commit': _git_commit(),
        'date': _build_date(),
        'catalog': cat,
        'corpus': corpus(model_paths, catalog_bases, schema_dir),
        'generators': [generator(path, cat, model_paths) for path in manifest_paths],
    }


def _record(path):
    py = os.path.join(os.path.dirname(path), 'verified.py')
    js = os.path.join(os.path.dirname(path), 'verified.json')
    if os.path.isfile(py):
        return runpy.run_path(py), py
    if os.path.isfile(js):
        with open(js, encoding='utf-8') as stream:
            return json.load(stream), js
    return {}, None


def _default_tolerance(record):
    agreement = str(record.get('AGREEMENT', ''))
    match = re.search(r'atol\s+([0-9.eE+-]+)\s*/\s*rtol\s+([0-9.eE+-]+)', agreement)
    return (match.group(1), match.group(2)) if match else ('recorded by the suite', '')


def _verification_lines(item):
    manifest = item['manifest']
    record, source = _record(item['path'])
    out = [f"### `{manifest['generator']['name']}`", '']
    if not source:
        return out + ['No verification record is published by this generator.', '']

    out += [f"Record: `{os.path.relpath(source, ROOT)}`.", '']
    fixtures = record.get('FIXTURES', [])
    if fixtures:
        default_atol, default_rtol = _default_tolerance(record)
        rows = []
        fixture_dir = os.path.join(os.path.dirname(source), 'fixtures')
        checkpoint_ids = record.get('CHECKPOINT_IDS', {})
        for entry in fixtures:
            fixture, model, artifact = entry[:3]
            tolerance = entry[3] if len(entry) > 3 else (default_atol, default_rtol)
            meta = _metadata(os.path.join(fixture_dir, fixture))
            provenance = ', '.join(
                value for value in (
                    checkpoint_ids.get(artifact, artifact),
                    f"transformers {meta.get('transformers')}" if meta.get('transformers') else None,
                    f"torch {meta.get('torch')}" if meta.get('torch') else None,
                ) if value)
            rows.append([
                f'`{model}`', f'`{fixture}`', f"`{meta.get('dtype', '—')}`",
                f"atol {tolerance[0]}" + (f" / rtol {tolerance[1]}" if tolerance[1] else ''),
                f"`{json.dumps(meta.get('tokens', []), separators=(',', ':'))}`",
                provenance or '—',
            ])
        out += ['#### Integration fixtures', '']
        out += _table(['Model', 'Fixture', 'Compute', 'Tolerance', 'Recorded tokens', 'Provenance'], rows)

    full = record.get('FULL', [])
    if full:
        out += ['#### Whole-model generation', '']
        out += _table(
            ['Model', 'Prompt token ids', 'Expected greedy token ids'],
            [[f'`{model}`', f'`{json.dumps(prompt, separators=(",", ":"))}`',
              f'`{json.dumps(tokens, separators=(",", ":"))}`']
             for model, _checkpoint, prompt, tokens, *_rest in full],
        )

    measurements = record.get('MEASUREMENTS', [])
    if measurements:
        out += ['#### Recorded measurements', '']
        if isinstance(measurements, dict):
            measurements = [{'case': key, 'result': value} for key, value in measurements.items()]
        keys = []
        for measurement in measurements:
            for key in measurement:
                if key not in keys:
                    keys.append(key)
        out += _table([key.replace('_', ' ').title() for key in keys],
                      [[measurement.get(key, '—') for key in keys] for measurement in measurements])
    return out


def render_status(state):
    cat = state['catalog']
    corpus_rows = state['corpus']
    valid = sum(row['validation'] == 'valid' for row in corpus_rows)
    located = sum(row['located'] == 'all tensors' for row in corpus_rows)
    templates = len(cat.get('templates', {}))
    out = [
        '# TensorSpine status', '',
        f"*Generated from commit `{state['commit']}` on {state['date']}. Not tracked; rebuilt by the documentation action.*",
        '',
        'This page reports what the repository validates, derives and runs at build time. '
        'The specification remains the authority on validity and meaning.', '',
        '## Catalog and corpus', '',
    ]
    out += _table(
        ['Catalog contracts', 'Axes', 'Precision roles', 'Concrete documents',
         'Valid as written', 'Fully located', 'Templates'],
        [[len(cat['contracts']), len(cat['axes']), len(cat['precision']), len(corpus_rows),
          valid, located, templates]],
    )
    out += ['### Corpus', '']
    out += _table(
        ['Document', 'Validation', 'Checkpoint locations', 'First refusal or required input'],
        [[f"`{row['name']}`", row['validation'], row['located'], row['detail'] or '—']
         for row in corpus_rows],
    )
    out += ['## Generator coverage', '',
            'A manifest is checked against the catalog before its coverage is reported. '
            'The detailed model-and-generator to-do list is the [branch ledger](../branch-ledger/).', '']
    for item in state['generators']:
        manifest = item['manifest']
        gen = manifest['generator']
        can_run = sum(ok for ok, _reasons in item['verdicts'].values())
        out += [f"### `{gen['name']}`", '',
                f"Manifest `{os.path.relpath(item['path'], ROOT)}`, generated by `{gen['generator']}`.", '']
        out += _table(
            ['Contract entries', 'Contracts without an entry', 'Entries with branch gaps',
             'Corpus documents runnable', 'Contracts without a witness'],
            [[len(manifest['contracts']), len(item['missing']), len(item['branches']),
              f"{can_run}/{len(item['verdicts'])}", len(item['unwitnessed'])]],
        )
        gaps = [[f'`{identity}`', ', '.join(f'`{gap}`' for gap in values)]
                for identity, values in sorted(item['branches'].items())]
        gaps = [[f'`{identity}`', 'contract entry absent'] for identity in item['missing']] + gaps
        out += ['#### Catalog gaps', '']
        out += (_table(['Contract', 'Not admitted'], gaps) if gaps else ['None.', ''])
        out += ['#### Corpus admission', '']
        out += _table(
            ['Document', 'Verdict', 'First reason'],
            [[f'`{name}`', 'can run' if ok else 'cannot run',
              '—' if ok else capabilities_mod.condensed(reasons)[0]]
             for name, (ok, reasons) in item['verdicts'].items()],
        )
    out += ['## Verification', '',
            'Verification data comes from the records consumed by each generator test suite; '
            'the site build does not rerun weight-dependent measurements.', '']
    for item in state['generators']:
        out += _verification_lines(item)
    return '\n'.join(out).rstrip() + '\n'


def render_ledger(state, selected=None):
    items = state['generators']
    if selected:
        items = [item for item in items if item['manifest']['generator']['name'] == selected]
    out = [
        '# TensorSpine branch ledger', '',
        f"*Generated from commit `{state['commit']}` on {state['date']}. Not tracked; rebuilt by the documentation action.*",
        '',
        'For each model-and-generator pair, this is the implementation to-do list: absent contract '
        'entries, unadmitted enum values, record fields and optional arguments, followed by the '
        'admission result for every corpus document.', '',
    ]
    for item in items:
        gen = item['manifest']['generator']
        out += [f"## `{gen['name']}`", '', '### Contract and branch gaps', '']
        gaps = [[f'`{identity}`', 'contract entry absent'] for identity in item['missing']]
        gaps += [[f'`{identity}`', '<br>'.join(f'`{gap}`' for gap in values)]
                 for identity, values in sorted(item['branches'].items())]
        out += (_table(['Contract', 'Work remaining'], gaps) if gaps else ['None.', ''])
        out += ['### Model pairs', '']
        out += _table(
            ['Model', 'Admission', 'Work remaining'],
            [[f'`{name}`', 'can run' if ok else 'cannot run',
              '—' if ok else '<br>'.join(capabilities_mod.condensed(reasons))]
             for name, (ok, reasons) in item['verdicts'].items()],
        )
    return '\n'.join(out).rstrip() + '\n'


def run(kind, model_paths, catalog_bases, schema_dir, manifest_paths, output=None,
        selected_generator=None):
    """Write a status or ledger Markdown document. Return a CLI exit status."""
    try:
        state = facts(model_paths, catalog_bases, schema_dir, manifest_paths)
        text = (render_status(state) if kind == 'status'
                else render_ledger(state, selected=selected_generator))
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            catalog_mod.CatalogError) as error:
        print(f'  status not readable: {error}', file=sys.stderr)
        return 1
    filename = 'status.md' if kind == 'status' else 'branch-ledger.md'
    target = os.path.join(output, filename) if output and os.path.isdir(output) else output
    stream = sys.stderr if target is None else sys.stdout
    if target is None:
        sys.stdout.write(text)
        target = '<stdout>'
    else:
        with open(target, 'w', encoding='utf-8') as destination:
            destination.write(text)
    print(f"  {len(state['corpus'])} documents, {len(state['generators'])} generator(s) -> {target}",
          file=stream)
    return 0

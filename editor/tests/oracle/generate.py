#!/usr/bin/env python3
"""The parity oracle: what the language core must reproduce, produced by the tools themselves.

The editor's `packages/lang` is a TypeScript port of this repository's `tools/`, held to parity
with them (the editor plan's D2). This script runs the tools and records their
answers so that the parity suites of `packages/lang/test/parity/` can compare, document by
document, without a Python process of their own.

Python runs here and nowhere else in the editor: in CI, and on a developer machine before the
parity suites — never at runtime (the implementation plan's §0.2). It reads the repository and
writes only under `editor/tests/oracle/out/`, which is gitignored and regenerated, never
committed.

What it writes (the implementation plan's §0.5):

    out/manifest.json                 everything below, with the digests of tools/ and schemas/
    out/d1/<name>.d1.json             `--d1` for every corpus document and the template
    out/derive/<name>.derived.json    `--derive` for the same
    out/validate/<set>.txt            `--validate` output, one file per set of documents
    out/lint/<set>.txt                `--lint` output, one file per set of documents
    out/primitive-schema/<p>_<v>.json `--document primitive-schema` for the reference base
    out/rejections/*.json             the rejection suite's expectations, copied
    out/signatures/*.json             the signature suite's recorded signatures, copied
    out/structural/index.json         the schema stage's lines for every document and unit
    out/structural/mutations/*.json   documents mutated one place at a time, for the same

The structural stage is recorded on its own because it is read two different ways: a model
document through `validate.structural`, which keeps every top-level error, and a library unit
through `schema.deepest`, which keeps the leaf behind each one — the two readings the tools
themselves take (`validate.py` and `primitive_library._units`). Beside the repository's own
documents it records *mutations*: one corpus document or unit with one value changed, deleted or
added, a hundred at a time, so that the port's message mapping is held to more than the twenty
cases `tests/rejections` happens to carry.

`--validate` and `--lint` are read for a *set* of documents: the corpus in one invocation, as
the repository itself runs them, and the template in another, since a template is read under an
assignment (§4.6). Lint findings name the set ("called by none of the N model(s) linted"), so a
parity test must lint the same set to compare the same words.

The template's assignment is not written here: it is read from `tests/signature.py`, the one
the repository's own suites use.

Usage:  pnpm oracle            (from editor/)
        python3 tests/oracle/generate.py [--out DIR]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EDITOR = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(EDITOR)

TOOL = os.path.join(ROOT, 'tools', 'tensorspine')
MODELS = os.path.join(ROOT, 'data', 'models')
REJECTIONS = os.path.join(ROOT, 'tests', 'rejections')
SIGNATURES = os.path.join(ROOT, 'tests', 'signatures')


def die(message):
    print(f"oracle: {message}", file=sys.stderr)
    raise SystemExit(2)


def repository():
    """The repository this editor is part of, or a clear refusal. The oracle never falls back
    on stale output: without the tools there is no oracle."""
    for path in (TOOL, MODELS, REJECTIONS, SIGNATURES, os.path.join(ROOT, 'schemas')):
        if not os.path.exists(path):
            die(f"{path} is missing — run this from a checkout of the repository")
    try:
        import jsonschema                                          # noqa: F401
    except ImportError:
        die("the tools need jsonschema: python3 -m pip install 'jsonschema==4.25.0'")
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    try:
        from signature import ASSIGNMENTS, corpus, name_of
    except Exception as exc:                                       # pragma: no cover - a broken tree
        die(f"tests/signature.py could not be read ({exc})")
    return ASSIGNMENTS, corpus, name_of


def run(arguments, capture):
    """One invocation of the tools. A refusal is the oracle's refusal: the corpus is expected to
    pass, and recording a broken state as the expectation would make the parity job lie."""
    result = subprocess.run([sys.executable, TOOL, *arguments], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        die(f"`tensorspine {' '.join(arguments)}` exited {result.returncode}")
    return result.stdout if capture else None


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def digest(directory, suffixes):
    """A digest over a directory's files, so that a recorded oracle can be told stale (§9 Q2)."""
    sha = hashlib.sha256()
    for base, _, names in sorted(os.walk(directory)):
        if '__pycache__' in base:
            continue
        for name in sorted(names):
            if not name.endswith(suffixes):
                continue
            path = os.path.join(base, name)
            sha.update(os.path.relpath(path, ROOT).encode())
            with open(path, 'rb') as handle:
                sha.update(handle.read())
    return sha.hexdigest()


def commit():
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def jsonschema_version():
    import importlib.metadata
    return importlib.metadata.version('jsonschema')


# --- the structural stage, recorded both ways (feature 1.1) -----------------

SCHEMAS = os.path.join(ROOT, 'schemas')
PRIMITIVE_LIBRARY = os.path.join(ROOT, 'data', 'primitive-library')

# The documents whose mutations are recorded, and how many of each: a small model, a template, a
# composite, and units of three sizes, so that every shape of the two schemas is reached.
MUTATED = (
    ('data/models/llama3-8b.json', 'model', 60),
    ('data/models/shieldstral-3b-composite.json', 'model', 60),
    ('data/primitive-library/primitives/norm/rms/1.0.0.json', 'primitive-library-unit', 40),
    ('data/primitive-library/primitives/sequence/gated_delta/1.0.0.json',
     'primitive-library-unit', 60),
    ('data/primitive-library/primitives/attention/dense/1.0.0.json',
     'primitive-library-unit', 60),
    ('data/primitive-library/axes/attention/heads.json', 'primitive-library-unit', 20),
    ('data/primitive-library/primitive-library.json', 'primitive-library-unit', 20),
)

MUTATIONS = ('delete', 'number', 'string', 'list', 'object', 'null', 'true', 'extra',
             'duplicate', 'negative', 'empty', 'long')


_UNIT_REGISTRY = {}


def structural_lines(path, role):
    """The schema stage's lines for one file, read the way the tools read that role.

    A model document goes through `validate.structural` — the JSON layer first (V12), then every
    top-level error of the grammar — and so does a derived document, which `d1.check` reads the
    same way. A library unit goes through `schema.deepest`, which is what
    `primitive_library._units` prints: the leaf behind each error rather than the branch point.
    """
    import schema as schema_mod
    import validate as validate_mod
    if role != 'primitive-library-unit':
        return validate_mod.structural(path, SCHEMAS, role=role)
    if role not in _UNIT_REGISTRY:
        _UNIT_REGISTRY[role] = (schema_mod.locate(SCHEMAS, role), schema_mod.registry(SCHEMAS))
    unit_schema, registry = _UNIT_REGISTRY[role]
    return [schema_mod.format_error(e)
            for e in schema_mod.deepest(schema_mod.check(unit_schema, path, registry))]


def _places(node, path=()):
    """Every place inside a document, depth first, the root left out."""
    if isinstance(node, dict):
        for name, value in node.items():
            yield path + (name,)
            yield from _places(value, path + (name,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield path + (index,)
            yield from _places(value, path + (index,))


def _mutate(document, place, operation):
    """One document with one place changed, or None when the operation does not apply there."""
    document = json.loads(json.dumps(document))
    parent = document
    for step in place[:-1]:
        parent = parent[step]
    last = place[-1]
    if operation == 'delete':
        if isinstance(parent, dict):
            del parent[last]
        else:
            parent.pop(last)
        return document
    if operation == 'extra':
        target = parent[last]
        if not isinstance(target, dict):
            return None
        target['zzz'] = 1
        return document
    if operation == 'duplicate':
        if not isinstance(parent, list):
            return None
        parent.insert(last, json.loads(json.dumps(parent[last])))
        return document
    parent[last] = {'number': 7, 'string': 'mutant', 'list': [], 'object': {},
                    'null': None, 'true': True, 'negative': -1, 'empty': '',
                    'long': 'x' * 600}[operation]
    return document


def mutations(document, count):
    """`count` mutations of one document, spread over its places, one operation each in turn."""
    places = list(_places(document))
    if not places:
        return []
    stride = max(1, len(places) // count)
    made = []
    for index, place in enumerate(places[::stride]):
        if len(made) >= count:
            break
        operation = MUTATIONS[index % len(MUTATIONS)]
        mutated = _mutate(document, place, operation)
        if mutated is None:
            continue
        made.append(('/'.join(str(step) for step in place), operation, mutated))
    return made


def structural(out, derived):
    """Every document and unit of the repository, and a spread of mutations, through the stage."""
    cases = []
    files = [(os.path.relpath(path, ROOT), 'model')
             for path in sorted(glob.glob(os.path.join(MODELS, '**', '*.json'), recursive=True))]
    files += [(os.path.relpath(path, ROOT), 'primitive-library-unit')
              for path in sorted(glob.glob(os.path.join(PRIMITIVE_LIBRARY, '**', '*.json'),
                                           recursive=True))]
    files += [(os.path.join('tests', 'rejections', 'models', name), 'model')
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]
    for base in sorted(os.listdir(os.path.join(REJECTIONS, 'primitive-library'))):
        root = os.path.join(REJECTIONS, 'primitive-library', base)
        if not os.path.isdir(root):
            continue
        files += [(os.path.relpath(path, ROOT), 'primitive-library-unit')
                  for path in sorted(glob.glob(os.path.join(root, '**', '*.json'),
                                               recursive=True))]
    for relative, role in files:
        cases.append({'document': relative, 'file': None, 'role': role,
                      'lines': structural_lines(os.path.join(ROOT, relative), role)})

    # The derived products, against the schema of their own role: the one place the cross-file
    # references of `derived.json` — into the model schema and the unit schema — are exercised.
    for relative in derived:
        cases.append({'document': None, 'file': relative, 'role': 'derived',
                      'lines': structural_lines(os.path.join(out, relative), 'derived')})

    made = os.path.join(out, 'structural', 'mutations')
    os.makedirs(made)
    number = 0
    for relative, role, count in MUTATED + tuple(
            (name, 'derived', 40) for name in derived if name.startswith('d1/colbert')):
        source = ROOT if os.path.exists(os.path.join(ROOT, relative)) else out
        with open(os.path.join(source, relative), encoding='utf-8') as handle:
            document = json.load(handle)
        for place, operation, mutated in mutations(document, count):
            number += 1
            name = f"{number:04d}.json"
            path = os.path.join(made, name)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(mutated, handle)
            cases.append({'document': None, 'file': f"structural/mutations/{name}",
                          'role': role, 'source': relative, 'place': place,
                          'mutation': operation,
                          'lines': structural_lines(path, role)})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=os.path.join(HERE, 'out'),
                        help='where to write (default: editor/tests/oracle/out)')
    options = parser.parse_args()

    assignments, corpus, name_of = repository()
    out = os.path.abspath(options.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    # The tools decide between one file and one file per model by asking whether the -o path is
    # a directory, so every directory is made before they are called.
    for directory in ('d1', 'derive', 'validate', 'lint', 'primitive-schema'):
        os.makedirs(os.path.join(out, directory))

    models = [path for path in corpus() if os.path.dirname(path) == MODELS]
    templates = [path for path in corpus() if os.path.dirname(path) != MODELS]
    documents = []

    print(f"oracle: {len(models)} model(s) and {len(templates)} template(s) -> {out}")

    # The corpus, in one invocation each: the tools load the primitive library once.
    run(['--d1', '-o', os.path.join(out, 'd1'), MODELS], capture=False)
    run(['--derive', '-o', os.path.join(out, 'derive'), MODELS], capture=False)
    write(os.path.join(out, 'validate', 'corpus.txt'), run(['--validate', MODELS], capture=True))
    write(os.path.join(out, 'lint', 'corpus.txt'), run(['--lint', MODELS], capture=True))
    for path in models:
        name = name_of(path)
        documents.append({
            'name': name,
            'path': os.path.relpath(path, ROOT),
            'assignment': None,
            'd1': f"d1/{name}.d1.json",
            'derived': f"derive/{name}.derived.json",
            'validate': 'validate/corpus.txt',
            'lint': 'lint/corpus.txt',
        })

    # A template denotes one graph per admissible assignment (§4.6), so it is read alone under
    # the assignment the repository's suites use.
    for path in templates:
        name = name_of(path)
        assignment = assignments.get(name)
        if assignment is None:
            die(f"{name}: tests/signature.py records no assignment for this template")
        assigned = ['--assign', json.dumps(assignment)]
        run(['--d1', '-o', os.path.join(out, 'd1', f"{name}.d1.json"), *assigned, path],
            capture=False)
        run(['--derive', '-o', os.path.join(out, 'derive', f"{name}.derived.json"), *assigned,
             path], capture=False)
        write(os.path.join(out, 'validate', f"{name}.txt"),
              run(['--validate', *assigned, path], capture=True))
        write(os.path.join(out, 'lint', f"{name}.txt"), run(['--lint', *assigned, path],
                                                            capture=True))
        documents.append({
            'name': name,
            'path': os.path.relpath(path, ROOT),
            'assignment': assignment,
            'd1': f"d1/{name}.d1.json",
            'derived': f"derive/{name}.derived.json",
            'validate': f"validate/{name}.txt",
            'lint': f"lint/{name}.txt",
        })

    # The generated artifacts the editor consumes and never regenerates (plan §7 F5): here as
    # the parity material for the argument sheet's schemas.
    run(['--document', 'primitive-schema', '-o', os.path.join(out, 'primitive-schema')],
        capture=False)
    primitive_schemas = sorted(os.listdir(os.path.join(out, 'primitive-schema')))

    # The rejection suite's expectations and the signature suite's recorded signatures: the
    # wording contract (plan §7 F1) and the denoted graphs (D2).
    os.makedirs(os.path.join(out, 'rejections'))
    for name in sorted(os.listdir(REJECTIONS)):
        if name.endswith('.json'):
            shutil.copy(os.path.join(REJECTIONS, name), os.path.join(out, 'rejections', name))
    os.makedirs(os.path.join(out, 'signatures'))
    for name in sorted(os.listdir(SIGNATURES)):
        if name.endswith('.json'):
            shutil.copy(os.path.join(SIGNATURES, name), os.path.join(out, 'signatures', name))

    rejection_documents = sorted(
        os.path.join('models', name)
        for name in os.listdir(os.path.join(REJECTIONS, 'models')) if name.endswith('.json'))
    rejection_bases = sorted(
        os.path.join('primitive-library', name)
        for name in os.listdir(os.path.join(REJECTIONS, 'primitive-library'))
        if os.path.isdir(os.path.join(REJECTIONS, 'primitive-library', name)))

    derived_products = sorted(
        [f"d1/{name}" for name in os.listdir(os.path.join(out, 'd1'))]
        + [f"derive/{name}" for name in os.listdir(os.path.join(out, 'derive'))])
    structural_cases = structural(out, derived_products)
    print(f"oracle: {len(structural_cases)} structural case(s)")

    manifest = {
        'generated_by': 'editor/tests/oracle/generate.py',
        'repository_commit': commit(),
        'tools_sha256': digest(os.path.join(ROOT, 'tools'), ('.py', 'tensorspine')),
        'schemas_sha256': digest(os.path.join(ROOT, 'schemas'), ('.json',)),
        'python': '.'.join(str(part) for part in sys.version_info[:3]),
        'jsonschema': jsonschema_version(),
        'note': ('--validate and --lint are read for a set of documents: lint findings name the '
                 'set they were computed over, so a comparison must use the same set.'),
        'documents': documents,
        'primitive_schemas': {'directory': 'primitive-schema', 'files': primitive_schemas},
        'rejections': {
            'models': 'rejections/models.json',
            'primitive_library': 'rejections/primitive-library.json',
            'documents': rejection_documents,
            'bases': rejection_bases,
        },
        'signatures': [f"signatures/{name}" for name in
                       sorted(os.listdir(os.path.join(out, 'signatures')))],
        'structural': {
            'index': 'structural/index.json',
            'note': ('a model document is read through validate.structural, a library unit '
                     'through schema.deepest, which is how the tools read each of them.'),
            'cases': len(structural_cases),
        },
    }
    write(os.path.join(out, 'structural', 'index.json'),
          json.dumps({'cases': structural_cases}, indent=1) + '\n')
    write(os.path.join(out, 'manifest.json'), json.dumps(manifest, indent=2) + '\n')
    print(f"oracle: {len(documents)} document(s), {len(primitive_schemas)} primitive schema(s), "
          f"{len(rejection_documents)} rejection document(s), "
          f"{len(rejection_bases)} rejection base(s)")


if __name__ == '__main__':
    main()

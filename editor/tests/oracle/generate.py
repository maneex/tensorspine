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
    }
    write(os.path.join(out, 'manifest.json'), json.dumps(manifest, indent=2) + '\n')
    print(f"oracle: {len(documents)} document(s), {len(primitive_schemas)} primitive schema(s), "
          f"{len(rejection_documents)} rejection document(s), "
          f"{len(rejection_bases)} rejection base(s)")


if __name__ == '__main__':
    main()

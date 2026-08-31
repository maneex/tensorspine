#!/usr/bin/env python3
"""The ZML generator's test harness — run from the repository root.

Builds `@tensorspine//:tspl` in a ZML checkout, then checks the generator against the
language's own tools: for every corpus document, what `tspl` reads out of the derived
document must equal what `tools/derive.py` put in it.

    generators/zml/tests/run_zml.py [--zml DIR] [--model NAME] [--keep]

`--zml` defaults to $ZML_HOME, then ~/work/perso/zml. Prints `skip` and exits 0 when the
checkout or its Bazel wrapper is absent, so the suite is runnable on a machine without one.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # generators/zml/tests
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
MODELS = os.path.join(ROOT, 'data', 'models')
TARGET = '@tensorspine//:tspl'

COUNTS = re.compile(
    r'(\d+) occurrences, (\d+) values, (\d+) parameter tensors, '
    r'(\d+) states, (\d+) edges, (\d+) ordered')


def expected(path):
    """What the language put in the document, counted here."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return (len(d['d1']['nodes']), len(d['d2']['values']), len(d['d3']['tensors']),
            len(d['d4']['states']), len(d['d1']['edges']), len(d['d1']['topological_order']))


def derive(out_dir, only=None):
    """The derived documents, through the language's own tool — never re-implemented here."""
    args = [os.path.join(ROOT, 'tools', 'tensorspine'), '--derive', '-o', out_dir]
    args.append(os.path.join(MODELS, f'{only}.json') if only else MODELS)
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return sorted(f for f in os.listdir(out_dir) if f.endswith('.derived.json'))


def evaluate(binary, derived, checkpoint, out_dir):
    """The numbers, against an oracle computed here from the checkpoint alone.

    Not against the reference generator: these two values are short enough to write
    down independently, so the check does not inherit anybody's reading of them. The
    reference's fixtures take over at the first layer cut, where the mathematics stops
    being one line."""
    try:
        import numpy as np
        import torch
        from safetensors.torch import safe_open
    except ImportError as e:
        print(f'skip: evaluation needs numpy, torch and safetensors ({e})')
        return 0

    index = os.path.join(checkpoint, 'model.safetensors.index.json')
    if not os.path.isfile(index):
        print(f'skip: no checkpoint at {checkpoint}')
        return 0
    weight_map = json.load(open(index, encoding='utf-8'))['weight_map']

    def weight(name):
        with safe_open(os.path.join(checkpoint, weight_map[name]), framework='pt') as f:
            return f.get_tensor(name).to(torch.float32).numpy()

    ids = [128000, 791, 6864, 315, 9822, 374]
    with open(derived, encoding='utf-8') as f:
        doc = json.load(f)
    eps = doc['d1']['nodes']['decoder/attn_n[layer=0]']['arguments']['eps']

    rows = weight('model.embed_tokens.weight')[ids]
    scale = weight('model.layers.0.input_layernorm.weight')
    normed = rows / np.sqrt((rows ** 2).mean(-1, keepdims=True) + eps) * scale

    failed = 0
    for value, want, tol in (('embed.output', rows, 0.0),
                             ('decoder/attn_n[layer=0].output', normed, 1e-6)):
        path = os.path.join(out_dir, 'value.bin')
        run = subprocess.run([binary, f'--derived={derived}', f'--checkpoint={checkpoint}',
                              f'--until={value}', f'--out={path}'], capture_output=True)
        if run.returncode != 0:
            print(f'FAIL {value}: {run.stderr.decode(errors="replace")[-600:]}')
            failed += 1
            continue
        got = np.fromfile(path, dtype=np.float32).reshape(want.shape)
        err = float(np.abs(got - want).max())
        ok = err <= tol
        print(f'{"OK  " if ok else "FAIL"} {value}: max abs {err:.3e} (tolerance {tol:.0e})')
        failed += 0 if ok else 1
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zml', default=os.environ.get('ZML_HOME', os.path.expanduser('~/work/perso/zml')))
    ap.add_argument('--model', help='one document by name, instead of the whole corpus')
    ap.add_argument('--checkpoint', help='also evaluate llama3-8b against an oracle computed from this checkpoint')
    ap.add_argument('--keep', action='store_true', help='keep the derived documents')
    a = ap.parse_args()

    bazel = os.path.join(a.zml, 'bazel.sh')
    if not os.path.isfile(bazel):
        print(f'skip: no ZML checkout at {a.zml} (--zml or $ZML_HOME)')
        return 0
    if not os.path.isfile(os.path.join(a.zml, 'MODULE.bazel')):
        print(f'skip: {a.zml} has no MODULE.bazel')
        return 0
    with open(os.path.join(a.zml, 'MODULE.bazel'), encoding='utf-8') as f:
        if 'tensorspine' not in f.read():
            print(f'FAIL: {a.zml}/MODULE.bazel does not declare the tensorspine repository.\n'
                  f'      See generators/zml/README.md for the two lines it needs.')
            return 1

    build = subprocess.run([bazel, 'build', TARGET], cwd=a.zml,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if build.returncode != 0:
        sys.stderr.write(build.stderr.decode(errors='replace')[-4000:])
        print('FAIL: build')
        return 1
    binary = os.path.join(a.zml, 'bazel-bin', 'external', '+local_repository+tensorspine', 'tspl')
    if not os.path.isfile(binary):
        print(f'FAIL: built, but no binary at {binary}')
        return 1

    out_dir = tempfile.mkdtemp(prefix='tspl-derived-')
    try:
        names = derive(out_dir, a.model)
        failed = 0
        for name in names:
            path = os.path.join(out_dir, name)
            model = name[:-len('.derived.json')]
            run = subprocess.run([binary, f'--derived={path}'], capture_output=True)
            out = (run.stdout + run.stderr).decode(errors='replace')
            m = COUNTS.search(out)
            if run.returncode != 0 or not m:
                print(f'FAIL {model}: {out.strip().splitlines()[-1] if out.strip() else "no output"}')
                failed += 1
                continue
            got, want = tuple(int(x) for x in m.groups()), expected(path)
            if got != want:
                print(f'FAIL {model}: tspl {got} != derive {want}')
                failed += 1
            else:
                print(f'OK   {model}: {got[0]} occurrences, {got[2]} tensors, {got[3]} states')
        checked = len(names)
        if a.checkpoint:
            llama = os.path.join(out_dir, 'llama3-8b.derived.json')
            if os.path.isfile(llama):
                print()
                before = failed
                failed += evaluate(binary, llama, a.checkpoint, out_dir)
                checked += 2
                if failed == before:
                    pass
        print(f'\n{checked - failed} passed, {failed} failed')
        return 1 if failed else 0
    finally:
        if a.keep:
            print(f'derived documents kept in {out_dir}')
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())

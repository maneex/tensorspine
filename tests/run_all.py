#!/usr/bin/env python3
"""One command for every suite (README §3): the twelve language suites, then the generator
harnesses by what the environment offers — the reference harness (the witness inside it) needs
`torch`, the ONNX harness `onnxruntime`, the ZML harness a checkout at `$ZML_HOME` — every skip
printed with its reason, exit 1 on any failure. Checkpoints and samples come from
`$TENSORSPINE_MODEL_ARTIFACTS`; without it every fixture check inside a harness says `skip`, as
it does when run alone. `tests/requirements.txt` pins the environment the evidence was produced
in; the language suites need `jsonschema` alone.

    python3 tests/run_all.py [--language-only] [--no-strict-provenance]

`--no-strict-provenance` is passed to the reference harness (docs/TENSORSPINE-FIXTURE.md §5): the
witness's regeneration from the seed is printed, not required exact — the rule away from the box
the fixtures were recorded on.
"""
import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LANGUAGE = ('run_rejections', 'run_templates', 'run_states', 'run_expressions', 'run_signatures', 'run_costs',
            'run_derived', 'run_artifact', 'run_fixtures', 'run_capabilities', 'run_status', 'run_harness')


def generators(strict_provenance):
    """(name, command, reason it is skipped or None) for every generator harness."""
    torch = importlib.util.find_spec('torch') is not None
    ort = importlib.util.find_spec('onnxruntime') is not None
    zml = os.environ.get('ZML_HOME')
    return [
        ('reference', [os.path.join('generators', 'reference', 'tests', 'run_reference.py')] + ([] if strict_provenance else ['--no-strict-provenance']),
         None if torch else 'torch is not importable'),
        ('onnx', [os.path.join('generators', 'onnx', 'tests', 'run_onnx.py')],
         None if torch and ort else ('onnxruntime is not importable' if torch else 'torch is not importable')),
        ('zml', [os.path.join('generators', 'zml', 'tests', 'run_zml.py')],
         None if zml else 'ZML_HOME is not set'),
    ]


def run(name, command):
    t0 = time.time()
    p = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and 'Warning' not in l]
    ok = p.returncode == 0
    print(f"  {'ok  ' if ok else 'FAIL'} {name:12s} {lines[-1] if lines else '(no output)'}  ({time.time() - t0:.0f}s)")
    if not ok:
        for l in lines[-30:]:
            print(f"         {l}")
    return ok


def main(argv):
    language_only = '--language-only' in argv
    strict = '--no-strict-provenance' not in argv
    ok = True
    print(f"language  {len(LANGUAGE)} suite(s)")
    for name in LANGUAGE:
        ok &= run(name, [os.path.join('tests', name + '.py')])
    if language_only:
        print("all: language suites green" if ok else "all: FAILED")
        return 0 if ok else 1
    print("generators")
    for name, command, reason in generators(strict):
        if reason:
            print(f"  skip {name:12s} {reason}")
            continue
        ok &= run(name, command)
    print("all: green" if ok else "all: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

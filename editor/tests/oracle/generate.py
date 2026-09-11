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
    out/expressions/index.json        every expression and condition of the corpus and of the
                                      reference base, with the environment it was evaluated in
                                      and the value `tools/expr.py` answered
    out/library/index.json            the reference base gathered, the 33 library rejection cases
                                      refused word for word, and `primitive_references` over one
                                      mutation at a time of every unit of the reference base
    out/model/index.json              `model.load` over every model document of the repository and
                                      over edited ones, with the refusals it raises
    out/model/normalised/*.json       each of those documents with its scoped bindings hoisted
    out/quantities/index.json         `check_quantities`, `check_assignment` and the assignment
                                      report over every model document, and over the edited
                                      documents and assignments of `quantity_cases.py`
    out/arguments/index.json          `resolve_arguments`, the invariants (V8) and the facts the
                                      argument sheet reads, per site of every model document and
                                      of every template it instantiates, plus the synthetic
                                      declarations of `argument_cases.py`
    out/graph/index.json              `analyse` as far as V19 over every model document: the
                                      refusals, the counters, the public interface ports, and the
                                      expanded graph — sites, guards, edges, domains, order
    out/bindings/index.json           the whole `analyse` over the same documents, and over edited
                                      ones: the refusals, the counters, the advisories, the
                                      instance keys, what is carried, the physical names, the
                                      bound slots and the identity instances
    out/bindings/documents/*.json     each edited document as the tools were handed it
    out/expansion/index.json          `d1.emit` over every model document of the repository and
                                      over edited ones: the emitted document or the exception
    out/expansion/emitted/*.json      each emitted D1, as `--d1` writes one
    out/expansion/documents/*.json    each edited document as the tools were handed it
    out/artifact/index.json           `artifact.check` over every model document against a
                                      checkpoint synthesised from its own D3, over mutations of
                                      those headers, and over synthetic D3s for every location form
    out/artifact/headers/*.json       each synthesised checkpoint, recorded literally

The expressions are recorded as *cases* rather than as a walk: each one carries the expression,
the quantities, the index environment or the resolved arguments it was evaluated against, and
the answer — so the parity suite replays them without a walk of its own, and a walk that moves
(the validator's, D1's) cannot silently change what the evaluator is held to. Beside the
corpus's own expressions it records a synthetic table over every operator and every comparison,
because the corpus writes five of the eleven operators and no `divide`, `min`, `max`, `negate`
or `absolute` at all.

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
import copy
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EDITOR = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(EDITOR)

TOOL = os.path.join(ROOT, 'tools', 'tensorspine')
MODELS = os.path.join(ROOT, 'data', 'models')
REJECTIONS = os.path.join(ROOT, 'tests', 'rejections')
# The base every suite of the repository reads a model under: `tests/run_rejections.py` loads it
# explicitly, and every corpus document pins it through its own `primitive_libraries`.
REFERENCE_BASE = os.path.join('data', 'primitive-library')
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


# --- expressions and conditions, evaluated where they stand (feature 1.2) --

EXPRESSION_TAGS = ({'literal'}, {'quantity'}, {'index'}, {'argument'}, {'op', 'args'},
                   {'if', 'then', 'else'})
CONDITION_TAGS = ({'boolean'}, {'not'}, {'all'}, {'any'}, {'present'}, {'compare'})

# How many index environments of a grid are evaluated, and how many of a site's argument maps
# are handed to the primitive side. The first few cover the residues a periodic guard tests
# (`layer mod 5 = 4`), the last two the boundary a `layers - 1` override reaches.
ENVIRONMENTS = 6
LAST = 2
ARGUMENT_ENVIRONMENTS = 3


def encode(value):
    """A value as the fixture carries it: tagged, and exact.

    An integer is written in decimal digits and a float as `repr` writes it, both as text, so
    that neither the fixture's own JSON nor the reader's number parsing can lose the distinction
    the language reads (V3) — which is the whole point of comparing the two implementations
    here."""
    from expr import UNRESOLVED
    if value is UNRESOLVED:
        return {'unresolved': True}
    if value is None:
        return {'none': True}
    if isinstance(value, bool):
        return {'bool': value}
    if isinstance(value, int):
        return {'int': str(value)}
    if isinstance(value, float):
        return {'float': repr(value)}
    if isinstance(value, str):
        return {'str': value}
    if isinstance(value, dict):
        return {'record': {name: encode(one) for name, one in value.items()}}
    if isinstance(value, (list, tuple)):
        return {'list': [encode(one) for one in value]}
    die(f"the expression fixture cannot encode {value!r}")


def encode_map(mapping):
    return {name: encode(one) for name, one in mapping.items()}


def _classified(node):
    """`expression`, `condition`, or None: what a node is, by the keys it carries.

    The tags are the grammar's own — the required keys of the alternatives of `scalar_expression`
    and of `condition`, on both sides of the language — and a node is one only when its key set
    is exactly an alternative's, so a map whose names happen to include `literal` is not mistaken
    for one."""
    if not isinstance(node, dict):
        return None
    keys = set(node)
    if keys == {'literal'} and not isinstance(node['literal'], (dict, list)):
        return 'expression'
    if keys in ({'quantity'}, {'index'}, {'argument'}) and isinstance(next(iter(node.values())), str):
        return 'expression'
    if keys in EXPRESSION_TAGS:
        return 'expression'
    if keys == {'present'} and isinstance(node['present'], str):
        return 'condition'
    if keys in CONDITION_TAGS:
        return 'condition'
    return None


def _nodes(node, path=''):
    """Every expression and condition of a subtree, outermost first; a node's own parts are left
    to the evaluator, which recurses into them."""
    kind = _classified(node)
    if kind is not None:
        yield path, kind, node
        return
    if isinstance(node, dict):
        for name, one in node.items():
            yield from _nodes(one, f"{path}/{name}")
    elif isinstance(node, list):
        for index, one in enumerate(node):
            yield from _nodes(one, f"{path}/{index}")


def _envs(names, ranges):
    """The index environments of a grid that are evaluated, first few and last few."""
    import itertools
    combos = list(itertools.product(*ranges))
    kept = combos[:ENVIRONMENTS] + combos[-LAST:] if len(combos) > ENVIRONMENTS else combos
    seen, out = set(), []
    for combo in kept:
        if combo in seen:
            continue
        seen.add(combo)
        out.append(dict(zip(names, combo)))
    return out


class Cases:
    """The cases collected, deduplicated by what they are: form, expression, environment."""

    def __init__(self):
        self.cases = []
        self.environments = []
        self._interned = {}
        self._seen = set()

    def intern(self, arguments):
        encoded = encode_map(arguments)
        key = json.dumps(encoded, sort_keys=True)
        if key not in self._interned:
            self._interned[key] = len(self.environments)
            self.environments.append(encoded)
        return self._interned[key]

    def add(self, where, form, node, result, **rest):
        key = json.dumps([form, node, rest], sort_keys=True, default=str)
        if key in self._seen:
            return
        self._seen.add(key)
        self.cases.append({'where': where, 'form': form, 'expression': node,
                           'result': result, **rest})


def _model_cases(cases, name, quantities, subtree, env, where):
    """Every expression and condition of a subtree, evaluated in one index environment."""
    from expr import model_condition, model_value
    for path, kind, node in _nodes(subtree):
        form = 'model_value' if kind == 'expression' else 'model_condition'
        answer = (model_value(node, quantities, env) if kind == 'expression'
                  else model_condition(node, quantities, env))
        cases.add(f"{name}{where}{path}", form, node, encode(answer),
                  document=name, env=encode_map(env))


def _arguments_cases(cases, name, quantities, arguments, env, where):
    """`static_argument` over an instance's argument map, records included."""
    from expr import static_argument
    for argument, value in arguments.items():
        cases.add(f"{name}{where}/arguments/{argument}", 'static_argument', value,
                  encode(static_argument(value, quantities, env)),
                  document=name, env=encode_map(env))


def _primitive_cases(cases, name, definition, arguments, where):
    """Every expression and condition of a primitive declaration, against resolved arguments."""
    from expr import argument_references, primitive_condition, primitive_value
    index = cases.intern(arguments)
    for path, kind, node in _nodes(definition):
        if kind == 'expression':
            cases.add(f"{name}{where}{path}", 'primitive_value', node,
                      encode(primitive_value(node, arguments)), arguments=index)
            continue
        try:
            answer = encode(primitive_condition(node, arguments))
        except TypeError as error:                       # this side does not catch what the
            answer = {'error': f"TypeError: {error}"}    # model side catches; recorded as it is
        cases.add(f"{name}{where}{path}", 'primitive_condition', node, answer, arguments=index)
        cases.add(f"{name}{where}{path}", 'argument_references', node,
                  sorted(argument_references(node)))


def _literal(value):
    return {'literal': value}


def _algebra(cases):
    """Every operator and every comparison over the kinds of value the algebra admits.

    The corpus writes five of the eleven operators; this table writes all of them, over the
    integers, the floats, the booleans and the strings, on both sides of zero — so that
    `ceil_divide` on a negative, a division by zero and Python's numeric tower are held to the
    tools rather than to a reading of them."""
    from expr import model_condition, model_value
    unary = ['negate', 'absolute']
    binary = ['subtract', 'divide', 'floor_divide', 'ceil_divide', 'modulo']
    nary = ['add', 'multiply', 'min', 'max']
    comparisons = ['equal', 'not_equal', 'less', 'less_or_equal', 'greater', 'greater_or_equal']
    scalars = [0, 1, 2, 7, -7, 32, 4096, True, False, 0.0, -0.0, 1.0, 1.5, -1.5, 7.5, -7.5,
               2.0, 1e-05, 'ab', 'a']
    for op in unary:
        for value in scalars:
            node = {'op': op, 'args': [_literal(value)]}
            cases.add(f"algebra/{op}/{value!r}", 'model_value', node,
                      encode(model_value(node, {})), document=None, env={})
    for op in binary + nary:
        for left in scalars:
            for right in scalars:
                node = {'op': op, 'args': [_literal(left), _literal(right)]}
                cases.add(f"algebra/{op}/{left!r},{right!r}", 'model_value', node,
                          encode(model_value(node, {})), document=None, env={})
    for op in nary:
        for third in scalars:
            node = {'op': op, 'args': [_literal(3), _literal(-2), _literal(third)]}
            cases.add(f"algebra/{op}/3,-2,{third!r}", 'model_value', node,
                      encode(model_value(node, {})), document=None, env={})
    for operator in comparisons:
        for left in scalars:
            for right in scalars:
                node = {'compare': {'operator': operator, 'left': _literal(left),
                                    'right': _literal(right)}}
                cases.add(f"algebra/{operator}/{left!r},{right!r}", 'model_condition', node,
                          encode(model_condition(node, {})), document=None, env={})
    for truth in (True, False):
        node = {'if': {'boolean': truth}, 'then': {'literal': 1}, 'else': {'literal': 2.0}}
        cases.add(f"algebra/if/{truth}", 'model_value', node, encode(model_value(node, {})),
                  document=None, env={})


def expressions(assignments, corpus, name_of):
    """Every expression of the corpus and of the reference base, where it stands."""
    import model as model_mod
    import primitive_library as primitive_library_mod
    import validate as validate_mod
    from expr import (index_grid, missing_assignment, resolve_quantities, static_argument)

    cases = Cases()
    documents = []
    _algebra(cases)
    for path in corpus():
        name = name_of(path)
        document = model_mod.load(path)
        assignment = assignments.get(name)
        quantities = resolve_quantities(document, assignment)
        cat = primitive_library_mod.load_for(path, document)
        grids = {}

        # (a) the quantities themselves: derivations, declared defaults, domain bounds.
        for quantity, declaration in document['quantities'].items():
            _model_cases(cases, name, quantities, declaration, {},
                         f"/quantities/{quantity}")
        # (b) everything written at the top level in no index scope.
        for section in ('constants', 'interfaces'):
            _model_cases(cases, name, quantities, document.get(section, {}), {},
                         f"/{section}")
        # (c) the top-level instances, and their arguments through `static_argument`.
        for instance, o in document['instances'].items():
            _model_cases(cases, name, quantities, o, {}, f"/instances/{instance}")
            _arguments_cases(cases, name, quantities, o['arguments'], {}, f"/instances/{instance}")
        # (d) the compositions, at the index environments their grid unrolls to.
        for composition, definition in document.get('compositions', {}).items():
            names, ranges = index_grid(definition['indices'], quantities)
            grids[composition] = {'names': names,
                                  'ranges': [[encode(v) for v in r] for r in ranges]}
            for env in _envs(names, ranges):
                for site, o in definition['instances'].items():
                    where = f"/compositions/{composition}/instances/{site}"
                    _model_cases(cases, name, quantities, o, env, where)
                    _arguments_cases(cases, name, quantities, o['arguments'], env, where)
        # (e) the bindings, normalised, each under its own `for_each` grid.
        for kind, rules in document['bindings'].items():
            for rule, binding in rules.items():
                where = f"/bindings/{kind}/{rule}"
                if 'for_each' in binding:
                    names, ranges = index_grid(binding['for_each'], quantities)
                    envs = _envs(names, ranges)
                else:
                    envs = [{}]
                for env in envs:
                    _model_cases(cases, name, quantities, binding, env, where)

        # (f) the primitive side: every declaration, against the arguments its instances give it.
        sites = [('', instance, o) for instance, o in document['instances'].items()]
        for composition, definition in document.get('compositions', {}).items():
            names, ranges = index_grid(definition['indices'], quantities)
            for env in _envs(names, ranges)[:ARGUMENT_ENVIRONMENTS]:
                sites += [(env, f"{composition}.{site}", o)
                          for site, o in definition['instances'].items()]
        for env, instance, o in sites:
            definition = primitive_library_mod.primitive(cat, o['primitive'])
            if definition is None:
                continue
            try:
                arguments, _problems = validate_mod.resolve_arguments(
                    definition, o['arguments'],
                    lambda v, _e=env: static_argument(v, quantities, _e))
            except (ValueError, TypeError, KeyError):
                continue                       # a document the validator refuses; not our case
            _primitive_cases(cases, name, definition, arguments, f"/primitive/{instance}")

        documents.append({
            'name': name,
            'path': os.path.relpath(path, ROOT),
            'assignment': encode_map(assignment) if assignment else None,
            'quantities': encode_map(quantities),
            'missing': missing_assignment(document, assignment),
            'grids': grids,
        })
    return {'documents': documents, 'environments': cases.environments, 'cases': cases.cases}


# --- model normalisation (feature 1.4) --------------------------------------


def _edit(document, pointer, operation, value):
    """One edit at a JSON pointer, applied in place.

    No member name of a model document holds `/` or `~`, so the pointer is split and nothing is
    unescaped — the same reading the library step's mutations take, so that both implementations
    walk the same path."""
    steps = pointer.split('/')[1:]
    cursor = document
    for step in steps[:-1]:
        cursor = cursor[int(step)] if isinstance(cursor, list) else cursor[step]
    last = steps[-1]
    if operation == 'delete':
        if isinstance(cursor, list):
            cursor.pop(int(last))
        else:
            del cursor[last]
        return
    if isinstance(cursor, list):
        cursor[int(last)] = value
    else:
        cursor[last] = value


def _no_floats(value, where):
    """A fixture value carries no float: the integer/float distinction is what is being compared,
    and a float in the fixture would make the fixture's own JSON decide it."""
    if isinstance(value, float):
        die(f"{where}: an edit value carries a float ({value!r}); write whole numbers only")
    if isinstance(value, dict):
        for name, one in value.items():
            _no_floats(one, where)
    elif isinstance(value, list):
        for one in value:
            _no_floats(one, where)


def _normalised(out, slug, document):
    """`json.dumps(model.normalise(document))`, written where the parity suite reads it."""
    import model as model_mod
    text = json.dumps(model_mod.normalise(document), indent=2, ensure_ascii=False) + '\n'
    write(os.path.join(out, 'model', 'normalised', f"{slug}.json"), text)
    return f"model/normalised/{slug}.json"


def model(out, corpus, name_of):
    """`model.load` over every model document of the repository, and over edited ones.

    Every document of `data/models/` and of `tests/rejections/models/` is recorded normalised —
    the whole text, so that a parity failure names the line that moved — or, for the three that
    refuse, with the `ModelError` the tools raise. Beside them the edited cases of
    `model_cases.py` reach the branches no document of the repository takes: a scoped `constants`
    rule, a declared `tensor` or `identity`, a second index, an endpoint written as a selector,
    and the refusal `tests/rejections` has no case for.
    """
    import model as model_mod
    from model_cases import CASES, SOURCE

    def refusal(path):
        try:
            return None, model_mod.load(path)
        except model_mod.ModelError as error:
            return {'type': 'ModelError', 'message': str(error)}, None

    documents = []
    paths = [(name_of(path), os.path.relpath(path, ROOT)) for path in corpus()]
    paths += [(f"rejection-{name[:-len('.json')]}",
               os.path.join('tests', 'rejections', 'models', name))
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]
    for slug, relative in paths:
        error, document = refusal(os.path.join(ROOT, relative))
        if error is not None:
            documents.append({'path': relative, 'error': error, 'normalised': None})
            continue
        documents.append({'path': relative, 'error': None,
                          'normalised': _normalised(out, slug, document)})

    cases = []
    source = os.path.join(ROOT, SOURCE)
    for name, edits in CASES:
        for edit in edits:
            if edit['op'] == 'set':
                _no_floats(edit['value'], name)
        with open(source, encoding='utf-8') as handle:
            document = json.load(handle, object_pairs_hook=model_mod._pairs)
        for edit in edits:
            _edit(document, edit['pointer'], edit['op'], edit.get('value'))
        case = {'name': name, 'source': SOURCE, 'edits': edits}
        try:
            case['normalised'] = _normalised(out, f"case-{name}", document)
            case['error'] = None
        except model_mod.ModelError as error:
            case['normalised'] = None
            case['error'] = {'type': 'ModelError', 'message': str(error)}
        cases.append(case)

    return {'source': SOURCE, 'documents': documents, 'cases': cases}


# --- quantities and assignments (feature 1.5) -------------------------------


def _facts(model, raw, assignment):
    """What feature 1.5's four functions answer about one document, in one fixed order.

    `check_quantities` is handed the *normalised* reading, as `analyse` hands it one, and
    `check_assignment`, `external_names` and `missing_assignment` the document as read, as `run`,
    `--d1`, `--derive`, `--lint` and the status page hand it. Normalisation moves only `bindings`
    and `compositions` (feature 1.4), which no rule of this module reads, and the parity suite
    pins that the two readings agree on every corpus document.

    The order matters: a document off the grammar raises somewhere, and the fixture records the
    *first* refusal, so the port has to reach it at the same point.
    """
    import validate
    from expr import external_names, missing_assignment, resolve_quantities
    external = sorted(external_names(raw))
    required = sorted(external_names(raw, with_defaults=False))
    missing = missing_assignment(raw, assignment)
    unassigned = missing_assignment(raw, None)
    variable = sorted(validate.variable_quantities(model))
    resolved = resolve_quantities(model, assignment)
    problems = [[code, message] for code, message in validate.check_quantities(model, resolved)]
    errors = validate.check_assignment(raw, assignment)
    return {
        'external': external,
        'required': required,
        'missing': missing,
        'unassigned_missing': unassigned,
        # The fragment `--validate`, `--d1` and `--derive` each print around their own words.
        'report': f"needs --assign for {missing}" if missing else None,
        'variable': variable,
        'resolved': encode_map(resolved),
        'problems': problems,
        'assignment_errors': errors,
    }


def _raised(error):
    return {'type': type(error).__name__, 'message': str(error)}


def _encoded_edit(edit):
    """One edit as the fixture carries it: the value tagged, so its float-ness survives."""
    written = {'pointer': edit['pointer'], 'op': edit['op']}
    if edit['op'] == 'set':
        written['value'] = encode(edit['value'])
    return written


def quantities(corpus, name_of, assignments):
    """`check_quantities`, `check_assignment`, `variable_quantities` and the assignment report.

    Over every model document of the repository — the corpus, the template under its documented
    assignment, and the 73 of `tests/rejections/models/` under the assignment each case names —
    and over the edited documents and assignments of `quantity_cases.py`, which reach the branches
    the corpus does not take: no corpus document declares a variable quantity, a boolean or a
    physical quantity, and eight domains in fifteen documents leave most of the domain table
    unvisited.
    """
    import model as model_mod
    from quantity_cases import CASES, SOURCE, TEMPLATE

    with open(os.path.join(REJECTIONS, 'models.json'), encoding='utf-8') as handle:
        rejection_cases = json.load(handle)['cases']
    named = {os.path.join('tests', 'rejections', case['document']): case.get('assign')
             for case in rejection_cases}

    documents = []
    paths = [(name_of(path), os.path.relpath(path, ROOT)) for path in corpus()]
    paths += [(f"rejection-{name[:-len('.json')]}",
               os.path.join('tests', 'rejections', 'models', name))
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]
    for slug, relative in paths:
        assignment = assignments.get(slug, named.get(relative))
        record = {'path': relative,
                  'assignment': None if assignment is None else encode_map(assignment),
                  'refused': None, 'error': None}
        full = os.path.join(ROOT, relative)
        try:
            model = model_mod.load(full)
        except model_mod.ModelError as error:
            record['refused'] = str(error)
            documents.append(record)
            continue
        with open(full, encoding='utf-8') as handle:
            raw = json.load(handle)
        try:
            record.update(_facts(model, raw, assignment))
        except Exception as error:                    # whatever it is, it is the answer
            record['error'] = _raised(error)
        documents.append(record)

    cases = []
    for name, relative, edits, assignment in CASES:
        with open(os.path.join(ROOT, relative), encoding='utf-8') as handle:
            document = json.load(handle, object_pairs_hook=model_mod._pairs)
        for edit in edits:
            _edit(document, edit['pointer'], edit['op'], edit.get('value'))
        record = {'name': name, 'document': relative,
                  'edits': [_encoded_edit(edit) for edit in edits],
                  'assignment': None if assignment is None else encode_map(assignment),
                  'error': None}
        try:
            # An edited document is not normalised: every edit is inside `quantities`, which
            # normalisation never touches, and both implementations read it as it stands.
            record.update(_facts(document, document, assignment))
        except Exception as error:                    # whatever it is, it is the answer
            record['error'] = _raised(error)
        cases.append(record)

    return {'source': SOURCE, 'template': TEMPLATE, 'documents': documents, 'cases': cases}


# --- arguments, their types, domains and invariants (feature 1.6a) ----------

# How many index environments of one composition site are recorded when they all resolve to the
# same thing. Every environment is *walked* — the lines are checked against the tools' own — and
# the cases the fixture carries are the distinct outcomes, plus the first and the last of the
# grid, so a `layer`-valued argument keeps one case per layer and a constant one keeps one case.
FIRST_ENVIRONMENT = 0
LAST_ENVIRONMENT = -1


def _written_fields(written):
    """The map of expressions a record argument's fields are written in, or None."""
    if isinstance(written, dict) and isinstance(written.get('record'), dict):
        return written['record']
    return None


def _resolve_with_facts(declared, given, evaluate, root, path, written, problems, facts,
                        into=None):
    """`validate._resolve_record`, transcribed once more, recording each row's verdict as the
    walk decides it.

    The transcription is what the fixture needs and what the tools do not expose: `applicable` is
    read *during* the loop, and an argument the domain refused is `UNRESOLVED` in the map that
    comes out, so neither can be recovered from the answer afterwards (finding F2 of the plan).
    It is not a second opinion: the caller requires its values and its problems to equal
    `validate.resolve_arguments`' own on every site of every document, so a transcription that
    drifted would fail the oracle rather than the port. Every check it makes is the tools' own
    function — `_check_type`, `_check_argument_domain`, `primitive_condition`, `primitive_value`.
    """
    from expr import UNRESOLVED, primitive_condition, primitive_value
    import validate as validate_mod

    values = {} if into is None else into
    if root is None:
        root = values
    supplied = set()
    for name, value in given.items():
        if name not in declared:
            problems.append(('V2', f"unknown argument '{path}{name}'"))
        else:
            supplied.add(name)
            values[name] = evaluate(value)
    pending = [n for n, d in declared.items() if n not in values and 'default' in d]
    while pending:
        progress = False
        for name in list(pending):
            v = primitive_value(declared[name]['default'], root)
            if v is not None and v is not UNRESOLVED:
                values[name] = v
                pending.remove(name)
                progress = True
        if not progress:
            for name in pending:
                problems.append(('V2', f"default of '{path}{name}' does not resolve"))
            break
    for name, decl in declared.items():
        label = f"{path}{name}"
        expression = written[name] if isinstance(written, dict) and name in written else None
        fact = {'path': label, 'applicable': True, 'source': 'absent',
                'domain': 'unchecked' if 'domain' in decl else 'undeclared',
                'indices': set()}
        if name in supplied and expression is not None:
            fact['written'] = encode(expression)
        facts.append(fact)
        first_problem = len(problems)
        first_fact = len(facts)

        def close(fact=fact, first_problem=first_problem, first_fact=first_fact):
            """The problems of this row: those appended for it, less the fields' own."""
            claimed = set()
            for one in facts[first_fact:]:
                claimed |= one['indices']
            own = set(range(first_problem, len(problems))) - claimed
            fact['problems'] = [list(problems[i]) for i in sorted(own)]
            fact['indices'] = own | claimed

        if 'present_when' in decl:
            fact['applicable'] = bool(primitive_condition(decl['present_when'], root))
        if name not in values:
            if decl['required'] and fact['applicable']:
                problems.append(('V2', f"required argument missing '{label}'"))
            close()
            continue
        fact['source'] = 'given' if name in supplied else 'default'
        if not fact['applicable']:
            fact['value'] = encode(values[name])
            problems.append(('V3', f"argument '{label}' is present but inapplicable "
                                   f"for these arguments"))
            close()
            continue
        if decl['type']['kind'] == 'record' and isinstance(values[name], dict):
            # `_check_type`'s record branch, unrolled so that the fields' rows are recorded too:
            # the record is cleared and filled in place, as the tools fill it.
            held = values[name]
            inner = dict(held)
            held.clear()
            _resolve_with_facts(decl['type']['fields'], inner, lambda x: x, root, label + '.',
                                _written_fields(expression), problems, facts, into=held)
        else:
            validate_mod._check_type(values[name], decl['type'], label, lambda x: x, root,
                                     problems, values)
        if len(problems) == first_problem and 'domain' in decl and values[name] is not UNRESOLVED:
            before = len(problems)
            validate_mod._check_argument_domain(values[name], decl['domain'], label, problems,
                                                root)
            fact['domain'] = 'refused' if len(problems) > before else 'ok'
        if len(problems) > first_problem and decl['type']['kind'] != 'record':
            values[name] = UNRESOLVED
        fact['value'] = encode(values[name])
        close()
    return values


def _invariant_verdicts(definition, values):
    """The V8 block of `analyse`, with a verdict for every invariant and not only for the ones
    that fail — the block the argument sheet shows under its rows (§4.12)."""
    from expr import UNRESOLVED, argument_references, primitive_condition
    import validate as validate_mod

    out = []
    for invariant in definition.get('invariants', []):
        references = argument_references(invariant['holds'])
        reads = sorted(references)
        shown = ', '.join(
            f"{path} = {validate_mod._resolve_path(path, values)}" for path in reads
            if validate_mod._resolve_path(path, values) is not None
            and validate_mod._resolve_path(path, values) is not UNRESOLVED)
        record = {'description': invariant['description'], 'reads': reads, 'shown': shown}
        if any(validate_mod._resolve_path(path, values) is UNRESOLVED for path in references):
            out.append(dict(record, verdict='skipped'))
        elif primitive_condition(invariant['holds'], values):
            out.append(dict(record, verdict='holds'))
        else:
            out.append(dict(record, verdict='fails', message=(
                f"'{invariant['description']}' does not hold" + (f" ({shown})" if shown else ""))))
    return out


def _argument_case(definition, instance, env, quantities):
    """One site resolved: what `resolve_arguments` answers, the V8 verdicts, and the facts."""
    from expr import static_argument
    import validate as validate_mod

    problems = []
    facts = []
    values = _resolve_with_facts(definition['arguments'], instance['arguments'],
                                 lambda v: static_argument(v, quantities, env), None, '',
                                 instance['arguments'], problems, facts)
    expected, expected_problems = validate_mod.resolve_arguments(
        definition, instance['arguments'], lambda v: static_argument(v, quantities, env))
    written = json.dumps(encode_map(values))
    if written != json.dumps(encode_map(expected)):
        die(f"the transcription resolves {written} where validate.resolve_arguments answers "
            f"{json.dumps(encode_map(expected))}")
    if [tuple(one) for one in problems] != [tuple(one) for one in expected_problems]:
        die(f"the transcription refuses {problems} where validate.resolve_arguments refuses "
            f"{expected_problems}")
    for fact in facts:
        fact.pop('indices', None)
        fact.setdefault('problems', [])
    invariants = _invariant_verdicts(definition, values)
    refusals = [list(one) for one in problems]
    refusals += [['V8', one['message']] for one in invariants if one['verdict'] == 'fails']
    return {'values': encode_map(values),
            'problems': refusals,
            'facts': facts,
            'invariants': invariants}, values


def _sites_of(document, quantities):
    """The sites `analyse` resolves, in its own order: the root instances whose guard fires, and
    every index environment of every composition's, `where(key)` and the pointer of each."""
    import itertools
    from expr import UNRESOLVED, index_grid, model_condition

    sites = []
    for name, instance in document['instances'].items():
        if 'when' in instance:
            truth = model_condition(instance['when'], quantities, {})
            if truth is UNRESOLVED or not truth:
                continue
        sites.append({'at': f"instances/{name}", 'where': name, 'env': {}, 'instance': instance,
                      'grid': None})
    for composition, definition in document.get('compositions', {}).items():
        names, ranges = index_grid(definition['indices'], quantities)
        grid = [dict(zip(names, combo)) for combo in itertools.product(*ranges)]
        for position, env in enumerate(grid):
            for site, instance in definition['instances'].items():
                if 'when' in instance:
                    truth = model_condition(instance['when'], quantities, env)
                    if truth is UNRESOLVED or not truth:
                        continue
                where = f"{composition}/{site}[" + ",".join(
                    f"{k}={v}" for k, v in sorted(env.items())) + "]"
                sites.append({
                    'at': f"compositions/{composition}/instances/{site}",
                    'where': where, 'env': env, 'instance': instance,
                    'grid': (position, len(grid))})
    return sites


def _walk_document(path, assignment, cat, bases, depth, within, cases, lines, messages, seen):
    """Every site of one document, and of every template it instantiates, as `analyse` walks
    them: the same guards, the same interface for a template primitive, the same sub-assignment
    at a call site, and the same cache — a template analysed once per assignment (§4.6)."""
    import model as model_mod
    import primitive_library as primitive_library_mod
    import validate as validate_mod
    from expr import UNRESOLVED, resolve_quantities

    document = model_mod.load(path)
    quantities = resolve_quantities(document, assignment)
    relative = os.path.relpath(path, ROOT)
    grouped = {}
    for site in _sites_of(document, quantities):
        instance = site['instance']
        name = instance['primitive']['name']
        definition = primitive_library_mod.primitive(cat, instance['primitive'])
        if definition is None:
            continue                       # V1, and no arguments to resolve (feature 1.6b)
        template_file = None
        if 'template' in definition:
            template_file = primitive_library_mod.template_path(cat, definition)
            definition = validate_mod.template_interface(definition,
                                                        model_mod.load(template_file))
        answer, values = _argument_case(definition, instance, site['env'], quantities)
        resolution_problems = [one for one in answer['problems'] if one[0] != 'V8']
        for code, message in answer['problems']:
            lines.append(f"[{code}] {name} @{site['where']}: {message}")
            messages.append(f"[{code}] {message}")
        case = dict(answer, document=relative, at=site['at'], where=site['where'],
                    within=within, bases=bases,
                    assignment=None if assignment is None else encode_map(assignment),
                    primitive=instance['primitive'], template=template_file is not None,
                    env=encode_map(site['env']))
        # One case per distinct outcome, plus the first and the last environment of the grid.
        outcome = json.dumps([answer['values'], answer['problems'], answer['facts'],
                              answer['invariants']], sort_keys=True)
        group = grouped.setdefault((site['at'], outcome), {'case': case, 'environments': 0})
        group['environments'] += 1
        if site['grid'] is not None and site['grid'][0] == site['grid'][1] - 1:
            group['case'] = case            # the boundary the last environment reaches
        if template_file is not None and not resolution_problems:
            if depth + 1 > validate_mod.MAX_PRIMITIVE_DEPTH:
                continue
            sub_assignment = {k: v for k, v in values.items() if v is not UNRESOLVED}
            key = (template_file, json.dumps(sub_assignment, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            _walk_document(template_file, sub_assignment, cat, bases, depth + 1,
                           f"{within + ' / ' if within else ''}{name} @{site['where']}",
                           cases, lines, messages, seen)
    for group in grouped.values():
        cases.append(dict(group['case'], environments=group['environments']))


def arguments(corpus, name_of, assignments):
    """`resolve_arguments`, the V8 block, and the per-argument facts of `describe`, over every
    site of every model document of the repository — the corpus, the template under its
    documented assignment, the 73 of `tests/rejections/models/`, and the sites of every template
    a document instantiates, expanded at the call site as `analyse` expands them.

    Beside them, the synthetic declarations of `argument_cases.py`: the reference base declares 13
    `present_when`s, no `seconds` argument, no set domain and no default that fails to resolve, so
    the corpus alone leaves half of `_resolve_record` unvisited.

    Every line the walk produces is checked against `validate.analyse`'s own errors for the same
    document, so the fixture is the tools' answer and not a transcription's.
    """
    import model as model_mod
    import primitive_library as primitive_library_mod
    import validate as validate_mod
    from argument_cases import CASES
    from expr import static_argument

    with open(os.path.join(REJECTIONS, 'models.json'), encoding='utf-8') as handle:
        rejection_cases = json.load(handle)['cases']
    named = {os.path.join('tests', 'rejections', case['document']): case.get('assign')
             for case in rejection_cases}

    paths = [(name_of(path), os.path.relpath(path, ROOT)) for path in corpus()]
    paths += [(f"rejection-{name[:-len('.json')]}",
               os.path.join('tests', 'rejections', 'models', name))
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]

    documents = []
    cases = []
    gathered = {}
    for slug, relative in paths:
        full = os.path.join(ROOT, relative)
        record = {}
        assignment = assignments.get(slug, named.get(relative))
        record.update({'name': slug, 'path': relative,
                       'assignment': None if assignment is None else encode_map(assignment),
                       'sites': 0, 'error': None})
        mine = []
        lines = []
        messages = []
        try:
            document = model_mod.load(full)
            # A corpus document is read under the bases it declares; a rejection document under
            # the reference base, which is what `tests/run_rejections.py` hands `validate.semantic`
            # — 21 of the 73 declare `../primitive-library/`, which from `tests/rejections/models/`
            # names the directory of *rejection bases* and gathers nothing.
            bases = ([REFERENCE_BASE] if relative.startswith('tests/rejections/')
                     else [os.path.relpath(base, ROOT)
                           for base in primitive_library_mod.bases_of(full, document)])
            # `load_for` memoises; `load` does not, and every document of the repository names
            # the same base — 88 gatherings of it would cost more than the whole step.
            key = tuple(bases)
            if key not in gathered:
                gathered[key] = primitive_library_mod.load(
                    *[os.path.join(ROOT, base) for base in bases])
            cat = gathered[key]
            record['bases'] = bases
            _walk_document(full, assignment, cat, bases, 0, '', mine, lines, messages, set())
        except Exception as error:                        # whatever it is, it is the answer
            record['error'] = _raised(error)
            record['sites'] = len(mine)
            documents.append(record)
            cases.extend(mine)
            continue
        record['sites'] = len(mine)
        # The refusals without the instance and the site `analyse` prefixes them with: a
        # deduplicated case stands for every environment that resolves to it, so the *messages*
        # are what both implementations can compare exactly.
        record['messages'] = sorted(set(messages))
        # The tools' own answer for the same document: every line the walk produced must be one
        # of `analyse`'s, and every V2 and V8 line of `analyse`'s must be one of the walk's (V3
        # is shared with the quantities, feature 1.5).
        try:
            errors = validate_mod.analyse(full, cat, assignment)['errors']
        except Exception:
            record['analysed'] = False
        else:
            record['analysed'] = True
            for line in lines:
                if not any(one == line or one.startswith(line + '  (in instance ')
                           for one in errors):
                    die(f"{relative}: the walk produced a line the tools do not: {line}")
            for one in errors:
                bare = one.split('  (in instance ')[0]
                if one.startswith('[V2] ') or one.startswith('[V8] '):
                    if bare not in lines:
                        die(f"{relative}: the tools refuse a line the walk does not: {one}")
        documents.append(record)
        cases.extend(mine)

    synthetic = []
    for name, definition, given, quantities, env in CASES:
        record = {'name': name, 'definition': definition, 'given': given,
                  'quantities': encode_map(quantities), 'env': encode_map(env), 'error': None}
        try:
            record.update(_argument_case(definition, {'arguments': given}, env, quantities)[0])
        except Exception as error:                        # whatever it is, it is the answer
            record['error'] = _raised(error)
        synthetic.append(record)

    return {'documents': documents, 'cases': cases, 'synthetic': synthetic}

# --- the graph: sites, edges, domains and interfaces (feature 1.6b) ---------


def _partial_analyse():
    """`validate.analyse` truncated at the parameter bindings: feature 1.6b's half of it.

    The feature ports `analyse` as far as V19 — sites and guards, references, ports and shapes,
    edges, interfaces, indexing domains — and feature 1.6c continues from there. The tools have
    one function, so the expectation is built by *taking their own source* and cutting it at the
    comment that opens the bindings, then appending the two blocks that close it: the interface
    ports a caller reads back, and the merge of an expanded template's counters. Nothing is
    transcribed but those two blocks, and both are copied from the same source.

    The cut and the tail are found by their comments, so a `validate.py` that moves them makes
    the oracle die rather than record a half-truth. What the truncation answers is held to the
    whole function on every document: its errors are a subsequence of `analyse`'s, and the
    counters and the interface ports it computes are `analyse`'s own.
    """
    import inspect
    import validate as validate_mod

    lines = inspect.getsource(validate_mod.analyse).splitlines()

    def find(prefix):
        hits = [i for i, line in enumerate(lines) if line.strip().startswith(prefix)]
        if len(hits) != 1:
            die(f"validate.analyse: {prefix!r} appears {len(hits)} time(s), expected once")
        return hits[0]

    cut = find('# --- V7/V14/V15: parameters')
    ports = find('# What this document exposes to a caller')
    ret = find("return {'errors': errors,")
    if not (cut < ports < ret):
        die('validate.analyse: the blocks feature 1.6b ends with are no longer in order')
    body = lines[:cut] + lines[ports:ret] + [
        "    return {'errors': errors, 'stats': stats, 'ports': ports,",
        "            'advisories': advisories, 'where': where,",
        "            'graph': {'resolved': resolved, 'edges': edges, 'domains': domains,",
        "                      'own': own, 'absent': absent, 'order': order, 'sites': sites,",
        "                      'seeds': seeds, 'fragmented': fragmented, 'producers': producers,",
        "                      'consumed': consumed, 'weights_prefixes': weights_prefixes,",
        "                      'sub_results': sub_results}}",
    ]
    scope = dict(validate_mod.__dict__)
    exec('\n'.join(body), scope)                                   # noqa: S102 - the tools' own
    return scope['analyse']


def _subsequence(inner, outer):
    """Every line of `inner`, in order, among the lines of `outer`."""
    walk = iter(outer)
    return all(any(line == one for one in walk) for line in inner)


def _graph_facts(answer):
    """The expanded graph as the fixture carries it: names, not tuples."""
    where = answer['where']
    graph = answer['graph']
    return {
        'sites': [where(key) for key in graph['resolved']],
        'absent': sorted(where(key) for key in graph['absent']),
        'edges': [[where(src), sp, where(dst), dp, bid]
                  for src, sp, dst, dp, bid in graph['edges']],
        'order': [where(key) for key in graph['order']],
        'domains': [[where(key), port, kind, stream]
                    for (key, port), (kind, stream) in graph['domains'].items()],
        'own': [[where(key), None if mine is None else [mine[0], mine[1]]]
                for key, mine in graph['own'].items()],
        'seeds': [[where(key), port, kind, stream]
                  for (key, port), (kind, stream) in graph['seeds'].items()],
        'fragmented': sorted(graph['fragmented']),
        'producers': [[where(key), port, bid]
                      for (key, port), bid in graph['producers'].items()],
        'consumed': sorted([where(key), port] for key, port in graph['consumed']),
        'weights_prefixes': [[where(key), prefix]
                             for key, prefix in graph['weights_prefixes'].items()],
        'ports': _interface_ports(answer['ports']),
        'stats': {name: encode(one) for name, one in answer['stats'].items()},
        'advisories': list(answer['advisories']),
    }


def _interface_ports(ports):
    """The public interface ports, with their evaluated shapes tagged."""
    return {side: {name: {'kind': encode(entry['kind']), 'stream': encode(entry['stream']),
                          'shape': (None if entry['shape'] is None
                                    else [[axis, encode(extent)] for axis, extent in entry['shape']])}
                   for name, entry in entries.items()}
            for side, entries in ports.items()}


def graph(corpus, name_of, assignments):
    """`analyse` as far as V19, over every model document of the repository.

    The corpus and the template under its documented assignment, and the 73 of
    `tests/rejections/models/` — the last read under the reference base, which is what
    `tests/run_rejections.py` hands `validate.semantic`: 21 of them declare
    `../primitive-library/`, which from `tests/rejections/models/` names the directory of
    rejection *bases* and gathers nothing (feature 1.6a's finding).

    Recorded per document: the refusals, the counters, the public interface ports, and the graph
    itself — the resolved sites, the sites a guard removed, the edges, the topological order, the
    indexing domain of every port, what feeds and consumes each one. A template instance's own
    expansion is recorded beside it, which is what makes the composite document's counts checkable
    against the flat document's (`tests/run_templates.py`'s claim).
    """
    import model as model_mod
    import primitive_library as primitive_library_mod
    import validate as validate_mod

    partial = _partial_analyse()

    with open(os.path.join(REJECTIONS, 'models.json'), encoding='utf-8') as handle:
        rejection_cases = json.load(handle)['cases']
    named = {os.path.join('tests', 'rejections', case['document']): case.get('assign')
             for case in rejection_cases}

    paths = [(name_of(path), os.path.relpath(path, ROOT)) for path in corpus()]
    paths += [(f"rejection-{name[:-len('.json')]}",
               os.path.join('tests', 'rejections', 'models', name))
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]

    gathered = {}
    documents = []
    for slug, relative in paths:
        full = os.path.join(ROOT, relative)
        assignment = assignments.get(slug, named.get(relative))
        record = {'name': slug, 'path': relative,
                  'assignment': None if assignment is None else encode_map(assignment),
                  'error': None}
        try:
            document = model_mod.load(full)
            bases = ([REFERENCE_BASE] if relative.startswith('tests/rejections/')
                     else [os.path.relpath(base, ROOT)
                           for base in primitive_library_mod.bases_of(full, document)])
        except Exception:
            bases = [REFERENCE_BASE]
        record['bases'] = bases
        key = tuple(bases)
        if key not in gathered:
            gathered[key] = primitive_library_mod.load(
                *[os.path.join(ROOT, base) for base in bases])
        cat = gathered[key]
        try:
            answer = partial(full, cat, assignment)
        except Exception as error:                     # whatever it is, it is the answer
            record['error'] = _raised(error)
            documents.append(record)
            continue
        record['errors'] = list(answer['errors'])
        if 'graph' not in answer:
            # `model.load` refused the document: one line, and nothing else (`empty`).
            record.update({'stats': {}, 'advisories': [],
                           'ports': _interface_ports(answer['ports']),
                           'read': False, 'whole': True, 'expansions': [],
                           'arbitrary_own': False})
            documents.append(record)
            continue
        record.update(_graph_facts(answer))
        record['read'] = True
        # `mine = next(iter(agree))` reads a Python *set*, so an instance whose inputs disagree
        # takes an arbitrary one of them as its own domain — hash-seeded, and different between
        # two runs of the tools themselves (`PYTHONHASHSEED=3` answers `audio` where 0, 1, 2 and
        # 4 answer `tokens` on `v5-fusion-without-join.json`). The refusals are stable, since the
        # V5 line sorts what it lists; what follows the choice is not, so it is not recorded.
        record['arbitrary_own'] = any('inputs in different domains' in line
                                      for line in answer['errors'])
        record['expansions'] = [
            [answer['where'](key), {name: encode(one) for name, one in sub['stats'].items()},
             _interface_ports(sub['ports'])]
            for key, sub in answer['graph']['sub_results'].items()]
        # The truncation is held to the whole function: its lines are `analyse`'s, in order, and
        # the counters and interface ports it computes are `analyse`'s own.
        try:
            whole = validate_mod.analyse(full, cat, assignment)
        except Exception:
            record['whole'] = False
            documents.append(record)
            continue
        record['whole'] = True

        if not _subsequence(answer['errors'], whole['errors']):
            die(f"{relative}: the truncated analyse refuses a line validate.analyse does not")
        for name, one in answer['stats'].items():
            if whole['stats'].get(name) != one:
                die(f"{relative}: the truncated analyse counts {name} = {one!r} where "
                    f"validate.analyse counts {whole['stats'].get(name)!r}")
        if json.dumps(_interface_ports(whole['ports']), sort_keys=True) != \
                json.dumps(record['ports'], sort_keys=True):
            die(f"{relative}: the truncated analyse exposes other interface ports")
        if record['arbitrary_own']:
            for name in ('own', 'domains', 'ports'):
                record.pop(name)
        documents.append(record)
    return {'documents': documents}

# --- the bindings: slots, identities, locations and states (feature 1.6c) ---


def _where_of_analyse():
    """`analyse`'s own `where(key)`, taken from its source.

    The whole `analyse` does not return it, and the bindings step names every site with it. It
    closes over nothing, so the block is lifted out of `inspect.getsource(validate.analyse)` and
    executed as it stands — the tools' own four lines, not a transcription — and a `validate.py`
    that moves or changes it makes the oracle die rather than record a half-truth.
    """
    import inspect
    import textwrap
    import validate as validate_mod

    lines = inspect.getsource(validate_mod.analyse).splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == 'def where(key):']
    if len(starts) != 1:
        die(f"validate.analyse: `def where(key):` appears {len(starts)} time(s), expected once")
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (not lines[end].strip()
                                or len(lines[end]) - len(lines[end].lstrip()) > indent):
        end += 1
    scope = {}
    exec(textwrap.dedent('\n'.join(lines[start:end])), scope)          # noqa: S102 - the tools' own
    where = scope['where']
    if where(('root', 'embed')) != 'embed' or \
            where(('gen', 'decoder', 'attn', (('layer', 3),))) != 'decoder/attn[layer=3]':
        die('validate.analyse: `where` no longer names a site the way the fixture reads it')
    return where


def _location(evaluated):
    """One evaluated location, its integers tagged: D3 writes them and V17 compares them."""
    if evaluated is None:
        return None
    if 'tensor' in evaluated:
        return {'tensor': evaluated['tensor']}
    for form in ('stack', 'concat'):
        if form in evaluated:
            part = evaluated[form]
            return {form: {'axis': part['axis'], 'dim': encode(part['dim']),
                           'parts': [_location(one) for one in part['parts']]}}
    part = evaluated['slice']
    return {'slice': {'tensor': part['tensor'], 'axis': part['axis'],
                      'dim': encode(part['dim']), 'offset': encode(part['offset']),
                      'extent': encode(part['extent'])}}


SENTINEL = re.compile(r'<object object at 0x[0-9a-f]+>')


def _elide_sentinel(lines):
    """`repr(expr.UNRESOLVED)` carries the sentinel's address, which changes between two runs of
    the tools themselves (measured: three runs of `analyse` on
    `tests/rejections/models/v3-streams-below-two.json` print three addresses). One line of the
    bindings interpolates a value with `!r` and can be handed the sentinel — a slot whose
    multiplicity reads an argument V3 refused — so the address is elided here and the port writes
    `<object object>` on its side. The rest of the line is contract; the address is not.
    """
    written = [SENTINEL.sub('<object object>', line) for line in lines]
    return written, written != list(lines)


def _binding_facts(answer, where):
    """What `analyse` answers about the bindings, with every site named rather than keyed.

    `mine = next(iter(agree))` reads a Python *set* (feature 1.6b's finding), so an instance whose
    inputs disagree takes a hash-seeded one of them as its own domain — and at *this* stage the
    choice reaches the refusals themselves, not only the state behind them: measured on
    `tests/rejections/models/v5-fusion-without-join.json`, `PYTHONHASHSEED=0` prints 131 lines with
    26 of V16 where seed 3 prints 105 with none, because one reading puts the state on a fragmented
    stream and the other does not. The three answers that follow the choice — the refusals, the
    advisories and what is carried — are therefore not recorded for such a document; everything
    else here is independent of it.
    """
    graph = answer['graph']
    errors, elided = _elide_sentinel(answer['errors'])
    advisories, elided_too = _elide_sentinel(answer['advisories'])
    arbitrary = any('inputs in different domains' in line for line in errors)
    settled = {} if arbitrary else {
        'errors': errors,
        'advisories': advisories,
        'carried': [[rule, None if mine is None else [mine[0], mine[1]]]
                    for rule, mine in answer['carried'].items()],
    }
    return {
        **settled,
        'arbitrary_own': arbitrary,
        'elided_sentinel': elided or elided_too,
        'stats': {name: encode(one) for name, one in answer['stats'].items()},
        'instance_keys': [[key, list(axes)] for key, axes in answer['instance_keys'].items()],
        'physical': {
            'whole': [[name, identity] for name, identity in answer['physical']['whole'].items()],
            'slices': [[name, [[encode(offset), encode(extent), identity]
                               for offset, extent, identity in intervals]]
                       for name, intervals in answer['physical']['slices'].items()],
        },
        'slots': [[where(key), slot, rule] for (key, slot), rule in graph['slots'].items()],
        'state_slots': [[where(key), port, rule]
                        for (key, port), rule in graph['state_slots'].items()],
        'tensors': [{'identity': one['identity'], 'rule': one['rule'],
                     'members': [[where(key), name] for key, name in one['members']],
                     'dtype': one['dtype'], 'location': _location(one.get('location'))}
                    for one in graph['tensor_instances']],
        'states': [{'identity': one['identity'], 'rule': one['rule'],
                    'members': [[where(key), name] for key, name in one['members']],
                    'dtype': one['dtype'], 'indices': list(one['indices']),
                    'writer': (None if one.get('writer') is None
                               else [where(one['writer'][0]), one['writer'][1]])}
                   for one in graph['state_instances']],
    }


def bindings(out, corpus, name_of, assignments):
    """The whole `validate.analyse` over every model document of the repository, and over edited
    ones reaching the branches nothing in the repository takes.

    Feature 1.6b recorded `analyse` truncated at the parameter bindings; this records the function
    itself, so the two together state that the port is the whole of it. Beside the refusals and
    the counters it records what the bindings answer — the instance keys of §4.4, what is carried
    across fragments (§5.3), the physical names, every bound slot and state port, and the identity
    instances D3 and D4 will read.

    The 73 documents of `tests/rejections/models/` are read under the reference base, which is
    what `tests/run_rejections.py` hands `validate.semantic`: 21 of them declare
    `../primitive-library/`, which from `tests/rejections/models/` names the directory of
    rejection *bases* and gathers nothing (feature 1.6a's finding).
    """
    import model as model_mod
    import primitive_library as primitive_library_mod
    import validate as validate_mod
    from binding_cases import CASES, LLAMA, SHORT, VOX, VOXTRAL

    where = _where_of_analyse()
    written = os.path.join(out, 'bindings', 'documents')
    os.makedirs(written, exist_ok=True)

    with open(os.path.join(REJECTIONS, 'models.json'), encoding='utf-8') as handle:
        rejection_cases = json.load(handle)['cases']
    named = {os.path.join('tests', 'rejections', case['document']): case.get('assign')
             for case in rejection_cases}

    paths = [(name_of(path), os.path.relpath(path, ROOT)) for path in corpus()]
    paths += [(f"rejection-{name[:-len('.json')]}",
               os.path.join('tests', 'rejections', 'models', name))
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]

    gathered = {}

    def catalogue(bases):
        key = tuple(bases)
        if key not in gathered:
            gathered[key] = primitive_library_mod.load(
                *[os.path.join(ROOT, base) for base in bases])
        return gathered[key]

    documents = []
    for slug, relative in paths:
        full = os.path.join(ROOT, relative)
        assignment = assignments.get(slug, named.get(relative))
        record = {'name': slug, 'path': relative,
                  'assignment': None if assignment is None else encode_map(assignment),
                  'error': None}
        try:
            document = model_mod.load(full)
            bases = ([REFERENCE_BASE] if relative.startswith('tests/rejections/')
                     else [os.path.relpath(base, ROOT)
                           for base in primitive_library_mod.bases_of(full, document)])
        except Exception:
            bases = [REFERENCE_BASE]
        record['bases'] = bases
        try:
            answer = validate_mod.analyse(full, catalogue(bases), assignment)
        except Exception as error:                     # whatever it is, it is the answer
            record['error'] = _raised(error)
            documents.append(record)
            continue
        if 'graph' not in answer:
            # `model.load` refused it: one line, and the empty answer beside it.
            lines, elided = _elide_sentinel(answer['errors'])
            record.update({'errors': lines, 'elided_sentinel': elided, 'arbitrary_own': False,
                           'stats': {}, 'advisories': [], 'read': False})
            documents.append(record)
            continue
        record.update(_binding_facts(answer, where))
        record['read'] = True
        documents.append(record)

    cases = []
    shortened = {LLAMA: SHORT, VOXTRAL: VOX}
    for name, relative, edits in CASES:
        applied = shortened[relative] + list(edits)
        with open(os.path.join(ROOT, relative), encoding='utf-8') as handle:
            document = json.load(handle)
        for edit in applied:
            _no_floats(edit.get('value'), f"{name} {edit['pointer']}")
            _edit(document, edit['pointer'], edit['op'], edit.get('value'))
        # `analyse` reads a path, so the edited document is written where the fixture can name it;
        # the parity suite applies the same edits to the same source and never reads this file.
        path = os.path.join(written, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
        record = {'name': name, 'source': relative,
                  'edits': [_encoded_edit(edit) for edit in applied], 'error': None}
        try:
            answer = validate_mod.analyse(path, catalogue([REFERENCE_BASE]))
        except Exception as error:                     # whatever it is, it is the answer
            record['error'] = _raised(error)
            cases.append(record)
            continue
        record.update(_binding_facts(answer, where))
        record['read'] = 'graph' in answer
        cases.append(record)

    return {'documents': documents, 'cases': cases}


# --- D1, the expanded graph (feature 1.7) -----------------------------------

def expansion(out, corpus, name_of, assignments):
    """`d1.emit` over every model document of the repository, and over edited ones.

    The emitter is not the validator: it resolves an instance's arguments *as written* — the
    document's values, then the declared defaults, records included — where `validate.analyse`
    resolves them typed, and it raises where the validator would have written a line. So its
    answers are recorded on their own, against `d1.emit` and nothing else; `derive.products` is
    where the two resolutions are required to agree, and that is feature 1.8's.

    What is recorded per document is the emitted document *itself*, as `--d1` writes it
    (`json.dumps(indent=2, ensure_ascii=False)`), so the parity suite compares bytes and a failure
    names a line. The corpus and the template are not written twice: their record points at the
    file `--d1` already wrote, and the step re-emits each one and dies if the two texts differ —
    the expectation must be the tool's own output, not a second reading of it.

    The 73 documents of `tests/rejections/models/` are read under the reference base, which is what
    `tests/run_rejections.py` hands the validator (feature 1.6a's finding); `d1.emit` emits a graph
    for most of them, since a document V2 or V8 refuses still denotes one.
    """
    import model as model_mod
    import primitive_library as primitive_library_mod
    import d1 as d1_mod
    from d1_cases import CASES, SHORTENED

    emitted_dir = os.path.join(out, 'expansion', 'emitted')
    documents_dir = os.path.join(out, 'expansion', 'documents')
    os.makedirs(emitted_dir, exist_ok=True)
    os.makedirs(documents_dir, exist_ok=True)

    with open(os.path.join(REJECTIONS, 'models.json'), encoding='utf-8') as handle:
        rejection_cases = json.load(handle)['cases']
    named = {os.path.join('tests', 'rejections', case['document']): case.get('assign')
             for case in rejection_cases}

    paths = [(name_of(path), os.path.relpath(path, ROOT)) for path in corpus()]
    paths += [(f"rejection-{name[:-len('.json')]}",
               os.path.join('tests', 'rejections', 'models', name))
              for name in sorted(os.listdir(os.path.join(REJECTIONS, 'models')))
              if name.endswith('.json')]

    gathered = {}

    def catalogue(bases):
        key = tuple(bases)
        if key not in gathered:
            gathered[key] = primitive_library_mod.load(
                *[os.path.join(ROOT, base) for base in bases])
        return gathered[key]

    def text_of(document):
        return json.dumps(document, indent=2, ensure_ascii=False) + '\n'

    def facts(document, where):
        graph = document['d1']
        return {'nodes': len(graph['nodes']), 'edges': len(graph['edges']),
                'instances': list(graph.get('instances') or {}), 'emitted': where}

    documents = []
    for slug, relative in paths:
        full = os.path.join(ROOT, relative)
        assignment = assignments.get(slug, named.get(relative))
        try:
            document = model_mod.load(full)
            bases = ([REFERENCE_BASE] if relative.startswith('tests/rejections/')
                     else [os.path.relpath(base, ROOT)
                           for base in primitive_library_mod.bases_of(full, document)])
        except Exception:
            bases = [REFERENCE_BASE]
        record = {'name': slug, 'path': relative, 'bases': bases,
                  'assignment': None if assignment is None else encode_map(assignment),
                  'error': None}
        try:
            answer = d1_mod.emit(full, catalogue(bases), assignment)
        except Exception as error:                     # whatever it is, it is the answer
            record['error'] = _raised(error)
            documents.append(record)
            continue
        text = text_of(answer)
        if relative.startswith(os.path.join('data', 'models')):
            # `--d1` wrote it already; the record points there, and the two must be the same text.
            where = f"d1/{slug}.d1.json"
            with open(os.path.join(out, where), encoding='utf-8') as handle:
                if handle.read() != text:
                    die(f"{slug}: re-emitting D1 does not reproduce what --d1 wrote")
        else:
            where = f"expansion/emitted/{slug}.d1.json"
            write(os.path.join(out, where), text)
        record.update(facts(answer, where))
        documents.append(record)

    cases = []
    for name, relative, edits in CASES:
        applied = SHORTENED[relative] + list(edits)
        with open(os.path.join(ROOT, relative), encoding='utf-8') as handle:
            document = json.load(handle)
        for edit in applied:
            _no_floats(edit.get('value'), f"{name} {edit['pointer']}")
            _edit(document, edit['pointer'], edit['op'], edit.get('value'))
        # `emit` reads a path, so the edited document is written where the tools can be handed it;
        # the parity suite applies the same edits to the same source and never reads this file.
        path = os.path.join(documents_dir, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
        record = {'name': name, 'source': relative,
                  'edits': [_encoded_edit(edit) for edit in applied], 'error': None}
        try:
            answer = d1_mod.emit(path, catalogue([REFERENCE_BASE]))
        except Exception as error:                     # whatever it is, it is the answer
            record['error'] = _raised(error)
            cases.append(record)
            continue
        where = f"expansion/emitted/case-{name}.d1.json"
        write(os.path.join(out, where), text_of(answer))
        record.update(facts(answer, where))
        cases.append(record)

    return {'documents': documents, 'cases': cases}


# --- the primitive library loader (feature 1.3) -----------------------------

# What a mutation replaces a value by. A string keeps the shape it had — `attention.heads` becomes
# `attention.headsz`, which is the fixtures' own idiom (`attention.headz`) — and a second pass puts
# an axis the base *does* hold in its place, which is what reaches the checks that ask what an
# existing name is (a key axis that is a value axis, a unit axis that is not an axis of its slot).
IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]*$')
UNMUTATED = {'description', 'summary', 'note', 'title', 'url'}
SWAP_FOR = 'model.width'


def _mutation_sites(node, path='', parent=None, out=None):
    """Every place in a definition a mutation can change without changing its shape: a string that
    reads as a name, and a boolean. Documentation text is left alone — it is not a reference."""
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            _mutation_sites(value, f"{path}/{key}", key, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _mutation_sites(value, f"{path}/{index}", parent, out)
    elif isinstance(node, bool):
        out.append(('flip', path, not node))
    elif isinstance(node, str) and parent not in UNMUTATED and IDENTIFIER.match(node):
        out.append(('suffix', path, node + 'z'))
        if node != SWAP_FOR:
            out.append(('swap', path, SWAP_FOR))
    return out


def _apply(node, pointer, value):
    """The value at a JSON pointer, replaced. The pointer is the fixture's own, so the port applies
    the same change to the same file and the two run on one input."""
    parts = pointer.split('/')[1:]
    cursor = node
    for step in parts[:-1]:
        cursor = cursor[int(step)] if isinstance(cursor, list) else cursor[step]
    last = parts[-1]
    if isinstance(cursor, list):
        cursor[int(last)] = value
    else:
        cursor[last] = value


def library():
    """The loader's answers: the reference base gathered, the rejection suite's refusals word for
    word, `primitive_references` over thousands of one-value mutations, and the interface the one
    template primitive presents.

    The mutations are what makes this more than the 33 cases the rejection suite carries: those
    reach about fifteen of the checker's forty message sites, and one substitution per name reaches
    ninety-six of them. Each is recorded as a *pointer into a repository file*, so the port applies
    the same change to the same bytes and neither side owns the input."""
    import copy
    import primitive_library as pl
    import validate as validate_mod
    import model as model_mod

    cat = pl.load(PRIMITIVE_LIBRARY)
    origin = {}
    for section, _kind in pl.SECTIONS:
        root = os.path.join(PRIMITIVE_LIBRARY, section)
        for path in sorted(glob.glob(os.path.join(root, '**', '*.json'), recursive=True)):
            parts = os.path.relpath(path, root)[:-len('.json')].split(os.sep)
            key = ('.'.join(parts[:-1]), parts[-1]) if section == 'primitives' \
                else ('.'.join(parts), None)
            origin[(section, key)] = os.path.relpath(path, ROOT)

    reference = {
        'base': os.path.relpath(PRIMITIVE_LIBRARY, ROOT),
        'by_id': [[name, version] for name, version in sorted(cat['by_id'])],
        'axes': list(cat['axes']),
        'precision': list(cat['precision']),
        'primitives': {name: d['version'] for name, d in cat['primitives'].items()},
        'templates': {f"{name}@{version}": os.path.relpath(path, ROOT)
                      for (name, version), path in cat['templates'].items()},
        'files': {f"{section}:{key[0]}@{key[1]}" if key[1] else f"{section}:{key[0]}": path
                  for (section, key), path in origin.items()},
    }

    cases = []
    with open(os.path.join(REJECTIONS, 'primitive-library.json'), encoding='utf-8') as handle:
        manifest = json.load(handle)
    for case in manifest['cases']:
        base = os.path.join(REJECTIONS, case['base'])
        try:
            pl.load(base, PRIMITIVE_LIBRARY)
        except pl.PrimitiveLibraryError as refusal:
            # The paths are the ones `load` was given, so they are recorded relative to the root
            # and the reader puts its own prefix back.
            text = str(refusal).replace(ROOT + os.sep, '')
            cases.append({'base': case['base'], 'match': case['match'], 'error': text})
        else:
            die(f"{case['base']} was accepted; the rejection suite says it must not be")

    mutations = []
    for (name, version), definition in sorted(cat['by_id'].items()):
        unit = origin[('primitives', (name, version))]
        # The unmutated definition first: the checker must say nothing about the base as it stands.
        mutations.append({'unit': unit, 'pointer': None, 'kind': 'none', 'value': None,
                          'lines': pl.primitive_references(definition, cat)})
        for kind, pointer, value in _mutation_sites(definition):
            mutated = copy.deepcopy(definition)
            _apply(mutated, pointer, value)
            try:
                lines = pl.primitive_references(mutated, cat)
            except Exception as exc:                                   # pragma: no cover
                die(f"{name}@{version} {pointer} ({kind}) raised {type(exc).__name__}: {exc}")
            mutations.append({'unit': unit, 'pointer': pointer, 'kind': kind, 'value': value,
                              'lines': lines})

    interfaces = {}
    for (name, version), path in cat['templates'].items():
        definition = cat['by_id'][(name, version)]
        template = model_mod.load(path)
        interfaces[f"{name}@{version}"] = encode(
            validate_mod.template_interface(definition, template))

    return {'reference': reference, 'cases': cases, 'mutations': mutations,
            'template_interfaces': interfaces}


# --- the checkpoint check, V17 against safetensors headers (feature 1.9) ---------

# A checkpoint the corpus can be checked against does not exist in the repository — the weights
# are gigabytes and live outside it — so the oracle *synthesises* one per document, out of the
# document's own D3: every physical name a location binds, with the shape and dtype that location
# needs. That synthesis is the oracle's alone and is recorded literally, so the port never writes a
# second copy of it and both implementations read the same bytes. It is held to account by the
# clean case: a synthesis that got a shape wrong makes `artifact.check` refuse, and the step dies
# rather than record it.
#
# The four location forms are then reached twice over: the corpus reaches `tensor`, `stack` and
# `slice` (9 254 located identity instances over twelve documents, and no `concat` anywhere), and
# the synthetic D3s below reach every branch of `_check_part` — the ones `tests/run_artifact.py`
# writes, and the six it does not.

ARTIFACT_FILE = 'x'


def _place(ev, logical, dtype, out, sliced, identity):
    """One evaluated location, placed in a checkpoint that satisfies it."""
    import artifact as artifact_mod
    if 'tensor' in ev:
        out[ev['tensor']] = {'dtype': dtype, 'shape': list(logical), 'file': ARTIFACT_FILE}
    elif 'stack' in ev:
        dim = ev['stack']['dim']
        inner = logical[:dim] + logical[dim + 1:]
        for part in ev['stack']['parts']:
            _place(part, inner, dtype, out, sliced, identity)
    elif 'slice' in ev:
        s = ev['slice']
        want = artifact_mod.squeeze(logical)
        pos = [i for i, d in enumerate(logical) if d != 1].index(s['dim'])
        shape = list(want)
        shape[pos] = s['offset'] + s['extent']
        held = out.get(s['tensor'])
        if held is None:
            out[s['tensor']] = {'dtype': dtype, 'shape': shape, 'file': ARTIFACT_FILE}
            sliced[s['tensor']] = pos
        else:
            held['shape'][pos] = max(held['shape'][pos], shape[pos])
    else:
        die(f"{identity}: a location form the synthesis has no answer for: {sorted(ev)}")


def _synthesise(d3):
    """A checkpoint that holds exactly what a D3 says, `tests/run_artifact.py`'s `headers_of`
    widened to the forms the corpus writes — and, beside it, the tensors a `slice` binds, with the
    position of the axis it is sliced along, so that a case can make one too short."""
    out, sliced = {}, {}
    for t in d3['tensors']:
        ev = t.get('location')
        if ev is None:
            continue
        _place(ev, [a['extent'] for a in t['shape']], t['dtype'], out, sliced, t['identity'])
    return out, sliced


def _edited(headers, edits):
    """The recorded headers with the case's edits applied — the same two operations on both
    sides, so an edit cannot mean one thing here and another there."""
    out = copy.deepcopy(headers)
    for edit in edits:
        if edit['op'] == 'delete':
            del out[edit['tensor']]
        elif edit['op'] == 'set':
            out[edit['tensor']] = copy.deepcopy(edit['entry'])
        else:
            die(f"unknown header edit {edit['op']}")
    return out


def _artifact_case(name, product, checkpoint, **rest):
    """One case: what the tools answered over that D3 and those headers."""
    import artifact as artifact_mod
    errors, advisories, stats = artifact_mod.check(product, checkpoint)
    return dict(name=name, errors=errors, advisories=advisories, stats=stats, **rest)


def _artifact_forms():
    """The synthetic D3s: every branch of `_check_part`, `tests/run_artifact.py`'s own first.

    Its cases are the stack, the concat, the slice and the multiplicity; the six beside them are
    the branches it does not write — a stack whose count is not the axis's extent, a concat part
    that is absent or of another rank, a slice that is absent, of another rank, or of another
    dtype — each of which is a line of the tools nothing else of the repository produces.
    """
    def d3_of(location, shape, dtype='bf16'):
        return {'tensors': [{'identity': 't', 'dtype': dtype, 'location': location,
                             'shape': [{'axis': 'a', 'extent': n} for n in shape]}]}

    def header(dtype, shape):
        return {'dtype': dtype, 'shape': list(shape), 'file': ARTIFACT_FILE}

    stack = {'stack': {'axis': 'e', 'dim': 0,
                       'parts': [{'tensor': f"w.{i}"} for i in range(3)]}}
    three = {f"w.{i}": header('bf16', [4]) for i in range(3)}
    concat = {'concat': {'axis': 'r', 'dim': 0,
                         'parts': [{'tensor': 'g'}, {'tensor': 'u'}]}}
    parts = {'g': header('bf16', [2, 4]), 'u': header('bf16', [4, 4])}
    sliced = {'slice': {'tensor': 'big', 'axis': 'r', 'dim': 0, 'offset': 3, 'extent': 4}}
    beyond = {'slice': {'tensor': 'big', 'axis': 'r', 'dim': 0, 'offset': 8, 'extent': 4}}
    big = {'big': header('bf16', [10, 4])}
    copies = {'stack': {'axis': 'multiplicity', 'dim': 0,
                        'parts': [{'tensor': f"p.{i}.weight"} for i in range(3)]}}
    three_copies = {f"p.{i}.weight": header('bf16', [2, 2]) for i in range(3)}

    forms = [
        ('stack-holds', d3_of(stack, [3, 4]), three),
        ('stack-part-absent', d3_of(stack, [3, 4]), {k: v for k, v in three.items()
                                                     if k != 'w.1'}),
        ('stack-count-wrong', d3_of(stack, [4, 4]), {**three, 'w.3': header('bf16', [4])}),
        ('concat-holds', d3_of(concat, [6, 4]), parts),
        ('concat-sum-wrong', d3_of(concat, [5, 4]), parts),
        ('concat-part-absent', d3_of(concat, [6, 4]), {'g': parts['g']}),
        ('concat-part-rank', d3_of(concat, [6, 4]), {**parts, 'g': header('bf16', [2])}),
        ('slice-fits', d3_of(sliced, [4, 4]), big),
        ('slice-does-not-fit', d3_of(beyond, [4, 4]), big),
        ('slice-absent', d3_of(sliced, [4, 4]), {}),
        ('slice-rank', d3_of(sliced, [4, 4]), {'big': header('bf16', [10])}),
        ('slice-dtype', d3_of(sliced, [4, 4]), {'big': header('f32', [10, 4])}),
        ('multiplicity-stacked', d3_of(copies, [3, 2, 2]), three_copies),
        ('multiplicity-copy-absent', d3_of(copies, [3, 2, 2]),
         {k: v for k, v in three_copies.items() if k != 'p.2.weight'}),
        ('multiplicity-fused', d3_of({'tensor': 'p.weight'}, [3, 2, 2]),
         {'p.weight': header('bf16', [3, 2, 2])}),
        ('multiplicity-fused-count', d3_of({'tensor': 'p.weight'}, [3, 2, 2]),
         {'p.weight': header('bf16', [2, 2, 2])}),
        ('multiplicity-one', d3_of({'tensor': 'p.weight'}, [1, 2, 2]),
         {'p.weight': header('bf16', [2, 2])}),
        ('unit-axes-dropped', d3_of({'tensor': 'w'}, [4096]),
         {'w': header('bf16', [1, 4096, 1])}),
    ]
    return [_artifact_case(name, d3, headers, d3=d3, headers=headers)
            for name, d3, headers in forms]


def artifact(out, corpus, name_of, assignments):
    """`artifact.check` over the corpus against a checkpoint synthesised from each document's own
    D3, over mutations of those headers, and over synthetic D3s for every location form.

    The mutations are the cases `tests/run_artifact.py` writes against llama3-8b — an absent
    tensor, a wrong shape, a wrong dtype, unit axes the logical shape lacks, a tensor no location
    names — applied to *every* located document of the corpus, so that the wording is held over
    the composite, over the documents whose locations are slices, and over the one with stacks.
    """
    import primitive_library as primitive_library_mod
    import derive as derive_mod

    written = os.path.join(out, 'artifact', 'headers')
    os.makedirs(written, exist_ok=True)

    documents, cases = [], []
    for path in corpus():
        slug = name_of(path)
        relative = os.path.relpath(path, ROOT)
        assignment = assignments.get(slug)
        with open(path, encoding='utf-8') as handle:
            document = json.load(handle)
        cat = primitive_library_mod.load_for(path, document, None)
        d3 = derive_mod.products(path, cat, assignment)['d3']
        headers, sliced = _synthesise(d3)
        record = {'name': slug, 'path': relative,
                  'assignment': None if assignment is None else encode_map(assignment),
                  'tensors': len(d3['tensors']),
                  'located': sum(1 for t in d3['tensors'] if 'location' in t),
                  'physical': len(headers),
                  'headers': f"artifact/headers/{slug}.json"}
        documents.append(record)
        write(os.path.join(written, f"{slug}.json"),
              json.dumps(headers, separators=(',', ':')) + '\n')

        clean = _artifact_case(f"{slug}/clean", d3, headers, document=slug, headers_of=slug,
                               edits=[])
        if record['located'] and (clean['errors'] or clean['advisories']):
            die(f"{slug}: the synthesised checkpoint does not satisfy its own D3 "
                f"({(clean['errors'] + clean['advisories'])[:1]})")
        cases.append(clean)
        if not record['located']:
            continue

        names = sorted(headers)
        first = names[0]
        entry = headers[first]
        other = 'f64' if entry['dtype'] == 'f32' else 'f32'
        edited = [
            ('absent', [{'op': 'delete', 'tensor': first}]),
            ('shape-wrong', [{'op': 'set', 'tensor': first,
                              'entry': dict(entry, shape=[n + 1 for n in entry['shape']])}]),
            ('dtype-wrong', [{'op': 'set', 'tensor': first, 'entry': dict(entry, dtype=other)}]),
            ('unit-axes', [{'op': 'set', 'tensor': first,
                            'entry': dict(entry, shape=[1] + list(entry['shape']) + [1])}]),
            ('unnamed', [{'op': 'set', 'tensor': 'oracle.unused.weight',
                          'entry': {'dtype': 'f32', 'shape': [64], 'file': ARTIFACT_FILE}}]),
        ]
        # A tensor a `slice` binds, made one element too short along the sliced axis: the branch
        # only the documents whose locations are slices reach — 162 of them each in the three
        # qwen documents, and none anywhere else.
        if sliced:
            short = sorted(sliced)[0]
            shape = list(headers[short]['shape'])
            shape[sliced[short]] -= 1
            edited.append(('slice-short', [{'op': 'set', 'tensor': short,
                                            'entry': dict(headers[short], shape=shape)}]))
        for suffix, edits in edited:
            cases.append(_artifact_case(f"{slug}/{suffix}", d3, _edited(headers, edits),
                                        document=slug, headers_of=slug, edits=edits))

    # The composite against the flat document's checkpoint (§3.4): its instance's tensors,
    # prefixed, are the flat document's names.
    flat = {one['name']: one for one in documents}
    if 'shieldstral-3b' in flat and 'shieldstral-3b-composite' in flat:
        with open(os.path.join(written, 'shieldstral-3b.json'), encoding='utf-8') as handle:
            headers = json.load(handle)
        path = os.path.join(ROOT, flat['shieldstral-3b-composite']['path'])
        with open(path, encoding='utf-8') as handle:
            document = json.load(handle)
        cat = primitive_library_mod.load_for(path, document, None)
        d3 = derive_mod.products(path, cat)['d3']
        cases.append(_artifact_case('shieldstral-3b-composite/against-the-flat-document', d3,
                                    headers, document='shieldstral-3b-composite',
                                    headers_of='shieldstral-3b', edits=[]))

    return {'documents': documents, 'cases': cases, 'forms': _artifact_forms()}


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

    expression_cases = expressions(assignments, corpus, name_of)
    print(f"oracle: {len(expression_cases['cases'])} expression case(s) over "
          f"{len(expression_cases['documents'])} document(s)")

    library_cases = library()
    print(f"oracle: {len(library_cases['cases'])} library rejection case(s) and "
          f"{len(library_cases['mutations'])} cross-reference case(s)")

    model_cases = model(out, corpus, name_of)
    print(f"oracle: {len(model_cases['documents'])} document(s) normalised and "
          f"{len(model_cases['cases'])} edited case(s)")

    quantity_cases = quantities(corpus, name_of, assignments)
    print(f"oracle: {len(quantity_cases['documents'])} document(s) and "
          f"{len(quantity_cases['cases'])} case(s) of quantities and assignments")

    argument_cases = arguments(corpus, name_of, assignments)
    print(f"oracle: {len(argument_cases['cases'])} argument site(s) over "
          f"{len(argument_cases['documents'])} document(s), and "
          f"{len(argument_cases['synthetic'])} synthetic declaration(s)")

    graph_cases = graph(corpus, name_of, assignments)
    print(f"oracle: {len(graph_cases['documents'])} document(s) analysed as far as V19, "
          f"{sum(len(one.get('sites', [])) for one in graph_cases['documents'])} resolved site(s) "
          f"and {sum(len(one.get('edges', [])) for one in graph_cases['documents'])} edge(s)")

    expansion_cases = expansion(out, corpus, name_of, assignments)
    print(f"oracle: {len(expansion_cases['documents'])} document(s) expanded and "
          f"{len(expansion_cases['cases'])} edited case(s), "
          f"{sum(one.get('nodes', 0) for one in expansion_cases['documents'])} node(s) and "
          f"{sum(one.get('edges', 0) for one in expansion_cases['documents'])} edge(s)")

    binding_cases = bindings(out, corpus, name_of, assignments)
    print(f"oracle: {len(binding_cases['documents'])} document(s) analysed in full and "
          f"{len(binding_cases['cases'])} edited case(s), "
          f"{sum(len(one.get('tensors', [])) for one in binding_cases['documents'])} tensor "
          f"identity instance(s) and "
          f"{sum(len(one.get('states', [])) for one in binding_cases['documents'])} state one(s)")

    artifact_cases = artifact(out, corpus, name_of, assignments)
    print(f"oracle: {len(artifact_cases['documents'])} document(s) checked against a synthesised "
          f"checkpoint, {sum(one['physical'] for one in artifact_cases['documents'])} physical "
          f"tensor(s), {len(artifact_cases['cases'])} case(s) and "
          f"{len(artifact_cases['forms'])} location form(s)")

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
        'expressions': {
            'index': 'expressions/index.json',
            'note': ('every expression and condition where it stands, with the environment it '
                     'was evaluated in and the value expr.py answered; values are tagged and '
                     'written as text, so the integer/float distinction survives the fixture.'),
            'cases': len(expression_cases['cases']),
        },
        'structural': {
            'index': 'structural/index.json',
            'note': ('a model document is read through validate.structural, a library unit '
                     'through schema.deepest, which is how the tools read each of them.'),
            'cases': len(structural_cases),
        },
        'model': {
            'index': 'model/index.json',
            'note': ('model.load over every model document of the repository, and over edited '
                     'ones reaching the branches no document takes; each normalised document is '
                     'the whole text json.dumps writes, so a parity failure names the line.'),
            'documents': len(model_cases['documents']),
            'cases': len(model_cases['cases']),
        },
        'quantities': {
            'index': 'quantities/index.json',
            'note': ('check_quantities over the normalised reading of every model document of '
                     'the repository, check_assignment and missing_assignment over the document '
                     'as read, and the edited documents and assignments that reach the branches '
                     'the corpus does not take; values are tagged, so an edit keeps its '
                     'float-ness and the resolved map its integer/float distinction.'),
            'documents': len(quantity_cases['documents']),
            'cases': len(quantity_cases['cases']),
        },
        'arguments': {
            'index': 'arguments/index.json',
            'note': ('resolve_arguments, the V8 block and the per-argument facts of describe, '
                     'over every site of every model document of the repository and of every '
                     'template it instantiates, and over synthetic declarations reaching the '
                     'branches the reference base does not declare; every line the walk produces '
                     'is checked against validate.analyse own errors for the same document.'),
            'documents': len(argument_cases['documents']),
            'sites': len(argument_cases['cases']),
            'synthetic': len(argument_cases['synthetic']),
        },
        'graph': {
            'index': 'graph/index.json',
            'note': ('validate.analyse truncated at the parameter bindings — its own source, cut '
                     'at the comment that opens them and closed with the interface-port block and '
                     'the merge of an expanded template counters — over every model document of '
                     'the repository; held to the whole function on each: its lines are a '
                     'subsequence of analyse own, and the counters and ports it computes are '
                     'analyse own.'),
            'documents': len(graph_cases['documents']),
        },
        'bindings': {
            'index': 'bindings/index.json',
            'note': ('the whole validate.analyse over every model document of the repository, and '
                     'over edited ones reaching the branches nothing in the repository takes: the '
                     'refusals, the counters, the advisories, the instance keys, what is carried '
                     'across fragments, the physical names, the bound slots and the identity '
                     'instances D3 and D4 read. Every site is named by analyse own `where`, '
                     'lifted from its source.'),
            'documents': len(binding_cases['documents']),
            'cases': len(binding_cases['cases']),
        },
        'expansion': {
            'index': 'expansion/index.json',
            'note': ('d1.emit over every model document of the repository and over edited ones '
                     'reaching the branches nothing in the repository takes: the emitted document '
                     'as --d1 writes it, or the exception the emitter raised. The corpus and the '
                     'template point at the file --d1 wrote, re-emitted and compared here so that '
                     'the expectation is the tool own output.'),
            'documents': len(expansion_cases['documents']),
            'cases': len(expansion_cases['cases']),
        },
        'artifact': {
            'index': 'artifact/index.json',
            'note': ('artifact.check over every model document of the repository against a '
                     'checkpoint synthesised from its own D3 — recorded literally, so that both '
                     'implementations read the same bytes — over five mutations of those headers '
                     'per located document, and over synthetic D3s reaching every branch of '
                     '_check_part. No weights are needed: the corpus checkpoints are gigabytes '
                     'and live outside the repository.'),
            'documents': len(artifact_cases['documents']),
            'cases': len(artifact_cases['cases']),
            'forms': len(artifact_cases['forms']),
        },
        'library': {
            'index': 'library/index.json',
            'note': ('the reference base gathered, the rejection suite refused word for word, and '
                     'primitive_references over one-value mutations of every unit — each recorded '
                     'as a pointer into a repository file, so both implementations read the same '
                     'bytes and apply the same change.'),
            'cases': len(library_cases['cases']),
            'mutations': len(library_cases['mutations']),
        },
    }
    write(os.path.join(out, 'structural', 'index.json'),
          json.dumps({'cases': structural_cases}, indent=1) + '\n')
    write(os.path.join(out, 'expressions', 'index.json'),
          json.dumps(expression_cases, indent=1) + '\n')
    write(os.path.join(out, 'library', 'index.json'),
          json.dumps(library_cases, indent=1) + '\n')
    write(os.path.join(out, 'model', 'index.json'),
          json.dumps(model_cases, indent=1) + '\n')
    write(os.path.join(out, 'quantities', 'index.json'),
          json.dumps(quantity_cases, indent=1) + '\n')
    write(os.path.join(out, 'arguments', 'index.json'),
          json.dumps(argument_cases, indent=1) + '\n')
    write(os.path.join(out, 'graph', 'index.json'),
          json.dumps(graph_cases, indent=1) + '\n')
    write(os.path.join(out, 'bindings', 'index.json'),
          json.dumps(binding_cases, indent=1) + '\n')
    write(os.path.join(out, 'expansion', 'index.json'),
          json.dumps(expansion_cases, indent=1) + '\n')
    write(os.path.join(out, 'artifact', 'index.json'),
          json.dumps(artifact_cases, indent=1) + '\n')
    write(os.path.join(out, 'manifest.json'), json.dumps(manifest, indent=2) + '\n')
    print(f"oracle: {len(documents)} document(s), {len(primitive_schemas)} primitive schema(s), "
          f"{len(rejection_documents)} rejection document(s), "
          f"{len(rejection_bases)} rejection base(s)")


if __name__ == '__main__':
    main()

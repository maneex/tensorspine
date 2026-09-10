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
    write(os.path.join(out, 'manifest.json'), json.dumps(manifest, indent=2) + '\n')
    print(f"oracle: {len(documents)} document(s), {len(primitive_schemas)} primitive schema(s), "
          f"{len(rejection_documents)} rejection document(s), "
          f"{len(rejection_bases)} rejection base(s)")


if __name__ == '__main__':
    main()

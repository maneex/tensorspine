"""The edited documents and the assignments the `quantities` step records (feature 1.5).

`check_quantities` and `check_assignment` decide four rules over one map, and the corpus exercises
about a third of what they can say. Measured over the fifteen documents: 215 quantities, of which
186 are cardinalities, 15 reals and 14 enums — **no boolean and no physical quantity anywhere** —
202 literal, 5 derived and 8 external (the template's, all of them); ten literals declare a
derivation (V11's path); eight domains in all, seven intervals and one set, with no
upper-inclusive bound; and **not one variable quantity outside the template**, so
`variable_quantities`' fixpoint never takes its second step and the V3 domain rule it feeds fires
in a single rejection fixture.

So beside the repository's own documents the step records *edited* ones, in the idiom features 1.3
and 1.4 established: each case is a list of edits at JSON pointers into a repository file, and both
implementations apply the same edits to the same bytes — neither owns the input. Each reaches one
branch, and the table below is meant to be read as the branch list of `validate.py`'s last three
functions: every type, every unit, every domain edge, every source kind, the fixpoint's transitive
step, the two derivation refusals that no fixture carries, and the four places where the tools
*raise* rather than answering (an unorderable comparison, `int()` of an infinity or a NaN,
`float()` of an astronomical integer, and a member the grammar requires).

An edit's value is written as a plain Python value and encoded by the generator, so `4096` stays an
`int` and `4096.0` a `float` all the way into the fixture — which is the distinction V3 reads, and
the one thing a fixture written as bare JSON would lose.

Cases marked *off the grammar* are documents no schema would admit. They are here because these
three functions never see the schema: `analyse` runs after the structural stage, and what it does
with a document that never crossed it is a fact about the port's faithfulness, not about the
language. Where the tools raise, the fixture records the exception, and the port raises the same.
"""

SOURCE = 'data/models/llama3-8b.json'
TEMPLATE = 'data/models/decoder-causal-yarn/1.0.0.json'

# The README's `--assign` example, which is also `tests/signature.py`'s ASSIGNMENTS entry and
# `tests/run_templates.py`'s ASSIGNMENT: the template's documented assignment.
ASSIGNMENT = {"width": 3072, "layers": 26, "heads": 32, "kv_heads": 8, "head_dim": 128,
              "inner": 9216, "eps": 0.00001, "precision": "bf16"}

Q = '/quantities'


def _set(pointer, value):
    return {'pointer': pointer, 'op': 'set', 'value': value}


def _delete(pointer):
    return {'pointer': pointer, 'op': 'delete'}


def _value(name, value):
    return _set(f'{Q}/{name}/source/value', value)


def _type(name, declared):
    return _set(f'{Q}/{name}/type', declared)


def _domain(name, declared):
    return _set(f'{Q}/{name}/domain', declared)


def _source(name, declared):
    return _set(f'{Q}/{name}/source', declared)


def _interval(lower=None, upper=None, lower_inclusive=True, upper_inclusive=True):
    domain = {'kind': 'interval'}
    if lower is not None:
        domain['lower'] = {'value': lower, 'inclusive': lower_inclusive}
    if upper is not None:
        domain['upper'] = {'value': upper, 'inclusive': upper_inclusive}
    return domain


def _literal(value):
    return {'literal': value}


def _physical(unit):
    return {'kind': 'physical', 'unit': unit}


def _derived(expression):
    return {'kind': 'derived', 'expression': expression}


def _multiply(*args):
    return {'op': 'multiply', 'args': list(args)}


def _floor_divide(left, right):
    return {'op': 'floor_divide', 'args': [left, right]}


def _divide(left, right):
    return {'op': 'divide', 'args': [left, right]}


def _quantity(name):
    return {'quantity': name}


DERIVATION = f'{Q}/head_dim/source/derivation'

# Each case: a name, the document it edits, the edits, and the assignment it is read under.
# `llama3-8b` declares d=4096, ffn=14336, heads=32, kv_heads=8, head_dim=128 (with the derivation
# floor_divide(d, heads)), layers=32, vocab=128256, eps=1e-05 (real) and precision='bf16' (an enum
# over ['bf16']); every one of them is a literal, and none declares a domain.
CASES = [
    # --- the type table (V3), on a cardinality, a real and an enum -----------
    ('cardinality-real-value', SOURCE, [_value('d', 4096.0)], None),
    ('cardinality-negative', SOURCE, [_value('d', -1)], None),
    ('cardinality-string', SOURCE, [_value('d', 'four thousand')], None),
    ('cardinality-boolean', SOURCE, [_value('d', True)], None),
    ('cardinality-null', SOURCE, [_value('d', None)], None),
    ('cardinality-list', SOURCE, [_value('d', [4096])], None),
    ('real-string', SOURCE, [_value('eps', 'small')], None),
    ('real-boolean', SOURCE, [_value('eps', True)], None),
    ('real-integer-accepted', SOURCE, [_value('eps', 1)], None),
    ('enum-out-of-set', SOURCE, [_value('precision', 'fp8')], None),
    ('enum-integer', SOURCE, [_value('precision', 1)], None),
    ('boolean-accepted', SOURCE,
     [_type('precision', {'kind': 'boolean'}), _value('precision', True)], None),
    ('boolean-given-a-string', SOURCE, [_type('precision', {'kind': 'boolean'})], None),

    # --- physical units: whole numbers everywhere but seconds (V3, R07) -----
    ('physical-tokens-whole', SOURCE, [_type('d', _physical('tokens'))], None),
    ('physical-tokens-fraction', SOURCE,
     [_type('d', _physical('tokens')), _value('d', 2.5)], None),
    ('physical-tokens-whole-real', SOURCE,
     [_type('d', _physical('tokens')), _value('d', 4096.0)], None),
    ('physical-seconds-real', SOURCE,
     [_type('d', _physical('seconds')), _value('d', 0.5)], None),
    ('physical-bytes-string', SOURCE,
     [_type('d', _physical('bytes')), _value('d', 'plenty')], None),
    ('physical-elements-boolean', SOURCE,
     [_type('d', _physical('elements')), _value('d', True)], None),
    ('physical-operations-whole', SOURCE, [_type('d', _physical('operations'))], None),
    # `float(v) != int(v)` compares a rounded double with an exact integer, so an integer past
    # 2**53 is refused as "not a whole number" — the tools' own arithmetic, reproduced.
    ('physical-tokens-inexact-integer', SOURCE,
     [_type('d', _physical('tokens')), _value('d', 2 ** 53 + 1)], None),

    # --- where the type table raises (off the grammar) ----------------------
    ('physical-tokens-infinity', SOURCE,
     [_type('d', _physical('tokens')), _value('d', float('inf'))], None),
    ('physical-tokens-nan', SOURCE,
     [_type('d', _physical('tokens')), _value('d', float('nan'))], None),
    ('physical-tokens-astronomical-integer', SOURCE,
     [_type('d', _physical('tokens')), _value('d', 10 ** 400)], None),
    ('physical-without-unit', SOURCE, [_type('d', {'kind': 'physical'})], None),
    ('record-type-given-a-scalar', SOURCE,
     [_type('d', {'kind': 'record', 'fields': {}})], None),
    ('unknown-type-kind', SOURCE, [_type('d', {'kind': 'colour'})], None),
    ('type-missing', SOURCE, [_delete(f'{Q}/d/type')], None),
    ('source-missing', SOURCE, [_delete(f'{Q}/d/source')], None),
    ('quantities-missing', SOURCE, [_delete(Q)], None),
    ('derived-without-expression', SOURCE, [_source('ffn', {'kind': 'derived'})], None),

    # --- domains (V3): every edge, inclusive and exclusive ------------------
    ('domain-lower-inclusive-below', SOURCE,
     [_domain('d', _interval(lower=_literal(8192)))], None),
    ('domain-lower-inclusive-equal', SOURCE,
     [_domain('d', _interval(lower=_literal(4096)))], None),
    ('domain-lower-exclusive-equal', SOURCE,
     [_domain('d', _interval(lower=_literal(4096), lower_inclusive=False))], None),
    ('domain-upper-inclusive-above', SOURCE,
     [_domain('d', _interval(upper=_literal(1024)))], None),
    ('domain-upper-exclusive-equal', SOURCE,
     [_domain('d', _interval(upper=_literal(4096), upper_inclusive=False))], None),
    ('domain-both-bounds-violated', SOURCE,
     [_domain('d', _interval(lower=_literal(8192), upper=_literal(1024)))], None),
    ('domain-bound-is-real', SOURCE,
     [_domain('d', _interval(lower=_literal(4096.5)))], None),
    ('domain-bound-is-an-expression', SOURCE,
     [_domain('d', _interval(lower=_multiply(_literal(2), _literal(4096))))], None),
    # A bound written over another quantity is undecidable in the empty scope `_check_domain`
    # evaluates in, and is skipped rather than resolved — even though `d` and `ffn` both have a
    # value at this point.
    ('domain-bound-reads-a-quantity', SOURCE,
     [_domain('d', _interval(lower=_quantity('ffn')))], None),
    ('domain-interval-with-no-bound', SOURCE, [_domain('d', {'kind': 'interval'})], None),
    ('domain-set-outside', SOURCE,
     [_domain('d', {'kind': 'set', 'values': [1024, 2048]})], None),
    ('domain-set-inside', SOURCE, [_domain('d', {'kind': 'set', 'values': [4096]})], None),
    # `4096 in [4096.0]` is true in Python: `in` compares with `==`, which crosses int and float.
    ('domain-set-member-is-real', SOURCE,
     [_domain('d', {'kind': 'set', 'values': [4096.0]})], None),
    # The domain is checked only when the type held: one refusal per value, with its reason.
    ('domain-after-a-type-refusal', SOURCE,
     [_value('d', 'four thousand'), _domain('d', _interval(lower=_literal(1)))], None),
    # An enum's string against a numeric bound: Python cannot order them and raises.
    ('domain-unorderable', SOURCE,
     [_domain('precision', _interval(lower=_literal(1)))], None),

    # --- variability (§2.1) and its fixpoint --------------------------------
    ('external-without-domain', SOURCE, [_source('d', {'kind': 'external'})], None),
    ('external-with-domain', SOURCE,
     [_source('d', {'kind': 'external'}), _domain('d', _interval(lower=_literal(1)))], None),
    ('external-with-domain-assigned', SOURCE,
     [_source('d', {'kind': 'external'}), _domain('d', _interval(lower=_literal(1)))],
     {'d': 4096}),
    ('external-with-default', SOURCE,
     [_source('d', {'kind': 'external', 'default': _literal(4096)}),
      _domain('d', _interval(lower=_literal(1)))], None),
    ('external-assigned-below-its-domain', SOURCE,
     [_source('d', {'kind': 'external'}), _domain('d', _interval(lower=_literal(8192)))],
     {'d': 4096}),
    ('derived-of-a-literal-is-not-variable', SOURCE,
     [_source('ffn', _derived(_multiply(_quantity('d'), _literal(4))))], None),
    ('derived-of-an-external-is-variable', SOURCE,
     [_source('d', {'kind': 'external'}), _domain('d', _interval(lower=_literal(1))),
      _source('ffn', _derived(_multiply(_quantity('d'), _literal(4))))],
     {'d': 4096}),
    # The fixpoint's second step: `vocab` reads `ffn`, which reads the external `d`.
    ('derived-variable-transitively', SOURCE,
     [_source('d', {'kind': 'external'}), _domain('d', _interval(lower=_literal(1))),
      _source('ffn', _derived(_multiply(_quantity('d'), _literal(4)))),
      _source('vocab', _derived(_multiply(_quantity('ffn'), _literal(2))))],
     {'d': 4096}),
    # `src.get('derivation')` is read for every source kind but `derived`, this one included.
    ('external-carrying-a-derivation', SOURCE,
     [_source('d', {'kind': 'external', 'derivation': _quantity('nope')})], None),

    # --- derivations: V1, V10 and V11 ---------------------------------------
    ('derived-reads-undeclared', SOURCE,
     [_source('ffn', _derived(_quantity('nope')))], None),
    # `quantity_references` answers a set, so one name read twice is one line.
    ('derived-reads-undeclared-twice', SOURCE,
     [_source('ffn', _derived(_multiply(_quantity('nope'), _quantity('nope'))))], None),
    # Two undeclared names come out sorted, not in the order the expression reads them.
    ('derived-reads-two-undeclared', SOURCE,
     [_source('ffn', _derived(_multiply(_quantity('zeta'), _quantity('alpha'))))], None),
    ('literal-derivation-agrees-across-kinds', SOURCE,
     [_set(DERIVATION, _divide(_quantity('d'), _quantity('heads')))], None),
    ('literal-derivation-disagrees-by-a-real', SOURCE,
     [_set(DERIVATION, _divide(_quantity('d'), _quantity('kv_heads')))], None),
    ('literal-derivation-does-not-resolve', SOURCE,
     [_set(DERIVATION, _floor_divide(_quantity('d'), _literal(0)))], None),
    ('literal-derivation-on-a-string', SOURCE,
     [_set(f'{Q}/precision/source/derivation', _literal('f16'))], None),
    ('literal-derivation-on-a-string-agreeing', SOURCE,
     [_set(f'{Q}/precision/source/derivation', _literal('bf16'))], None),

    # --- assignments (§4.6), against the template's external quantities -----
    ('template-documented-assignment', TEMPLATE, [], ASSIGNMENT),
    ('template-without-an-assignment', TEMPLATE, [], None),
    ('template-inadmissible', TEMPLATE, [], dict(ASSIGNMENT, layers=0, precision='fp4')),
    ('template-wrong-types', TEMPLATE, [],
     dict(ASSIGNMENT, width=3072.5, kv_heads=True, eps='small')),
    ('template-eps-at-its-upper-bound', TEMPLATE, [], dict(ASSIGNMENT, eps=1.0)),
    ('template-eps-at-zero', TEMPLATE, [], dict(ASSIGNMENT, eps=0.0)),
    # An assignment name the document does not declare is not refused: `check_assignment` walks
    # the document's quantities and never looks at the assignment's own names.
    ('template-unknown-name', TEMPLATE, [], dict(ASSIGNMENT, wid7h=1)),
    ('template-partial', TEMPLATE, [], {'width': 3072}),
    # An assignment cannot stand for a literal quantity: only external sources are read.
    ('model-assignment-is-ignored', SOURCE, [], {'d': 1, 'precision': 'fp8'}),
]

"""The synthetic declarations the `arguments` step records beside the corpus (feature 1.6a).

`_resolve_record` decides V2 and V3 over a *declaration*, and a declaration is a primitive
library unit's, not a document's: an edited document — the idiom of features 1.3, 1.4 and 1.5 —
cannot reach a branch the reference base never declares, because the base is read-only (Q6) and
the editor may not touch it. So this table is the expression step's other idiom instead: a
synthetic algebra, written here and evaluated by the tools' own functions.

What the reference base leaves unvisited, measured over its 36 primitive versions: **13
`present_when`s in all**, every one of them on `attention.dense`, `attention.latent_compressed`,
`ffn.gated`, `moe` or `sequence.gated_delta`; **no `seconds` argument and no `elements`, `bytes`
or `operations` one** — every physical argument is in `tokens`; **no set domain** (every domain
is an interval, all but two of them a lower bound alone); **no default that fails to resolve**,
no default chain of length two, and no default naming a record field; **no argument type the
grammar admits and the base does not use** — no `record` nested three deep. Half of the branches
of `_resolve_record`, `_check_type` and `_check_argument_domain` are therefore reached by nothing
the repository holds.

Each case is `(name, definition, given, quantities, env)`: a definition carrying the `arguments`
the walk resolves and the `invariants` V8 reads, the argument map a document would write, and
the quantities and index environment `static_argument` evaluates it in. Numbers keep their Python
type — `4096` an `int`, `4096.0` a `float` — all the way into the fixture, and the parity suite
reads the fixture with the lexeme-preserving parser, so the distinction V3 turns on survives.

Cases marked *off the grammar* are declarations no unit schema would admit. They are here because
`_resolve_record` never sees the schema: the loader has, and what the walk does with a
declaration that never crossed it is a fact about the port's faithfulness, not about the
language. Where the tools raise, the fixture records the exception and the port raises the same.
"""

# --- the vocabulary, written once ------------------------------------------


def _argument(kind, required=False, **rest):
    declaration = {'type': dict(kind), 'required': required, 'structural': False}
    declaration.update(rest)
    return declaration


CARDINALITY = {'kind': 'cardinality'}
REAL = {'kind': 'real'}
BOOLEAN = {'kind': 'boolean'}


def _physical(unit):
    return {'kind': 'physical', 'unit': unit}


def _enum(*values):
    return {'kind': 'enum', 'values': list(values)}


def _record(**fields):
    return {'kind': 'record', 'fields': fields}


def _interval(lower=None, upper=None, lower_inclusive=True, upper_inclusive=True):
    domain = {'kind': 'interval'}
    if lower is not None:
        domain['lower'] = {'value': lower, 'inclusive': lower_inclusive}
    if upper is not None:
        domain['upper'] = {'value': upper, 'inclusive': upper_inclusive}
    return domain


def _set(*values):
    return {'kind': 'set', 'values': list(values)}


def _literal(value):
    return {'literal': value}


def _argument_ref(path):
    return {'argument': path}


def _quantity(name):
    return {'quantity': name}


def _index(name):
    return {'index': name}


def _record_value(**fields):
    return {'record': fields}


def _compare(operator, left, right):
    return {'compare': {'operator': operator, 'left': left, 'right': right}}


def _present(path):
    return {'present': path}


def _invariant(holds, description):
    return {'holds': holds, 'description': description}


def _definition(arguments, invariants=None):
    definition = {'arguments': arguments}
    if invariants is not None:
        definition['invariants'] = invariants
    return definition


# The quantities every case is read under unless it says otherwise: `llama3-8b`'s own, so that a
# value written as `{"quantity": "d"}` means what it means in the corpus.
QUANTITIES = {'d': 4096, 'heads': 32, 'kv_heads': 8, 'head_dim': 128, 'layers': 32,
              'eps': 0.00001, 'ratio': 0.5, 'precision': 'bf16', 'gated': True,
              'span': 4096, 'seconds': 0.02, 'inexact': 4096.0}
ENV = {'layer': 3}

# `heads` and `kv_heads`, as `attention.dense` declares them: the default that names another
# argument, and the bound that names one (§4.6, unit guide §4).
HEADS = _argument(CARDINALITY, required=True, domain=_interval(lower=_literal(1)))
KV_HEADS = _argument(CARDINALITY, default=_argument_ref('heads'),
                     domain=_interval(lower=_literal(1), upper=_argument_ref('heads')))

CASES = [
    # --- V2: presence, unknown names, defaults ------------------------------
    ('required-present', _definition({'heads': HEADS}), {'heads': _quantity('heads')},
     QUANTITIES, {}),
    ('required-missing', _definition({'heads': HEADS}), {}, QUANTITIES, {}),
    ('unknown-argument', _definition({'heads': HEADS}),
     {'heads': _literal(32), 'kernel_hint': _literal('flash')}, QUANTITIES, {}),
    ('unknown-argument-alone', _definition({}), {'heads': _literal(32)}, QUANTITIES, {}),
    ('optional-absent', _definition({'scale': _argument(REAL)}), {}, QUANTITIES, {}),
    # A declared default is not a silent default (I7): applied before every check (V2).
    ('default-names-another-argument', _definition({'heads': HEADS, 'kv_heads': KV_HEADS}),
     {'heads': _literal(32)}, QUANTITIES, {}),
    ('default-overridden', _definition({'heads': HEADS, 'kv_heads': KV_HEADS}),
     {'heads': _literal(32), 'kv_heads': _literal(8)}, QUANTITIES, {}),
    # The fixpoint's second step: `c` reads `b`, `b` reads `a`, and they are declared backwards.
    ('default-chain-backwards', _definition({
        'c': _argument(CARDINALITY, default=_argument_ref('b')),
        'b': _argument(CARDINALITY, default=_argument_ref('a')),
        'a': _argument(CARDINALITY, required=True)}), {'a': _literal(4)}, QUANTITIES, {}),
    ('default-reads-a-record-field', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True)), required=True),
        'theta_again': _argument(REAL, default=_argument_ref('rope.theta'))}),
     {'rope': _record_value(theta=_literal(500000.0))}, QUANTITIES, {}),
    # What never resolves is a refusal, never a guess: an absent argument, and a cycle.
    ('default-does-not-resolve', _definition({
        'scale': _argument(REAL, default=_argument_ref('missing'))}), {}, QUANTITIES, {}),
    ('default-cycle', _definition({
        'a': _argument(CARDINALITY, default=_argument_ref('b')),
        'b': _argument(CARDINALITY, default=_argument_ref('a'))}), {}, QUANTITIES, {}),
    ('default-of-a-field-does-not-resolve', _definition({
        'rope': _argument(_record(
            theta=_argument(REAL, required=True),
            layout=_argument(_enum('split', 'interleaved'),
                             default=_argument_ref('rope.missing'))), required=True)},),
     {'rope': _record_value(theta=_literal(500000.0))}, QUANTITIES, {}),
    # A default that resolves to `False` is applied: `v is not None and v is not UNRESOLVED`,
    # not the truth of the value.
    ('default-false', _definition({'cross': _argument(BOOLEAN, default=_literal(False))}),
     {}, QUANTITIES, {}),
    ('default-zero', _definition({'offset': _argument(CARDINALITY, default=_literal(0))}),
     {}, QUANTITIES, {}),

    # --- V3: the type table, through an argument ----------------------------
    ('cardinality-real', _definition({'heads': HEADS}), {'heads': _literal(32.0)},
     QUANTITIES, {}),
    ('cardinality-negative', _definition({'heads': HEADS}), {'heads': _literal(-8)},
     QUANTITIES, {}),
    ('cardinality-string', _definition({'heads': HEADS}), {'heads': _literal('thirty-two')},
     QUANTITIES, {}),
    ('cardinality-boolean', _definition({'heads': HEADS}), {'heads': _literal(True)},
     QUANTITIES, {}),
    ('boolean-string', _definition({'cross': _argument(BOOLEAN)}), {'cross': _literal('yes')},
     QUANTITIES, {}),
    ('boolean-quantity', _definition({'cross': _argument(BOOLEAN)}),
     {'cross': _quantity('gated')}, QUANTITIES, {}),
    ('real-string', _definition({'scale': _argument(REAL)}), {'scale': _literal('banana')},
     QUANTITIES, {}),
    ('real-integer-accepted', _definition({'scale': _argument(REAL)}), {'scale': _literal(1)},
     QUANTITIES, {}),
    ('enum-out-of-set', _definition({'mask': _argument(_enum('causal', 'chunked', 'none'),
                                                       required=True)}),
     {'mask': _literal('sideways')}, QUANTITIES, {}),
    ('enum-from-a-quantity', _definition({'precision': _argument(_enum('bf16', 'fp8'))}),
     {'precision': _quantity('precision')}, QUANTITIES, {}),
    ('unresolved-quantity', _definition({'heads': HEADS}), {'heads': _quantity('absent')},
     QUANTITIES, {}),
    ('index-valued-argument', _definition({'layer': _argument(CARDINALITY, required=True)}),
     {'layer': _index('layer')}, QUANTITIES, ENV),
    ('index-out-of-scope', _definition({'layer': _argument(CARDINALITY, required=True)}),
     {'layer': _index('expert')}, QUANTITIES, ENV),
    # A physical value is a whole number in every unit but seconds (V3, R07).
    ('physical-tokens-whole', _definition({'span': _argument(_physical('tokens'))}),
     {'span': _quantity('span')}, QUANTITIES, {}),
    ('physical-tokens-whole-real', _definition({'span': _argument(_physical('tokens'))}),
     {'span': _quantity('inexact')}, QUANTITIES, {}),
    ('physical-tokens-fraction', _definition({'span': _argument(_physical('tokens'))}),
     {'span': _literal(2.5)}, QUANTITIES, {}),
    ('physical-seconds-real', _definition({'window': _argument(_physical('seconds'))}),
     {'window': _quantity('seconds')}, QUANTITIES, {}),
    ('physical-seconds-fraction', _definition({'window': _argument(_physical('seconds'))}),
     {'window': _literal(0.5)}, QUANTITIES, {}),
    ('physical-elements-fraction', _definition({'count': _argument(_physical('elements'))}),
     {'count': _literal(2.5)}, QUANTITIES, {}),
    ('physical-bytes-fraction', _definition({'size': _argument(_physical('bytes'))}),
     {'size': _literal(1.5)}, QUANTITIES, {}),
    ('physical-operations-fraction', _definition({'work': _argument(_physical('operations'))}),
     {'work': _literal(1.5)}, QUANTITIES, {}),
    ('physical-string', _definition({'span': _argument(_physical('tokens'))}),
     {'span': _literal('plenty')}, QUANTITIES, {}),

    # --- V3: the domain, at the call site -----------------------------------
    ('domain-lower-inclusive-equal', _definition({
        'heads': _argument(CARDINALITY, domain=_interval(lower=_literal(32)))}),
     {'heads': _literal(32)}, QUANTITIES, {}),
    ('domain-lower-inclusive-below', _definition({
        'heads': _argument(CARDINALITY, domain=_interval(lower=_literal(33)))}),
     {'heads': _literal(32)}, QUANTITIES, {}),
    ('domain-lower-exclusive-equal', _definition({
        'eps': _argument(REAL, domain=_interval(lower=_literal(0), lower_inclusive=False))}),
     {'eps': _literal(0)}, QUANTITIES, {}),
    ('domain-upper-inclusive-above', _definition({
        'heads': _argument(CARDINALITY, domain=_interval(upper=_literal(16)))}),
     {'heads': _literal(32)}, QUANTITIES, {}),
    ('domain-upper-exclusive-equal', _definition({
        'ratio': _argument(REAL, domain=_interval(upper=_literal(1), upper_inclusive=False))}),
     {'ratio': _literal(1.0)}, QUANTITIES, {}),
    ('domain-both-bounds-violated', _definition({
        'heads': _argument(CARDINALITY,
                           domain=_interval(lower=_literal(64), upper=_literal(16)))}),
     {'heads': _literal(32)}, QUANTITIES, {}),
    ('domain-set-inside', _definition({
        'streams': _argument(CARDINALITY, domain=_set(2, 4, 8))}),
     {'streams': _literal(4)}, QUANTITIES, {}),
    ('domain-set-outside', _definition({
        'streams': _argument(CARDINALITY, domain=_set(2, 4, 8))}),
     {'streams': _literal(3)}, QUANTITIES, {}),
    # A bound naming another argument: `kv_heads <= heads`, satisfied, violated, and read when
    # the argument it names is itself defaulted.
    ('domain-bound-names-an-argument', _definition({'heads': HEADS, 'kv_heads': KV_HEADS}),
     {'heads': _literal(32), 'kv_heads': _literal(8)}, QUANTITIES, {}),
    ('domain-bound-names-an-argument-violated',
     _definition({'heads': HEADS, 'kv_heads': KV_HEADS}),
     {'heads': _literal(8), 'kv_heads': _literal(32)}, QUANTITIES, {}),
    # An undecidable bound is skipped, not read as a limit: the argument it names was refused.
    ('domain-bound-names-a-refused-argument',
     _definition({'heads': HEADS, 'kv_heads': KV_HEADS}),
     {'heads': _literal('many'), 'kv_heads': _literal(32)}, QUANTITIES, {}),
    ('domain-bound-names-an-absent-argument', _definition({
        'kv_heads': _argument(CARDINALITY, domain=_interval(upper=_argument_ref('heads')))}),
     {'kv_heads': _literal(32)}, QUANTITIES, {}),
    # The domain is read after the type, and only when the type held.
    ('domain-after-a-type-refusal', _definition({
        'heads': _argument(CARDINALITY, domain=_interval(lower=_literal(1)))}),
     {'heads': _literal(-1)}, QUANTITIES, {}),

    # --- V3: records --------------------------------------------------------
    ('record-fields', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True),
                                  layout=_argument(_enum('split', 'interleaved'),
                                                   default=_literal('split'))))}),
     {'rope': _record_value(theta=_literal(500000.0))}, QUANTITIES, {}),
    ('record-given-a-scalar', _definition({
        'temperature': _argument(_record(floor=_argument(CARDINALITY, required=True)))}),
     {'temperature': _literal(0.1)}, QUANTITIES, {}),
    ('record-missing-required-field', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True)))}),
     {'rope': _record_value()}, QUANTITIES, {}),
    ('record-unknown-field', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True)))}),
     {'rope': _record_value(theta=_literal(500000.0), thetta=_literal(1.0))}, QUANTITIES, {}),
    ('record-field-refused', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True),
                                  layout=_argument(_enum('split'), required=True)))}),
     {'rope': _record_value(theta=_literal('big'), layout=_literal('split'))}, QUANTITIES, {}),
    ('record-nested-three-deep', _definition({
        'a': _argument(_record(b=_argument(_record(c=_argument(CARDINALITY, required=True)),
                                           required=True)), required=True)}),
     {'a': _record_value(b=_record_value(c=_literal(2)))}, QUANTITIES, {}),
    ('record-from-a-quantity', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True)))}),
     {'rope': _quantity('d')}, QUANTITIES, {}),
    ('record-unresolved', _definition({
        'rope': _argument(_record(theta=_argument(REAL, required=True)))}),
     {'rope': _quantity('absent')}, QUANTITIES, {}),

    # --- V3: `present_when`, and I2's refusal of an inapplicable value ------
    ('present-when-applicable', _definition({
        'mask': _argument(_enum('causal', 'chunked', 'none'), required=True),
        'chunk': _argument(_record(span=_argument(_physical('tokens'), required=True)),
                           required=True,
                           present_when=_compare('equal', _argument_ref('mask'),
                                                 _literal('chunked')))}),
     {'mask': _literal('chunked'), 'chunk': _record_value(span=_literal(4096))}, QUANTITIES, {}),
    ('present-when-inapplicable-and-given', _definition({
        'mask': _argument(_enum('causal', 'chunked', 'none'), required=True),
        'chunk': _argument(_record(span=_argument(_physical('tokens'), required=True)),
                           required=True,
                           present_when=_compare('equal', _argument_ref('mask'),
                                                 _literal('chunked')))}),
     {'mask': _literal('causal'), 'chunk': _record_value(span=_literal(4096))}, QUANTITIES, {}),
    # A required argument that does not apply is not missing: `required and applicable`.
    ('present-when-inapplicable-and-absent', _definition({
        'mask': _argument(_enum('causal', 'chunked', 'none'), required=True),
        'chunk': _argument(_record(span=_argument(_physical('tokens'), required=True)),
                           required=True,
                           present_when=_compare('equal', _argument_ref('mask'),
                                                 _literal('chunked')))}),
     {'mask': _literal('causal')}, QUANTITIES, {}),
    ('present-when-applicable-and-missing', _definition({
        'mask': _argument(_enum('causal', 'chunked', 'none'), required=True),
        'chunk': _argument(_record(span=_argument(_physical('tokens'), required=True)),
                           required=True,
                           present_when=_compare('equal', _argument_ref('mask'),
                                                 _literal('chunked')))}),
     {'mask': _literal('chunked')}, QUANTITIES, {}),
    # A field's `present_when` reads an absolute path — the record being filled in place is what
    # makes it decidable while its siblings are still being resolved.
    ('present-when-reads-a-sibling-field', _definition({
        'rope': _argument(_record(
            scaling=_argument(_record(
                kind=_argument(_enum('none', 'yarn'), required=True),
                beta_fast=_argument(REAL, required=True,
                                    present_when=_compare('equal',
                                                          _argument_ref('rope.scaling.kind'),
                                                          _literal('yarn')))))))}),
     {'rope': _record_value(scaling=_record_value(kind=_literal('yarn'),
                                                  beta_fast=_literal(32.0)))}, QUANTITIES, {}),
    ('present-when-reads-a-sibling-field-inapplicable', _definition({
        'rope': _argument(_record(
            scaling=_argument(_record(
                kind=_argument(_enum('none', 'yarn'), required=True),
                beta_fast=_argument(REAL, required=True,
                                    present_when=_compare('equal',
                                                          _argument_ref('rope.scaling.kind'),
                                                          _literal('yarn')))))))}),
     {'rope': _record_value(scaling=_record_value(kind=_literal('none'),
                                                  beta_fast=_literal(32.0)))}, QUANTITIES, {}),
    # An undecidable `present_when` is false on the primitive side: "the guard it protects simply
    # does not fire" — which is why the loader refuses a comparison of a maybe-absent argument
    # outside a `present` test (feature 1.3).
    ('present-when-undecidable', _definition({
        'window': _argument(CARDINALITY,
                            present_when=_compare('equal', _argument_ref('absent'),
                                                  _literal(True)))}),
     {'window': _literal(4096)}, QUANTITIES, {}),
    ('present-when-guarded-by-present', _definition({
        'mask': _argument(_enum('causal', 'none')),
        'window': _argument(CARDINALITY,
                            present_when={'all': [_present('mask'),
                                                  _compare('equal', _argument_ref('mask'),
                                                           _literal('causal'))]})}),
     {'window': _literal(4096)}, QUANTITIES, {}),

    # --- V8: the invariants -------------------------------------------------
    ('invariant-holds', _definition(
        {'heads': HEADS, 'kv_heads': KV_HEADS},
        [_invariant(_compare('equal',
                             {'op': 'modulo', 'args': [_argument_ref('heads'),
                                                       _argument_ref('kv_heads')]},
                             _literal(0)), 'heads is a multiple of kv_heads')]),
     {'heads': _literal(32), 'kv_heads': _literal(8)}, QUANTITIES, {}),
    ('invariant-fails', _definition(
        {'heads': HEADS, 'kv_heads': KV_HEADS},
        [_invariant(_compare('equal',
                             {'op': 'modulo', 'args': [_argument_ref('heads'),
                                                       _argument_ref('kv_heads')]},
                             _literal(0)), 'heads is a multiple of kv_heads')]),
     {'heads': _literal(32), 'kv_heads': _literal(3)}, QUANTITIES, {}),
    # An invariant reading an argument V3 refused is skipped: the V3 line already covers it.
    ('invariant-skipped', _definition(
        {'heads': HEADS, 'kv_heads': KV_HEADS},
        [_invariant(_compare('equal',
                             {'op': 'modulo', 'args': [_argument_ref('heads'),
                                                       _argument_ref('kv_heads')]},
                             _literal(0)), 'heads is a multiple of kv_heads')]),
     {'heads': _literal(32), 'kv_heads': _literal(0.5)}, QUANTITIES, {}),
    # An invariant reading nothing at all: the message carries no parenthesis.
    ('invariant-without-references', _definition(
        {'heads': HEADS}, [_invariant({'boolean': False}, 'nothing ever holds')]),
     {'heads': _literal(32)}, QUANTITIES, {}),
    # A path that resolves to nothing is left out of the values shown.
    ('invariant-reads-an-absent-path', _definition(
        {'mask': _argument(_enum('causal', 'none'), required=True),
         'streaming': _argument(BOOLEAN, default=_literal(False))},
        [_invariant({'any': [{'not': _present('window')},
                             _compare('equal', _argument_ref('mask'), _literal('causal'))]},
                    'a window needs a causal mask')]),
     {'mask': _literal('none')}, QUANTITIES, {}),
    ('invariant-guarded-by-present-vacuous', _definition(
        {'window': _argument(CARDINALITY)},
        [_invariant({'any': [{'not': _present('window')},
                             _compare('greater', _argument_ref('window'), _literal(0))]},
                    'a window spans at least one position')]),
     {}, QUANTITIES, {}),
    ('invariant-guarded-by-present-fires', _definition(
        {'window': _argument(CARDINALITY)},
        [_invariant({'any': [{'not': _present('window')},
                             _compare('greater', _argument_ref('window'), _literal(0))]},
                    'a window spans at least one position')]),
     {'window': _literal(0)}, QUANTITIES, {}),
    ('invariant-over-a-record-field', _definition(
        {'rope': _argument(_record(low=_argument(REAL, required=True),
                                   high=_argument(REAL, required=True)), required=True)},
        [_invariant(_compare('less_or_equal', _argument_ref('rope.low'),
                             _argument_ref('rope.high')),
                    'the correction range low does not exceed high')]),
     {'rope': _record_value(low=_literal(4.0), high=_literal(1.0))}, QUANTITIES, {}),
    ('invariant-shows-a-record', _definition(
        {'rope': _argument(_record(low=_argument(REAL, required=True)), required=True)},
        [_invariant(_compare('equal', _argument_ref('rope'), _literal(0)),
                    'rope is zero, which it never is')]),
     {'rope': _record_value(low=_literal(4.0))}, QUANTITIES, {}),
    ('invariants-two-of-three-fail', _definition(
        {'heads': HEADS, 'kv_heads': KV_HEADS},
        [_invariant(_compare('greater', _argument_ref('heads'), _literal(0)), 'heads is positive'),
         _invariant(_compare('greater', _argument_ref('heads'), _literal(64)),
                    'heads exceeds sixty-four'),
         _invariant(_compare('less', _argument_ref('kv_heads'), _literal(4)),
                    'kv_heads is below four')]),
     {'heads': _literal(32), 'kv_heads': _literal(8)}, QUANTITIES, {}),

    # --- names, and the one JavaScript reads as a prototype -----------------
    ('proto-named-argument', _definition({
        '__proto__': _argument(CARDINALITY, required=True),
        'twice': _argument(CARDINALITY, default={'op': 'multiply',
                                                 'args': [_argument_ref('__proto__'),
                                                          _literal(2)]})}),
     {'__proto__': _literal(21)}, QUANTITIES, {}),
    ('proto-named-field', _definition({
        'rope': _argument(_record(__proto__=_argument(CARDINALITY, required=True)),
                          required=True)}),
     {'rope': _record_value(__proto__=_literal(7))}, QUANTITIES, {}),

    # --- off the grammar: where the walk raises or falls through ------------
    ('unknown-type-kind', _definition({'heads': _argument({'kind': 'colour'})}),
     {'heads': _literal('red')}, QUANTITIES, {}),
    ('type-missing', _definition({'heads': {'required': True, 'structural': False}}),
     {'heads': _literal(32)}, QUANTITIES, {}),
    ('required-missing-from-the-declaration',
     _definition({'heads': {'type': CARDINALITY, 'structural': False}}), {}, QUANTITIES, {}),
    ('arguments-missing', {'invariants': []}, {}, QUANTITIES, {}),
    ('physical-without-unit', _definition({'span': _argument({'kind': 'physical'})}),
     {'span': _literal(4096)}, QUANTITIES, {}),
    ('enum-without-values', _definition({'mask': _argument({'kind': 'enum'})}),
     {'mask': _literal('causal')}, QUANTITIES, {}),
    ('record-without-fields', _definition({'rope': _argument({'kind': 'record'})}),
     {'rope': _record_value()}, QUANTITIES, {}),
    # A domain compared with a value it cannot order raises out of `analyse`, uncaught — the
    # quantity side's own hole (feature 1.5), here on the argument side.
    ('domain-on-an-enum-value', _definition({
        'mask': _argument(_enum('causal'), domain=_interval(lower=_literal(1)))}),
     {'mask': _literal('causal')}, QUANTITIES, {}),
    ('invariant-compares-a-string-with-a-number', _definition(
        {'mask': _argument(_enum('causal'), required=True)},
        [_invariant(_compare('greater', _argument_ref('mask'), _literal(1)),
                    'a mask exceeds one, which cannot be decided')]),
     {'mask': _literal('causal')}, QUANTITIES, {}),
]

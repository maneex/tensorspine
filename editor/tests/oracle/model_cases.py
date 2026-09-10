"""The edited documents the `model` step of the oracle normalises (feature 1.4).

`tools/model.py` has branches the corpus never takes: no document declares a scoped `constants`
rule, none declares the `tensor` or the `identity` of a scoped rule (all 471 leave it implied),
none has a composition with two indices, and only three endpoints are written as an explicit
selector. The three refusals are worse served still — `tests/rejections` carries a case for two of
them and none for the third.

So beside the corpus the step records *transformations* of one corpus document, each reaching one
branch. Each is a list of edits at JSON pointers into `data/models/llama3-8b.json`, and the parity
suite applies the same edits to the same bytes: neither implementation owns the input, exactly as
the library step's mutations are pointers rather than documents (feature 1.3).

The pointers need no escaping — no member name of a model document holds `/` or `~` — and the
generator refuses a value that carries a float, because the fixture's own JSON would then have to
carry the integer/float distinction that the two implementations are being compared on.
"""

SOURCE = 'data/models/llama3-8b.json'

SCOPED = '/compositions/decoder/bindings'
TOP = '/bindings'

# A range over `[0, 2)`, for the second index a corpus composition never has.
RANGE = {'start': {'literal': 0}, 'stop': {'literal': 2}, 'step': {'literal': 1}}

# A top-level rule of each kind, well enough shaped to collide with a hoisted one.
TOP_VALUE = {
    'from': {'instance': {'kind': 'root', 'instance': 'embed'}, 'port': 'output'},
    'to': {'instance': {'kind': 'root', 'instance': 'final_n'}, 'port': 'input'},
}
TOP_PARAMETER = {
    'tensor': {'name': 'collide'},
    'members': [{'instance': {'kind': 'root', 'instance': 'embed'}, 'parameter': 'weight'}],
}
TOP_STATE = {
    'identity': {'name': 'collide'},
    'members': [{'instance': {'kind': 'root', 'instance': 'embed'}, 'state': 'kv'}],
}
TOP_CONSTANT = {
    'constant': 'mask',
    'members': [{'instance': {'kind': 'root', 'instance': 'embed'}, 'constant': 'mask'}],
}

SCOPED_CONSTANT = {
    'constant': 'mask',
    'members': [{'site': 'attn', 'constant': 'mask'}],
}


def _set(pointer, value):
    return {'pointer': pointer, 'op': 'set', 'value': value}


def _delete(pointer):
    return {'pointer': pointer, 'op': 'delete'}


# Each case: a name, and the edits that make it. The name says what branch it reaches.
CASES = [
    # --- the three refusals ---------------------------------------------------
    ('collision-value', [_set(f'{TOP}/values/decoder.attn.norm_in', TOP_VALUE)]),
    ('collision-parameter', [_set(f'{TOP}/parameters/decoder.attn.q', TOP_PARAMETER)]),
    ('collision-state', [_set(f'{TOP}/states/decoder.attn.kv', TOP_STATE)]),
    ('collision-constant', [
        _set(f'{SCOPED}/constants', {'attn.mask': SCOPED_CONSTANT}),
        _set(f'{TOP}/constants/decoder.attn.mask', TOP_CONSTANT),
    ]),
    ('unknown-site-value-from', [_set(f'{SCOPED}/values/attn.norm_in/from/site', 'attention')]),
    ('unknown-site-value-to', [_set(f'{SCOPED}/values/attn.norm_in/to/site', 'attention')]),
    ('unknown-site-parameter-member',
     [_set(f'{SCOPED}/parameters/attn.q/members/0/site', 'attention')]),
    ('unknown-site-state-member', [_set(f'{SCOPED}/states/attn.kv/members/0/site', 'attention')]),
    ('unknown-index-value',
     [_set(f'{SCOPED}/values/attn_n.carry/from/indices', {'lyaer': {'index': 'layer'}})]),
    ('unknown-index-parameter',
     [_set(f'{SCOPED}/parameters/attn.q/members/0/indices', {'head': {'literal': 0}})]),

    # --- endpoints written as explicit selectors ------------------------------
    ('explicit-root-endpoint', [
        _set(f'{SCOPED}/values/attn.norm_in/from',
             {'instance': {'kind': 'root', 'instance': 'embed'}, 'port': 'output'}),
    ]),
    ('explicit-generated-endpoint', [
        _set(f'{SCOPED}/values/attn.norm_in/from',
             {'instance': {'kind': 'generated', 'composition': 'decoder', 'instance': 'ffn_r',
                           'indices': {'layer': {'literal': 0}}},
              'port': 'output'}),
    ]),
    ('explicit-instance-member', [
        _set(f'{SCOPED}/parameters/attn.q/members/0',
             {'instance': {'kind': 'root', 'instance': 'embed'}, 'parameter': 'weight'}),
    ]),

    # --- identities the corpus always leaves implied --------------------------
    ('declared-tensor', [
        _set(f'{SCOPED}/parameters/attn.q/tensor',
             {'name': 'shared.q', 'indices': {'layer': {'index': 'layer'}}}),
    ]),
    ('declared-tensor-without-indices',
     [_set(f'{SCOPED}/parameters/attn.q/tensor', {'name': 'shared.q'})]),
    ('declared-identity', [_set(f'{SCOPED}/states/attn.kv/identity', {'name': 'shared.kv'})]),
    ('parameter-without-location', [_delete(f'{SCOPED}/parameters/attn.q/location')]),
    ('parameter-without-dtype', [_delete(f'{SCOPED}/parameters/attn.q/dtype')]),
    ('state-dtype', [_set(f'{SCOPED}/states/attn.kv/dtype', 'bf16')]),
    ('state-when', [
        _set(f'{SCOPED}/states/attn.kv/when',
             {'compare': {'operator': 'greater_or_equal', 'left': {'index': 'layer'},
                          'right': {'literal': 0}}}),
    ]),
    ('parameter-when', [
        _set(f'{SCOPED}/parameters/attn.q/when',
             {'compare': {'operator': 'not_equal', 'left': {'index': 'layer'},
                          'right': {'literal': 0}}}),
    ]),

    # --- a kind no corpus document writes -------------------------------------
    ('scoped-constants', [_set(f'{SCOPED}/constants', {'attn.mask': SCOPED_CONSTANT})]),
    ('scoped-constants-with-dtype', [
        _set(f'{SCOPED}/constants',
             {'attn.mask': dict(SCOPED_CONSTANT, dtype='u8')}),
    ]),
    ('scoped-constants-with-when', [
        _set(f'{SCOPED}/constants',
             {'attn.mask': {'when': {'boolean': True}, **SCOPED_CONSTANT}}),
    ]),
    ('scoped-constants-index-override', [
        _set(f'{SCOPED}/constants',
             {'attn.mask': {'constant': 'mask',
                            'members': [{'site': 'attn',
                                         'indices': {'layer': {'literal': 0}},
                                         'constant': 'mask'}]}}),
    ]),

    # --- a composition with more than one index -------------------------------
    ('two-index-composition', [_set('/compositions/decoder/indices/head', RANGE)]),
    ('two-index-override-first', [
        _set('/compositions/decoder/indices/head', RANGE),
        _set(f'{SCOPED}/values/attn.norm_in/from/indices', {'layer': {'literal': 0}}),
    ]),
    ('two-index-override-second', [
        _set('/compositions/decoder/indices/head', RANGE),
        _set(f'{SCOPED}/values/attn.norm_in/from/indices', {'head': {'literal': 1}}),
    ]),
    ('two-index-override-both', [
        _set('/compositions/decoder/indices/head', RANGE),
        _set(f'{SCOPED}/values/attn.norm_in/from/indices',
             {'head': {'literal': 1}, 'layer': {'literal': 0}}),
    ]),
    ('index-named-under-double-underscore', [
        _set('/compositions/decoder/indices/__proto__', RANGE),
    ]),

    # --- the shape of the scoped map itself -----------------------------------
    ('scoped-bindings-empty', [_set(SCOPED, {})]),
    ('scoped-kind-empty', [_set(SCOPED, {'values': {}})]),
    ('scoped-bindings-absent', [_delete(SCOPED)]),
    ('compositions-empty', [_set('/compositions', {})]),
    ('rule-name-without-a-dot', [
        _set(f'{SCOPED}/values/link',
             {'from': {'site': 'attn_n', 'port': 'output'},
              'to': {'site': 'attn', 'port': 'input'}}),
        _delete(f'{SCOPED}/values/attn.norm_in'),
    ]),
    ('rule-written-in-another-order', [
        _set(f'{SCOPED}/values/attn.norm_in',
             {'when': {'boolean': True},
              'to': {'site': 'attn', 'port': 'input'},
              'from': {'site': 'attn_n', 'port': 'output'}}),
    ]),
]

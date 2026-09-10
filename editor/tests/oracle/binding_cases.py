"""The edited documents the `bindings` step records (feature 1.6c).

`analyse`'s parameter and state blocks carry about thirty-five distinct refusals; the corpus and
the 73 documents of `tests/rejections/models/` reach twenty-two of them. Measured over the
repository, the ones nothing reaches are: both markers of the dtype selector (`unknown`, `not an
enum`), a physical tensor bound whole *and* sliced, four of V9's five refusals, and five of the
seven places `evaluate_location` refuses — an unresolvable index, a coordinate outside a stack,
a stack whose extent is not a count, a negative slice offset and a location form the grammar does
not carry. Nothing in the repository produces an **advisory** either, so `--lint`'s one model
finding has no document behind it.

So beside the repository's own documents the step records *edited* ones, in the idiom features
1.3, 1.4 and 1.5 established: each case is a list of edits at JSON pointers into a repository
file, and both implementations apply the same edits to the same bytes — neither owns the input.

Every case is built on a **shortened** corpus document: `llama3-8b` with its decoder cut to two
layers, and `voxtral-realtime` with both of its compositions cut to two. A composition of two
still carries an index in its physical names, a carry edge and a guard, and it makes each case's
answer readable — the same edit on the full document repeats every line thirty-two times.
Shortening is itself a case (`short`, `vox-short`), so the fixture states that the base each case
edits is clean.

What no edit of a corpus document can reach, because the reference base declares nothing that
would: a state port with several key-axis sets (all eleven of the base's state ports are keyed
`instance.session, instance.branch`), a state port whose rules do not cover its own
`present_when` (both ports with a conditional last rule are exhaustive), a template that locates
none of its identities (the one template of the repository locates all of them), a stack over an
axis whose extent *resolves* to something that is not a count (every extent of the base is an
argument with a lower bound of 1, so a bad one is `UNRESOLVED` and the tools print its address),
and a constant slot (no unit declares one, and `analyse` reads `bindings.constants` nowhere).
Those are the unit suite's, over a base built for them.
"""

LLAMA = 'data/models/llama3-8b.json'
VOXTRAL = 'data/models/voxtral-realtime.json'

# `llama3-8b`'s decoder runs `layer` over the literal 0..32 and its `final_n` edge names
# `layers - 1`, so both move together (the design canvas's erratum E10: the range is a literal,
# only the template names a quantity).
SHORT = [
    {'pointer': '/compositions/decoder/indices/layer/stop/literal', 'op': 'set', 'value': 2},
    {'pointer': '/quantities/layers/source/value', 'op': 'set', 'value': 2},
]

# `voxtral-realtime` has two compositions and names each one's length as a quantity.
VOX = [
    {'pointer': '/compositions/encoder/indices/layer/stop/literal', 'op': 'set', 'value': 2},
    {'pointer': '/compositions/decoder/indices/layer/stop/literal', 'op': 'set', 'value': 2},
    {'pointer': '/quantities/enc_layers/source/value', 'op': 'set', 'value': 2},
    {'pointer': '/quantities/dec_layers/source/value', 'op': 'set', 'value': 2},
]

PARAMETERS = '/bindings/parameters'
SCOPED = '/compositions/decoder/bindings'


def _set(pointer, value):
    return {'pointer': pointer, 'op': 'set', 'value': value}


def _delete(pointer):
    return {'pointer': pointer, 'op': 'delete'}


def _generated(composition, instance, layer):
    return {'kind': 'generated', 'composition': composition, 'instance': instance,
            'indices': {'layer': {'literal': layer}}}


def _root(instance):
    return {'kind': 'root', 'instance': instance}


def _tie_states(name, first, second):
    """A root state binding naming two ports as one identity: what V9 is written over."""
    return _set(f'/bindings/states/{name}', {
        'identity': {'name': name},
        'members': [{'instance': first, 'state': 'kv'}, {'instance': second, 'state': 'kv'}],
    })


def _name(*parts):
    """A physical name as tagged data: literal strings and `{"index": …}`."""
    return list(parts)


LAYER = {'index': 'layer'}

# Each case: a name, the document it edits, and the edits — read under no assignment, since every
# source is a closed document.
CASES = [
    # --- the shortened bases, which every case below edits ------------------
    ('short', LLAMA, []),
    ('vox-short', VOXTRAL, []),

    # --- the dtype selector (V14) -------------------------------------------
    ('dtype-selector-unknown', LLAMA,
     [_set(f'{PARAMETERS}/embed.weight/dtype', {'quantity': 'nowhere'})]),
    ('dtype-selector-not-an-enum', LLAMA,
     [_set(f'{PARAMETERS}/embed.weight/dtype', {'quantity': 'd'})]),
    # A literal selector: `_dtype_values` answers the one value, and the role refuses it.
    ('dtype-literal-inadmissible', LLAMA,
     [_set(f'{PARAMETERS}/embed.weight/dtype', 'fp8_e4m3')]),
    # The only branch that reads the enum's own values: a quantity whose source is not a literal.
    ('dtype-enum-values-of-a-derived-quantity', LLAMA, [
        _set('/quantities/mixed', {
            'type': {'kind': 'enum', 'values': ['bf16', 'fp8_e4m3']},
            'source': {'kind': 'derived', 'expression': {'quantity': 'precision'}}}),
        _set(f'{PARAMETERS}/embed.weight/dtype', {'quantity': 'mixed'}),
    ]),
    ('state-dtype-selector-unknown', LLAMA,
     [_set(f'{SCOPED}/states/attn.kv/dtype', {'quantity': 'nowhere'})]),
    # One line per payload component per member: `checked` counts each component.
    ('state-dtype-inadmissible', LLAMA, [_set(f'{SCOPED}/states/attn.kv/dtype', 'fp8_e4m3')]),

    # --- slots and ports (V1, V7) -------------------------------------------
    ('parameter-slot-unknown', LLAMA,
     [_set(f'{SCOPED}/parameters/attn.q/members/0/parameter', 'nowhere')]),
    # `q_gated` is present only under `output_gate`, which this instance does not set.
    ('parameter-slot-absent', LLAMA,
     [_set(f'{SCOPED}/parameters/attn.q/members/0/parameter', 'q_gated')]),
    ('parameter-member-instance-absent', LLAMA,
     [_set(f'{PARAMETERS}/embed.weight/members/0/instance/instance', 'nowhere')]),
    ('state-port-unknown', LLAMA,
     [_set(f'{SCOPED}/states/attn.kv/members/0/state', 'nowhere')]),
    ('state-port-bound-twice', LLAMA, [_set(f'{SCOPED}/states/attn.kv2', {
        'identity': {'name': 'attn.kv2'},
        'members': [{'site': 'attn', 'state': 'kv'}]})]),
    ('state-member-instance-absent', VOXTRAL,
     [_set('/bindings/states/conv_frontend.conv1_history/members/0/instance/instance', 'nowhere')]),

    # --- `loop_envs` is called twice over the parameter bindings ------------
    # Once for the identities, once for the resident count, so an undecidable guard is refused
    # twice — the second line after the V7 totality lines, which is where the second call stands.
    ('parameter-binding-when-unresolved', LLAMA, [
        _set('/quantities/maybe', {
            'type': {'kind': 'boolean'}, 'source': {'kind': 'external'},
            'domain': {'kind': 'set', 'values': [True, False]}}),
        _set(f'{PARAMETERS}/embed.weight/when', {'compare': {
            'operator': 'equal', 'left': {'quantity': 'maybe'}, 'right': {'literal': True}}}),
    ]),

    # --- tying (V15): the three refusals, on one identity -------------------
    ('tie-incompatible-roles-and-shapes', LLAMA, [
        _set(f'{PARAMETERS}/embed.weight/members', [
            {'instance': _root('embed'), 'parameter': 'weight'},
            {'instance': _root('final_n'), 'parameter': 'weight'}]),
        _delete(f'{PARAMETERS}/final_n.weight'),
    ]),

    # --- locations (V17): what `evaluate_location` refuses ------------------
    ('location-index-unresolved', LLAMA,
     [_set(f'{SCOPED}/parameters/attn.q/location/tensor/1', {'index': 'nowhere'})]),
    ('location-coordinate-outside-stack', LLAMA,
     [_set(f'{SCOPED}/parameters/attn.q/location/tensor/1', {'coordinate': 'feature'})]),
    # Off the grammar: the `location` union is closed, and `analyse` never sees the schema.
    ('location-unknown-form', LLAMA,
     [_set(f'{SCOPED}/parameters/attn.q/location', {'nowhere': True})]),
    # A `concat` that holds: the corpus writes none, and the one in `tests/rejections` is refused
    # for the slice inside it. This is the evaluated form D3 will write.
    ('location-concat', LLAMA, [_set(f'{SCOPED}/parameters/attn.q/location', {'concat': {
        'axis': 'heads_flat',
        'parts': [{'tensor': _name('model.layers.', LAYER, '.self_attn.q_a.weight')},
                  {'tensor': _name('model.layers.', LAYER, '.self_attn.q_b.weight')}]}})]),
    ('location-slice-negative-offset', LLAMA, [_set(f'{SCOPED}/parameters/attn.q/location', {
        'slice': {'tensor': _name('model.layers.', LAYER, '.self_attn.qkv.weight'),
                  'axis': 'heads_flat', 'offset': {'literal': -1}}})]),
    # A name bound whole by one identity and sliced by another.
    ('location-whole-and-slice', LLAMA, [_set(f'{SCOPED}/parameters/attn.k/location', {
        'slice': {'tensor': _name('model.layers.', LAYER, '.self_attn.q_proj.weight'),
                  'axis': 'kv_heads_flat', 'offset': {'literal': 0}}})]),
    # Two regions of one tensor, bound in the order that makes the sort load-bearing: `k` takes
    # the upper half and `v` the lower, so a pairwise walk over the insertion order would report
    # an overlap where `sorted(intervals)` finds none. `kv_heads_flat` is 8 x 128 = 1024 here.
    ('location-slices-out-of-order', LLAMA, [
        _set(f'{SCOPED}/parameters/attn.k/location', {
            'slice': {'tensor': _name('model.layers.', LAYER, '.self_attn.kv_proj.weight'),
                      'axis': 'kv_heads_flat', 'offset': {'literal': 1024}}}),
        _set(f'{SCOPED}/parameters/attn.v/location', {
            'slice': {'tensor': _name('model.layers.', LAYER, '.self_attn.kv_proj.weight'),
                      'axis': 'kv_heads_flat', 'offset': {'literal': 0}}}),
    ]),

    # --- state identities (V9, V20) and the advisory (§5.3) -----------------
    # Two members with everything in common: no V9 line, and V20's two writers.
    ('state-tie-compatible', VOXTRAL, [
        _tie_states('shared.kv', _generated('encoder', 'attn', 0), _generated('encoder', 'attn', 1)),
        _delete('/compositions/encoder/bindings/states/attn.kv'),
    ]),
    # The encoder's and the decoder's attention: different head dimensions, and the decoder's
    # stream is the same `audio` at another kind.
    ('state-members-different-payloads', VOXTRAL,
     [_tie_states('cross.kv', _generated('encoder', 'attn', 0), _generated('decoder', 'attn', 0))]),
    # One member chunked, the other windowed: rule 1 against rule 2, same payload, same stream.
    ('state-members-different-rules', VOXTRAL, [
        _set('/compositions/encoder/instances/attn/arguments/mask', {
            'if': {'compare': {'operator': 'equal', 'left': {'index': 'layer'},
                               'right': {'literal': 0}}},
            'then': {'literal': 'chunked'}, 'else': {'literal': 'causal'}}),
        _tie_states('shared.kv', _generated('encoder', 'attn', 0), _generated('encoder', 'attn', 1)),
        _delete('/compositions/encoder/bindings/states/attn.kv'),
    ]),
    # `streaming: false` drops the carrying condition, so the encoder's cache is a self-indexed
    # state on a fragmented stream that nothing carries: the advisory, and V18 beside it.
    ('advisory-self-indexed-not-carried', VOXTRAL,
     [_set('/compositions/encoder/instances/attn/arguments/streaming', {'literal': False})]),
]

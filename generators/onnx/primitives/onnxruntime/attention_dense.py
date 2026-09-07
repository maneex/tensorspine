"""attention.dense@2.0.0 for the onnxruntime target — causal attention over an `append` state as one
`GroupQueryAttention` node (com.microsoft): the rotary embedding inside it from cos/sin caches
(YaRN's frequencies and attention factor folded into the caches, the rotation being linear), the
cache growing along the sequence, GQA by its `kv_num_heads`. Every branch the fused kernel does
not take — a non-causal mask, no state, a partial or interleaved rope, an output gate, a head
size that is not a multiple of 16 — keeps the portable form.
"""
import numpy as np
from onnx import TensorProto

from primitives import attention_dense as portable
from primitives._common import linear
from primitives.attention_dense import inv_freq

PRIMITIVE = portable.PRIMITIVE
CAPABILITIES = dict(portable.CAPABILITIES, notes=["causal attention over an append state with a full-head rope (or none) as one GroupQueryAttention node, rotary inside"])


def supports(arguments):
    return portable.supports(arguments)


ROTARY_CACHE = 16384      # positions the fused form's cos/sin caches cover (a physical parameter `rotary_cache` overrides)


def fused(ctx, arguments, inputs, params, states, q, k, v):
    """The onnxruntime form: one GroupQueryAttention over the state, the rotation applied inside from
    cos/sin caches, which take YaRN's frequencies and attention factor as a scale of the caches
    (the rotation is linear: (x·cos + rot(x)·sin)·f = x·(f·cos) + rot(x)·(f·sin))."""
    b = ctx.b
    h, d, kv = arguments['heads'], arguments['head_dim'], arguments['kv_heads']
    i64 = lambda *v: b.const(np.array(v, dtype=np.int64), 'i')
    st = states['kv']
    past = st.read()
    past_k = b.node('Transpose', [b.unsqueeze0(past['k'], f"{ctx.node}.pk4")], hint=f"{ctx.node}.past_k", perm=[0, 2, 1, 3])   # [1, kv, len, d]
    past_v = b.node('Transpose', [b.unsqueeze0(past['v'], f"{ctx.node}.pv4")], hint=f"{ctx.node}.past_v", perm=[0, 2, 1, 3])
    n = b.node('Gather', [b.node('Shape', [q], hint=f"{ctx.node}.qshape"), b.const(np.array(0, dtype=np.int64), 'i0')], hint=f"{ctx.node}.n")
    held = b.node('Gather', [b.node('Shape', [past['k']], hint=f"{ctx.node}.pshape"), b.const(np.array(0, dtype=np.int64), 'i0')], hint=f"{ctx.node}.held")
    total = b.node('Cast', [b.node('Add', [held, n], hint=f"{ctx.node}.total64")], hint=f"{ctx.node}.total", to=TensorProto.INT32)
    seqlens = b.node('Unsqueeze', [b.node('Sub', [total, b.const(np.int32(1), 'one32')], hint=f"{ctx.node}.last"), i64(0)], hint=f"{ctx.node}.seqlens")
    r = arguments.get('rope')
    cache = int((ctx.physical or {}).get('rotary_cache', ROTARY_CACHE))
    if r:
        inv = inv_freq(d, r['theta'], r.get('scaling'))
        angles = np.arange(cache, dtype=np.float32)[:, None] * inv[None, :]
        factor = float(r['scaling']['attention_factor']) if r.get('scaling') else 1.0
        cos, sin = b.const((np.cos(angles) * factor).astype(np.float32), 'cos_cache'), b.const((np.sin(angles) * factor).astype(np.float32), 'sin_cache')
        rotary = [cos, sin]
    else:
        rotary = []
    out, present_k, present_v = b.node('GroupQueryAttention',
                                       [b.unsqueeze0(q, f"{ctx.node}.q3"), b.unsqueeze0(k, f"{ctx.node}.k3"), b.unsqueeze0(v, f"{ctx.node}.v3"),
                                        past_k, past_v, seqlens, total] + rotary,
                                       outputs=3, hint=f"{ctx.node}.gqa", domain='com.microsoft',
                                       num_heads=h, kv_num_heads=kv, do_rotary=1 if r else 0, rotary_interleaved=0, local_window_size=-1)
    st.write({'k': b.squeeze0(b.node('Transpose', [present_k], hint=f"{ctx.node}.present_k", perm=[0, 2, 1, 3]), f"{ctx.node}.k_out"),
              'v': b.squeeze0(b.node('Transpose', [present_v], hint=f"{ctx.node}.present_v", perm=[0, 2, 1, 3]), f"{ctx.node}.v_out")})
    return b.squeeze0(out, f"{ctx.node}.ctx")


def fused_aligned(ctx, arguments, inputs, params, states, q, k, v):
    """The onnxruntime form on the aligned layout (B04): GroupQueryAttention over the state's
    buffers [b, kv, capacity, d] shared between past and present, `seqlens_k` each session's held
    positions plus the new ones less one, `total_sequence_length` the largest; the rotation inside
    from the cos/sin caches. The runtime takes a fresh prefill of several sessions and a one-token
    decode of several, not a multi-token continuation of several (batch-plan finding 3)."""
    b = ctx.b
    h, d, kv = arguments['heads'], arguments['head_dim'], arguments['kv_heads']
    st = states['kv']
    past = st.read()
    held = st.held or ctx.held()
    past_k = b.node('Transpose', [past['k']], hint=f"{ctx.node}.past_k", perm=[0, 2, 1, 3])       # [b, kv, cap, d]
    past_v = b.node('Transpose', [past['v']], hint=f"{ctx.node}.past_v", perm=[0, 2, 1, 3])
    n = b.dim(q, 1, hint=f"{ctx.node}.n")
    total64 = b.node('Add', [b.node('ReduceMax', [held], hint=f"{ctx.node}.maxheld", keepdims=0), n], hint=f"{ctx.node}.total64")
    total = b.node('Cast', [total64], hint=f"{ctx.node}.total", to=TensorProto.INT32)
    seqlens = b.node('Cast', [b.node('Sub', [b.node('Add', [held, n], hint=f"{ctx.node}.lens"), b.i64(1)], hint=f"{ctx.node}.last")],
                     hint=f"{ctx.node}.seqlens", to=TensorProto.INT32)                             # [b]
    r = arguments.get('rope')
    cache = int((ctx.physical or {}).get('rotary_cache', ROTARY_CACHE))
    rotary = []
    if r:
        inv = inv_freq(d, r['theta'], r.get('scaling'))
        angles = np.arange(cache, dtype=np.float32)[:, None] * inv[None, :]
        factor = float(r['scaling']['attention_factor']) if r.get('scaling') else 1.0
        rotary = [b.const((np.cos(angles) * factor).astype(np.float32), 'cos_cache'), b.const((np.sin(angles) * factor).astype(np.float32), 'sin_cache')]
    out, present_k, present_v = b.node('GroupQueryAttention', [q, k, v, past_k, past_v, seqlens, total] + rotary,
                                       outputs=3, hint=f"{ctx.node}.gqa", domain='com.microsoft',
                                       num_heads=h, kv_num_heads=kv, do_rotary=1 if r else 0, rotary_interleaved=0, local_window_size=-1)
    st.write({'k': b.node('Transpose', [present_k], hint=f"{ctx.node}.present_k", perm=[0, 2, 1, 3]),
              'v': b.node('Transpose', [present_v], hint=f"{ctx.node}.present_v", perm=[0, 2, 1, 3])})
    return out


def fusable(arguments, states):
    """The branches the fused form covers: causal, an append state, a full-head rotate-half rope or none,
    no gate, a head size onnxruntime's kernel takes (a multiple of 16); the rest is the portable form."""
    r = arguments.get('rope')
    return (arguments['mask'] == 'causal' and 'kv' in states and not arguments.get('output_gate')
            and (not r or (r.get('layout', 'split') == 'split' and not r.get('partial') and not r.get('mrope')))
            and arguments['head_dim'] % 16 == 0)



def emit(ctx, arguments, inputs, params, states):
    if not fusable(arguments, states):
        return portable.emit(ctx, arguments, inputs, params, states)
    q, k, v = portable.projections(ctx, arguments, inputs, params)
    form = fused_aligned if ctx.layout == 'aligned' else fused
    out = form(ctx, arguments, inputs, params, states, q, k, v)
    return {'output': linear(ctx, out, ctx.param(params['out']), ctx.param(params['out_bias']) if arguments.get('out_bias') else None)}

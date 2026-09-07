"""attention.dense@1.0.0 — dense or grouped-query attention over an `append` KV state, causal.

| branch / record                 | status                                                   |
|---------------------------------|----------------------------------------------------------|
| mask causal, append state       | emitted: the keys and values of the invocation are appended to the state (Concat under the `none` layout; scattered at each session's held length into a buffer of the capacity under `aligned`), every query attends to the positions at or before its own, masked by explicit positions |
| mask none (stateless)           | emitted: every query attends to every key of the invocation |
| mask chunked, window, streaming, cross | refused                                            |
| rope: theta, layout split, partial, scaling yarn | emitted (rotate-half; YaRN's frequencies computed at emission as the reference computes them) |
| rope: layout interleaved / 2d, mrope, scaling llama3 / linear | refused                     |
| qk_norm, temperature, output_gate | refused                                                |
| scale, kv_source shared         | refused                                                  |
| q/k/v/out biases                | emitted                                                  |

Conventions as the reference reads them: keys of the current elements join the state before the
queries attend (a query sees itself); the scale is head_dim⁻½; rope `split` pairs channel i with
i + rotary/2 over the rotated channels, whose base frequencies are computed on the rotated width.
"""
import math

import numpy as np
from onnx import TensorProto

from primitives._common import linear, supports_from

PRIMITIVE = ("attention.dense", "1.0.0")
CAPABILITIES = {"arguments": {"width": "any", "heads": "any", "head_dim": "any", "kv_heads": "any", "scale": "absent",
                              "mask": ["causal", "none"], "window": "absent", "chunk": "absent", "cross": [False],
                              "streaming": [False], "kv_source": ["own"], "temperature": "absent",
                              "rope": {"absent": True, "fields": {"theta": "any", "layout": ["split"], "partial": "any",
                                       "mrope": "absent",
                                       "scaling": {"absent": True, "fields": {"kind": ["yarn"], "factor": "any", "orig_ctx": "any",
                                                   "beta_fast": "any", "beta_slow": "any", "attention_factor": "any",
                                                   "low": "absent", "high": "absent"}}}},
                              "qk_norm": "absent",
                              "q_bias": "any", "k_bias": "any", "v_bias": "any", "out_bias": "any", "output_gate": [False]},
                "states": ["append"]}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def inv_freq(r, theta, scaling=None):
    """The r/2 inverse frequencies over the rotated width: the trained ones, or YaRN's, as the reference computes them."""
    inv = 1.0 / (theta ** (np.arange(0, r, 2, dtype=np.float32) / r))
    if not scaling:
        return inv.astype(np.float32)
    factor, orig = float(scaling['factor']), float(scaling['orig_ctx'])

    def dim_of(rotations):
        return (r * math.log(orig / (rotations * 2 * math.pi))) / (2 * math.log(theta))
    low = max(math.floor(dim_of(scaling['beta_fast'])), 0)
    high = min(math.ceil(dim_of(scaling['beta_slow'])), r - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(r // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    return ((inv / factor) * ramp + inv * (1 - ramp)).astype(np.float32)


def rope(ctx, x, positions, d, theta, partial=None, scaling=None):
    """Rotate-half RoPE on x [(b,) n, heads, d] at positions [(b,) n]: the first `partial · d`
    channels rotated. Axes are counted from the end, so the `none` and `aligned` layouts share it."""
    b = ctx.b
    r = d if not partial else int(d * partial)
    inv = b.const(inv_freq(r, theta, scaling), 'inv_freq')
    i64 = lambda *v: b.const(np.array(v, dtype=np.int64), 'i')
    pos = b.node('Unsqueeze', [b.node('Cast', [positions], hint=f"{ctx.node}.posf", to=TensorProto.FLOAT), i64(-1)], hint=f"{ctx.node}.pos")
    freqs = b.node('Mul', [pos, inv], hint=f"{ctx.node}.freqs")                       # [(b,) n, r/2]
    emb = b.node('Unsqueeze', [b.node('Concat', [freqs, freqs], hint=f"{ctx.node}.emb", axis=-1), i64(-2)], hint=f"{ctx.node}.emb3")   # [(b,) n, 1, r]
    cos, sin = b.node('Cos', [emb], hint=f"{ctx.node}.cos"), b.node('Sin', [emb], hint=f"{ctx.node}.sin")
    xr = b.node('Slice', [x, i64(0), i64(r), i64(-1)], hint=f"{ctx.node}.xr") if r < d else x
    x1 = b.node('Slice', [xr, i64(0), i64(r // 2), i64(-1)], hint=f"{ctx.node}.x1")
    x2 = b.node('Slice', [xr, i64(r // 2), i64(r), i64(-1)], hint=f"{ctx.node}.x2")
    rot = b.node('Concat', [b.node('Neg', [x2], hint=f"{ctx.node}.negx2"), x1], hint=f"{ctx.node}.rot", axis=-1)
    y = b.node('Add', [b.node('Mul', [xr, cos], hint=f"{ctx.node}.xc"), b.node('Mul', [rot, sin], hint=f"{ctx.node}.rs")], hint=f"{ctx.node}.roped")
    if scaling:
        y = b.node('Mul', [y, b.const(np.float32(scaling['attention_factor']), 'attention_factor')], hint=f"{ctx.node}.yarn")
    if r < d:
        y = b.node('Concat', [y, b.node('Slice', [x, i64(r), i64(d), i64(-1)], hint=f"{ctx.node}.xp")], hint=f"{ctx.node}.roped_full", axis=-1)
    return y


LOWEST = np.float32(np.finfo(np.float32).min)     # a masked score: the dtype's lowest, never −inf, so a masked row stays finite (B07)


def scatter_positions(ctx, buffer, new, held, hint):
    """`new` [b, n, *payload] written into `buffer` [b, capacity, *payload] at rows held_b … held_b + n − 1
    of each session: the append evolution on the aligned layout."""
    b = ctx.b
    n = b.dim(new, 1, hint=f"{hint}.n")
    rows = b.node('Add', [b.node('Unsqueeze', [held, b.i64(-1)], hint=f"{hint}.held2"),
                          b.node('Unsqueeze', [b.node('Range', [b.i64(0), n, b.i64(1)], hint=f"{hint}.arange"), b.i64(0)], hint=f"{hint}.arange2")],
                  hint=f"{hint}.rows")                                                              # [b, n]
    idx = rows
    for _ in range(2):                                                                             # -> [b, n, 1, 1] for a [kv, d] payload
        idx = b.node('Unsqueeze', [idx, b.i64(-1)], hint=f"{hint}.idx")
    idx = b.node('Expand', [idx, b.shape_of(new, hint=f"{hint}.newshape")], hint=f"{hint}.idx4")   # [b, n, kv, d]
    return b.node('ScatterElements', [buffer, idx, new], hint=hint, axis=1)


def emit_aligned(ctx, arguments, inputs, params, states):
    """The aligned layout (B04): x [b, n, width]; the state a buffer [b, capacity, kv, d] per
    component, the new keys and values scattered at each session's held length; every query
    attends to the buffer's positions at or before its own (the rows beyond a session's length
    are masked with them)."""
    b = ctx.b
    h, d, kv = arguments['heads'], arguments['head_dim'], arguments['kv_heads']
    q, k, v = projections(ctx, arguments, inputs, params)
    q = b.reshape_tail(q, 2, [h, d], hint=f"{ctx.node}.q")
    k = b.reshape_tail(k, 2, [kv, d], hint=f"{ctx.node}.k")
    v = b.reshape_tail(v, 2, [kv, d], hint=f"{ctx.node}.v")
    r = arguments.get('rope')
    if r:
        q = rope(ctx, q, ctx.positions, d, r['theta'], r.get('partial'), r.get('scaling'))
        k = rope(ctx, k, ctx.positions, d, r['theta'], r.get('partial'), r.get('scaling'))
    causal = arguments['mask'] == 'causal'
    n = b.dim(q, 1, hint=f"{ctx.node}.n")
    if 'kv' in states:
        st = states['kv']
        past = st.read()
        held = st.held or ctx.held()
        K = scatter_positions(ctx, past['k'], k, held, f"{ctx.node}.K")
        V = scatter_positions(ctx, past['v'], v, held, f"{ctx.node}.V")
        st.write({'k': K, 'v': V})
        limit = b.node('Add', [held, n], hint=f"{ctx.node}.limit")                                  # [b]: the rows a session fills
    else:
        K, V, limit = k, v, None
    m = b.dim(K, 1, hint=f"{ctx.node}.m")
    kpos = b.node('Range', [b.i64(0), m, b.i64(1)], hint=f"{ctx.node}.kpos")                     # [m]

    def heads_first(t, groups):
        t = b.node('Transpose', [t], hint=f"{ctx.node}.hf", perm=[0, 2, 1, 3])                     # [b, kv, m, d]
        if groups > 1:
            t = b.node('Unsqueeze', [t, b.i64(2)], hint=f"{ctx.node}.u")                           # [b, kv, 1, m, d]
            shape = b.shape_of(t, hint=f"{ctx.node}.ushape")
            target = b.node('Concat', [b.node('Slice', [shape, b.i64(0), b.i64(2)], hint=f"{ctx.node}.bkv"), b.i64(groups),
                                       b.node('Slice', [shape, b.i64(3), b.i64(5)], hint=f"{ctx.node}.md")], hint=f"{ctx.node}.target", axis=0)
            t = b.node('Expand', [t, target], hint=f"{ctx.node}.ex")                               # [b, kv, groups, m, d]
            shape2 = b.shape_of(t, hint=f"{ctx.node}.exshape")
            merged = b.node('Concat', [b.node('Slice', [shape2, b.i64(0), b.i64(1)], hint=f"{ctx.node}.b1"), b.i64(h),
                                       b.node('Slice', [shape2, b.i64(3), b.i64(5)], hint=f"{ctx.node}.md2")], hint=f"{ctx.node}.merged", axis=0)
            t = b.node('Reshape', [t, merged], hint=f"{ctx.node}.rep")                             # [b, h, m, d]
        return t
    Qh = b.node('Transpose', [q], hint=f"{ctx.node}.Qh", perm=[0, 2, 1, 3])                        # [b, h, n, d]
    Kh, Vh = heads_first(K, h // kv), heads_first(V, h // kv)
    scores = b.node('MatMul', [Qh, b.node('Transpose', [Kh], hint=f"{ctx.node}.KhT", perm=[0, 1, 3, 2])], hint=f"{ctx.node}.scores")   # [b, h, n, m]
    scores = b.node('Mul', [scores, b.const(np.float32(1.0 / math.sqrt(d)), 'scale')], hint=f"{ctx.node}.scaled")
    allowed = None
    kpos4 = b.node('Unsqueeze', [kpos, b.i64(0, 1, 2)], hint=f"{ctx.node}.kpos4")                # [1, 1, 1, m]
    if causal:
        qpos = b.node('Unsqueeze', [ctx.positions, b.i64(1, 3)], hint=f"{ctx.node}.qpos")         # [b, 1, n, 1]
        allowed = b.node('LessOrEqual', [kpos4, qpos], hint=f"{ctx.node}.allowed")                # [b, 1, n, m]
    if limit is not None:
        valid = b.node('Less', [kpos4, b.node('Unsqueeze', [limit, b.i64(1, 2, 3)], hint=f"{ctx.node}.limit4")], hint=f"{ctx.node}.valid")   # [b, 1, 1, m]
        allowed = valid if allowed is None else b.node('And', [allowed, valid], hint=f"{ctx.node}.allowed_valid")
    if allowed is not None:
        scores = b.node('Where', [allowed, scores, b.const(LOWEST, 'lowest')], hint=f"{ctx.node}.masked")
    p = b.node('Softmax', [scores], hint=f"{ctx.node}.p", axis=-1)
    out = b.node('MatMul', [p, Vh], hint=f"{ctx.node}.ctx")                                        # [b, h, n, d]
    out = b.reshape_tail(b.node('Transpose', [out], hint=f"{ctx.node}.ctxT", perm=[0, 2, 1, 3]), 2, [h * d], hint=f"{ctx.node}.flat")
    return {'output': linear(ctx, out, ctx.param(params['out']), ctx.param(params['out_bias']) if arguments.get('out_bias') else None)}


def projections(ctx, arguments, inputs, params):
    """q, k, v of the invocation's elements, [n, heads·d] each, biases added."""
    x = inputs['input']
    q = linear(ctx, x, ctx.param(params['q']), ctx.param(params['q_bias']) if arguments.get('q_bias') else None)
    k = linear(ctx, x, ctx.param(params['k']), ctx.param(params['k_bias']) if arguments.get('k_bias') else None)
    v = linear(ctx, x, ctx.param(params['v']), ctx.param(params['v_bias']) if arguments.get('v_bias') else None)
    return q, k, v


def emit(ctx, arguments, inputs, params, states):
    if ctx.layout == 'aligned':
        return emit_aligned(ctx, arguments, inputs, params, states)
    b = ctx.b
    h, d, kv = arguments['heads'], arguments['head_dim'], arguments['kv_heads']
    i64 = lambda *v: b.const(np.array(v, dtype=np.int64), 'i')
    q, k, v = projections(ctx, arguments, inputs, params)
    q = b.node('Reshape', [q, i64(-1, h, d)], hint=f"{ctx.node}.q")
    k = b.node('Reshape', [k, i64(-1, kv, d)], hint=f"{ctx.node}.k")
    v = b.node('Reshape', [v, i64(-1, kv, d)], hint=f"{ctx.node}.v")
    r = arguments.get('rope')
    if r:
        q = rope(ctx, q, ctx.positions, d, r['theta'], r.get('partial'), r.get('scaling'))
        k = rope(ctx, k, ctx.positions, d, r['theta'], r.get('partial'), r.get('scaling'))
    causal = arguments['mask'] == 'causal'
    if 'kv' in states:
        st = states['kv']
        past = st.read()
        K = b.node('Concat', [past['k'], k], hint=f"{ctx.node}.K", axis=0)     # every position held, this invocation's last
        V = b.node('Concat', [past['v'], v], hint=f"{ctx.node}.V", axis=0)
        st.write({'k': K, 'v': V})
    else:
        K, V = k, v
    m = b.node('Gather', [b.node('Shape', [K], hint=f"{ctx.node}.Kshape"), i64(0)], hint=f"{ctx.node}.m")
    kpos = b.node('Range', [i64(0), b.node('Cast', [m], hint=f"{ctx.node}.m64", to=TensorProto.INT64), i64(1)], hint=f"{ctx.node}.kpos") if 'kv' in states or True else None

    def heads_first(t, groups):
        t = b.node('Transpose', [t], hint=f"{ctx.node}.hf", perm=[1, 0, 2])          # [kv, m, d]
        if groups > 1:
            t = b.node('Unsqueeze', [t, i64(1)], hint=f"{ctx.node}.u")               # [kv, 1, m, d]
            shape = b.node('Shape', [t], hint=f"{ctx.node}.ushape")
            target = b.node('Concat', [i64(kv, groups), b.node('Slice', [shape, i64(2), i64(4)], hint=f"{ctx.node}.md")], hint=f"{ctx.node}.target", axis=0)
            t = b.node('Expand', [t, target], hint=f"{ctx.node}.ex")                 # [kv, groups, m, d]
            t = b.node('Reshape', [t, i64(h, -1, d)], hint=f"{ctx.node}.rep")        # [h, m, d]
        return t
    Qh = b.node('Transpose', [q], hint=f"{ctx.node}.Qh", perm=[1, 0, 2])             # [h, n, d]
    Kh, Vh = heads_first(K, h // kv), heads_first(V, h // kv)
    scores = b.node('MatMul', [Qh, b.node('Transpose', [Kh], hint=f"{ctx.node}.KhT", perm=[0, 2, 1])], hint=f"{ctx.node}.scores")
    scores = b.node('Mul', [scores, b.const(np.float32(1.0 / math.sqrt(d)), 'scale')], hint=f"{ctx.node}.scaled")
    if causal:
        qpos = b.node('Unsqueeze', [ctx.positions, i64(1)], hint=f"{ctx.node}.qpos")          # [n, 1]
        allowed = b.node('LessOrEqual', [b.node('Unsqueeze', [kpos, i64(0)], hint=f"{ctx.node}.kpos2"), qpos], hint=f"{ctx.node}.allowed")   # [n, m]
        scores = b.node('Where', [b.node('Unsqueeze', [allowed, i64(0)], hint=f"{ctx.node}.allowed3"), scores, b.const(np.float32(-np.inf), 'neg_inf')], hint=f"{ctx.node}.masked")
    p = b.node('Softmax', [scores], hint=f"{ctx.node}.p", axis=-1)
    out = b.node('MatMul', [p, Vh], hint=f"{ctx.node}.ctx")                          # [h, n, d]
    out = b.node('Reshape', [b.node('Transpose', [out], hint=f"{ctx.node}.ctxT", perm=[1, 0, 2]), i64(-1, h * d)], hint=f"{ctx.node}.flat")
    return {'output': linear(ctx, out, ctx.param(params['out']), ctx.param(params['out_bias']) if arguments.get('out_bias') else None)}

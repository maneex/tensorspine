"""attention.dense@1.0.0 — dense or grouped-query attention over a KV state; self or cross.

| branch / record                 | status                                          |
|---------------------------------|-------------------------------------------------|
| mask causal                     | implemented                                     |
| mask none (stateless encoder)   | implemented                                     |
| mask chunked (`chunk`)          | refused                                         |
| cross (`source_values`)         | implemented under `mask: none`: keys and values are projected from the source elements the invocation delivers and appended to `kv` along the source stream — a delivery of nothing (every decode step) appends nothing — and every query attends to the whole cache; with `rope` refused (the source positions are not delivered to the kernel); with `mask: causal` refused (a query's position and a source position are on different streams) |
| streaming                       | implemented: a carrying property — the cache survives the fragments, the computation is the mask's and the window's |
| window (`span`)                 | implemented with `mask: causal`: the cache is a ring of `span` positions; a query at `p` attends to the keys at `j` with `p − span < j ≤ p` — itself and the `span − 1` before it, `transformers`' sliding mask — read from the ring's valid entries (whose positions are the last before the fragment) followed by the fragment's own, the ring appended after, so a fragment longer than the span is served too; with `mask: none` refused (a bidirectional window is not what any document declares) |
| rope: theta, layout split       | implemented (rotate-half)                       |
| rope: layout interleaved / 2d   | refused                                         |
| rope: partial                   | implemented (the first `partial · head_dim` channels) |
| rope: mrope (sections contiguous or interleaved) | implemented for one position stream (t = h = w: the two layouts are one computation; an image would tell them apart) |
| rope: scaling yarn              | implemented: the original implementation's correction range (rotations → dimensions, truncated to integers) and linear ramp over the r/2 frequencies; the document's `attention_factor` on the rotated queries and keys |
| rope: scaling llama3 / linear   | refused                                         |
| qk_norm kind rms (eps, scale, zero_centered) | implemented, before RoPE           |
| qk_norm kind layer              | refused                                         |
| temperature                     | refused                                         |
| q/k/v/out biases                | implemented                                     |
| output_gate (`q_gated`)         | implemented: per head, query rows then gate rows |

Conventions the contract leaves open, as read here: keys of the current elements are
appended to the state before the queries attend (a query sees itself; a cross-attention query
sees every source element delivered so far) — under `window`, read from the ring first and
appended after, which comes to the same for the queries and lets a fragment longer than the span
through; the scale is head_dim^-1/2; rope `split` pairs channel i with i + rotary/2 (rotate-half) over the rotated
channels only, whose base frequencies are computed on the rotated width; `qk_norm` is an RMS
norm over head_dim applied before RoPE, with the learned scales `qk_norm.scale` declares, zero-centred
when it says so. These
readings are now stated by the contract (finding 1, 30 Aug 2026); `mrope.sections` is declared
by the document and, for a single position stream, both layouts are plain RoPE.
"""
import math
import torch
from kernels._common import present, refuse_unknown, supports_from, w

CONTRACT = ("attention.dense", "1.0.0")
KNOWN = {'width', 'heads', 'head_dim', 'kv_heads', 'mask', 'window', 'chunk', 'cross', 'streaming', 'rope',
         'qk_norm', 'temperature', 'q_bias', 'k_bias', 'v_bias', 'out_bias', 'output_gate'}


CAPABILITIES = {"arguments": {"width": "any", "heads": "any", "head_dim": "any", "kv_heads": "any",
                              "mask": ["causal", "none"], "window": {"absent": True, "fields": {"span": "any"}}, "chunk": "absent",
                              "cross": [False, True], "streaming": "any", "temperature": "absent",
                              "rope": {"absent": True, "fields": {"theta": "any", "layout": ["split"], "partial": "any",
                                       "mrope": {"absent": True, "fields": {"t": "any", "h": "any", "w": "any",
                                                                              "sections": ["contiguous", "interleaved"]}},
                                       "scaling": {"absent": True, "fields": {"kind": ["yarn"], "factor": "any", "orig_ctx": "any",
                                                   "beta_fast": "any", "beta_slow": "any", "attention_factor": "any",
                                                   "low": "absent", "high": "absent"}}}},
                              "qk_norm": {"absent": True, "fields": {"kind": ["rms"], "eps": "any",
                                          "scale": {"absent": True, "fields": {"zero_centered": "any"}}}},
                              "q_bias": "any", "k_bias": "any", "v_bias": "any", "out_bias": "any", "output_gate": "any"},
                "states": ["append", "window"],
                "excluding": [{"cross": True, "mask": "causal"}],
                "transforms": ["align"],
                "notes": ["mrope for one position stream only: an image would need the sections to differ",
                          "cross attention with rope is refused at run time: the source stream's positions are not delivered to the kernel",
                          "a window with mask none is refused at run time: a query attends to itself and the span − 1 positions before it, the causal reading"]}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities.
FIXTURES = [
    {"case": "causal-rope-gqa", "seed": 11, "invocations": [{"input": 5}, {"input": 3}],
     "arguments": {"width": 64, "heads": 4, "kv_heads": 2, "head_dim": 16, "mask": "causal", "rope": {"theta": 10000.0}}},
    {"case": "bidirectional-biased", "seed": 12, "invocations": [{"input": 6}, {"input": 2}],
     "arguments": {"width": 64, "heads": 4, "head_dim": 16, "mask": "none",
                   "q_bias": True, "k_bias": True, "v_bias": True, "out_bias": True}},
    {"case": "gated-qknorm-partial-mrope", "seed": 13, "invocations": [{"input": 5}, {"input": 1}],
     "arguments": {"width": 64, "heads": 4, "kv_heads": 2, "head_dim": 16, "mask": "causal", "output_gate": True,
                   "qk_norm": {"kind": "rms", "eps": 1e-6, "scale": {"zero_centered": True}},
                   "rope": {"theta": 10000000.0, "partial": 0.5, "mrope": {"t": 2, "h": 1, "w": 1, "sections": "interleaved"}}}},
    {"case": "yarn", "seed": 14, "invocations": [{"input": 5}, {"input": 3}],
     "arguments": {"width": 64, "heads": 4, "head_dim": 16, "mask": "causal",
                   "rope": {"theta": 1000000.0, "scaling": {"kind": "yarn", "factor": 16.0, "orig_ctx": 16384,
                                                             "beta_fast": 32.0, "beta_slow": 1.0, "attention_factor": 1.0}}}},
    # two streams: seven source elements cached in the first invocation, none delivered in the second,
    # whose three queries attend to the seven cached (Whisper's biases: q, v and out, none on k)
    {"case": "cross", "seed": 15, "invocations": [{"input": 5, "source_values": 7}, {"input": 3}],
     "arguments": {"width": 64, "heads": 4, "head_dim": 16, "mask": "none", "cross": True,
                   "q_bias": True, "v_bias": True, "out_bias": True}},
    # a ring of four: the second invocation's queries read the ring, the third's six exceed the span,
    # so its first query sees three ring entries and its last none of them — the boundary p − span < j ≤ p
    {"case": "window", "seed": 16, "invocations": [{"input": 3}, {"input": 3}, {"input": 6}],
     "arguments": {"width": 64, "heads": 4, "kv_heads": 2, "head_dim": 16, "mask": "causal", "window": {"span": 4},
                   "rope": {"theta": 1000000.0}}},
    # the same on a fragmented input: the ring is carried across the fragments (Voxtral's encoder)
    {"case": "window-streaming", "seed": 17, "invocations": [{"input": 3}, {"input": 3}, {"input": 6}],
     "arguments": {"width": 64, "heads": 4, "head_dim": 16, "mask": "causal", "streaming": True, "window": {"span": 4},
                   "rope": {"theta": 1000000.0}, "q_bias": True, "v_bias": True, "out_bias": True}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def inv_freq(r, theta, scaling=None, device=None):
    """The r/2 inverse frequencies over the rotated width r: the trained ones, or YaRN's —
    the original implementation's correction range (a number of rotations mapped to a dimension,
    truncated to integers, clamped to [0, r-1]), a linear ramp over the r/2 frequencies, the
    frequencies interpolated by `factor` below the ramp and left as trained above it."""
    inv = 1.0 / (theta ** (torch.arange(0, r, 2, device=device, dtype=torch.float32) / r))
    if not scaling:
        return inv
    assert scaling['kind'] == 'yarn', scaling
    factor, orig = float(scaling['factor']), float(scaling['orig_ctx'])

    def dim_of(rotations):
        return (r * math.log(orig / (rotations * 2 * math.pi))) / (2 * math.log(theta))
    low = max(math.floor(dim_of(scaling['beta_fast'])), 0)
    high = min(math.ceil(dim_of(scaling['beta_slow'])), r - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(r // 2, device=device, dtype=torch.float32) - low) / (high - low), 0, 1)
    return (inv / factor) * ramp + inv * (1 - ramp)


def rope_split(x, positions, theta, partial=None, scaling=None):
    """Rotate-half RoPE over the first `partial · d` channels of each head (all of them when
    `partial` is absent): x [n, h, d], positions [n]. The base frequencies are computed on the
    rotated width, as the reference does; under YaRN the rotated channels are then multiplied by
    the document's `attention_factor` (queries and keys alike, as the reference scales cos and sin)."""
    d = x.shape[-1]
    r = d if not partial else int(d * partial)
    inv = inv_freq(r, theta, scaling, x.device)
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().to(x.dtype)[:, None, :], emb.sin().to(x.dtype)[:, None, :]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
    xr = xr * cos + torch.cat([-x2, x1], dim=-1) * sin
    if scaling:
        xr = xr * scaling['attention_factor']
    return torch.cat([xr, xp], dim=-1) if r < d else xr


def rms_norm(x, scale, eps, zero_centered):
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    y = (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype)
    return y * (scale + 1 if zero_centered else scale)


def attend(q, K, V, length, qpos, causal, static=False):
    """Scores of q [n, h, d] against the first `length` positions of K/V [cap, kv, d]
    (their positions are 0..length-1); GQA by repeating KV heads. `static` keeps
    the whole buffer and masks (the compiled form); otherwise the buffer is sliced."""
    n, h, d = q.shape
    kv = K.shape[1]
    if not static:
        K, V = K[:length], V[:length]
    m = K.shape[0]
    if h != kv:
        K = K.repeat_interleave(h // kv, dim=1)
        V = V.repeat_interleave(h // kv, dim=1)
    scores = torch.einsum('nhd,mhd->hnm', q, K) * (1.0 / math.sqrt(d))
    kpos = torch.arange(m, device=q.device)
    allowed = kpos[None, :] < length
    if causal:
        allowed = allowed & (kpos[None, :] <= qpos[:, None])
    scores = scores.masked_fill(~allowed[None, :, :], float('-inf'))
    p = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
    return torch.einsum('hnm,mhd->nhd', p, V).reshape(n, h * d)


def attend_positions(q, K, V, qpos, kpos, causal, span):
    """Scores of q [n, h, d] at positions qpos [n] against K/V [m, kv, d] at positions kpos [m],
    every key masked by its position: causal keeps j ≤ p, the window keeps j > p − span (the query
    among its own span). GQA by repeating KV heads."""
    n, h, d = q.shape
    kv = K.shape[1]
    if h != kv:
        K = K.repeat_interleave(h // kv, dim=1)
        V = V.repeat_interleave(h // kv, dim=1)
    scores = torch.einsum('nhd,mhd->hnm', q, K) * (1.0 / math.sqrt(d))
    allowed = torch.ones((n, K.shape[0]), dtype=torch.bool, device=q.device)
    if causal:
        allowed = allowed & (kpos[None, :] <= qpos[:, None])
    if span is not None:
        allowed = allowed & (kpos[None, :] > qpos[:, None] - span)
    scores = scores.masked_fill(~allowed[None, :, :], float('-inf'))
    p = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
    return torch.einsum('hnm,mhd->nhd', p, V).reshape(n, h * d)


def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    n = x.shape[0]
    h, d, kv = arguments['heads'], arguments['head_dim'], arguments['kv_heads']
    cross = bool(arguments.get('cross'))
    src = inputs['source_values'] if cross else x        # cross: the source elements this invocation delivers, none at a decode step
    m = src.shape[0]
    gate = None
    if arguments.get('output_gate'):
        qg = (x @ w(ctx, params['q_gated']).T).view(n, h, 2, d)   # per head: query rows, then gate rows
        q, gate = qg[:, :, 0, :], qg[:, :, 1, :]
    else:
        q = x @ w(ctx, params['q']).T
    k = src @ w(ctx, params['k']).T
    v = src @ w(ctx, params['v']).T
    if arguments.get('q_bias'):
        q = q + w(ctx, params['q_bias'])
    if arguments.get('k_bias'):
        k = k + w(ctx, params['k_bias'])
    if arguments.get('v_bias'):
        v = v + w(ctx, params['v_bias'])
    q, k, v = q.reshape(n, h, d), k.view(m, kv, d), v.view(m, kv, d)
    qk_norm = arguments.get('qk_norm')
    if qk_norm and qk_norm['kind'] == 'rms':
        eps = qk_norm['eps']
        scale = qk_norm.get('scale')                    # present: q_norm and k_norm are declared
        zc = bool(scale and scale.get('zero_centered'))
        one = torch.ones(d, device=x.device, dtype=x.dtype)
        qs = w(ctx, params['q_norm']) if scale is not None else one
        ks = w(ctx, params['k_norm']) if scale is not None else one
        q = rms_norm(q, qs, eps, zc)
        k = rms_norm(k, ks, eps, zc)
    rope = arguments.get('rope')
    if rope:
        if cross:
            raise ValueError("cross attention with rope: the source stream's positions are not delivered to this kernel")
        q = rope_split(q, ctx.positions, rope['theta'], rope.get('partial'), rope.get('scaling'))
        k = rope_split(k, ctx.positions, rope['theta'], rope.get('partial'), rope.get('scaling'))
    causal = arguments['mask'] == 'causal'
    window = arguments.get('window')
    if window is not None:
        if not causal:
            raise ValueError("attention.dense: a window with mask none is not implemented (the reading is causal: a query and the span − 1 before it)")
        st = states['kv']                                    # a ring of `span` positions: the last before this fragment
        prev, n_prev = st.tail()
        p0 = int(ctx.positions[0]) if n else 0
        kpos = torch.cat([torch.arange(p0 - n_prev, p0, device=q.device), ctx.positions])
        K = torch.cat([prev['k'][:n_prev].to(q.dtype), k], dim=0)
        V = torch.cat([prev['v'][:n_prev].to(q.dtype), v], dim=0)
        out = attend_positions(q, K, V, ctx.positions, kpos, causal, int(window['span']))
        st.append({'k': k, 'v': v})
    elif 'kv' in states:
        st = states['kv']
        if m:                                                # a cross cache is appended along the source stream, when it delivers
            st.append({'k': k, 'v': v})
        bufs, length = st.read()
        out = attend(q, bufs['k'].to(q.dtype), bufs['v'].to(q.dtype), length, ctx.positions, causal, static=ctx.static)
    else:
        out = attend(q, k, v, n, ctx.positions, causal)
    if gate is not None:
        out = out * torch.sigmoid(gate.reshape(n, h * d))
    y = out @ w(ctx, params['out']).T
    if arguments.get('out_bias'):
        y = y + w(ctx, params['out_bias'])
    return {'output': y}

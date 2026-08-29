"""attention.dense@1.0.0 — dense or grouped-query attention over a KV state.

| branch / record                 | status                                          |
|---------------------------------|-------------------------------------------------|
| mask causal                     | implemented                                     |
| mask none (stateless encoder)   | implemented                                     |
| mask chunked (`chunk`)          | refused                                         |
| cross (`source_values`)         | refused                                         |
| streaming                       | refused                                         |
| window                          | refused                                         |
| rope: theta, layout split       | implemented (rotate-half)                       |
| rope: layout interleaved / 2d   | refused                                         |
| rope: partial                   | implemented (the first `partial · head_dim` channels) |
| rope: mrope (sections contiguous or interleaved) | implemented for one position stream (t = h = w: the two layouts are one computation; an image would tell them apart) |
| rope: scaling                   | refused                                         |
| qk_norm rms (+ weight, zero-centred) | implemented, before RoPE                   |
| qk_norm layer                   | refused                                         |
| temperature                     | refused                                         |
| q/k/v/out biases                | implemented                                     |
| output_gate (`q_gated`)         | implemented: per head, query rows then gate rows |

Conventions the contract leaves open, as read here: keys of the current elements are
appended to the state before the queries attend (a query sees itself); the scale is
head_dim^-1/2; rope `split` pairs channel i with i + rotary/2 (rotate-half) over the rotated
channels only, whose base frequencies are computed on the rotated width; `qk_norm` is an RMS
norm over head_dim applied before RoPE, its scale zero-centred when the argument says so. These
readings are now stated by the contract (finding 1, 30 Aug 2026); `mrope.sections` is declared
by the document and, for a single position stream, both layouts are plain RoPE.
"""
import math
import torch
from kernels._common import present, refuse_unknown, w

CONTRACT = ("attention.dense", "1.0.0")
KNOWN = {'width', 'heads', 'head_dim', 'kv_heads', 'mask', 'window', 'chunk', 'cross', 'streaming', 'rope',
         'qk_norm', 'temperature', 'q_bias', 'k_bias', 'v_bias', 'out_bias', 'output_gate', 'qk_norm_weight',
         'qk_norm_zero_centered'}


def supports(arguments):
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    if arguments.get('mask') not in ('causal', 'none'):
        reasons.append(f"mask={arguments.get('mask')}")
    for flag in ('cross', 'streaming'):
        if arguments.get(flag):
            reasons.append(f"{flag}=true")
    for rec in ('window', 'chunk', 'temperature'):
        if present(arguments, rec):
            reasons.append(f"{rec}={arguments[rec]}")
    if present(arguments, 'qk_norm') and arguments['qk_norm'] != 'rms':
        reasons.append(f"qk_norm={arguments['qk_norm']}")
    rope = arguments.get('rope')
    if rope:
        if rope.get('layout', 'split') != 'split':
            reasons.append(f"rope.layout={rope.get('layout')}")
        if rope.get('scaling') is not None:
            reasons.append(f"rope.scaling={rope['scaling']}")
    return reasons


def rope_split(x, positions, theta, partial=None):
    """Rotate-half RoPE over the first `partial · d` channels of each head (all of them when
    `partial` is absent): x [n, h, d], positions [n]. The base frequencies are computed on the
    rotated width, as the reference does."""
    d = x.shape[-1]
    r = d if not partial else int(d * partial)
    inv = 1.0 / (theta ** (torch.arange(0, r, 2, device=x.device, dtype=torch.float32) / r))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().to(x.dtype)[:, None, :], emb.sin().to(x.dtype)[:, None, :]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
    xr = xr * cos + torch.cat([-x2, x1], dim=-1) * sin
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


def run(ctx, arguments, inputs, params, states):
    x = inputs['input']
    n = x.shape[0]
    h, d, kv = arguments['heads'], arguments['head_dim'], arguments['kv_heads']
    gate = None
    if arguments.get('output_gate'):
        qg = (x @ w(ctx, params['q_gated']).T).view(n, h, 2, d)   # per head: query rows, then gate rows
        q, gate = qg[:, :, 0, :], qg[:, :, 1, :]
    else:
        q = x @ w(ctx, params['q']).T
    k = x @ w(ctx, params['k']).T
    v = x @ w(ctx, params['v']).T
    if arguments.get('q_bias'):
        q = q + w(ctx, params['q_bias'])
    if arguments.get('k_bias'):
        k = k + w(ctx, params['k_bias'])
    if arguments.get('v_bias'):
        v = v + w(ctx, params['v_bias'])
    q, k, v = q.reshape(n, h, d), k.view(n, kv, d), v.view(n, kv, d)
    if arguments.get('qk_norm') == 'rms':
        eps = ctx.eps
        zc = bool(arguments.get('qk_norm_zero_centered'))
        one = torch.ones(d, device=x.device, dtype=x.dtype)
        qs = w(ctx, params['q_norm']) if arguments.get('qk_norm_weight') else one
        ks = w(ctx, params['k_norm']) if arguments.get('qk_norm_weight') else one
        q = rms_norm(q, qs, eps, zc and arguments.get('qk_norm_weight'))
        k = rms_norm(k, ks, eps, zc and arguments.get('qk_norm_weight'))
    rope = arguments.get('rope')
    if rope:
        q = rope_split(q, ctx.positions, rope['theta'], rope.get('partial'))
        k = rope_split(k, ctx.positions, rope['theta'], rope.get('partial'))
    causal = arguments['mask'] == 'causal'
    if 'kv' in states:
        st = states['kv']
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

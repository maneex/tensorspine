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
| rope: partial, mrope, scaling   | refused (M2)                                    |
| qk_norm, qk_norm_weight         | refused (M2)                                    |
| temperature                     | refused                                         |
| q/k/v/out biases                | implemented                                     |
| output_gate (`q_gated`)         | refused (M2)                                    |

Conventions the contract leaves open, as read here: keys of the current elements are
appended to the state before the queries attend (a query sees itself); the scale is
head_dim^-1/2; rope `split` pairs channel i with i + head_dim/2 (rotate-half).
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
    for flag in ('cross', 'streaming', 'output_gate'):
        if arguments.get(flag):
            reasons.append(f"{flag}=true")
    for rec in ('window', 'chunk', 'temperature', 'qk_norm'):
        if present(arguments, rec):
            reasons.append(f"{rec}={arguments[rec]}")
    rope = arguments.get('rope')
    if rope:
        if rope.get('layout', 'split') != 'split':
            reasons.append(f"rope.layout={rope.get('layout')}")
        for f in ('partial', 'mrope', 'scaling'):
            if rope.get(f) is not None:
                reasons.append(f"rope.{f}={rope[f]}")
    return reasons


def rope_split(x, positions, theta):
    """Rotate-half RoPE over the whole head: x [n, h, d], positions [n]."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().to(x.dtype)[:, None, :], emb.sin().to(x.dtype)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


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
    q = x @ w(ctx, params['q']).T
    k = x @ w(ctx, params['k']).T
    v = x @ w(ctx, params['v']).T
    if arguments.get('q_bias'):
        q = q + w(ctx, params['q_bias'])
    if arguments.get('k_bias'):
        k = k + w(ctx, params['k_bias'])
    if arguments.get('v_bias'):
        v = v + w(ctx, params['v_bias'])
    q, k, v = q.view(n, h, d), k.view(n, kv, d), v.view(n, kv, d)
    rope = arguments.get('rope')
    if rope:
        q = rope_split(q, ctx.positions, rope['theta'])
        k = rope_split(k, ctx.positions, rope['theta'])
    causal = arguments['mask'] == 'causal'
    if 'kv' in states:
        st = states['kv']
        st.append({'k': k, 'v': v})
        bufs, length = st.read()
        out = attend(q, bufs['k'].to(q.dtype), bufs['v'].to(q.dtype), length, ctx.positions, causal, static=ctx.static)
    else:
        out = attend(q, k, v, n, ctx.positions, causal)
    y = out @ w(ctx, params['out']).T
    if arguments.get('out_bias'):
        y = y + w(ctx, params['out_bias'])
    return {'output': y}

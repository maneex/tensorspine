"""sequence.gated_delta@1.0.0 — gated DeltaNet: a per-head matrix state updated by the
delta rule behind a short causal convolution (Qwen 3.5 / Qwen3-Next).

| branch / record            | status                                         |
|----------------------------|------------------------------------------------|
| conv (kernel, history)     | implemented, depthwise causal, `window` state  |
| no conv                    | refused                                        |
| out_gate silu              | implemented                                    |
| out_gate none/sigmoid/swish| refused                                        |
| value_heads > key_heads    | implemented (q, k repeated per value head)     |

Conventions the contract leaves open, as read from the reference implementation: q and k are
L2-normalised (eps 1e-6) inside the rule and q is scaled by head_dim^-1/2; β = σ(b);
g = −exp(A_log) · softplus(a + dt_bias); the state decays by exp(g) then takes the delta
update k ⊗ ((v − Sᵀk) β); the read-out is normalised per head by an RMS norm whose scale is
`norm` (not zero-centred) and gated by silu(z); the convolution reads the `history` previous
inputs then appends the current ones. The recurrent form, position by position, is the
reference; it is exact and slow.
"""
import math
import torch
import torch.nn.functional as F
from kernels._common import present, refuse_unknown, supports_from, w

CONTRACT = ("sequence.gated_delta", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "key_heads": "any", "value_heads": "any", "head_dim": "any",
                              "out_gate": ["silu"],
                              "conv": {"absent": False, "fields": {"width": "any", "kernel": "any", "history": "any"}}},
                "states": ["window", "fixed"],
                "notes": ["the recurrent form, position by position: exact and slow"]}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    n = x.shape[0]
    kh, vh, d = arguments['key_heads'], arguments['value_heads'], arguments['head_dim']
    conv = arguments['conv']
    kernel, history = conv['kernel'], conv['history']
    mixed = x @ w(ctx, params['qkv']).T                     # [n, 2·kh·d + vh·d]
    z = x @ w(ctx, params['z']).T                            # [n, vh·d]
    b = x @ w(ctx, params['b']).T                            # [n, vh]
    a = x @ w(ctx, params['a']).T                            # [n, vh]
    # causal depthwise convolution over the last `history` inputs and the current ones
    st = states['conv']
    hist, length = st.read()                                 # [history, C], zero-padded
    st.append({'w': mixed})
    seq = torch.cat([hist['w'].to(mixed.dtype), mixed], dim=0)        # [history + n, C]
    weight = w(ctx, params['conv_weight'])                   # [C, kernel]
    assert weight.shape[1] == kernel and history == kernel - 1
    windows = seq.unfold(0, kernel, 1)                       # [n, C, kernel]
    mixed = F.silu((windows * weight[None]).sum(-1))         # [n, C]
    q, k, v = torch.split(mixed, [kh * d, kh * d, vh * d], dim=-1)
    q, k, v = q.view(n, kh, d), k.view(n, kh, d), v.view(n, vh, d)
    if vh > kh:
        q = q.repeat_interleave(vh // kh, dim=1)
        k = k.repeat_interleave(vh // kh, dim=1)
    beta = torch.sigmoid(b)                                  # [n, vh]
    g = -torch.exp(w(ctx, params['A_log']).to(torch.float32)) * F.softplus(a.to(torch.float32) + w(ctx, params['dt_bias']).to(torch.float32))
    q = l2norm(q) * (1.0 / math.sqrt(d))
    k = l2norm(k)
    rec = states['recurrent']
    S, _ = rec.read()
    S = S['s'].to(torch.float32)                             # [vh, d, d]: rows k, columns v
    out = torch.empty(n, vh, d, dtype=torch.float32, device=x.device)
    qf, kf, vf, bf = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), beta.to(torch.float32)
    for i in range(n):
        S = S * torch.exp(g[i])[:, None, None]
        kv_mem = (S * kf[i][:, :, None]).sum(-2)              # [vh, d]
        delta = (vf[i] - kv_mem) * bf[i][:, None]
        S = S + kf[i][:, :, None] * delta[:, None, :]
        out[i] = (S * qf[i][:, :, None]).sum(-2)
    rec.write({'s': S})
    # normalise the read-out per head, gate by silu(z), project back
    o = out.to(x.dtype)
    var = o.pow(2).mean(-1, keepdim=True)
    o = o * torch.rsqrt(var + 1e-6) * w(ctx, params['norm'])
    o = o * F.silu(z.view(n, vh, d))
    y = o.reshape(n, vh * d) @ w(ctx, params['out']).T
    return {'output': y}

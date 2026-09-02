"""moe@1.0.0 — mixture of experts: `router` scores every token against `experts` experts, the
`top_k` best are activated, each a gated FFN read from the fused `in` (gate rows first, then up
rows, as the reference stores routed experts) and `out`; `shared` experts are applied to every
token, their sum optionally weighted through a sigmoid of `shared_output_gate`.

| branch / record                   | status                                                    |
|-----------------------------------|-----------------------------------------------------------|
| routing learned, scoring softmax  | implemented: softmax over all experts in fp32, then top-k |
| norm_topk                         | implemented both ways (the selected scores renormalised to one) |
| shared, shared_inner, shared_output_gate | implemented (`shared` ≥ 0; the gate is one sigmoid per token) |
| scale                             | implemented (absent: 1)                                   |
| routing hash, scoring sigmoid / sqrtsoftplus, score_bias, swiglu_limit | refused until a document uses them |

Conventions the contract leaves open, as read here: an expert's activation is SiLU (a "gated FFN"
as Qwen's); the routing weight multiplies the expert's output before the sum over the activated
experts; the experts are evaluated one activated expert at a time with index gathers — the
reference measures agreement, not speed.
"""
import torch
import torch.nn.functional as F
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("moe", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "experts": "any", "top_k": "any", "inner": "any", "shared": "any",
                              "shared_inner": "any", "norm_topk": "any", "routing": ["learned"], "scale": "any",
                              "scoring": ["softmax"], "swiglu_limit": "absent", "shared_output_gate": "any",
                              "score_bias": [False], "hash_vocabulary": "absent"},
                "states": [],
                "notes": ["an expert's activation is SiLU; one activated expert at a time"]}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities.
FIXTURES = [
    {"case": "softmax-topk-shared", "seed": 51, "invocations": [{"input": 6}, {"input": 2}],
     "arguments": {"width": 64, "experts": 8, "top_k": 2, "inner": 32, "shared": 1, "shared_inner": 32,
                   "norm_topk": True, "shared_output_gate": True}},
    {"case": "plain-scaled", "seed": 52, "invocations": [{"input": 6}],
     "arguments": {"width": 64, "experts": 4, "top_k": 1, "inner": 32, "scale": 1.5}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def _per_copy(t):
    """A multiplicity slot as [copies, …]: one copy stays a plain tensor."""
    return t if t.dim() == 3 else t[None]

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    n = x.shape[0]
    k = arguments['top_k']
    logits = (x.to(torch.float32) @ params['router'].to(torch.float32).T)
    probs = torch.softmax(logits, dim=-1)                                  # [n, experts], fp32
    top, idx = torch.topk(probs, k, dim=-1)                                # [n, top_k]
    if arguments.get('norm_topk'):
        top = top / top.sum(dim=-1, keepdim=True)
    top = top.to(ctx.dtype)
    out = torch.zeros(n, x.shape[-1], dtype=ctx.dtype, device=x.device)
    fused, down = params['in'], params['out']                              # [experts, 2·inner, width], [experts, width, inner]
    for e in torch.unique(idx).tolist():
        rows, slots = (idx == e).nonzero(as_tuple=True)                    # the tokens routed to e, and where in their top-k
        h = x[rows] @ fused[e].to(ctx.dtype).T
        gate, up = h.chunk(2, dim=-1)
        y = (F.silu(gate) * up) @ down[e].to(ctx.dtype).T
        out.index_add_(0, rows, y * top[rows, slots][:, None])
    scale = arguments.get('scale')
    if scale is not None:
        out = out * scale
    if arguments.get('shared'):
        G, U, O = (_per_copy(params[s]) for s in ('shared_gate', 'shared_up', 'shared_out'))
        shared = torch.zeros_like(out)
        for s in range(arguments['shared']):
            a = F.silu(x @ G[s].to(ctx.dtype).T) * (x @ U[s].to(ctx.dtype).T)
            shared = shared + a @ O[s].to(ctx.dtype).T
        if arguments.get('shared_output_gate'):
            shared = shared * torch.sigmoid(x @ params['shared_output_gate'].to(ctx.dtype))[:, None]   # one gate per token
        out = out + shared
    return {'output': out}

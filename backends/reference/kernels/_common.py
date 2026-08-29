"""Helpers shared by the kernels: reading resolved D1 arguments (defaults applied),
building refusals, upcasting parameters to the compute dtype."""


def present(arguments, name):
    return arguments.get(name) is not None


def refuse_unknown(arguments, known, reasons):
    for k in arguments:
        if k not in known:
            reasons.append(f"{k}={arguments[k]!r} (argument unknown to this kernel)")


def w(ctx, t):
    """A parameter in the compute dtype (upcast per operation; storage stays at the D3 dtype)."""
    return t.to(ctx.dtype)


UPCAST_CHUNK_BYTES = 256 * 2**20     # the largest fp32 temporary a chunked projection makes


def chunked_matmul(ctx, x, weight):
    """x @ weightᵀ with `weight` upcast to the compute dtype in row chunks bounded by
    UPCAST_CHUNK_BYTES — the same products, a bounded temporary. For a projection whose
    weight is the model's largest tensor (an output head), this is what keeps the resident
    set small."""
    import torch
    width = torch.tensor([], dtype=ctx.dtype).element_size()
    rows = max(1, UPCAST_CHUNK_BYTES // (weight.shape[1] * width))
    if rows >= weight.shape[0]:
        return x @ weight.to(ctx.dtype).T
    return torch.cat([x @ weight[i:i + rows].to(ctx.dtype).T for i in range(0, weight.shape[0], rows)], dim=-1)

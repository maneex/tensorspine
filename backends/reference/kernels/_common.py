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

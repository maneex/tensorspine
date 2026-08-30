"""embedding.token_position_type@1.0.0 — BERT's embedding: the token's row of `weight`, the
position's row of `position` (the stream position, 0 upwards) and the segment type's row of
`token_type` summed, then a LayerNorm with `norm`, `norm_bias` and `eps`.

| branch / record             | status                                                        |
|-----------------------------|---------------------------------------------------------------|
| token + position + type sum | implemented                                                   |
| segment type                | 0 for every token, as the contract states for this version    |
| LayerNorm (eps)             | implemented                                                   |

A position at or beyond `positions` has no row and is refused at run time.
"""
import torch.nn.functional as F
from kernels._common import refuse_unknown, supports_from, w

CONTRACT = ("embedding.token_position_type", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any", "positions": "any", "token_types": "any", "eps": "any"},
                "states": [],
                "notes": ["the segment type is 0 for every token, as the contract states for this version"]}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    ids = inputs['tokens']
    pos = ctx.positions
    if int(pos.max()) >= arguments['positions']:
        raise ValueError(f"position {int(pos.max())} has no row: the table holds {arguments['positions']} positions")
    x = params['weight'][ids].to(ctx.dtype) + params['position'][pos].to(ctx.dtype) + params['token_type'][0].to(ctx.dtype)
    return {'output': F.layer_norm(x, (x.shape[-1],), w(ctx, params['norm']), w(ctx, params['norm_bias']), arguments['eps'])}

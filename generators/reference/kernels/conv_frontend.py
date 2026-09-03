"""conv_frontend@1.0.0 — the audio stem: `frames` [n, mels] through the first convolution
(`conv1_weight` [width, mels, kernel], the file's own layout, stride one), an activation, the
second (`conv2_weight` [width, width, kernel]) with stride `stride`, an activation, then the row
of the learned position table for each output position. n frames make n / stride positions of
the same stream (the contract's `merge`, §5.3).

| branch / record                       | status                                                    |
|---------------------------------------|-----------------------------------------------------------|
| symmetric padding (`causal` false)    | implemented: (kernel − 1) / 2 frames each side, both convolutions |
| bias                                  | implemented, both ways                                    |
| position (learned table)              | implemented; absent, nothing is added (rotary positions belong to the next occurrence) |
| causal (left padding, Voxtral)        | implemented: `kernel − stride` zero frames before the first of each convolution (`kernel − 1` for the first, whose stride is one) |
| streaming (conv1_history, conv2_history) | implemented: the left padding of each convolution is its history ring — the frames of the previous fragments it keeps, zeros before the first write — and the fragment's frames (the first convolution's outputs, for the second) are appended after; the fragments compute what the whole signal would |

Conventions the contract leaves to the witness, as read here: the activation after each
convolution is the erf GELU (Whisper's stem, Voxtral Realtime's); an even kernel has no symmetric
padding and is refused. The positions the kernel receives are the frames' (the node's own domain,
count 1); the output positions are theirs divided by `stride` — integers, since a delivery is
aligned to the stride (§5.3) — and the table's row for each is added, whole window or not, which
is where a delivery shorter than the table differs from a delivery implementation that adds the
whole table. Under `streaming`, a ring holds exactly the frames its convolution's next output
needs (`transformers`' `padding_cache`: the last `kernel − stride` inputs of each convolution).
"""
import torch
import torch.nn.functional as F
from kernels._common import present, supports_from, w

CONTRACT = ("conv_frontend", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "mels": "any", "stride": "any", "kernel": "any", "bias": "any", "position": "any",
                              "causal": "any", "streaming": "any"},
                "states": ["window"],
                "transforms": ["merge"],
                "notes": ["the activation after each convolution is the erf GELU, the convention the contract leaves to the witness",
                          "an even kernel has no symmetric padding and is refused at run time"]}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities. Twelve frames make six positions.
FIXTURES = [
    {"case": "basic", "seed": 91, "invocations": [{"frames": 12}],
     "arguments": {"width": 16, "mels": 8, "stride": 2, "kernel": 3, "position": 8}},
    # two fragments of eight frames: the second's first outputs read the histories the first left
    {"case": "causal-streaming", "seed": 92, "invocations": [{"frames": 8}, {"frames": 8}],
     "arguments": {"width": 16, "mels": 8, "stride": 2, "kernel": 3, "causal": True, "streaming": True}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['frames']                                   # [n, mels]
    kernel, stride = arguments['kernel'], arguments['stride']
    bias = bool(arguments.get('bias'))
    w1, b1 = w(ctx, params['conv1_weight']), (w(ctx, params['conv1_bias']) if bias else None)
    w2, b2 = w(ctx, params['conv2_weight']), (w(ctx, params['conv2_bias']) if bias else None)
    if not arguments.get('causal'):
        if (kernel - 1) % 2:
            raise ValueError(f"kernel {kernel}: an even kernel has no symmetric padding of (kernel − 1) / 2 frames each side")
        pad = (kernel - 1) // 2
        h = x.T[None]                                      # [1, mels, n]: torch's channels-first convolution
        h = F.gelu(F.conv1d(h, w1, b1, padding=pad))
        h = F.gelu(F.conv1d(h, w2, b2, stride=stride, padding=pad))
        y = h[0].T                                         # [n / stride, width]
    else:
        # left padding alone: the history ring under streaming (zeros before the first write, then the
        # last kernel − stride inputs of the convolution), else kernel − stride zero frames
        def left(history, rows, width_in):
            if history is not None:
                bufs, _n = history.read()                  # chronological, zero-padded to the span
                return bufs['w'].to(ctx.dtype)
            return torch.zeros((rows, width_in), device=x.device, dtype=x.dtype)
        h1_in = torch.cat([left(states.get('conv1_history'), kernel - 1, x.shape[1]), x], dim=0)
        h1 = F.gelu(F.conv1d(h1_in.T[None], w1, b1))[0].T                           # [n, width]
        h2_in = torch.cat([left(states.get('conv2_history'), kernel - stride, h1.shape[1]), h1], dim=0)
        y = F.gelu(F.conv1d(h2_in.T[None], w2, b2, stride=stride))[0].T             # [n / stride, width]
        if 'conv1_history' in states:
            states['conv1_history'].append({'w': x})
        if 'conv2_history' in states:
            states['conv2_history'].append({'w': h1})
    if present(arguments, 'position'):
        pos = ctx.positions[::stride] // stride            # the output positions: the frames' divided by the stride
        if pos.numel() and int(pos.max()) >= arguments['position']:
            raise ValueError(f"position {int(pos.max())} has no row: the table holds {arguments['position']} positions")
        y = y + params['position'][pos].to(ctx.dtype)
    return {'output': y}

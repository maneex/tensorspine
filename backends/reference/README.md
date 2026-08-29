# Reference backend

A loader and a reference implementation of the primitives, in PyTorch, that turn a model document,
its derived products and the checkpoint it names into a module with one `forward`, a comparison
against the official implementation at every layer boundary, and a small chat. Not a serving
system. The plan is [`docs/reference-backend-plan.md`](../../docs/reference-backend-plan.md); the
weight locations it loads by are [`docs/location-plan.md`](../../docs/location-plan.md).

**Status: M0.** Random weights, no checkpoint: the derived document builds a module, prefill and
decode run with every value checked against D2, the values at every layer cut are dumped. Six
kernels (`embed`, `norm.rms`, `attention.dense` causal/GQA/RoPE, `residual.add`, `ffn.gated`,
`lm_head`), one state law exercised (`append`; `window` and `fixed` implemented, unexercised).

```sh
python3 backends/reference/ref.py info data/models/llama3-8b.json --capacity 4096
python3 backends/reference/ref.py run  data/models/llama3-8b.json --random --truncate decoder.layer=3 \
        --ids 1,2,3,4 --steps 2 --dump /tmp/llama.3layers.safetensors
python3 backends/reference/ref.py run  data/models/llama3-8b.json --random --set quantities.d.source.value=64 …
python3 backends/reference/tests/run_reference.py [--compile]
```

What it reads: the derived document only (D1 graph and arguments, D2 shapes, D3 tensors, D4
states, D6 cuts) — derived in-process from a model document through `tools/derive.py` — never
the model source or the catalog. What it knows about a model: nothing; `--truncate` and `--set`
edit the *document* before deriving and are a test convenience.

Conventions: a value's tensor has the element axis first, then the port's axes in the contract's
order; one sequence, no batch axis; parameters stay at their D3 dtype and are upcast per
operation (`--compute f32` on CPU, `bf16` on CUDA); `append` states are buffers of `--capacity`
positions written at a cursor, and a kernel masks beyond the length. Each kernel's docstring lists
the contract's branches it implements or refuses, and the conventions the contract leaves open.

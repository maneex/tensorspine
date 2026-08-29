#!/usr/bin/env python3
"""The official implementation at the same boundaries (R07, R11): `transformers` runs the
checkpoint — optionally with `num_hidden_layers` overridden so the dump stays small — and
writes, under the keys the reference backend's `--dump` uses, the output of every decoder
layer, the KV cache after prefill, the last position's logits, the argmax per position and
the greedy tokens. The hook → D1-value map is data in the header: the only place HF names
meet D1 names.

    python3 fixtures/dump_hf.py --model DIR --layers 3 --ids 128000,791,… --steps 3 --out F
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare import write_dump   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--layers', type=int, help='num_hidden_layers override (the truncated fixture)')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--dtype', default='f32', choices=['f32', 'bf16'])
    ap.add_argument('--composition', default='decoder', help="the D1 composition the layers belong to")
    ap.add_argument('--layer-output', default='ffn_r', help="the D1 site whose output closes a layer")
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    from transformers import AutoConfig, AutoModelForCausalLM
    dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[args.dtype]
    config = AutoConfig.from_pretrained(args.model)
    kwargs = {}
    if args.layers:
        kwargs['num_hidden_layers'] = args.layers
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"loaded {args.model}: {n_layers} layers in {dtype} ({time.time() - t0:.0f}s)")
    ids = [int(x) for x in args.ids.split(',')]
    dump, hooks, hook_map = {}, [], {}
    layers = model.model.layers
    for i, layer in enumerate(layers):
        key = f"value/{args.composition}/{args.layer_output}[layer={i}].output"
        hook_map[f"model.layers.{i}"] = key

        def hook(module, inputs, output, key=key):
            out = output[0] if isinstance(output, tuple) else output
            if key not in dump:                     # prefill only
                dump[key] = out[0].detach().to(torch.float32).cpu()
        hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        x = torch.tensor([ids])
        t0 = time.time()
        out = model(input_ids=x, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0].to(torch.float32)
        dump['logits/last'] = logits[-1].cpu()
        dump['logits/argmax'] = logits.argmax(-1).cpu()
        for i in range(n_layers):
            try:
                k, v = cache.layers[i].keys, cache.layers[i].values
            except AttributeError:
                k, v = cache[i]
            dump[f"state/{args.composition}.attn.kv[layer={i}]/k"] = k[0].permute(1, 0, 2).to(torch.float32).cpu()
            dump[f"state/{args.composition}.attn.kv[layer={i}]/v"] = v[0].permute(1, 0, 2).to(torch.float32).cpu()
        hook_map['past_key_values.layers[i].keys[0].permute(1,0,2)'] = f"state/{args.composition}.attn.kv[layer=i]/k"
        nxt = int(logits[-1].argmax())
        tokens = [nxt]
        print(f"prefill {len(ids)} -> {nxt} ({time.time() - t0:.1f}s)")
        for _ in range(args.steps):
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
        print("tokens:", tokens)
    for h in hooks:
        h.remove()
    header = {'model': args.model, 'layers': n_layers, 'dtype': args.dtype, 'ids': ids, 'tokens': tokens,
              'hook_map': hook_map, 'torch': torch.__version__,
              'transformers': __import__('transformers').__version__,
              'index_sha256': _sha(os.path.join(args.model, 'model.safetensors.index.json'))}
    write_dump(args.out, dump, header)
    print(f"dumped {len(dump)} tensors -> {args.out}")


def _sha(path):
    import hashlib
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


if __name__ == '__main__':
    main()

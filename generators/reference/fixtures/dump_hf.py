#!/usr/bin/env python3
"""The official implementation at the same boundaries (R07, R11): `transformers` runs the
checkpoint — optionally with `num_hidden_layers` overridden so the dump stays small — and
writes, under the keys the reference generator's `--dump` uses, the output of every decoder
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
    ap.add_argument('--attn-site', default='attn', help="the D1 site of the attention (its kv state)")
    ap.add_argument('--gdn-site', default='gdn', help="the D1 site of the gated delta net (recurrent, conv states)")
    ap.add_argument('--conv-history', type=int, default=3, help="positions the D4 conv state keeps")
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[args.dtype]
    config = AutoConfig.from_pretrained(args.model)
    text = getattr(config, 'text_config', None) or config
    if args.layers:
        text.num_hidden_layers = args.layers
        if getattr(text, 'layer_types', None):
            text.layer_types = list(text.layer_types)[:args.layers]
    t0 = time.time()
    # the class the config maps to: causal-LM when transformers lists the type there, else the
    # image-text-to-text wrapper (a multimodal checkpoint run on text; its decoder is `language_model`)
    cls = AutoModelForCausalLM if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES else AutoModelForImageTextToText
    model = cls.from_pretrained(args.model, config=config, dtype=dtype)
    model.eval()
    n_layers = text.num_hidden_layers
    print(f"loaded {args.model}: {n_layers} layers in {dtype} ({time.time() - t0:.0f}s)")
    ids = [int(x) for x in args.ids.split(',')]
    dump, hooks, hook_map = {}, [], {}
    inner = model.model
    layers = getattr(inner, 'layers', None) or inner.language_model.layers
    for i, layer in enumerate(layers):
        key = f"value/{args.composition}/{args.layer_output}[layer={i}].output"
        hook_map[f"model.layers.{i}"] = key

        def hook(module, inputs, output, key=key):
            out = output[0] if isinstance(output, tuple) else output
            if key not in dump:                     # prefill only
                dump[key] = out[0].detach().to(torch.float32).cpu().clone()
        hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        x = torch.tensor([ids])
        t0 = time.time()
        out = model(input_ids=x, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0].to(torch.float32)
        dump['logits/last'] = logits[-1].cpu().clone()
        dump['logits/argmax'] = logits.argmax(-1).cpu().clone()
        for i in range(n_layers):
            layer = cache.layers[i]
            conv = getattr(layer, 'conv_states', None)
            if conv is not None and len(conv) and conv[0] is not None:
                c = conv[0][0] if conv[0].dim() == 3 else conv[0]          # [conv_dim, kernel]
                dump[f"state/{args.composition}.{args.gdn_site}.conv[layer={i}]/w"] = c[:, -args.conv_history:].T.to(torch.float32).cpu().clone()
                r = layer.recurrent_states[0]
                r = r[0] if r.dim() == 4 else r                              # [heads, k_dim, v_dim]
                dump[f"state/{args.composition}.{args.gdn_site}.recurrent[layer={i}]/s"] = r.to(torch.float32).cpu().clone()
                continue
            try:
                k, v = layer.keys, layer.values
            except AttributeError:
                k, v = cache[i]
            dump[f"state/{args.composition}.{args.attn_site}.kv[layer={i}]/k"] = k[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
            dump[f"state/{args.composition}.{args.attn_site}.kv[layer={i}]/v"] = v[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
        hook_map['past_key_values.layers[i].keys[0].permute(1,0,2)'] = f"state/{args.composition}.{args.attn_site}.kv[layer=i]/k"
        hook_map['past_key_values.layers[i].conv_states[0][0][:, -history:].T'] = f"state/{args.composition}.{args.gdn_site}.conv[layer=i]/w"
        hook_map['past_key_values.layers[i].recurrent_states[0][0]'] = f"state/{args.composition}.{args.gdn_site}.recurrent[layer=i]/s"
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
              **_provenance(args.model)}
    write_dump(args.out, dump, header)
    print(f"dumped {len(dump)} tensors -> {args.out}")


def _provenance(model_dir):
    """What identifies the weights: the index file's hash, or — for a checkpoint that is one
    file without an index — the hash of that file's safetensors header."""
    import hashlib
    import struct
    index = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(index):
        return {'index_sha256': hashlib.sha256(open(index, 'rb').read()).hexdigest()}
    single = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(single):
        with open(single, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            return {'header_sha256': hashlib.sha256(f.read(n)).hexdigest()}
    return {'index_sha256': None}


if __name__ == '__main__':
    main()

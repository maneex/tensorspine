#!/usr/bin/env python3
"""An integration fixture (docs/TENSORSPINE-FIXTURE.md): the delivery implementation at the
same boundaries as the reference generator's `--dump`. `transformers` runs the checkpoint —
optionally with `num_hidden_layers` overridden so the fixture stays small — and the file holds
the output of every decoder layer, the KV cache after prefill, the last position's logits, the
argmax per position and the greedy tokens, under the language's fixture schema: the corpus
document it is for, the artifact and its hash, the truncation, the library versions and the
tolerance a conformer must meet. The hook → D1-value map is data in the metadata: the only
place HF names meet D1 names.

    python3 fixtures/dump_hf.py --model DIR --document llama3-8b --layers 3 --ids 128000,791,… --steps 3 --out F
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare import write_fixture   # noqa: E402

PROGRAM = 'generators/reference/fixtures/dump_hf.py'
# What a conformer must meet against a fixture, per compute dtype: fp32 against an fp32 dump
# holds to a few 1e-6 in practice; bf16 was measured at 8.2e-2 absolute on the MoE fixture.
TOLERANCE = {'f32': {'atol': 1e-3, 'rtol': 1e-2}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='the checkpoint directory (the artifact)')
    ap.add_argument('--document', required=True, help='the corpus document this fixture is for, by name (llama3-8b)')
    ap.add_argument('--artifact-id', help='the published identifier of the artifact (NousResearch/Meta-Llama-3-8B)')
    ap.add_argument('--atol', type=float, help='an f32 conformer\'s absolute tolerance against this fixture (default: the f32 entry of TOLERANCE)')
    ap.add_argument('--rtol', type=float, help='its relative tolerance')
    ap.add_argument('--layers', type=int, help='num_hidden_layers override (the truncated fixture)')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--dtype', default='f32', choices=['f32', 'bf16'])
    ap.add_argument('--composition', default='decoder', help="the D1 composition the layers belong to")
    ap.add_argument('--layer-output', default='ffn_r', help="the D1 site whose output closes a layer")
    ap.add_argument('--attn-site', default='attn', help="the D1 site of the attention (its kv state)")
    ap.add_argument('--gdn-site', default='gdn', help="the D1 site of the gated delta net (recurrent, conv states)")
    ap.add_argument('--conv-history', type=int, default=3, help="positions the D4 conv state keeps")
    ap.add_argument('--encoder', action='store_true', help="a document without a generative output: the base model (AutoModel), "
                                                       "its encoder layers, one invocation, no cache, no tokens")
    ap.add_argument('--head', metavar='TENSOR:VALUE', help="with --encoder: a physical tensor applied to the final hidden state and "
                                                        "L2-normalised — a head transformers has no class for — recorded as the D1 value VALUE")
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
    if args.encoder:
        from transformers import AutoModel
        cls = AutoModel                                # the base model: the encoder and nothing on top
    else:
        cls = AutoModelForCausalLM if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES else AutoModelForImageTextToText
    model = cls.from_pretrained(args.model, config=config, dtype=dtype)
    model.eval()
    n_layers = text.num_hidden_layers
    print(f"loaded {args.model}: {n_layers} layers in {dtype} ({time.time() - t0:.0f}s)")
    ids = [int(x) for x in args.ids.split(',')]
    dump, hooks, hook_map = {}, [], {}
    inner = getattr(model, 'model', model)             # a base model is its own inner model
    layers = getattr(inner, 'layers', None) or getattr(getattr(inner, 'language_model', None), 'layers', None) \
        or inner.encoder.layer                          # BERT: encoder.layer
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
        if args.encoder:
            out = model(input_ids=x)
            tokens = []
            if args.head:
                tname, vname = args.head.split(':')
                W = _read_tensor(args.model, tname).to(torch.float32)
                h = out.last_hidden_state[0].to(torch.float32)
                dump[f"value/{vname}"] = torch.nn.functional.normalize(h @ W.T, dim=-1).cpu().clone()
                hook_map[f"normalize({tname} · last_hidden_state)"] = f"value/{vname}"
            print(f"encoded {len(ids)} ({time.time() - t0:.1f}s)")
            for h_ in hooks:
                h_.remove()
            write_fixture(args.out, dump, metadata(args, n_layers, ids, tokens, hook_map))
            print(f"dumped {len(dump)} tensors -> {args.out}")
            return
        # positions given explicitly, and — for a truncation without full-attention layers — the mask mapping
        # given empty: no layer consumes a mask, and HF's mask builder would ask an attention cache it does not have
        pos = torch.arange(len(ids))[None]
        types = list(getattr(text, 'layer_types', None) or [])
        no_mask = {t: None for t in set(types)} if types and 'full_attention' not in types else None
        out = model(input_ids=x, position_ids=pos, cache_position=pos[0], attention_mask=no_mask, use_cache=True)
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
            p = torch.tensor([len(ids) + len(tokens) - 1])
            out = model(input_ids=torch.tensor([[nxt]]), position_ids=p[None], cache_position=p, attention_mask=no_mask, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
        print("tokens:", tokens)
    for h in hooks:
        h.remove()
    write_fixture(args.out, dump, metadata(args, n_layers, ids, tokens, hook_map))
    print(f"dumped {len(dump)} tensors -> {args.out}")


def metadata(args, n_layers, ids, tokens, hook_map):
    """The fixture's metadata on the language's schema (docs/TENSORSPINE-FIXTURE.md)."""
    tolerance = {k: dict(v) for k, v in TOLERANCE.items()}
    if args.atol is not None or args.rtol is not None:
        tolerance['f32'] = {'atol': args.atol if args.atol is not None else TOLERANCE['f32']['atol'],
                            'rtol': args.rtol if args.rtol is not None else TOLERANCE['f32']['rtol']}
    if args.dtype == 'bf16':
        tolerance['f32'] = dict(TOLERANCE['bf16'])     # an fp32 conformer against a bf16 dump: the dump's rounding
    artifact = {'name': artifact_name(args.model), **_provenance(args.model)}
    if args.artifact_id:
        artifact['id'] = args.artifact_id
    return {'schema': 'tensorspine-fixture/1', 'kind': 'integration', 'document': args.document,
            'artifact': artifact,
            'delivery': {'implementation': 'transformers', 'program': PROGRAM,
                         'versions': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__}},
            'truncation': {'composition': args.composition, 'layers': n_layers},
            'ids': ids, 'tokens': tokens, 'hook_map': hook_map,
            'compute': args.dtype, 'tolerance': tolerance}


def _read_tensor(model_dir, name):
    """One physical tensor of the checkpoint, from the shard the index names or the single file."""
    from safetensors import safe_open
    index = os.path.join(model_dir, 'model.safetensors.index.json')
    file = json.load(open(index, encoding='utf-8'))['weight_map'][name] if os.path.exists(index) else 'model.safetensors'
    with safe_open(os.path.join(model_dir, file), framework='pt') as f:
        return f.get_tensor(name)



def artifact_name(path):
    """The artifact's directory name, never the path it was read from: a fixture is
    committed, and a committed file must not carry the layout of the machine that made
    it. The name is what `verified.py` and the harnesses key on anyway."""
    return os.path.basename(os.path.normpath(path))
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
    return {}


if __name__ == '__main__':
    main()

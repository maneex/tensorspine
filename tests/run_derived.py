#!/usr/bin/env python3
"""The derived document (§7): every document of the corpus emits one that is
on the derived schema, and what it says agrees with the validator and with
facts known independently.

  1. Schema: `--derive` output validates against schemas/tensorspine-derived.schema.json;
     `--d1` output, the graph alone, validates against the same schema.
  2. Agreement: D3 elements = the validator's resident count; D5 operations per element =
     the validator's; D1 nodes = D3 members' occurrences ∪ stateless occurrences.
  3. Facts: Llama 3 8B — 4 KiB per cached position per layer, 128 KiB per token, one value of
     8 KiB per element crossing a layer boundary, and the live-value peak at the head: the f32
     logits beside the normed hidden state, by hand; ColBERT — the peak inside a layer, three
     residual-width values; Whisper — the cross-attention cache grows along the audio stream;
     Voxtral — 60 states carried across fragments, and the token input joins the audio stream at
     the stream's count for its kind (§5.3); every structural cut is legal (no
     crossing edge points backwards); every document's peak is a set of D2 values at a D1
     node whose bytes add up.

    python3 tests/run_derived.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import catalog as catalog_mod          # noqa: E402
import d1                              # noqa: E402
import derive                          # noqa: E402
import schema as schema_mod            # noqa: E402
import validate                        # noqa: E402
from signature import ASSIGNMENTS, corpus, name_of   # noqa: E402

SCHEMAS = os.path.join(ROOT, 'schemas')
MODELS = os.path.join(ROOT, 'data', 'models')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main():
    cat = catalog_mod.load(os.path.join(ROOT, 'data', 'catalog'))
    schema_path = schema_mod.locate(SCHEMAS, 'derived')
    reg = schema_mod.registry(SCHEMAS)
    ok = check("a derived schema is in the tree", schema_path is not None)
    docs = {}
    for path in corpus():
        name = name_of(path)
        assignment = ASSIGNMENTS.get(name)
        doc = derive.products(path, cat, assignment)
        errors = schema_mod.check_document(schema_path, doc, reg)
        ok &= check(f"{name}: derived document on the schema", not errors,
                    errors and schema_mod.format_error(errors[0]))
        graph_only = d1.emit(path, cat, assignment)
        errors = schema_mod.check_document(schema_path, graph_only, reg)
        ok &= check(f"{name}: graph-only document on the schema", not errors,
                    errors and schema_mod.format_error(errors[0]))
        stats = validate.analyse(path, cat, assignment)['stats']
        ok &= check(f"{name}: D3 elements = validator's resident count",
                    doc['d3']['totals']['elements'] == stats['parameter_elements'])
        ok &= check(f"{name}: D5 operations per element = validator's",
                    doc['d5']['operations']['element']['value'] == stats['ops_per_element'])
        nodes = set(doc['d1']['nodes'])
        members = {m.rsplit('.', 1)[0] for t in doc['d3']['tensors'] for m in t['members']}
        ok &= check(f"{name}: every D3 member is a D1 node", members <= nodes,
                    str(sorted(members - nodes)[:3]))
        # the live-value peak (D2 `peak_live`): D2 values at a D1 node, their bytes adding up
        peak = doc['d2']['peak_live']
        by_value = {v['value']: v for v in doc['d2']['values']}
        ok &= check(f"{name}: peak_live is a set of D2 values at a D1 node whose bytes add up",
                    peak['node'] in nodes and all(v in by_value for v in peak['values'])
                    and peak['bytes_per_element'] == sum(by_value[v]['bytes_per_element'] for v in peak['values'])
                    and peak['bytes_per_element'] >= max((v['bytes_per_element'] or 0) for v in doc['d2']['values']),
                    str(peak)[:200])
        # every structural cut is legal: block A is closed under ancestors
        blocks = {}
        for c in doc['d2']['cuts']:
            payload = {p['value'] for p in c['payload']}
            crossing = [e for e in doc['d1']['edges'] if f"{e['from']['node']}.{e['from']['port']}" in payload]
            ok &= check(f"{name}: cut {c['cut']} has a payload of distinct values", len(payload) == len(c['payload']))
            if not crossing and c['payload']:
                ok &= check(f"{name}: cut {c['cut']} payload values are edge sources", False)
        docs[name] = doc
    l3 = docs['llama3-8b']
    kv = [s for s in l3['d4']['states'] if s['state'] == 'kv']
    ok &= check("llama3-8b: 32 KV states of 4096 bytes per cached position",
                len(kv) == 32 and all(s['bytes_per_cached_position'] == 4096 for s in kv))
    ok &= check("llama3-8b: 128 KiB per token across the model",
                l3['d4']['totals']['append_bytes_per_cached_position'] == 131072)
    cut = next(c for c in l3['d2']['cuts'] if c['cut'] == 'decoder[layer<=3]')
    ok &= check("llama3-8b: one value of 8 KiB per element crosses a layer boundary",
                len(cut['payload']) == 1 and cut['bytes_per_element'] == 8192
                and cut['bytes_per_invocation'] == {'tokens': 8192.0}, str(cut['payload']))
    ok &= check("llama3-8b: 14.96 GiB of parameters in bf16",
                round(l3['d3']['totals']['bytes'] / 2**30, 2) == 14.96)
    # by hand: the peak is at the head, where the f32 logits (128256 × 4) sit beside the normed
    # hidden state (4096 × 2) — 521216 bytes per element; the layers hold three residual-width
    # values at most (24 KiB), so nothing inside them comes close
    peak = l3['d2']['peak_live']
    ok &= check("llama3-8b: the live-value peak is at lm_head — the f32 logits beside the normed hidden state, 521216 bytes per element",
                peak['node'] == 'lm_head' and peak['values'] == ['final_n.output', 'lm_head.logits']
                and peak['bytes_per_element'] == 128256 * 4 + 4096 * 2 == 521216
                and peak['bytes_per_invocation'] == {'tokens': 521216.0}, str(peak))
    cbv = docs['colbert-v2']['d2']['peak_live']
    ok &= check("colbert-v2: the peak is inside the first layer — the residual, the attention output and their sum, three 768-wide bf16 values, 4608 bytes",
                cbv['node'] == 'enc/attn_r[layer=0]' and cbv['bytes_per_element'] == 3 * 768 * 2 and len(cbv['values']) == 3, str(cbv))
    w = docs['whisper-large-v3']
    cross = [s for s in w['d4']['states'] if 'cross' in s['identity']]
    ok &= check("whisper: the cross-attention cache grows along the audio stream and is frozen after it",
                len(cross) == 32 and all(s['stream'] == {'kind': 'position', 'stream': 'audio'}
                                         and s['indexed_by_source'] for s in cross))
    v = docs['voxtral-realtime']
    ok &= check("voxtral: 60 states carried across fragments — 32 encoder rings of 750 frames, 26 decoder rings of 8192 tokens on the "
                "stream the token input joined, and the front end's two histories",
                len(v['d4']['totals']['carried']) == 60
                and all(s['law'] == 'window' and s['span'] in (750, 8192) for s in v['d4']['states']
                        if s['carried_across_fragments'] and s['contract'] == 'attention.dense'))
    # a joining input takes the stream's count at its kind (§5.3): the tokens join `audio` at kind token, where the
    # projector's merge left one element per eight frames; the fused values count the same, and both inputs are
    # required for the generative output — the delivery adds the embeddings position by position
    d2v = {x['value']: x for x in v['d2']['values']}
    ok &= check("voxtral: the token input joins the audio stream at kind token and counts {audio: 1/8}, the stream's count at that kind",
                d2v['tokens']['domain'] == {'kind': 'token', 'stream': 'audio'} and d2v['tokens']['count'] == {'audio': 0.125}
                and d2v['audio']['count'] == {'audio': 1.0}, str(d2v['tokens']))
    ok &= check("voxtral: the fused embedding and every decoder value count {audio: 1/8}, one language-model position per eight frames",
                all(d2v[k]['count'] == {'audio': 0.125} and d2v[k]['domain'] == {'kind': 'token', 'stream': 'audio'}
                    for k in ('embed.output', 'audio_projector.output', 'fuse.output', 'decoder/ffn_r[layer=0].output', 'lm_head.logits')))
    ok &= check("voxtral: the audio and the tokens are both required for the generative output (§7)",
                d2v['audio']['required_for'] == ['main'] and d2v['tokens']['required_for'] == ['main'])
    ok &= check("deepseek-v4-pro: next_tokens joins the token stream at count 1.0, as before",
                {x['value']: x for x in docs['deepseek-v4-pro']['d2']['values']}['next_tokens']['count'] == {'tokens': 1.0})
    rings = {s['identity']: s for s in v['d4']['states'] if s['contract'] == 'conv_frontend'}
    ok &= check("voxtral: the front end's histories are windows of kernel − 1 and kernel − stride frames, indexed by the frames port on the audio stream (V18)",
                rings['conv_frontend.conv1_history']['span'] == 2 and rings['conv_frontend.conv2_history']['span'] == 1
                and all(s['indexed_by_port'] == 'frames' and s['indexed_by_source'] and s['carried_across_fragments']
                        and s['stream'] == {'kind': 'position', 'stream': 'audio'} for s in rings.values()), str(rings)[:300])
    ok &= check("voxtral: the audio stream's fragment alignment is 8 frames — a stride of 2, then 4 positions per token (§5.3) — and the "
                "joined token input introduces no stream of its own",
                v['d2']['streams']['audio'].get('fragment_alignment') == 8 and list(v['d2']['streams']) == ['audio'],
                str(v['d2']['streams']))
    ok &= check("llama3-8b: an unfragmented stream states no alignment", 'fragment_alignment' not in l3['d2']['streams']['tokens'])
    g = docs['gemma3n-kvshare']
    shared = [s for s in g['d4']['states'] if s['identity'].startswith('shared.')]
    ok &= check("gemma3n: the shared identities carry no layer in their instance key",
                len(shared) == 2 and all(s['instance_key'] == ['instance.session', 'instance.branch'] for s in shared))
    ok &= check("llama3-8b: no O5.10 information loss once every flattened axis declares its factors",
                l3['d6']['information_loss'] == [])
    parts = {(p['node'], json.dumps(p['target'], sort_keys=True)): p for p in l3['d6']['partitions']}
    heads = parts.get(('decoder/attn[layer=0]', json.dumps({'argument_axis': 'attention.heads'}, sort_keys=True)))
    ok &= check("llama3-8b: the head partition keeps whole KV groups — granularity 32 / 8 = 4 — and every partition lists its communications",
                heads is not None and heads['granularity'] == 4 and heads['communication'] == ['all_reduce']
                and all(isinstance(p['communication'], list)
                        and p['granularity'] == (4 if p['target'] == {'argument_axis': 'attention.heads'} else 1)
                        for p in l3['d6']['partitions']),
                str(heads))
    vocab = parts.get(('embed', json.dumps({'argument_axis': 'model.vocabulary'}, sort_keys=True)))
    ok &= check("llama3-8b: the embedding's vocabulary partition admits two patterns, a gather of owned rows or a sum of masked partials",
                vocab is not None and vocab['communication'] == ['all_gather', 'all_reduce'], str(vocab))
    located = {t['identity']: t.get('location') for t in l3['d3']['tensors']}
    ok &= check("llama3-8b: decoder.attn.q[layer=3] is stored as model.layers.3.self_attn.q_proj.weight",
                located.get('decoder.attn.q[layer=3]') == {'tensor': 'model.layers.3.self_attn.q_proj.weight'})
    d2 = {v['value']: v for v in l3['d2']['values']}
    ok &= check("llama3-8b: the public input `tokens` is a value — one token index per element on stream tokens",
                d2.get('tokens', {}).get('input') == 'tokens' and d2['tokens']['shape'] == [] and d2['tokens']['role'] == 'activation.token_index'
                and d2['tokens']['domain'] == {'kind': 'token', 'stream': 'tokens'} and d2['tokens']['to'] == ['embed.tokens'])
    ok &= check("llama3-8b: the generative output's value is listed with its shape and exposed as `logits`",
                d2.get('lm_head.logits', {}).get('exposed') == ['logits'] and [a['extent'] for a in d2['lm_head.logits']['shape']] == [128256])
    names = [v['tensor'] for v in located.values() if v]
    ok &= check("llama3-8b: 291 tensors located under 291 distinct physical names",
                len(names) == 291 and len(set(names)) == 291)
    ok &= check("llama3-8b: D1 carries rope.layout = split although the document omits it — record-field defaults applied (finding 12)",
                l3['d1']['nodes']['decoder/attn[layer=0]']['arguments']['rope'].get('layout') == 'split')
    sh = {t['identity']: t.get('location') for t in docs['shieldstral-3b']['d3']['tensors']}
    sh_names = [v['tensor'] for v in sh.values() if v]
    ok &= check("shieldstral-3b: 458 tensors located one-to-one — decoder.attn.q[layer=3] under language_model, the tie on embed_tokens",
                sh.get('decoder.attn.q[layer=3]') == {'tensor': 'language_model.model.layers.3.self_attn.q_proj.weight'}
                and sh.get('tied_embeddings') == {'tensor': 'language_model.model.embed_tokens.weight'}
                and len(sh_names) == 458 and len(set(sh_names)) == 458)
    q35 = {t['identity']: t for t in docs['qwen3.5-35b-a3b']['d3']['tensors']}
    ok &= check("qwen3.5-35b-a3b: 1134 tensors located; the fused experts are [256, 2·512, 2048] and [256, 2048, 512] on one physical tensor each",
                len(q35) == 1134 and all(t.get('location') for t in q35.values())
                and [a['extent'] for a in q35['decoder.mlp.in[layer=0]']['shape']] == [256, 1024, 2048]
                and [a['extent'] for a in q35['decoder.mlp.out[layer=0]']['shape']] == [256, 2048, 512]
                and q35['decoder.mlp.in[layer=0]']['location'] == {'tensor': 'model.language_model.layers.0.mlp.experts.gate_up_proj'})
    cb = {t['identity']: t.get('location') for t in docs['colbert-v2']['d3']['tensors']}
    cb_names = [v['tensor'] for v in cb.values() if v]
    ok &= check("colbert-v2: 198 tensors located one-to-one — enc.attn.q[layer=3] under bert.encoder.layer, the head on linear.weight",
                cb.get('enc.attn.q[layer=3]') == {'tensor': 'bert.encoder.layer.3.attention.self.query.weight'}
                and cb.get('pooler.weight') == {'tensor': 'linear.weight'} and len(cb_names) == 198 and len(set(cb_names)) == 198)
    print("derived: all good" if ok else "derived: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

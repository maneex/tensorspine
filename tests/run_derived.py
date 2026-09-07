#!/usr/bin/env python3
"""The derived document (§7): every document of the corpus emits one that is
on the derived schema, and what it says agrees with the validator and with
facts known independently.

  1. Schema: `--derive` output validates against schemas/tensorspine-derived.schema.json;
     `--d1` output, the graph alone, validates against the same schema.
  2. Agreement: D3 elements = the validator's resident count; D5 operations per element =
     the validator's; D1 nodes = D3 members' instances ∪ stateless instances.
  3. Facts: Llama 3 8B — 4 KiB per cached position per layer, 128 KiB per token, one value of
     8 KiB per element crossing a layer boundary, and the live-value peak at the head: the f32
     logits beside the normed hidden state, by hand; ColBERT — the peak inside a layer, three
     residual-width values; Whisper — the cross-attention cache grows along the audio stream;
     Voxtral — 60 states carried across fragments, and the token input joins the audio stream at
     the stream's count for its kind (§5.3); every structural graph_split is valid (no
     crossing edge points backwards); every document's peak is a set of D2 values at a D1
     node whose bytes add up.
  4. Across positions (§4.1, O9.5): every D1 node carries `across_positions`, the primitive's
     condition on the node's own arguments; a node reading across positions of a fragmented
     stream owns a state carried across its fragments (V18 as a property of the products); the
     counts on Llama, ColBERT, Whisper, Voxtral, Qwen 3.5 4B, DeepSeek and the template; a
     condition over an argument the document leaves unresolved refuses the derivation.
  5. The grammar before the meaning (the review's I2): a document off the schema — a foreign
     schema tag, a misspelt instance field — has no products; `derive.products` refuses it with
     `--validate`'s own first line, and `--lint` reports it unanalysed.

    python3 tests/run_derived.py
"""
import glob
import json
import math
import os
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import primitive_library as primitive_library_mod          # noqa: E402
import d1                              # noqa: E402
import derive                          # noqa: E402
import schema as schema_mod            # noqa: E402
import validate                        # noqa: E402
from expr import primitive_condition    # noqa: E402
from signature import ASSIGNMENTS, corpus, name_of   # noqa: E402

SCHEMAS = os.path.join(ROOT, 'schemas')
MODELS = os.path.join(ROOT, 'data', 'models')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def across(cat, node):
    """The primitive's across_positions condition on a D1 node's arguments (§4.1); false when absent."""
    effect = cat['primitives'][node['primitive']['name']].get('effects', {}).get('across_positions')
    return bool(effect and primitive_condition(effect['when'], node['arguments']))


def main():
    cat = primitive_library_mod.load(os.path.join(ROOT, 'data', 'primitive-library'))
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
        # every structural graph_split is valid: block A is closed under ancestors
        blocks = {}
        for c in doc['d2']['graph_splits']:
            payload = {p['value'] for p in c['payload']}
            crossing = [e for e in doc['d1']['edges'] if f"{e['from']['node']}.{e['from']['port']}" in payload]
            ok &= check(f"{name}: graph_split {c['graph_split']} has a payload of distinct values", len(payload) == len(c['payload']))
            if not crossing and c['payload']:
                ok &= check(f"{name}: graph_split {c['graph_split']} payload values are edge sources", False)
        # across_positions (§4.1, O9.5): every node carries it, equal to the primitive's condition on the
        # node's own arguments — false when the primitive declares none
        nodes_d1 = doc['d1']['nodes']
        wrong = [n for n, e in nodes_d1.items() if not isinstance(e.get('across_positions'), bool) or e['across_positions'] != across(cat, e)]
        ok &= check(f"{name}: every D1 node carries across_positions, the primitive's condition on its own arguments",
                    not wrong, str(wrong[:3]))
        # V18 as a property of the products: a node reading across positions of a fragmented stream owns a
        # state carried across that stream's fragments
        fragmented = {e.get('stream', n) for n, e in doc['d1']['interfaces']['inputs'].items() if e.get('fragmented')}
        reads = {}
        for v in doc['d2']['values']:
            for t in v.get('to', []):
                reads.setdefault(t.rsplit('.', 1)[0], set()).add(v['domain']['stream'])
        carried_on = {}
        for s in doc['d4']['states']:
            if s['carried_across_fragments'] and s['stream']:
                for m in s['members']:
                    carried_on.setdefault(m.rsplit('.', 1)[0], set()).add(s['stream']['stream'])
        bad = [(n, st) for n, e in nodes_d1.items() if e['across_positions']
               for st in reads.get(n, ()) & fragmented if st not in carried_on.get(n, ())]
        ok &= check(f"{name}: every node reading across positions of a fragmented stream carries a state across its fragments (V18)",
                    not bad, str(bad[:3]))
        docs[name] = doc
    l3 = docs['llama3-8b']
    kv = [s for s in l3['d4']['states'] if s['state'] == 'kv']
    ok &= check("llama3-8b: 32 KV states of 4096 bytes per cached position",
                len(kv) == 32 and all(s['bytes_per_cached_position'] == 4096 for s in kv))
    ok &= check("llama3-8b: 128 KiB per token across the model",
                l3['d4']['totals']['append_bytes_per_cached_position'] == 131072)
    graph_split = next(c for c in l3['d2']['graph_splits'] if c['graph_split'] == 'decoder[layer<=3]')
    ok &= check("llama3-8b: one value of 8 KiB per element crosses a layer boundary",
                len(graph_split['payload']) == 1 and graph_split['bytes_per_element'] == 8192
                and graph_split['bytes_per_invocation'] == {'tokens': 8192.0}, str(graph_split['payload']))
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
                and all(s['evolution'] == 'window' and s['span'] in (750, 8192) for s in v['d4']['states']
                        if s['carried_across_fragments'] and s['primitive'] == 'attention.dense'))
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
    ok &= check("voxtral: the audio, the tokens and the delay are all required for the generative output (§7)",
                all(d2v[k]['required_for'] == ['main'] for k in ('audio', 'tokens', 'delay')))
    caches = [s for s in v['d4']['states'] if s['state'] == 'condition_cache']
    ok &= check("voxtral: 26 condition caches, append states on the delay stream (kind sequence) indexed by the condition port, shared by_source, not carried",
                len(caches) == 26 and all(s['evolution'] == 'append' and s['sharing'] == 'by_source' and s['indexed_by_port'] == 'condition'
                                          and s['stream'] == {'kind': 'sequence', 'stream': 'delay'} and not s['carried_across_fragments']
                                          for s in caches), str(caches[:1])[:300])
    ok &= check("voxtral: the delay stream is one element per sequence, count 1.0, and the time embedding is a sequence-kind value on it",
                v['d2']['streams']['delay'] == {'kind': 'sequence', 'count': {'delay': 1.0}}
                and d2v['time_embed.embedding']['domain'] == {'kind': 'sequence', 'stream': 'delay'} and d2v['time_embed.embedding']['count'] == {'delay': 1.0})
    ok &= check("deepseek-v4-pro: next_tokens joins the token stream at count 1.0, as before",
                {x['value']: x for x in docs['deepseek-v4-pro']['d2']['values']}['next_tokens']['count'] == {'tokens': 1.0})
    rings = {s['identity']: s for s in v['d4']['states'] if s['primitive'] == 'conv_frontend'}
    ok &= check("voxtral: the front end's histories are windows of kernel − 1 and kernel − stride frames, indexed by the frames port on the audio stream (V18)",
                rings['conv_frontend.conv1_history']['span'] == 2 and rings['conv_frontend.conv2_history']['span'] == 1
                and all(s['indexed_by_port'] == 'frames' and s['indexed_by_source'] and s['carried_across_fragments']
                        and s['stream'] == {'kind': 'position', 'stream': 'audio'} for s in rings.values()), str(rings)[:300])
    ok &= check("voxtral: the audio stream's fragment alignment is 8 frames — a stride of 2, then 4 positions per token (§5.3) — and the "
                "joined token input introduces no stream of its own",
                v['d2']['streams']['audio'].get('fragment_alignment') == 8 and 'tokens' not in v['d2']['streams'],
                str(v['d2']['streams']))
    ok &= check("llama3-8b: an unfragmented stream states no alignment", 'fragment_alignment' not in l3['d2']['streams']['tokens'])
    g = docs['gemma3n-kvshare']
    shared = [s for s in g['d4']['states'] if s['identity'].startswith('shared.')]
    ok &= check("gemma3n: the shared identities carry no layer in their instance key",
                len(shared) == 2 and all(s['instance_key'] == ['instance.session', 'instance.branch'] for s in shared))
    ok &= check("gemma3n: D4 names the writer of each shared identity — layer 18 for the sliding ring, 19 for the full cache",
                {s['identity']: s['writer'] for s in shared} == {'shared.sliding.kv': 'decoder/attn[layer=18].kv', 'shared.full.kv': 'decoder/attn_full[layer=19].kv'})
    ok &= check("llama3-8b: no O5.10 information loss once every flattened axis declares its factors",
                l3['d6']['information_loss'] == [])
    parts = {(p['node'], json.dumps(p['target'], sort_keys=True)): p for p in l3['d6']['partition_options']}
    heads = parts.get(('decoder/attn[layer=0]', json.dumps({'argument_axis': 'attention.heads'}, sort_keys=True)))
    ok &= check("llama3-8b: the head partition keeps whole KV groups — granularity 32 / 8 = 4 — and every partition lists its communications",
                heads is not None and heads['granularity'] == 4 and heads['communication'] == ['all_reduce']
                and all(isinstance(p['communication'], list)
                        and p['granularity'] == (4 if p['target'] == {'argument_axis': 'attention.heads'} else 1)
                        for p in l3['d6']['partition_options']),
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
    # a slot with a multiplicity is stored with the storage axis first (§3.4, finding 30)
    ok &= check("every D3 tensor: elements is the product of its shape's extents — the storage axis carries the count once",
                all(t['elements'] == math.prod(a['extent'] for a in t['shape'])
                    for d in docs.values() for t in d['d3']['tensors'] if t['elements'] is not None))
    gp = {t['identity']: t for t in g['d3']['tensors']}
    ok &= check("gemma3n: expand.projection is stored [3, 2048, 2048], storage.multiplicity first, multiplicity 3, 12 582 912 elements; "
                "the totals (4 435 182 688 elements, 8 870 365 376 bytes with the correction scale stored bf16) and D6 (545 partition_options, 32 losses) unchanged",
                [(a['axis'], a['extent']) for a in gp['expand.projection']['shape']] == [('storage.multiplicity', 3), ('model.width', 2048), ('model.width', 2048)]
                and gp['expand.projection']['multiplicity'] == 3 and gp['expand.projection']['elements'] == 12582912
                and g['d3']['totals']['elements'] == 4435182688 and g['d3']['totals']['bytes'] == 8870365376    # the correction scale stored bf16 (S4.2)
                and len(g['d6']['partition_options']) == 545 and len(g['d6']['information_loss']) == 32,
                str(gp['expand.projection'].get('shape')))
    sg = q35['decoder.mlp.shared_gate[layer=0]']
    ok &= check("qwen3.5-35b-a3b: a declared multiplicity of one is an extent-one storage axis — shared_gate[layer=0] is [1, 512, 2048], "
                "multiplicity 1, 1 048 576 elements, located on the plain tensor; the totals (35 107 181 936 elements) and D6 (589 partition_options) unchanged",
                [(a['axis'], a['extent']) for a in sg['shape']] == [('storage.multiplicity', 1), ('ffn.inner', 512), ('model.width', 2048)]
                and sg['multiplicity'] == 1 and sg['elements'] == 1048576 and sg['location'] == {'tensor': 'model.language_model.layers.0.mlp.shared_expert.gate_proj.weight'}
                and docs['qwen3.5-35b-a3b']['d3']['totals']['elements'] == 35107181936 and len(docs['qwen3.5-35b-a3b']['d6']['partition_options']) == 589,
                str(sg['shape']))
    ok &= check("gemma3n: 697 tensors located under model.language_model — the two stream projections as stacks of three altup(_unembed)_projections.{c}.weight at dim 0 "
                "(the storage axis), the readers' k/v/k_norm without an identity from layer 20 on, the per-layer tables whole",
                all(t.get('location') for t in g['d3']['tensors']) and len(g['d3']['tensors']) == 697
                and gp['expand.projection']['location'] == {'stack': {'axis': 'multiplicity', 'dim': 0, 'parts': [{'tensor': f'model.language_model.altup_projections.{i}.weight'} for i in range(3)]}}
                and gp['unembed.projection']['location']['stack']['parts'][2] == {'tensor': 'model.language_model.altup_unembed_projections.2.weight'}
                and 'decoder.attn.k[layer=20]' not in gp and gp['decoder.attn.k[layer=17]']['location'] == {'tensor': 'model.language_model.layers.17.self_attn.k_proj.weight'}
                and gp['embed.per_layer_embed']['location'] == {'tensor': 'model.language_model.embed_tokens_per_layer.weight'}
                and [a['extent'] for a in gp['embed.per_layer_embed']['shape']] == [262144, 7680],
                str(gp['expand.projection'].get('location')))
    ok &= check("a slot that declares no multiplicity has no storage axis (llama3-8b, every tensor)",
                all(a['axis'] != 'storage.multiplicity' for t in l3['d3']['tensors'] for a in t['shape']) and l3['d3']['totals']['elements'] == 8030261248)
    cb = {t['identity']: t.get('location') for t in docs['colbert-v2']['d3']['tensors']}
    cb_names = [v['tensor'] for v in cb.values() if v]
    ok &= check("colbert-v2: 198 tensors located one-to-one — enc.attn.q[layer=3] under bert.encoder.layer, the head on linear.weight",
                cb.get('enc.attn.q[layer=3]') == {'tensor': 'bert.encoder.layer.3.attention.self.query.weight'}
                and cb.get('pooler.weight') == {'tensor': 'linear.weight'} and len(cb_names) == 198 and len(set(cb_names)) == 198)
    # across_positions on the corpus (§4.1, O9.5), counted by primitive
    def flagged(name):
        return Counter(e['primitive']['name'] for e in docs[name]['d1']['nodes'].values() if e['across_positions'])

    def stateless(name):
        members = {m.rsplit('.', 1)[0] for s in docs[name]['d4']['states'] for m in s['members']}
        return [n for n, e in docs[name]['d1']['nodes'].items() if e['across_positions'] and n not in members]
    ok &= check("llama3-8b: 32 instances read across positions — the attentions, nothing else",
                flagged('llama3-8b') == {'attention.dense': 32}, str(flagged('llama3-8b')))
    ok &= check("colbert-v2: the 12 attentions read across positions; the pooler with reduce none does not",
                flagged('colbert-v2') == {'attention.dense': 12} and docs['colbert-v2']['d1']['nodes']['pooler']['arguments']['reduce'] == 'none'
                and not docs['colbert-v2']['d1']['nodes']['pooler']['across_positions'], str(flagged('colbert-v2')))
    ok &= check("whisper-large-v3: the stem and 96 attentions read across positions; the stem and the encoder's 32 hold no state — "
                "the case a state-based proxy gets wrong",
                flagged('whisper-large-v3') == {'attention.dense': 96, 'conv_frontend': 1} and len(stateless('whisper-large-v3')) == 33
                and all(n == 'conv_frontend' or n.startswith('encoder/attn') for n in stateless('whisper-large-v3')),
                str((flagged('whisper-large-v3'), stateless('whisper-large-v3')[:3])))
    ok &= check("voxtral-realtime: the stem and 58 attentions read across positions; the temporal projector, a merge, does not",
                flagged('voxtral-realtime') == {'attention.dense': 58, 'conv_frontend': 1}
                and not v['d1']['nodes']['audio_projector']['across_positions'], str(flagged('voxtral-realtime')))
    ok &= check("qwen3.5-4b-text: 8 attentions and 24 gated deltas read across positions",
                flagged('qwen3.5-4b-text') == {'attention.dense': 8, 'sequence.gated_delta': 24}, str(flagged('qwen3.5-4b-text')))
    ok &= check("deepseek-v4-pro: 62 latent attentions and the MTP merge read across positions",
                flagged('deepseek-v4-pro') == {'attention.latent_compressed': 62, 'mtp.merge': 1}, str(flagged('deepseek-v4-pro')))
    ok &= check("decoder-causal-yarn: 26 attentions read across positions under the assignment",
                flagged('decoder-causal-yarn@1.0.0') == {'attention.dense': 26}, str(flagged('decoder-causal-yarn@1.0.0')))
    ok &= check("gemma3n: aux_select carries its layer index in D1 — an index-valued argument, evaluated in the site's environment",
                g['d1']['nodes']['decoder/aux_select[layer=7]']['arguments'].get('layer') == 7,
                str(g['d1']['nodes']['decoder/aux_select[layer=7]']['arguments']))
    # an undecidable condition is a refusal, never false: whisper's stem with its kernel left to an
    # external quantity that no assignment supplies
    with open(os.path.join(MODELS, 'whisper-large-v3.json'), encoding='utf-8') as f:
        undecided = json.load(f)
    undecided['quantities']['k'] = {"type": {"kind": "cardinality"}, "source": {"kind": "external"}}
    undecided['instances']['conv_frontend']['arguments']['kernel'] = {"quantity": "k"}
    tmp = tempfile.mkdtemp(prefix='tensorspine-derived-')
    path = os.path.join(tmp, 'whisper-undecided.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(undecided, f)
    try:
        d1.emit(path, cat, {})
        ok &= check("an across_positions condition over an argument the document leaves unresolved refuses the derivation", False, "emitted")
    except ValueError as e:
        ok &= check("an across_positions condition over an argument the document leaves unresolved refuses the derivation, naming the argument",
                    'across_positions' in str(e) and "'kernel'" in str(e), str(e)[:200])
    # the grammar before the meaning (I2): a document off the schema has no products, and the refusal
    # is --validate's own line — the derivation crosses the structural stage, not the semantic one alone
    import lint
    with open(os.path.join(MODELS, 'llama3-8b.json'), encoding='utf-8') as f:
        source = f.read()
    llama = json.loads(source)
    for label, mutate, expect in (
            ("a foreign schema tag", lambda d: d.__setitem__('schema', 'not-tensorspine/99'), "schema: 'tensorspine/2.0' was expected"),
            ("a misspelt instance field", lambda d: d['instances']['embed'].__setitem__('argumants', d['instances']['embed'].pop('arguments')),
             "instances/embed: 'arguments' is a required property")):
        mutated = json.loads(source)
        mutate(mutated)
        mutated['primitive_libraries'] = [{"base": os.path.join(ROOT, 'data', 'primitive-library') + os.sep}]
        path = os.path.join(tmp, f"llama-{label.split()[-1]}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mutated, f)
        line = validate.structural(path, SCHEMAS)
        ok &= check(f"{label}: --validate refuses it at the structural stage", bool(line) and expect in line[0], str(line[:1]))
        try:
            derive.products(path, cat)
            ok &= check(f"{label}: derive.products refuses it with --validate's line", False, "derived")
        except ValueError as e:
            ok &= check(f"{label}: derive.products refuses it with --validate's line",
                        str(e) == f"not valid, no products: {line[0]}", str(e)[:200])
        advisories = lint.model_advisories(cat, [path])
        ok &= check(f"{label}: --lint reports it off the schema and does not analyse it",
                    len(advisories) == 1 and 'off the schema' in advisories[0][1] and expect in advisories[0][1], str(advisories[:1]))
    # G (R20): the generated JSON Schema per primitive version (non-normative). Every corpus
    # instance's resolved arguments validate against its primitive's schema; a scalar domain
    # violation is caught by it, while a relation between arguments (an invariant) is invisible to
    # JSON Schema and kept as an x-tensorspine-* annotation, which the check states.
    import primitive_schema
    import jsonschema
    schemas = {cid: s for cid, s in primitive_schema.render(cat).items()}
    bad = []
    checked = 0
    for name, doc in docs.items():
        for node, e in doc['d1']['nodes'].items():
            sch = schemas.get(f"{e['primitive']['name']}@{e['primitive']['version']}")
            if sch is None:
                continue
            checked += 1
            errs = list(jsonschema.Draft202012Validator(sch).iter_errors(e['arguments']))
            if errs:
                bad.append(f"{name} {node}: {errs[0].message}")
    ok &= check(f"G: every corpus instance's resolved arguments validate against its primitive's generated schema ({checked} instances)",
                not bad, str(bad[:3]))
    att = schemas['attention.dense@1.0.0']
    v = jsonschema.Draft202012Validator(att)
    ok &= check("G: a scalar domain violation is caught by the schema — window.span 0 below minimum, kv_heads 0 below minimum, a fractional span not an integer",
                bool(list(v.iter_errors({'width': 8, 'heads': 4, 'head_dim': 8, 'mask': 'causal', 'window': {'span': 0}})))
                and bool(list(v.iter_errors({'width': 8, 'heads': 4, 'head_dim': 8, 'kv_heads': 0, 'mask': 'causal'})))
                and bool(list(v.iter_errors({'width': 8, 'heads': 4, 'head_dim': 8, 'mask': 'causal', 'window': {'span': 2.5}}))))
    ok &= check("G: a relation between arguments is invisible to JSON Schema — heads not a multiple of kv_heads validates, and is carried in x-tensorspine-invariants",
                not list(v.iter_errors({'width': 8, 'heads': 32, 'head_dim': 8, 'kv_heads': 3, 'mask': 'causal'}))
                and any('multiple' in i['description'] for i in att['x-tensorspine-invariants']))
    moe = schemas['moe@1.0.0']
    ok &= check("G: top_k above experts validates against the moe schema too — a relation, not a domain",
                not list(jsonschema.Draft202012Validator(moe).iter_errors({'width': 8, 'experts': 4, 'top_k': 8, 'inner': 8}))
                and any('experts' in i['description'] for i in moe['x-tensorspine-invariants']))
    print("derived: all good" if ok else "derived: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

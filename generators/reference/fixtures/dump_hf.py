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
    python3 fixtures/dump_hf.py --model DIR --document whisper-large-v3 --audio jfk.wav --attn-site self_attn \
                                --layers 3 --ids 50258,50259,50360,50364 --steps 7 --out F     # an encoder–decoder

An encoder–decoder (`--audio`) records the audio frames it was fed as `in/audio` with their
provenance (`--audio-origin`, `--audio-licence`: where the recording came from and under which
licence, required), and the encoder's output — the cross source every decoder layer reads — beside the
decoder's layer outputs and self-attention cache; its cross-attention cache is not recorded
(`hook_map` says so). A streaming model whose token stream joins the audio stream (Voxtral
Realtime, told by the config's `model_type`) takes the same flag: the artifact's processor builds
the prefill, the steps deliver a token and eight frames each, and the file records the frames
consumed, the delay as `in/delay`, the decoder caches, the convolution histories and the held
conditions after the prefill (`--encoder-rings` adds the encoder's caches).

    python3 fixtures/dump_hf.py --model DIR --document voxtral-realtime --audio la-cigale-et-la-fourmi.wav --layers 3 \
                                --steps 24 --audio-origin … --audio-licence … --out F               # a streaming model
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
    ap.add_argument('--ids', help="the prompt's token ids; on a streaming model the processor's prefill, checked against these when given")
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
    ap.add_argument('--audio', metavar='WAV', help="an encoder–decoder (Whisper): a mono 16-bit WAV at the extractor's rate, turned into the "
                                                   "`audio` frames by the checkpoint's own feature extractor and recorded as in/audio with its "
                                                   "provenance; --layers truncates the decoder (decoder_layers); the encoder runs whole")
    ap.add_argument('--audio-origin', metavar='TEXT', help="with --audio: where the recording came from (a repository, an archive item, a URL), recorded in its provenance")
    ap.add_argument('--audio-licence', metavar='TEXT', help="with --audio: the licence the recording is used under, recorded in its provenance")
    ap.add_argument('--encoder-output', default='enc_final_n.output', help="with --audio: the D1 value of the encoder's output, the cross source")
    ap.add_argument('--cross-site', default='cross_attn', help="with --audio: the D1 site of the cross attention (its kv state is not recorded)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--encoder-rings', action='store_true', help="with --audio on a streaming model: record the encoder's sliding-window caches after the prefill "
                                                                 "(32 layers of keys and values per encoder position: large; left out by default, every decode step exercises them)")
    args = ap.parse_args(argv)
    if args.audio:
        from transformers import AutoConfig
        if AutoConfig.from_pretrained(args.model).model_type == 'voxtral_realtime':
            return dump_streaming(args)
        return dump_encoder_decoder(args)
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


def _capture(dump, key):
    """A forward hook recording the module's output at `key` once: the prefill's."""
    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if key not in dump:
            dump[key] = out[0].detach().to(torch.float32).cpu().clone()
    return hook


def dump_encoder_decoder(args):
    """An encoder–decoder with an audio input (Whisper). The checkpoint's feature extractor turns
    the WAV into the frames the document's `audio` input takes — the delivery's preprocessing,
    outside the document as a tokenizer is — recorded element-major as `in/audio` with its
    provenance (docs/TENSORSPINE-FIXTURE.md §3). `--layers` truncates the decoder through
    `decoder_layers`, never the config's `num_hidden_layers`, which on Whisper names the encoder;
    the encoder runs whole and its final layer norm — the cross source every decoder layer reads —
    is recorded at `value/<--encoder-output>`. The decoder layers' outputs, the self-attention cache
    after the prefill, the logits and the greedy tokens follow the causal-LM path. The
    cross-attention cache is not recorded: a projection of the recorded encoder output that every
    decoder output exercises; `hook_map` states the omission."""
    import hashlib
    import wave
    import numpy as np
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq
    dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[args.dtype]
    config = AutoConfig.from_pretrained(args.model)
    if args.layers:
        config.decoder_layers = args.layers
    n_layers = config.decoder_layers
    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, config=config, dtype=dtype)
    model.eval()
    print(f"loaded {args.model}: {config.encoder_layers} encoder and {n_layers} decoder layers in {dtype} ({time.time() - t0:.0f}s)")
    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    with wave.open(args.audio) as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if (channels, width, rate) != (1, 2, extractor.sampling_rate):
            raise SystemExit(f"{args.audio}: {channels} channel(s), {8 * width}-bit, {rate} Hz — the extractor takes mono 16-bit at {extractor.sampling_rate} Hz")
        pcm = np.frombuffer(wav.readframes(frames), dtype='<i2').astype(np.float32) / 32768.0
    features = extractor(pcm, sampling_rate=rate, return_tensors='pt').input_features.to(dtype)     # [1, mels, frames]
    with open(args.audio, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    inputs = {'audio': _recording(args, digest,
                                  f"{type(extractor).__name__} from the checkpoint's preprocessor_config.json: "
                                  f"{extractor.feature_size} log-mel bins per frame of {extractor.hop_length} samples at {rate} Hz, "
                                  f"the {frames / rate:.1f} s signal zero-padded to {extractor.n_samples // rate} s "
                                  f"({features.shape[-1]} frames); one row per frame")}
    dump = {'in/audio': features[0].T.to(torch.float32).cpu().clone()}
    hooks, hook_map = [], {'input_features[0].T': 'in/audio'}
    inner = model.model
    for i, layer in enumerate(inner.decoder.layers):
        key = f"value/{args.composition}/{args.layer_output}[layer={i}].output"
        hook_map[f"model.decoder.layers.{i}"] = key
        hooks.append(layer.register_forward_hook(_capture(dump, key)))
    hook_map['model.encoder.layer_norm'] = f"value/{args.encoder_output}"
    hooks.append(inner.encoder.layer_norm.register_forward_hook(_capture(dump, f"value/{args.encoder_output}")))
    ids = [int(x) for x in args.ids.split(',')]
    with torch.no_grad():
        t0 = time.time()
        out = model(input_features=features, decoder_input_ids=torch.tensor([ids]), use_cache=True)
        cache = out.past_key_values
        self_cache = getattr(cache, 'self_attention_cache', cache)
        logits = out.logits[0].to(torch.float32)
        dump['logits/last'] = logits[-1].cpu().clone()
        dump['logits/argmax'] = logits.argmax(-1).cpu().clone()
        for i in range(n_layers):
            layer = self_cache.layers[i]
            dump[f"state/{args.composition}.{args.attn_site}.kv[layer={i}]/k"] = layer.keys[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
            dump[f"state/{args.composition}.{args.attn_site}.kv[layer={i}]/v"] = layer.values[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
        hook_map['past_key_values.self_attention_cache.layers[i].keys[0].permute(1,0,2)'] = f"state/{args.composition}.{args.attn_site}.kv[layer=i]/k"
        hook_map['past_key_values.cross_attention_cache'] = (f"not recorded: state/{args.composition}.{args.cross_site}.kv[layer=i] is a projection of "
                                                             f"value/{args.encoder_output}, which the fixture holds, and every decoder layer output exercises it")
        nxt = int(logits[-1].argmax())
        tokens = [nxt]
        print(f"encoded {features.shape[-1]} frames, prefill {len(ids)} -> {nxt} ({time.time() - t0:.1f}s)")
        encoder_outputs = (out.encoder_last_hidden_state,)
        for _ in range(args.steps):
            out = model(encoder_outputs=encoder_outputs, decoder_input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
        print("tokens:", tokens)
    for h in hooks:
        h.remove()
    write_fixture(args.out, dump, metadata(args, n_layers, ids, tokens, hook_map, inputs))
    print(f"dumped {len(dump)} tensors -> {args.out}")


def _recording(args, digest, processor):
    """The provenance of a recorded input (docs/TENSORSPINE-FIXTURE.md §3): the file's name and hash,
    where it came from and under which licence — both required, since a committed fixture carries
    open content — and how the tensor was made from it."""
    if not (args.audio_origin and args.audio_licence):
        raise SystemExit("--audio records open content: give --audio-origin and --audio-licence, which the fixture carries")
    return {'source': os.path.basename(args.audio), 'sha256': digest, 'origin': args.audio_origin, 'licence': args.audio_licence,
            'processor': processor}


def dump_streaming(args):
    """A streaming speech model whose token stream joins the audio stream (Voxtral Realtime). The
    artifact's processor — transformers' `VoxtralRealtimeProcessor`, which needs `mistral_common` —
    builds the streaming prefill: the prompt (a start token and the streaming pads of the left
    padding and the delay), the left-padded audio and `num_delay_tokens`; the checkpoint's feature
    extractor turns the audio into frames, eight per token. The prefill delivers the prompt with
    as many frames as its tokens take; every step delivers the token produced with the next eight
    frames — `transformers` run as `generate` runs it, the stem through its padding cache chunk by
    chunk (the streaming computation, which the whole-signal stem equals by construction), the
    encoder through its sliding-window cache, the decoder through its own. Recorded: `in/audio`
    (every frame the prefill and the steps consumed, with the recording's provenance), `in/delay`
    (the setting, with the processor's word for it), the decoder layer outputs, the encoder's
    final norm, the projector's output and the time embedding at the prefill, the decoder caches,
    the two convolution histories and the held conditions after it, the logits and the greedy
    tokens; the encoder rings with `--encoder-rings`. `--layers` truncates the decoder
    (`text_config.num_hidden_layers`); the encoder runs whole."""
    import hashlib
    import wave
    import numpy as np
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (VoxtralRealtimeConv1dPaddingCache,
                                                                                 VoxtralRealtimeForConditionalGeneration)
    dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[args.dtype]
    config = AutoConfig.from_pretrained(args.model)
    if args.layers:
        config.text_config.num_hidden_layers = args.layers
    n_layers = config.text_config.num_hidden_layers
    per_token = config.audio_length_per_tok                    # frames per token: the document's count, 8
    t0 = time.time()
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(args.model, config=config, dtype=dtype)
    model.eval()
    print(f"loaded {args.model}: {config.audio_config.num_hidden_layers} encoder and {n_layers} decoder layers in {dtype} ({time.time() - t0:.0f}s)")
    try:
        processor = AutoProcessor.from_pretrained(args.model)
    except ImportError as e:
        raise SystemExit(f"the artifact's processor needs mistral_common (and soundfile): {e}")
    extractor = processor.feature_extractor
    with wave.open(args.audio) as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if (channels, width, rate) != (1, 2, extractor.sampling_rate):
            raise SystemExit(f"{args.audio}: {channels} channel(s), {8 * width}-bit, {rate} Hz — the extractor takes mono 16-bit at {extractor.sampling_rate} Hz")
        pcm = np.frombuffer(wav.readframes(frames), dtype='<i2').astype(np.float32) / 32768.0
    with open(args.audio, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    enc = processor(pcm, is_streaming=True, is_first_audio_chunk=True, sampling_rate=rate, return_tensors='pt')
    ids = enc['input_ids'][0].tolist()
    if args.ids and [int(x) for x in args.ids.split(',')] != ids:
        raise SystemExit(f"--ids differ from the processor's streaming prefill {ids}: the prompt is the processor's")
    features = enc['input_features'].to(dtype)                  # [1, mels, frames], the left padding included
    delay = int(enc['num_delay_tokens'])
    total = features.shape[-1]
    steps = min(args.steps, total // per_token - len(ids))
    consumed = (len(ids) + steps) * per_token
    left_pad = processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.n_left_pad_tokens
    inputs = {'audio': _recording(args, digest,
                                  f"{type(extractor).__name__} from the checkpoint's preprocessor_config.json on the processor's streaming "
                                  f"first chunk — {left_pad} tokens of silence ({left_pad * per_token * extractor.hop_length} samples) before the "
                                  f"{frames / rate:.1f} s signal — {extractor.feature_size} log-mel bins per frame of {extractor.hop_length} samples "
                                  f"at {rate} Hz; the {consumed} frames the prefill ({len(ids)} tokens, {len(ids) * per_token} frames) and "
                                  f"{steps} steps of {per_token} consumed, one row per frame, of the {total} the signal makes"),
              'delay': {'processor': f"the processor's num_delay_tokens, {delay}: the checkpoint's default_num_delay_tokens "
                                     f"({config.default_num_delay_tokens}), a delivered setting with no file to hash"}}
    dump = {'in/audio': features[0].T[:consumed].to(torch.float32).cpu().clone(),
            'in/delay': torch.tensor([delay], dtype=torch.int32)}
    hooks, hook_map = [], {'input_features[0].T[:consumed]': 'in/audio', 'num_delay_tokens': 'in/delay'}
    inner = model.model
    for i, layer in enumerate(inner.language_model.layers):
        key = f"value/{args.composition}/{args.layer_output}[layer={i}].output"
        hook_map[f"model.language_model.layers.{i}"] = key
        hooks.append(layer.register_forward_hook(_capture(dump, key)))
    for module, name, key in ((inner.audio_tower.norm, 'model.audio_tower.norm', f"value/{args.encoder_output}"),
                              (inner.multi_modal_projector, 'model.multi_modal_projector', 'value/audio_projector.output')):
        hook_map[name] = key
        hooks.append(module.register_forward_hook(_capture(dump, key)))

    def time_hook(module, inputs_, output):
        if 'value/time_embed.embedding' not in dump:
            dump['value/time_embed.embedding'] = output.detach().to(torch.float32).reshape(1, -1).cpu().clone()
    hook_map['model.time_embedding'] = 'value/time_embed.embedding'
    hooks.append(inner.time_embedding.register_forward_hook(time_hook))
    padding_cache = VoxtralRealtimeConv1dPaddingCache()
    with torch.no_grad():
        t0 = time.time()
        # the stem chunk by chunk through its padding cache: the prefill's frames, then eight per step
        chunk = inner.audio_tower.embedder(features[:, :, :len(ids) * per_token], padding_cache=padding_cache)
        out = model(input_ids=torch.tensor([ids]), encoder_inputs_embeds=chunk, num_delay_tokens=delay, use_cache=True)
        cache, enc_cache = out.past_key_values, out.encoder_past_key_values
        logits = out.logits[0].to(torch.float32)
        dump['logits/last'] = logits[-1].cpu().clone()
        dump['logits/argmax'] = logits.argmax(-1).cpu().clone()
        for i in range(n_layers):
            layer = cache.layers[i]
            dump[f"state/{args.composition}.{args.attn_site}.kv[layer={i}]/k"] = layer.keys[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
            dump[f"state/{args.composition}.{args.attn_site}.kv[layer={i}]/v"] = layer.values[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
            dump[f"state/{args.composition}.time_scale.condition_cache[layer={i}]/c"] = dump['value/time_embed.embedding'].clone()
        hook_map['past_key_values.layers[i].keys[0].permute(1,0,2)'] = f"state/{args.composition}.{args.attn_site}.kv[layer=i]/k"
        hook_map['time_embedding output, held by every layer'] = f"state/{args.composition}.time_scale.condition_cache[layer=i]/c"
        for name, key in (('conv1', 'state/conv_frontend.conv1_history/w'), ('conv2', 'state/conv_frontend.conv2_history/w')):
            dump[key] = padding_cache.layers[name].cache[0].T.to(torch.float32).cpu().clone()     # [left_pad, channels]
            hook_map[f"padding_cache.layers[{name}].cache[0].T"] = key
        if args.encoder_rings:
            for i in range(config.audio_config.num_hidden_layers):
                layer = enc_cache.layers[i]
                dump[f"state/encoder.{args.attn_site}.kv[layer={i}]/k"] = layer.keys[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
                dump[f"state/encoder.{args.attn_site}.kv[layer={i}]/v"] = layer.values[0].permute(1, 0, 2).to(torch.float32).cpu().clone()
            hook_map['encoder_past_key_values.layers[i].keys[0].permute(1,0,2)'] = f"state/encoder.{args.attn_site}.kv[layer=i]/k"
        else:
            hook_map['encoder_past_key_values'] = (f"not recorded: state/encoder.{args.attn_site}.kv[layer=i] — the encoder's sliding-window caches, "
                                                   f"exercised by every decode step's four encoder positions; --encoder-rings records them")
        nxt = int(logits[-1].argmax())
        tokens = [nxt]
        print(f"prefill {len(ids)} tokens with {len(ids) * per_token} frames -> {nxt} ({time.time() - t0:.1f}s); {steps} steps of {per_token} frames")
        for k in range(steps):
            start = (len(ids) + k) * per_token
            chunk = inner.audio_tower.embedder(features[:, :, start:start + per_token], padding_cache=padding_cache)
            out = model(input_ids=torch.tensor([[nxt]]), encoder_inputs_embeds=chunk, num_delay_tokens=delay,
                        past_key_values=cache, encoder_past_key_values=enc_cache, use_cache=True)
            cache, enc_cache = out.past_key_values, out.encoder_past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
        print("tokens:", tokens)
    for h in hooks:
        h.remove()
    write_fixture(args.out, dump, metadata(args, n_layers, ids, tokens, hook_map, inputs))
    print(f"dumped {len(dump)} tensors -> {args.out}")


def metadata(args, n_layers, ids, tokens, hook_map, inputs=None):
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
    out = {'schema': 'tensorspine-fixture/1', 'kind': 'integration', 'document': args.document,
           'artifact': artifact,
           'delivery': {'implementation': 'transformers', 'program': PROGRAM,
                        'versions': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__}},
           'truncation': {'composition': args.composition, 'layers': n_layers},
           'ids': ids, 'tokens': tokens, 'hook_map': hook_map,
           'compute': args.dtype, 'tolerance': tolerance}
    if inputs:
        out['inputs'] = inputs          # the non-token inputs the prefill delivered, with their provenance
    return out


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

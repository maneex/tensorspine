"""A small chat on a session (R08, R09): the tokenizer and the chat template come from
the checkpoint (the tokenizer only — the model class is never instantiated); each turn
tokenises the whole transcript and feeds the suffix beyond what the session has consumed;
a template that rewrites the prefix restarts the session and says so. Greedy by default,
`--temperature` and `--top-p` optional, streaming.
"""
import json
import os
import sys
import time

import torch

import state as state_mod
from session import Session


def load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint)


def stop_ids(checkpoint, tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    path = os.path.join(checkpoint, 'generation_config.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            eos = json.load(f).get('eos_token_id')
        for e in (eos if isinstance(eos, list) else [eos]):
            if e is not None:
                ids.add(int(e))
    return ids


def render(tokenizer, transcript, closed=False):
    """Token ids of the whole transcript: the checkpoint's chat template when it has one,
    else a plain transcript (a base model completes text). `closed`: the last turn is the
    assistant's and no generation prompt follows — the form the next turn must extend."""
    if getattr(tokenizer, 'chat_template', None):
        ids = tokenizer.apply_chat_template(transcript, add_generation_prompt=not closed, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, 'keys'):       # transformers 5 returns a BatchEncoding
            ids = ids['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids], 'template'
    text = ''.join(f"{t['role'].capitalize()}: {t['content']}\n" for t in transcript)
    if not closed:
        text += "Assistant:"
    return tokenizer.encode(text), 'plain'


MARKERS = ("\nUser:", "\nAssistant:")


def turn_marker(text):
    """Where a plain transcript's next turn begins in generated text, if it does."""
    hits = [text.find(m) for m in MARKERS if m in text]
    return min(hits) if hits else None


def held_back(text):
    """Characters at the end of `text` that could be the start of a turn marker — not
    printed yet, so that a marker is never shown."""
    return max((k for m in MARKERS for k in range(1, len(m)) if text.endswith(m[:k])), default=0)


def sample(logits, temperature, top_p, generator):
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.to(torch.float32) / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, order = probs.sort(descending=True)
        keep = (sorted_p.cumsum(0) - sorted_p) < top_p
        probs = torch.zeros_like(probs).scatter(0, order[keep], sorted_p[keep])
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


def chat(model, graph, checkpoint, capacity, device, dtype, max_new_tokens=256, temperature=0.0, top_p=1.0,
         seed=0, out=sys.stdout, inp=sys.stdin, decode_model=None):
    tokenizer = load_tokenizer(checkpoint)
    stops = stop_ids(checkpoint, tokenizer)
    session = Session(model, capacity, device, dtype, decode_model=decode_model)
    rendered_prev = []                 # the transcript as rendered before this turn
    transcript = []
    generator = torch.Generator().manual_seed(seed)
    stream = graph.input_stream[graph.feedback_input]
    capacity = state_mod.capacity_of(capacity, stream)          # the token stream's, when given per stream
    print(f"chat: capacity {capacity} positions, "
          f"{'chat template' if getattr(tokenizer, 'chat_template', None) else 'plain transcript (base model)'}, "
          f"{'greedy' if temperature <= 0 else f'temperature {temperature}, top-p {top_p}'}; empty line to quit",
          file=out)
    while True:
        try:
            print("you> ", end='', file=out, flush=True)
            line = inp.readline()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or not line.strip():
            break
        transcript.append({'role': 'user', 'content': line.strip()})
        ids, mode = render(tokenizer, transcript)
        if ids[:len(rendered_prev)] != rendered_prev:
            print("  (the template rewrote the prefix: session restarted)", file=out)
            session.reset()
            rendered_prev = []
        new = ids[len(rendered_prev):] or ids[-1:]
        seen = session.consumed.get(stream, 0)
        if seen + len(new) + max_new_tokens > capacity:
            print(f"  refused: {seen} + {len(new)} + {max_new_tokens} positions exceed the capacity {capacity}", file=out)
            break
        t0 = time.time()
        logits = session.prefill(new)[graph.generative[0]][-1]
        generated = []
        text_so_far, shown = '', 0
        print("bot> ", end='', file=out, flush=True)
        for _ in range(max_new_tokens):
            nxt = sample(logits, temperature, top_p, generator)
            if nxt in stops:
                break
            generated.append(nxt)
            text = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
            cut = turn_marker(text) if mode == 'plain' else None
            if cut is not None:                     # a base model started the next turn itself
                text_so_far = text[:cut]
                break
            text_so_far = text
            safe = len(text) - held_back(text) if mode == 'plain' else len(text)
            if safe > shown:
                print(text[shown:safe], end='', file=out, flush=True)
                shown = safe
            logits = session.decode(nxt)[graph.generative[0]][-1]
        if len(text_so_far) > shown:
            print(text_so_far[shown:], end='', file=out, flush=True)
        dt = time.time() - t0
        print(f"\n  ({len(generated)} tokens, {len(generated) / dt if dt else 0:.2f} tok/s; {session.consumed.get(stream, 0)} positions)", file=out)
        transcript.append({'role': 'assistant', 'content': text_so_far})
        rendered_prev, _ = render(tokenizer, transcript, closed=True)
    return 0

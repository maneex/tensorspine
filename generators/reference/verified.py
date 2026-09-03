"""What the reference generator has been verified on: the committed integration fixtures (a
truncated model against a `transformers` dump at every legal cut and state, on the language's
fixture schema — docs/TENSORSPINE-FIXTURE.md) and the full-model greedy tokens. Read by the test.
The fixture's own metadata names its document, its artifact and its tolerance; the table below
lists the fixtures and what the test checks that metadata against."""

FIXTURES = [   # (fixture, model document, artifact directory under $TENSORSPINE_MODEL_ARTIFACTS/weights[, (atol, rtol) the fixture states for fp32 when not the default])
    ('llama3-8b.3layers.hf.safetensors', 'llama3-8b', 'Meta-Llama-3-8B'),
    ('qwen3.5-4b-text.4layers.hf.safetensors', 'qwen3.5-4b-text', 'Qwen3.5-4B'),
    ('qwen3.8-27b-text.4layers.hf.safetensors', 'qwen3.8-27b-text', 'Qwen3.8-27B'),
    ('shieldstral-3b.3layers.hf.safetensors', 'shieldstral-3b', 'Shieldstral-1.0-3B'),
    ('colbert-v2.12layers.hf.safetensors', 'colbert-v2', 'colbertv2.0'),      # the whole model: an encoder has nothing to decode
    ('qwen3.5-35b-a3b.2layers.hf.safetensors', 'qwen3.5-35b-a3b', 'Qwen3.5-35B-A3B'),   # two gated-delta layers with their MoE, fp32
    ('qwen3.5-35b-a3b.4layers.hf.safetensors', 'qwen3.5-35b-a3b', 'Qwen3.5-35B-A3B', (0.1, 0.02)),   # the attention layer too; transformers in bf16 (4 layers in fp32 exceed the memory here): measured 8.2e-2
    ('whisper-large-v3.3layers.hf.safetensors', 'whisper-large-v3', 'whisper-large-v3'),   # the encoder whole, the decoder at three layers; the audio the fixture carries (in/audio) is delivered with the prompt
    ('voxtral-realtime.3layers.hf.safetensors', 'voxtral-realtime', 'Voxtral-Mini-4B-Realtime-2602'),   # the encoder whole, the decoder at three layers; the token stream joins the audio stream: the prompt with its tokens' frames, then a token and eight frames per step, the prefill also replayed as fragments
]
FULL = [   # (model document, checkpoint directory, prompt ids, the greedy tokens transformers 5.14 produced in bf16, 29 Aug 2026[, the fixture whose in/ tensors are delivered with the prompt, or the sample the artifact's processor turns into a streaming delivery])
    ('llama3-8b', 'Meta-Llama-3-8B', [128000, 791, 6864, 315, 9822, 374],        # "<|begin_of_text|>The capital of France is"
     [12366, 13, 1102, 374, 7559, 304, 279, 10411]),                              # " Paris. It is located in the north"
    ('qwen3.5-4b-text', 'Qwen3.5-4B', [760, 6511, 314, 9338, 369],                # "The capital of France is"
     [11751, 13, 198, 32, 13, 2912, 198, 33]),                                    # " Paris.\nA. True\nB"
    ('qwen3.8-27b-text', 'Qwen3.8-27B', [760, 6511, 314, 9338, 369],
     [11751, 13, 198]),                                                           # " Paris.\n" — under --max-ram 8 and memory-mapped alike
    ('shieldstral-3b', 'Shieldstral-1.0-3B', [1, 22177, 1033, 4425, 1636, 3508, 1639, 4777, 1261, 36335, 8967, 1063],   # "<s>Hello! Can you help me plan a birthday party?"
     [2649, 2, 2649, 2, 2649, 2, 2649, 2]),                                       # "no</s>no</s>…" — a moderation fine-tune gives its verdict, then EOS; decoding past EOS repeats it.
                                                                                  # On "The capital of France is" bf16 ties "no" and "yes" at 28.25 and fp32 breaks the tie the other way.
    ('whisper-large-v3', 'whisper-large-v3', [50258, 50259, 50360, 50364],        # "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>", the audio from the fixture
     [400, 370, 452, 7177, 6280, 11, 1029, 406],                                  # " And so my fellow Americans, ask not" — transformers 5.14 in fp32, 2 Sep 2026
     'whisper-large-v3.3layers.hf.safetensors'),
    ('voxtral-realtime', 'Voxtral-Mini-4B-Realtime-2602', [1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],   # the processor's streaming prefill: <s> and 38 streaming pads (32 of left padding, 6 of delay)
     [32, 32, 32, 32, 32, 32, 32, 33, 2541, 32, 32, 32, 32, 33, 24515, 2068, 32, 32, 33, 1568, 1427, 32, 32, 32, 32, 32, 32, 33, 3946, 3391, 1046, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 2541, 32, 32, 32, 32, 32, 33, 24515, 2068,
      1044, 19548, 32, 32, 32, 32, 32, 33, 30611, 1337, 32, 32, 33, 5143, 32, 32, 32, 32, 33, 1295, 54902, 1044, 32, 32, 32, 32, 32, 33, 1412, 32, 32,
      32, 33, 97246, 32, 32, 32, 32, 33, 8067, 32, 32, 32, 32, 32, 32, 32, 33, 24486, 1479, 28512, 13415, 32, 33, 1427, 32, 32, 32, 32, 33, 1289,
      2087, 4505, 32, 32, 32, 32, 33, 34233, 1046, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 13161, 32, 33, 1457, 32, 32, 33, 18979, 32, 32, 33,
      16095, 32, 32, 32, 32, 33, 81856, 32, 33, 1311, 32, 32, 32, 33, 17312, 1837, 3194, 1311, 32, 32, 32, 32, 32, 32, 32, 32, 33, 14913, 53255, 1046,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 7157, 7553, 32, 32, 32, 33, 1272, 6180, 32, 32, 32, 32, 33, 91095, 32, 32, 32, 32,
      33, 14439, 32, 33, 1427, 32, 32, 32, 32, 32, 33, 3946, 3391, 1044, 1887, 32, 32, 32, 32, 32, 33, 102266, 1044, 32, 32, 32, 32, 32, 33, 1427, 32,
      32, 32, 33, 4135, 1450, 1311, 3952, 32, 32, 32, 32, 32, 33, 1517, 42279, 32, 32, 33, 12492, 32, 32, 32, 33, 42474, 32, 32, 32, 32, 33, 2388, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 33, 10517, 3020, 32, 32, 32, 32, 32, 32, 32, 33, 9759, 15868, 1427, 32, 32, 32, 32, 33, 14821, 32, 32, 32, 32,
      32, 32, 33, 15599, 1046, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 3582, 6790, 32, 32, 32, 32, 33, 4431, 1720, 2464, 32,
      32, 32, 32, 32, 32, 32, 33, 3952, 32, 32, 32, 32, 32, 33, 6549, 16960, 32, 32, 32, 32, 32, 32, 32, 32, 33, 8802, 32, 33, 108288, 32, 32, 32, 32,
      32, 32, 32, 33, 9011, 32, 32, 32, 32, 32, 32, 32, 32, 33, 1266, 23260, 5903, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 71061, 32, 32, 32, 32, 32,
      33, 1568, 32, 32],
     # the French sample la-cigale-et-la-fourmi.wav to its end — 416 steps of a token and eight frames after the prefill, transformers 5.14 in f32,
     # 3 Sep 2026: " La cigale et la fourmi. La cigale, ayant chanté tout l'été, se trouva fort dépourvue quand la bise fut venue. …" (274 streaming pads among
     # the 417 tokens, the transcript lagging the audio by the delay); the reference, 4.3 s per step on this CPU, produced every one of them
     'la-cigale-et-la-fourmi.wav'),
]
CHECKPOINT_IDS = {'Meta-Llama-3-8B': 'NousResearch/Meta-Llama-3-8B', 'Qwen3.5-4B': 'Qwen/Qwen3.5-4B', 'Qwen3.8-27B': 'Qwen/Qwen3.8-27B',
                  'Shieldstral-1.0-3B': 'mistralai/Shieldstral-1.0-3B', 'colbertv2.0': 'colbert-ir/colbertv2.0',
                  'Qwen3.5-35B-A3B': 'Qwen/Qwen3.5-35B-A3B', 'whisper-large-v3': 'openai/whisper-large-v3',
                  'Voxtral-Mini-4B-Realtime-2602': 'mistralai/Voxtral-Mini-4B-Realtime-2602'}
AGREEMENT = "values, states and logits within atol 1e-3 / rtol 1e-2 of transformers in fp32 (measured max |d| 8e-6); greedy tokens identical"

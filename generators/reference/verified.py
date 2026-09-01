"""What the reference generator has been verified on: the committed fixtures (a truncated model
against a `transformers` dump at every legal cut and state) and the full-model greedy tokens.
Read by the test and by `ref.py capabilities`."""

FIXTURES = [   # (fixture, model document, artifact directory under $TENSORSPINE_MODEL_ARTIFACTS/weights[, (atol, rtol) when the dump is not fp32])
    ('llama3-8b.3layers.hf.safetensors', 'llama3-8b', 'Meta-Llama-3-8B'),
    ('qwen3.5-4b-text.4layers.hf.safetensors', 'qwen3.5-4b-text', 'Qwen3.5-4B'),
    ('qwen3.8-27b-text.4layers.hf.safetensors', 'qwen3.8-27b-text', 'Qwen3.8-27B'),
    ('shieldstral-3b.3layers.hf.safetensors', 'shieldstral-3b', 'Shieldstral-1.0-3B'),
    ('colbert-v2.12layers.hf.safetensors', 'colbert-v2', 'colbertv2.0'),      # the whole model: an encoder has nothing to decode
    ('qwen3.5-35b-a3b.2layers.hf.safetensors', 'qwen3.5-35b-a3b', 'Qwen3.5-35B-A3B'),   # two gated-delta layers with their MoE, fp32
    ('qwen3.5-35b-a3b.4layers.hf.safetensors', 'qwen3.5-35b-a3b', 'Qwen3.5-35B-A3B', (0.1, 0.02)),   # the attention layer too; transformers in bf16 (4 layers in fp32 exceed the memory here): measured 8.2e-2
]
FULL = [   # (model document, checkpoint directory, prompt ids, the greedy tokens transformers 5.14 produced in bf16, 29 Aug 2026)
    ('llama3-8b', 'Meta-Llama-3-8B', [128000, 791, 6864, 315, 9822, 374],        # "<|begin_of_text|>The capital of France is"
     [12366, 13, 1102, 374, 7559, 304, 279, 10411]),                              # " Paris. It is located in the north"
    ('qwen3.5-4b-text', 'Qwen3.5-4B', [760, 6511, 314, 9338, 369],                # "The capital of France is"
     [11751, 13, 198, 32, 13, 2912, 198, 33]),                                    # " Paris.\nA. True\nB"
    ('qwen3.8-27b-text', 'Qwen3.8-27B', [760, 6511, 314, 9338, 369],
     [11751, 13, 198]),                                                           # " Paris.\n" — under --max-ram 8 and memory-mapped alike
    ('shieldstral-3b', 'Shieldstral-1.0-3B', [1, 22177, 1033, 4425, 1636, 3508, 1639, 4777, 1261, 36335, 8967, 1063],   # "<s>Hello! Can you help me plan a birthday party?"
     [2649, 2, 2649, 2, 2649, 2, 2649, 2]),                                       # "no</s>no</s>…" — a moderation fine-tune gives its verdict, then EOS; decoding past EOS repeats it.
                                                                                  # On "The capital of France is" bf16 ties "no" and "yes" at 28.25 and fp32 breaks the tie the other way.
]
CHECKPOINT_IDS = {'Meta-Llama-3-8B': 'NousResearch/Meta-Llama-3-8B', 'Qwen3.5-4B': 'Qwen/Qwen3.5-4B', 'Qwen3.8-27B': 'Qwen/Qwen3.8-27B',
                  'Shieldstral-1.0-3B': 'mistralai/Shieldstral-1.0-3B', 'colbertv2.0': 'colbert-ir/colbertv2.0',
                  'Qwen3.5-35B-A3B': 'Qwen/Qwen3.5-35B-A3B'}
AGREEMENT = "values, states and logits within atol 1e-3 / rtol 1e-2 of transformers in fp32 (measured max |d| 8e-6); greedy tokens identical"

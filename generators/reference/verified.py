"""What the reference backend has been verified on: the committed fixtures (a truncated model
against a `transformers` dump at every legal cut and state) and the full-model greedy tokens.
Read by the test and by `ref.py capabilities`."""

FIXTURES = [   # (fixture, model document, checkpoint directory under ~/work/perso/huggingface)
    ('llama3-8b.3layers.hf.safetensors', 'llama3-8b', 'Meta-Llama-3-8B'),
    ('qwen3.5-4b-text.4layers.hf.safetensors', 'qwen3.5-4b-text', 'Qwen3.5-4B'),
    ('qwen3.8-27b-text.4layers.hf.safetensors', 'qwen3.8-27b-text', 'Qwen3.8-27B'),
]
FULL = [   # (model document, checkpoint directory, prompt ids, the greedy tokens transformers 5.14 produced in bf16, 29 Aug 2026)
    ('llama3-8b', 'Meta-Llama-3-8B', [128000, 791, 6864, 315, 9822, 374],        # "<|begin_of_text|>The capital of France is"
     [12366, 13, 1102, 374, 7559, 304, 279, 10411]),                              # " Paris. It is located in the north"
    ('qwen3.5-4b-text', 'Qwen3.5-4B', [760, 6511, 314, 9338, 369],                # "The capital of France is"
     [11751, 13, 198, 32, 13, 2912, 198, 33]),                                    # " Paris.\nA. True\nB"
    ('qwen3.8-27b-text', 'Qwen3.8-27B', [760, 6511, 314, 9338, 369],
     [11751, 13, 198]),                                                           # " Paris.\n" — under --max-ram 8 and memory-mapped alike
]
CHECKPOINT_IDS = {'Meta-Llama-3-8B': 'NousResearch/Meta-Llama-3-8B', 'Qwen3.5-4B': 'Qwen/Qwen3.5-4B', 'Qwen3.8-27B': 'Qwen/Qwen3.8-27B'}
AGREEMENT = "values, states and logits within atol 1e-3 / rtol 1e-2 of transformers in fp32 (measured max |d| 8e-6); greedy tokens identical"

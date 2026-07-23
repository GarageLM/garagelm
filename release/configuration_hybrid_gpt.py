"""HF config for the hybrid local+global GPT (llm-arch-explore).

Uploaded alongside modeling_hybrid_gpt.py; loaded via trust_remote_code.
"""

from transformers import PretrainedConfig


class HybridGPTConfig(PretrainedConfig):
    model_type = "hybrid_gpt"
    # standard-name aliases so transformers utilities (cache setup,
    # generation, eval harnesses) can read the architecture
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_key_value_heads": "n_kv_head",
        "max_position_embeddings": "block_size",
    }

    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layer=16,
        n_head=16,
        n_kv_head=4,
        n_embd=1024,
        ffn_hidden=2816,
        dropout=0.0,
        rope_theta=10000.0,
        window=64,
        global_every=4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.ffn_hidden = ffn_hidden
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.window = window
        self.global_every = global_every
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)

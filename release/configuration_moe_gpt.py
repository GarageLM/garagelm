"""HF config for the hybrid local+global GPT with a top-2 MoE FFN
(llm-arch-explore, milestone 13).

Uploaded alongside modeling_moe_gpt.py; loaded via trust_remote_code.
"""

from transformers import PretrainedConfig


class MoEGPTConfig(PretrainedConfig):
    model_type = "moe_gpt"
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
        n_layer=12,
        n_head=12,
        n_kv_head=4,
        n_embd=768,
        ffn_hidden=1024,      # PER-EXPERT SwiGLU hidden dim
        n_expert=8,
        experts_per_token=2,   # research config calls this top_k (see below)
        capacity_factor=1.25,  # training-time fixed expert capacity (record)
        aux_loss_coeff=0.01,   # training-time load-balance loss (record; unused here)
        dropless=True,         # inference: route every token (see modeling file)
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
        self.n_expert = n_expert
        # The research config names this field `top_k`. That name collides with
        # transformers' generation parameter `top_k` (the GenerationConfig
        # validator reads it off the model config), so it is renamed here and
        # the research spelling is accepted on input.
        if "top_k" in kwargs:
            experts_per_token = kwargs.pop("top_k")
        self.experts_per_token = experts_per_token
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff
        self.dropless = dropless
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.window = window
        self.global_every = global_every
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)

from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 16
    n_head: int = 16
    n_kv_head: int = 4  # GQA: 16 query heads share 4 kv heads (group size 4)
    n_embd: int = 1024
    ffn_hidden: int = 2816  # SwiGLU hidden dim
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # Effective batch = micro_batch_size * grad_accum_steps = 32 sequences
    # of 512 tokens = 16,384 tokens/optimizer step. Micro-batch 8x512 stays
    # inside the MPS regime measured safe in 02 (batch32x512 triggered a
    # 63GB allocator pathology via the vocab-width logits tensor).
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    max_iters: int = 3000  # optimizer steps, not micro-steps
    eval_interval: int = 500
    eval_iters: int = 20
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    def as_dict(self):
        return asdict(self)

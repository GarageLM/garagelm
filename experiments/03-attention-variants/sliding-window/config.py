from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4  # same GQA width as 02 -- only the attention pattern varies here
    n_embd: int = 768
    ffn_hidden: int = 2048  # SwiGLU hidden dim
    dropout: float = 0.1
    rope_theta: float = 10000.0
    window: int = 64  # each query attends only to the last `window` positions (incl. itself)

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # identical to 02-baseline-small-lm's TrainConfig -- fair-comparison
    # rule: only the attention module (windowing here) may differ.
    batch_size: int = 16
    max_iters: int = 3000
    eval_interval: int = 500
    eval_iters: int = 30
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    def as_dict(self):
        return asdict(self)

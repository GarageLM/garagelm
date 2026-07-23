from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4  # GQA: 12 query heads share 4 kv heads (group size 3)
    n_embd: int = 768
    ffn_hidden: int = 2048  # SwiGLU hidden dim
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # batch/block sized from a measured MPS throughput probe: batch=32,
    # block=512 hit a pathological MPS allocator regime (63GB allocated,
    # 71.5s/iter) driven by the large vocab_size=50257 lm_head output
    # tensor; batch=16, block=256 measured a sane 848ms/iter at 14.7GB.
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

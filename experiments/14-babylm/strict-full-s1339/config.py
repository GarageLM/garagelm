from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024  # 2x 04's context; the long-range re-test needs room beyond the 64-token window
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768  # the proven 114M recipe from 02/03, at longer context
    ffn_hidden: int = 2048  # SwiGLU hidden dim
    dropout: float = 0.0  # one epoch, each token seen ~once: same rationale as 05/12b
    rope_theta: float = 10000.0

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # Identical across both 14-babylm arms of the strict track -- fair-comparison
    # rule. Only the attention module differs between hybrid and full.
    micro_batch_size: int = 4
    grad_accum_steps: int = 4  # 4 x 4 x 1024 = 16,384 tokens/optimizer step
    max_iters: int = 10041  # 1 epoch of strict (164,525,756 train tokens) at 16,384 tok/step
    eval_interval: int = 500
    eval_iters: int = 20
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1339

    def as_dict(self):
        return asdict(self)

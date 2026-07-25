from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    # The 05 winner (hybrid local+global) at the proven 232M recipe from 04.
    # 16 layers -> full attention at layers 3, 7, 11, 15; KV cache at T=1024
    # is (12*64 + 4*1024)/(16*1024) = ~30% of full attention's.
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 16
    n_head: int = 16
    n_kv_head: int = 4
    n_embd: int = 1024
    ffn_hidden: int = 2816  # SwiGLU hidden dim
    dropout: float = 0.0  # 500M-token budget from a 1.78B pool: sub-epoch, no regularization needed
    rope_theta: float = 10000.0
    window: int = 64
    global_every: int = 4

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # Same optimizer recipe proven at 232M in 04 and at this data/context in
    # 05 -- only the horizon changes. 30,500 steps x 16,384 tokens/step
    # = 1.0B tokens (~5.3 days at the 07-measured 7.4s/step).
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    max_iters: int = 18300  # ~1 epoch over the ~300M-token full-SmolTalk set
    eval_interval: int = 250
    eval_iters: int = 20
    learning_rate: float = 2e-5  # SFT: ~15x below pretrain peak
    min_lr: float = 2e-6
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    def as_dict(self):
        return asdict(self)

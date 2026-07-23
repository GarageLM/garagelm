import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim, max_seq_len, theta, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv_head, T, head_dim = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv_head, n_rep, T, head_dim)
    return x.reshape(B, n_kv_head * n_rep, T, head_dim)


def build_sliding_window_mask(block_size, window):
    """(T, T) bool mask, True where query i may attend to key j: causal
    (j <= i) AND within the window (i - j < window). Precomputed once at
    block_size and sliced to the actual T per forward.
    """
    i = torch.arange(block_size).unsqueeze(1)  # (T,1)
    j = torch.arange(block_size).unsqueeze(0)  # (1,T)
    causal = j <= i
    windowed = (i - j) < window
    return causal & windowed


class SlidingWindowAttention(nn.Module):
    """Same GQA structure as 02-baseline-small-lm's GQAAttention -- the
    only change is a fixed causal sliding-window mask instead of full
    causal attention (SDPA's attn_mask=, is_causal=False, since attn_mask
    and is_causal=True can't be combined).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, mask):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:T, :T], dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = SlidingWindowAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """Decoder-only GPT, identical to 02-baseline-small-lm except attention
    is restricted to a fixed sliding window (config.window) instead of full
    causal attention -- see SlidingWindowAttention / build_sliding_window_mask.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = build_sliding_window_mask(config.block_size, config.window)
        self.register_buffer("window_mask", mask, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence longer than block_size"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        mask = self.window_mask.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

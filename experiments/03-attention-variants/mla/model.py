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


def apply_rope_single(x, cos, sin):
    # x: (B, n_h, T, dim) where n_h may be 1 (shared, pre-expand) or n_head;
    # cos/sin: (T, dim) -- must be a table built for exactly this `dim`,
    # NOT a slice of a larger head_dim table (see mla/README.md).
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2-style), simplified: query is
    not compressed (only K/V are, since only K/V would be cached at
    inference). K/V are jointly compressed into a shared low-rank latent
    (kv_latent_dim, d_c) and reconstructed per-head; a small decoupled slice
    of each head (rope_dim) carries RoPE separately since rotation doesn't
    commute with the low-rank compression -- DeepSeek's actual design, not
    a simplification. The k_rope projection is intentionally SHARED across
    heads (one small vector per token position, broadcast to every head)
    while q_rope is per-head, so q_rope_h . k_rope still differs by head.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_dim = config.rope_dim
        self.nope_dim = self.head_dim - self.rope_dim
        self.d_c = config.kv_latent_dim
        self.dropout = config.dropout

        self.kv_down_proj = nn.Linear(config.n_embd, self.d_c, bias=False)
        self.k_up_proj = nn.Linear(self.d_c, self.n_head * self.nope_dim, bias=False)
        self.v_up_proj = nn.Linear(self.d_c, self.n_head * self.head_dim, bias=False)
        self.k_rope_proj = nn.Linear(config.n_embd, self.rope_dim, bias=False)  # shared across heads
        self.q_nope_proj = nn.Linear(config.n_embd, self.n_head * self.nope_dim, bias=False)
        self.q_rope_proj = nn.Linear(config.n_embd, self.n_head * self.rope_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos_r, sin_r):
        B, T, C = x.size()

        c_kv = self.kv_down_proj(x)  # (B,T,d_c) -- this is what a real KV-cache would store
        k_nope = self.k_up_proj(c_kv).view(B, T, self.n_head, self.nope_dim).transpose(1, 2)
        v = self.v_up_proj(c_kv).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_rope = self.k_rope_proj(x)  # (B,T,rope_dim), no head axis yet
        q_nope = self.q_nope_proj(x).view(B, T, self.n_head, self.nope_dim).transpose(1, 2)
        q_rope = self.q_rope_proj(x).view(B, T, self.n_head, self.rope_dim).transpose(1, 2)

        q_rope = apply_rope_single(q_rope, cos_r, sin_r)  # (B,n_head,T,rope_dim)
        k_rope = apply_rope_single(k_rope.unsqueeze(1), cos_r, sin_r)  # (B,1,T,rope_dim)
        k_rope = k_rope.expand(B, self.n_head, T, self.rope_dim)  # explicit expand, torch.cat won't broadcast

        q = torch.cat([q_nope, q_rope], dim=-1)  # (B,n_head,T,head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B,n_head,T,head_dim)

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
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
        self.attn = MLAAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos_r, sin_r):
        x = x + self.attn(self.attn_norm(x), cos_r, sin_r)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """Decoder-only GPT, identical to 02-baseline-small-lm except attention
    is Multi-head Latent Attention (MLA) instead of GQA -- see
    MLAAttention. Only a small rope_dim-width RoPE table is needed here
    (not a head_dim=64 table): RoPE is only ever applied to the decoupled
    rope slice, never to the nope/content slice reconstructed from the
    compressed latent.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        cos_r, sin_r = precompute_rope(config.rope_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos_r, persistent=False)
        self.register_buffer("rope_sin", sin_r, persistent=False)

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
        cos_r = self.rope_cos[:T].to(x.device)
        sin_r = self.rope_sin[:T].to(x.device)
        for block in self.blocks:
            x = block(x, cos_r, sin_r)
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

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


def apply_rope_single(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv_head, T, head_dim = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv_head, n_rep, T, head_dim)
    return x.reshape(B, n_kv_head * n_rep, T, head_dim)


def build_nsa_mask(T, window, compress_block, device=None):
    """(T, T + n_blocks) bool mask. Dense columns: causal sliding window.
    Compressed columns (one per complete block of `compress_block` tokens):
    causal at block granularity (`block_last_pos < i`) -- deliberately NOT
    excluding the window range, so the 1-2 newest compressed blocks may
    overlap the dense window. A stricter "compressed block must be fully
    outside the window" rule was tried and rejected: the window boundary
    slides by 1 each step while the compression grid is fixed in steps of
    `compress_block`, which leaves a silent coverage gap of up to
    compress_block-ish tokens that neither branch can see. Allowing
    overlap is what real NSA-style implementations do -- one joint softmax
    handles the redundancy fine.
    """
    n_blocks = T // compress_block
    i = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
    j = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
    dense_mask = (j <= i) & ((i - j) < window)
    if n_blocks == 0:
        return dense_mask, 0
    b = torch.arange(n_blocks, device=device).unsqueeze(0)  # (1,n_blocks)
    block_last_pos = b * compress_block + (compress_block - 1)
    compressed_mask = block_last_pos < i  # (T,1) vs (1,n_blocks) -> (T,n_blocks)
    return torch.cat([dense_mask, compressed_mask], dim=1), n_blocks


class NSALiteAttention(nn.Module):
    """Simplified 2-branch Native Sparse Attention: local sliding-window +
    compressed/pooled global context. Drops NSA's trainable top-k selective
    branch entirely (needs differentiable top-k machinery, out of scope).
    Same GQA KV width as 02-baseline-small-lm/sliding-window -- only the
    attention pattern varies.

    K/V for the compressed branch are mean-pooled BEFORE RoPE is applied
    (averaging already-rotated vectors from different positions has no
    well-defined position and destroys high-frequency RoPE channels via
    destructive interference) -- pool the raw projection, then RoPE the
    pooled vector once per block at a representative position (block
    start).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.compress_block = config.compress_block
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, compress_cos, compress_sin, mask, n_blocks):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_raw = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v_raw = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k_dense = apply_rope(q, k_raw, cos, sin)
        k_dense = repeat_kv(k_dense, self.n_rep)
        v_dense = repeat_kv(v_raw, self.n_rep)

        if n_blocks > 0:
            usable = n_blocks * self.compress_block
            k_blocks = k_raw[:, :, :usable, :].reshape(
                B, self.n_kv_head, n_blocks, self.compress_block, self.head_dim
            )
            v_blocks = v_raw[:, :, :usable, :].reshape(
                B, self.n_kv_head, n_blocks, self.compress_block, self.head_dim
            )
            k_compressed = apply_rope_single(k_blocks.mean(dim=3), compress_cos, compress_sin)
            v_compressed = v_blocks.mean(dim=3)
            k = torch.cat([k_dense, repeat_kv(k_compressed, self.n_rep)], dim=2)
            v = torch.cat([v_dense, repeat_kv(v_compressed, self.n_rep)], dim=2)
        else:
            k, v = k_dense, v_dense

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
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
        self.attn = NSALiteAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, compress_cos, compress_sin, mask, n_blocks):
        x = x + self.attn(self.attn_norm(x), cos, sin, compress_cos, compress_sin, mask, n_blocks)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """Decoder-only GPT, identical to 02-baseline-small-lm except attention
    is the simplified 2-branch NSA-lite pattern instead of full causal --
    see NSALiteAttention / build_nsa_mask. Mask and compressed-position
    tables are (re)computed per forward call from the actual sequence
    length T, not just a T==block_size fast path: benchmarks/run_quality_eval.py
    calls this model with arbitrary (usually much shorter) T, and getting
    that path wrong would silently corrupt quality numbers, not just crash.
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
        device = x.device
        cos = self.rope_cos[:T].to(device)
        sin = self.rope_sin[:T].to(device)

        mask, n_blocks = build_nsa_mask(T, self.config.window, self.config.compress_block, device=device)
        if n_blocks > 0:
            block_starts = torch.arange(
                0, n_blocks * self.config.compress_block, self.config.compress_block, device=device
            )
            compress_cos = self.rope_cos[block_starts].to(device)
            compress_sin = self.rope_sin[block_starts].to(device)
        else:
            compress_cos = compress_sin = None

        for block in self.blocks:
            x = block(x, cos, sin, compress_cos, compress_sin, mask, n_blocks)
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

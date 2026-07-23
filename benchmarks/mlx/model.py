"""MLX inference port of this repo's GPT (RoPE + RMSNorm + SwiGLU + GQA),
covering all three attention variants from a single class:

- full attention        (config has no `window`)          -> every layer global
- pure sliding window   (`window`, no `global_every`)     -> every layer local
- hybrid local+global   (`window` + `global_every`)       -> every Nth layer global

Inference uses real per-layer KV caches: global layers append forever
(FullCache), local layers keep only the last window-1 entries
(RotatingCache) -- the bounded-memory property this project's efficiency
claims rest on, measured directly via .nbytes().

Weight names deliberately mirror the PyTorch module tree 1:1 (tok_emb,
blocks.N.attn.q_proj, ...), so convert.py is a rename-free dump. The LM
head reuses the embedding matrix via Embedding.as_linear (weight tying),
matching PyTorch's tied tok_emb/lm_head.

RoPE is computed manually (cos/sin table + rotate_half) to match the
PyTorch implementation exactly rather than trusting nn.RoPE conventions;
logit parity against PyTorch is gated in convert.py --parity.
"""

import math

import mlx.core as mx
import mlx.nn as nn


def precompute_rope(head_dim, max_seq_len, theta):
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)  # (T, head_dim/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return mx.cos(emb), mx.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    # x: (B, heads, T, head_dim); cos/sin: (T, head_dim)
    return x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]


class FullCache:
    """Append-only KV cache for global (full-attention) layers."""

    def __init__(self):
        self.k = None
        self.v = None
        self.offset = 0  # absolute position of the next token

    def update(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        self.offset += k.shape[2]
        return self.k, self.v

    def nbytes(self):
        return 0 if self.k is None else self.k.nbytes + self.v.nbytes


class RotatingCache:
    """Bounded KV cache for sliding-window layers: after each update only the
    last window-1 entries are kept, so a decode step (T=1) attends to exactly
    the window-1 previous positions plus itself = the window.

    Multi-token updates are only valid on an empty cache (single prefill) --
    the banded mask handles intra-prefill windowing, and the trim afterwards
    leaves the correct decode state.
    """

    def __init__(self, window):
        self.window = window
        self.k = None
        self.v = None
        self.offset = 0

    def update(self, k, v):
        T = k.shape[2]
        if self.k is None:
            ks, vs = k, v
        else:
            assert T == 1, "RotatingCache only supports T>1 on an empty cache (prefill)"
            ks = mx.concatenate([self.k, k], axis=2)
            vs = mx.concatenate([self.v, v], axis=2)
        self.offset += T
        keep = self.window - 1
        self.k = ks[:, :, -keep:, :] if ks.shape[2] > keep else ks
        self.v = vs[:, :, -keep:, :] if vs.shape[2] > keep else vs
        return ks, vs

    def nbytes(self):
        return 0 if self.k is None else self.k.nbytes + self.v.nbytes


def additive_mask(T, window, offset, dtype):
    """(T, offset+T) additive mask for a prefill of T tokens at position
    offset: causal, and banded when window is not None."""
    i = mx.arange(offset, offset + T)[:, None]
    j = mx.arange(0, offset + T)[None, :]
    allowed = j <= i
    if window is not None:
        allowed = allowed & ((i - j) < window)
    return mx.where(allowed, mx.array(0.0, dtype), mx.array(-math.inf, dtype))


class Attention(nn.Module):
    def __init__(self, cfg, is_global):
        super().__init__()
        self.is_global = is_global
        self.n_head = cfg["n_head"]
        self.n_kv_head = cfg["n_kv_head"]
        self.head_dim = cfg["n_embd"] // cfg["n_head"]
        self.scale = self.head_dim ** -0.5
        n_embd = cfg["n_embd"]
        self.q_proj = nn.Linear(n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, n_embd, bias=False)

    def __call__(self, x, cos, sin, cache, mask):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE at absolute positions offset..offset+T-1 (cache.offset is
        # pre-update); keys in the cache were rotated when they were added.
        pos_cos = cos[cache.offset:cache.offset + T]
        pos_sin = sin[cache.offset:cache.offset + T]
        q = apply_rope(q, pos_cos, pos_sin)
        k = apply_rope(k, pos_cos, pos_sin)

        k, v = cache.update(k, v)
        # mx.fast SDPA handles GQA natively (n_kv_head < n_head)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["n_embd"], cfg["ffn_hidden"], bias=False)
        self.up_proj = nn.Linear(cfg["n_embd"], cfg["ffn_hidden"], bias=False)
        self.down_proj = nn.Linear(cfg["ffn_hidden"], cfg["n_embd"], bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg, is_global):
        super().__init__()
        self.is_global = is_global
        self.attn_norm = nn.RMSNorm(cfg["n_embd"], eps=1e-5)
        self.attn = Attention(cfg, is_global)
        self.ffn_norm = nn.RMSNorm(cfg["n_embd"], eps=1e-5)
        self.mlp = SwiGLU(cfg)

    def __call__(self, x, cos, sin, cache, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


def layer_is_global(cfg, i):
    if cfg.get("window") is None:
        return True
    ge = cfg.get("global_every")
    return ge is not None and i % ge == ge - 1


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.window = cfg.get("window")
        head_dim = cfg["n_embd"] // cfg["n_head"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.blocks = [Block(cfg, layer_is_global(cfg, i)) for i in range(cfg["n_layer"])]
        self.norm_f = nn.RMSNorm(cfg["n_embd"], eps=1e-5)
        cos, sin = precompute_rope(head_dim, cfg["block_size"], cfg["rope_theta"])
        self._cos, self._sin = cos, sin

    def make_cache(self):
        return [FullCache() if b.is_global else RotatingCache(self.window) for b in self.blocks]

    def __call__(self, idx, caches):
        B, T = idx.shape
        offset = caches[0].offset
        assert offset + T <= self.cfg["block_size"], "sequence longer than block_size"
        x = self.tok_emb(idx)
        if T > 1:
            global_mask = additive_mask(T, None, offset, x.dtype)
            local_mask = (additive_mask(T, self.window, offset, x.dtype)
                          if self.window is not None else global_mask)
        else:
            global_mask = local_mask = None  # decode: every cached entry is attendable
        cos = self._cos.astype(x.dtype)
        sin = self._sin.astype(x.dtype)
        for block, cache in zip(self.blocks, caches):
            x = block(x, cos, sin, cache, global_mask if block.is_global else local_mask)
        x = self.norm_f(x)
        return self.tok_emb.as_linear(x)  # weight-tied LM head

    def kv_bytes(self, caches):
        return sum(c.nbytes() for c in caches)

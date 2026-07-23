"""Self-contained HF transformers wrapper for the hybrid local+global GPT
from https://github.com/trevino293/llm-arch-explore.

Architecture: decoder-only, RoPE (rotate-half), RMSNorm, SwiGLU, GQA;
sliding-window attention on most layers with full causal attention every
`global_every`-th layer (layers 3/7/11/15 at 16 layers). Weight names match
the research repo 1:1, so checkpoints convert without remapping.

Notes for users:
- `use_cache` is not implemented (no KV cache in this wrapper); generation
  re-runs the full prefix each step. Fine for evaluation and small-scale
  use; for fast on-device decoding see the MLX build in the repo.
- Labels follow the HF convention (same as input_ids; shifted internally).
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_hybrid_gpt import HybridGPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim, max_seq_len, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)


def build_sliding_window_mask(block_size, window):
    i = torch.arange(block_size).unsqueeze(1)
    j = torch.arange(block_size).unsqueeze(0)
    return (j <= i) & ((i - j) < window)


class HybridAttention(nn.Module):
    def __init__(self, config, is_global):
        super().__init__()
        self.is_global = is_global
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

    def forward(self, x, cos, sin, window_mask):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        drop = self.dropout if self.training else 0.0
        if self.is_global:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=window_mask[:T, :T], dropout_p=drop, is_causal=False
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config, is_global):
        super().__init__()
        self.is_global = is_global
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = HybridAttention(config, is_global)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, window_mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, window_mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class HybridGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HybridGPTConfig
    _tied_weights_keys = {"lm_head.weight": "tok_emb.weight"}

    def __init__(self, config: HybridGPTConfig):
        super().__init__(config)
        head_dim = config.n_embd // config.n_head
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            Block(config, is_global=(i % config.global_every == config.global_every - 1))
            for i in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_theta)
        # persistent=True is load-bearing: from_pretrained builds the model on
        # the meta device, so buffers computed in __init__ are NOT materialized
        # -- they must round-trip through the checkpoint file.
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)
        mask = build_sliding_window_mask(config.block_size, config.window)
        self.register_buffer("window_mask", mask, persistent=True)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        B, T = input_ids.size()
        assert T <= self.config.block_size, "sequence longer than block_size"
        x = self.drop(self.tok_emb(input_ids))
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        mask = self.window_mask.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # no KV cache: re-run the (block_size-cropped) full prefix each step
        return {"input_ids": input_ids[:, -self.config.block_size:]}

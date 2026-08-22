"""Self-contained HF transformers wrapper for the hybrid local+global GPT
with a sparse top-2 Mixture-of-Experts FFN, from
https://github.com/trevino293/llm-arch-explore (milestone 13-moe).

Architecture: decoder-only, RoPE (rotate-half), RMSNorm, GQA;
sliding-window attention on most layers with full causal attention every
`global_every`-th layer (layers 3/7/11 at 12 layers). The FFN of every
layer is 8 SwiGLU experts (hidden 1024 each) with a linear router,
softmax top-2 selection, and gates renormalized over the selected two.
Weight names match the research repo 1:1, so checkpoints convert without
remapping.

Routing mode. The research model trained with a FIXED per-expert capacity
(capacity_factor x N x top_k / n_expert tokens per expert per forward,
overflow dropped) because the MPS allocator needs static tensor shapes.
Over the whole run the measured drop rate was 0.0004, so the capacity
ceiling never shaped the learned function. At inference that rule is a
liability: capacity scales with the number of tokens in the forward, so a
short prompt would drop most tokens from expert compute. This wrapper
therefore defaults to `config.dropless = True` (every token reaches both
of its experts). Set `config.dropless = False` to reproduce the research
model's capacity behaviour exactly (the release smoke test does this for
the logit-parity gate).

Notes for users:
- `use_cache` is not implemented (no KV cache in this wrapper); generation
  re-runs the full prefix each step. The expert dispatch is a plain
  per-expert gather loop, written for clarity, not speed.
- Labels follow the HF convention (same as input_ids; shifted internally).
  The training-time load-balance auxiliary loss is NOT added: the returned
  loss is plain cross-entropy, matching the repo's validation protocol.
- `blocks.N.mlp.ema_load` / `ema_drop` are the end-of-training router
  health buffers (per-expert EMA load fraction, EMA drop rate), shipped
  for the record. They do not affect the forward pass.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_moe_gpt import MoEGPTConfig


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


class ExpertSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SwiGLU(nn.Module):
    """MoE-SwiGLU (class name kept as SwiGLU so the checkpoint tree matches
    the research model: blocks.N.mlp.{router,experts.E.*,ema_load,ema_drop})."""

    def __init__(self, config):
        super().__init__()
        self.config = config  # read dropless / capacity at forward time
        self.n_expert = config.n_expert
        self.top_k = config.experts_per_token
        self.router = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList([ExpertSwiGLU(config) for _ in range(config.n_expert)])
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("ema_load", torch.full((config.n_expert,), 1.0 / config.n_expert),
                             persistent=True)
        self.register_buffer("ema_drop", torch.zeros(()), persistent=True)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        N = flat.size(0)
        logits = self.router(flat)                        # (N, E)
        probs = F.softmax(logits.float(), dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)     # (N, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)   # renormalize gates

        capacity = None
        if not getattr(self.config, "dropless", True):
            capacity = int(self.config.capacity_factor * N * self.top_k / self.n_expert)

        out = torch.zeros_like(flat)
        for e in range(self.n_expert):
            gate = (top_p * (top_i == e)).sum(dim=-1)     # (N,) 0 where not routed
            routed = gate.nonzero(as_tuple=True)[0]
            if capacity is not None and routed.numel() > capacity:
                keep = gate[routed].topk(capacity).indices  # research rule: keep top gates
                routed = routed[keep]
            if routed.numel() == 0:
                continue
            g = gate[routed].to(flat.dtype).unsqueeze(1)
            y = self.experts[e](flat.index_select(0, routed)) * g
            out = out.index_add(0, routed, y)
        return self.dropout(out.reshape(B, T, C))


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


class MoEGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MoEGPTConfig
    _tied_weights_keys = {"lm_head.weight": "tok_emb.weight"}

    def __init__(self, config: MoEGPTConfig):
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
        cos = self.rope_cos[:T].to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin[:T].to(device=x.device, dtype=x.dtype)
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

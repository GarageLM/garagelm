"""Muon optimizer (Keller Jordan's nanoGPT-speedrun formulation): momentum
SGD whose update is orthogonalized per 2D weight matrix via Newton-Schulz
iteration. Applied ONLY to hidden 2D matrices (attention/MLP projections);
embeddings (tied lm_head) and RMSNorm gains stay on AdamW.

Newton-Schulz runs in fp32 on MPS -- 5 iterations of <=1024x2048 matmuls
per matrix per step is negligible next to the model forward/backward.
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximately project G onto the nearest orthogonal matrix
    (polar factor). Quintic Newton-Schulz, coefficients from the speedrun."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                assert p.ndim == 2, "Muon is defined for 2D weight matrices only"

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(g.float(), steps=group["ns_steps"])
                # scale per the speedrun: compensates tall/wide aspect ratios
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-group["lr"] * scale)

"""Frozen run configuration for the harness runner.

One YAML per run in experiments/15-harness/configs/. A control and its harness
arm differ in exactly one key (normally `k`). The config's sha is recorded in
every results JSON so a number can always be traced back to the exact
sampler / prompt / strategy that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Sampler:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    max_tokens: int = 32768
    enable_thinking: bool = True


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    task: str                       # aime | gpqa | math500 | humaneval | mbpp | arc
    model: str                      # HF id served by mlx_lm.server
    quant: str = "q4"
    strategy: str = "cons"          # cons | best_of_n | repair | arc_loop
    k: int = 8
    sampler: Sampler = field(default_factory=Sampler)
    task_args: Dict[str, Any] = field(default_factory=dict)
    prompt_template: str = "default"
    server_url: str = "http://127.0.0.1:8421/v1"
    concurrency: int = 8            # requests in flight = server decode concurrency
    seed: int = 1234                # subsampling / bootstrap only; NEVER sent to the server
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def sha(self) -> str:
        d = self.to_dict()
        d.pop("notes", None)
        d.pop("run_id", None)
        d.pop("server_url", None)
        d.pop("concurrency", None)   # plumbing, not semantics
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def model_short(self) -> str:
        """'mlx-community/Qwen3.5-9B-4bit' -> 'qwen3.5-9b'."""
        name = self.model.split("/")[-1].lower()
        for suf in ("-4bit", "-8bit", "-bf16", "-fp16", "-6bit", "-5bit", "-3bit"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        return name

    def result_stem(self) -> str:
        strat = self.strategy if self.k == 1 else f"{self.strategy}{self.k}"
        if self.k == 1 and self.strategy == "cons":
            strat = "k1"
        return f"harness-{self.task}-{self.model_short()}-{self.quant}-{strat}"


def load_config(path: str) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    sampler = Sampler(**raw.pop("sampler", {}) or {})
    return RunConfig(sampler=sampler, **raw)


def dump_config(cfg: RunConfig, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

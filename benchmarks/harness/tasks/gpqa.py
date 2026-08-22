"""GPQA Diamond (Idavidrein/gpqa, gated: accept the terms on the Hub once).
Choice order is shuffled deterministically per item (sha of the question),
unlike lm-eval's unseeded random.shuffle, so reruns grade identically."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

from .base import Item

PROMPT = ("What is the correct answer to this question:\n{q}\n\nChoices:\n"
          "(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\n"
          "Think it through, then give your final answer as 'Answer: (X)' on its own line.")


def _pre(t):
    return " " if t is None else t.strip().replace(" [title]", ". ").replace("  ", " ")


def load(task_args: Dict[str, Any]) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", task_args.get("config", "gpqa_diamond"), split="train")
    items = []
    for i, row in enumerate(ds):
        q = _pre(row["Question"])
        choices = [_pre(row["Incorrect Answer 1"]), _pre(row["Incorrect Answer 2"]),
                   _pre(row["Incorrect Answer 3"]), _pre(row["Correct Answer"])]
        rng = random.Random(int(hashlib.sha256(q.encode()).hexdigest(), 16) % (2**32))
        rng.shuffle(choices)
        ans = "ABCD"[choices.index(_pre(row["Correct Answer"]))]
        rid = row.get("Record ID") or f"{i:03d}"
        items.append(Item(id=f"gpqa-{rid}", target=ans,
                          messages=[{"role": "user", "content": PROMPT.format(q=q, a=choices[0], b=choices[1], c=choices[2], d=choices[3])}],
                          meta={"domain": row.get("High-level domain")}))
    return items

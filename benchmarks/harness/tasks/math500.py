"""MATH-500 (HuggingFaceH4/MATH-500): quant gate, smoke tests, calibration."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Item, MATH_SUFFIX


def load(task_args: Dict[str, Any]) -> List[Item]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n = task_args.get("limit")
    levels = task_args.get("levels")          # e.g. [1, 2] for a short-trace smoke slice
    items = []
    for row in ds:
        if levels is not None and row.get("level") not in levels:
            continue
        if n is not None and len(items) >= n:
            break
        items.append(Item(id=f"math500-{row.get('unique_id', len(items))}".replace("/", "_"),
                          target=str(row["answer"]).strip(),
                          messages=[{"role": "user", "content": row["problem"].strip() + MATH_SUFFIX}],
                          meta={"level": row.get("level"), "subject": row.get("subject")}))
    return items

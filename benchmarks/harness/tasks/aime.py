"""AIME 2024 / 2025 / 2026 as chat items. Dataset ids follow lm-eval's
`aime24` / `aime25` tasks; 2026 is pinned via task_args['aime26_dataset']."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Item, MATH_SUFFIX

DATASETS = {2024: ("Maxwell-Jia/AIME_2024", None, "train"),
            2025: ("math-ai/aime25", None, "test")}


def _field(row: Dict[str, Any], *names):
    for n in names:
        for k in row:
            if k.lower() == n.lower():
                return row[k]
    raise KeyError(f"none of {names} in {list(row)}")


def load(task_args: Dict[str, Any]) -> List[Item]:
    from datasets import load_dataset
    years = task_args.get("years", [2024, 2025])
    items: List[Item] = []
    for y in years:
        if y in DATASETS:
            path, name, split = DATASETS[y]
        else:
            spec = task_args.get(f"aime{str(y)[-2:]}_dataset")
            if not spec:
                raise SystemExit(f"AIME {y}: pin task_args.aime{str(y)[-2:]}_dataset (path[:config][@split])")
            path, _, rest = spec.partition(":")
            name, _, split = rest.partition("@") if rest else (None, None, "test")
            name = name or None
            split = split or "test"
        ds = load_dataset(path, name, split=split)
        for i, row in enumerate(ds):
            problem = _field(row, "problem", "question")
            answer = str(_field(row, "answer", "solution"))
            rid = str(row.get("ID", row.get("id", f"{y}-{i:02d}")))
            items.append(Item(id=f"aime{y}-{rid}", target=answer.strip(),
                              messages=[{"role": "user", "content": problem.strip() + MATH_SUFFIX}],
                              meta={"year": y, "source": path}))
    return items

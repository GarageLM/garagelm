from typing import Any, Dict, List

from .base import Item


def load_items(task: str, task_args: Dict[str, Any]) -> List[Item]:
    if task == "aime":
        from . import aime as m
    elif task == "math500":
        from . import math500 as m
    elif task == "gpqa":
        from . import gpqa as m
    else:
        raise SystemExit(f"unknown task {task!r}")
    return m.load(task_args)


def grader_for(task: str):
    if task in ("aime", "math500"):
        from ..graders import math as g
    elif task == "gpqa":
        from ..graders import mcq as g
    else:
        raise SystemExit(f"no grader for task {task!r}")
    return g

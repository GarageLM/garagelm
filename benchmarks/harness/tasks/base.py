from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Item:
    id: str
    messages: List[Dict[str, str]]   # chat messages sent to the model
    target: str                      # grader target
    meta: Dict[str, Any] = field(default_factory=dict)


MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

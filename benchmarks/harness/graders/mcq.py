"""Multiple-choice letter grader (GPQA-Diamond). Extraction order:
'Answer: (X)' / 'Answer: X' -> \\boxed{X} -> 'answer is (X)' -> the last
standalone '(X)'. Only letters in the allowed set count."""

from __future__ import annotations

import re
from typing import Optional, Tuple

GRADER_VERSION = "mcq-1.0"


def answer_part(text: str) -> str:
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def extract(text: str, letters: str = "ABCD") -> Optional[str]:
    body = answer_part(text or "")
    L = f"[{letters}]"
    pats = [
        rf"(?i)answer\s*(?:is|:)?\s*\**\s*\(?\s*({L})\s*\)?",
        rf"\\boxed\{{\s*\(?\s*({L})\s*\)?\s*\}}",
        rf"(?i)\bcorrect (?:answer|choice|option) is\s*\(?\s*({L})\s*\)?",
        rf"\(({L})\)",
    ]
    for p in pats:
        found = re.findall(p, body)
        if found:
            return found[-1].upper()
    # last line that is a bare letter
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    if lines and re.fullmatch(rf"\**\s*\(?({L})\)?\s*\**\.?", lines[-1]):
        return re.sub(r"[^A-Z]", "", lines[-1].upper())[-1]
    return None


def normalize(pred: Optional[str]) -> Optional[str]:
    return pred.upper() if pred else None


def grade(text: str, target: str, letters: str = "ABCD") -> Tuple[bool, Optional[str]]:
    t = re.sub(r"[^A-Z]", "", str(target).upper())
    pred = extract(text, letters)
    return (pred is not None and pred == t), pred

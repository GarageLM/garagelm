"""Exact-match math grader. Normalisation is imported from lm-eval's
hendrycks_math utils (the same code lm-eval's `aime25` / `minerva_math` use),
not rewritten; this module only decides WHICH span of the response to grade
and adds a numeric-equivalence fallback (AIME answers are integers, and
"042" vs "42" must agree)."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from lm_eval.tasks.hendrycks_math.utils import (  # noqa: F401  (re-exported)
    is_equiv, last_boxed_only_string, remove_boxed, strip_string)

GRADER_VERSION = "math-1.0"

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def answer_part(text: str) -> str:
    """The part of a response after the last </think>, if any."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def extract(text: str) -> Optional[str]:
    """\\boxed{...} (last) -> last $...$ span -> last number in the final
    non-empty line -> None."""
    body = answer_part(text or "")
    boxed = last_boxed_only_string(body)
    if boxed is not None:
        try:
            inner = remove_boxed(boxed)
            if inner is not None and inner.strip():
                return inner.strip()
        except (AssertionError, IndexError):
            pass
    spans = re.findall(r"\$([^$]+)\$", body)
    if spans:
        return spans[-1].strip()
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if lines:
        m = re.search(r"(?i)answer\s*(?:is|:)?\s*\**\s*(-?\d+(?:\.\d+)?)", lines[-1])
        if m:
            return m.group(1)
        nums = _NUM_RE.findall(lines[-1])
        if nums:
            return nums[-1]
    return None


def _as_number(s: str):
    t = strip_string(s).replace(",", "")
    t = t.replace("\\%", "").replace("%", "")
    try:
        if re.fullmatch(r"-?\d+", t):
            return int(t)
        return float(t)
    except (ValueError, TypeError):
        return None


_TEXT_RE = re.compile(r"^\\(?:text|mathrm|textbf)\{(.*)\}$")


def _unwrap_text(s: str) -> str:
    s = s.strip()
    m = _TEXT_RE.match(s)
    return m.group(1).strip() if m else s


def equivalent(pred: Optional[str], target: str) -> bool:
    if pred is None:
        return False
    pred, target = _unwrap_text(pred), _unwrap_text(str(target))
    if is_equiv(pred, str(target)):
        return True
    a, b = _as_number(pred), _as_number(str(target))
    if a is not None and b is not None:
        return abs(a - b) < 1e-9 if isinstance(a, float) or isinstance(b, float) else a == b
    return False


def normalize(pred: Optional[str]) -> Optional[str]:
    """Canonical form for voting: numbers to int/float string, else strip_string."""
    if pred is None:
        return None
    pred = _unwrap_text(pred)
    n = _as_number(pred)
    if n is not None:
        return str(n)
    try:
        return strip_string(pred)
    except Exception:
        return pred.strip()


def grade(text: str, target: str) -> Tuple[bool, Optional[str]]:
    pred = extract(text)
    return equivalent(pred, target), pred

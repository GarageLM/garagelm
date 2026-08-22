"""Plain-python self-test for the harness substrate (no pytest in this env).

  uv run python benchmarks/harness/selftest.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from benchmarks.harness.config import RunConfig, Sampler          # noqa: E402
from benchmarks.harness.graders import code_exec, math as gm, mcq  # noqa: E402
from benchmarks.harness.report import mcnemar_exact, wilson, paired_diff_ci  # noqa: E402
from benchmarks.harness.strategies.cons import cons_at_k_curve, majority, pass_at_k  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    print("[selftest] math grader")
    cases = [
        ("The answer is \\boxed{042}.", "42", True),
        ("...</think>\n\nSo $x = \\boxed{\\frac{1}{2}}$", "\\frac12", True),
        ("\\boxed{1,000}", "1000", True),
        ("\\boxed{-7}", "-7", True),
        ("\\boxed{\\text{(B)}}", "(B)", True),        # strip_string drops \\text
        ("\\boxed{[0, 1)}", "[0,1)", True),
        ("\\boxed{3.50}", "3.5", True),
        ("Final answer: 17", "17", True),              # no box: last-line number
        ("\\boxed{41}", "42", False),
        ("no answer here", "5", False),
        ("\\boxed{\\frac{3}{4}}", "0.75", False),       # not claimed equal (string grader)
    ]
    for text, target, want in cases:
        got, ex = gm.grade(text, target)
        check(f"math {text[:28]!r} vs {target!r}", got == want, f"-> {got} (extracted {ex!r})")
    check("math normalize 042==42", gm.normalize("042") == gm.normalize("42"))

    print("[selftest] mcq grader")
    for text, target, want in [("Answer: (C)", "C", True), ("...the answer is B.", "B", True),
                               ("\\boxed{D}", "(D)", True), ("I pick (A) because (B) is wrong. Answer: (A)", "A", True),
                               ("no letter", "A", False), ("Answer: (E)", "A", False)]:
        got, ex = mcq.grade(text, target)
        check(f"mcq {text[:30]!r}", got == want, f"-> {got} ({ex})")

    print("[selftest] code executor")
    r = code_exec.passes_tests("def f(x):\n    return x + 1\n", "assert f(1) == 2\nassert f(-1) == 0")
    check("code pass", r.ok)
    r = code_exec.passes_tests("def f(x):\n    return x + 2\n", "assert f(1) == 2")
    check("code fail", (not r.ok) and "AssertionError" in r.stderr)
    r = code_exec.run_python("while True:\n    pass\n", timeout_s=1.0)
    check("code timeout", r.timed_out and r.wall_s < 3.0, f"{r.wall_s:.1f}s")
    r = code_exec.run_python("import os; print(os.environ.get('HOME','?'))")
    check("code env scrubbed", r.ok and "harness-exec" in r.stdout)

    print("[selftest] consensus + stats")
    check("majority", majority(["42", "41", "42", None]) == ("42", 2))
    check("majority tie first-seen", majority(["1", "2", "2", "1"]) == ("1", 2))
    per = [(["42", "42", "41", "42"], [True, True, False, True]),
           (["7", "8", "9", "7"], [False] * 4),
           ([None, "3", "3", "4"], [False, True, True, False])]
    curve = cons_at_k_curve(per, [1, 2, 4])
    check("cons@4 = 2/3", abs(curve[4] - 2 / 3) < 1e-9, f"{curve}")
    check("cons@1 = avg@1", abs(curve[1] - (3 / 4 + 0 + 2 / 4) / 3) < 1e-9)
    check("pass@2 = (1 + 0 + 5/6)/3", abs(pass_at_k(per, 2) - (1 + 0 + 5 / 6) / 3) < 1e-9, f"{pass_at_k(per, 2):.4f}")
    lo, hi = wilson(0.5, 90)
    check("wilson 0.5/90 ~ [0.40,0.60]", abs(lo - 0.399) < 0.01 and abs(hi - 0.601) < 0.01)
    check("mcnemar(10,3) ~0.092", abs(mcnemar_exact(10, 3) - 0.0923) < 0.002)
    d, ci = paired_diff_ci([1, 1, 1, 1], [0, 0, 0, 0], 500)
    check("paired diff 1.0", d == 1.0 and ci == (1.0, 1.0))

    print("[selftest] config")
    a = RunConfig(run_id="a", task="aime", model="mlx-community/Qwen3.5-9B-4bit", k=8)
    b = RunConfig(run_id="b", task="aime", model="mlx-community/Qwen3.5-9B-4bit", k=8,
                  notes="different notes", concurrency=4)
    c = RunConfig(run_id="c", task="aime", model="mlx-community/Qwen3.5-9B-4bit", k=1)
    check("sha ignores notes/concurrency", a.sha() == b.sha())
    check("sha sees k", a.sha() != c.sha())
    check("stems", a.result_stem() == "harness-aime-qwen3.5-9b-q4-cons8" and c.result_stem() == "harness-aime-qwen3.5-9b-q4-k1", a.result_stem())
    check("sampler default thinking on", Sampler().enable_thinking is True)

    print(f"[selftest] {'ALL OK' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

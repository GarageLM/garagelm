"""Subprocess code executor for code lanes and ARC program synthesis.

Isolation is BEST-EFFORT on macOS (no container): a fresh interpreter with
`-I -B`, an empty environment, a temp working directory, CPU/address-space/
file-size rlimits and a wall-clock timeout. It is not a security boundary and
network is not blocked; the README says so. Do not run untrusted code from
outside this lab's own model outputs through it."""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

GRADER_VERSION = "code-exec-1.0"


@dataclass
class ExecResult:
    ok: bool
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    wall_s: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _limits(mem_bytes: int, cpu_s: int):
    def fn():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        except (ValueError, OSError):
            pass
    return fn


def run_python(code: str, stdin: str = "", timeout_s: float = 10.0,
               mem_bytes: int = 1 << 30, cpu_s: Optional[int] = None,
               max_output: int = 64_000) -> ExecResult:
    import time
    cpu_s = cpu_s or int(timeout_s) + 1
    with tempfile.TemporaryDirectory(prefix="harness-exec-") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": d,
               "PYTHONDONTWRITEBYTECODE": "1"}
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-I", "-B", path], input=stdin,
                               capture_output=True, text=True, timeout=timeout_s,
                               cwd=d, env=env, preexec_fn=_limits(mem_bytes, cpu_s))
            return ExecResult(ok=(p.returncode == 0), returncode=p.returncode,
                              stdout=p.stdout[-max_output:], stderr=p.stderr[-max_output:],
                              timed_out=False, wall_s=time.time() - t0)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"")
            err = (e.stderr or b"")
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
            err = err.decode(errors="replace") if isinstance(err, bytes) else err
            return ExecResult(ok=False, returncode=None, stdout=out[-max_output:],
                              stderr=err[-max_output:], timed_out=True, wall_s=time.time() - t0)


def passes_tests(program: str, tests: str, timeout_s: float = 10.0) -> ExecResult:
    """Run `program` followed by `tests` (assert statements / a check() call)
    in one interpreter. ok == all asserts passed."""
    return run_python(program + "\n\n" + tests + "\n", timeout_s=timeout_s)

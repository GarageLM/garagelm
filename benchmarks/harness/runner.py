"""Harness runner: strategy x task x client, per-sample traces, aggregate JSON.

  uv run python benchmarks/harness/runner.py --config experiments/15-harness/configs/aime-9b-q4-cons8.yaml
  uv run python benchmarks/harness/runner.py --config ... --limit 5 --smoke      # preflight smoke, not a result
  uv run python benchmarks/harness/runner.py --config ... --regrade              # re-aggregate from stored traces

Per-sample records go to <out-dir>/<run_id>.jsonl as they complete (resumable:
finished (item, sample) pairs are skipped on restart). The aggregate lands in
benchmarks/results/<stem>.json (collision-checked) and the traces are gzipped
into benchmarks/results/harness-traces/<run_id>.jsonl.gz. A progress.json
heartbeat is rewritten after every sample for the stall watchdog.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from benchmarks.harness.client import ChatClient                    # noqa: E402
from benchmarks.harness.config import RunConfig, load_config        # noqa: E402
from benchmarks.harness.report import (bootstrap_mean_ci, wilson)   # noqa: E402
from benchmarks.harness.strategies.cons import (avg_at_1, cons_at_k_curve,  # noqa: E402
                                                majority, pass_at_k)
from benchmarks.harness.tasks import grader_for, load_items        # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "benchmarks", "results")
TRACES_DIR = os.path.join(RESULTS_DIR, "harness-traces")


# ----------------------------------------------------------------------------- helpers
def git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return None


def server_pid() -> Optional[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", "mlx_lm.server"], text=True).split()
        return int(out[0]) if out else None
    except Exception:
        return None


class RssMonitor(threading.Thread):
    """Samples the server's RSS at 1Hz (unified-memory proxy; MLX exposes no
    exact peak from outside the process)."""

    def __init__(self, pid: Optional[int]):
        super().__init__(daemon=True)
        self.pid, self.peak_bytes, self._stop = pid, 0, threading.Event()

    def run(self):
        while self.pid and not self._stop.is_set():
            try:
                rss_kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.pid)],
                                                     text=True).strip() or 0)
                self.peak_bytes = max(self.peak_bytes, rss_kb * 1024)
            except Exception:
                pass
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()


def read_traces(path: str) -> List[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    recs = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def grade_record(grader, rec: Dict[str, Any], target: str) -> Dict[str, Any]:
    text = rec.get("text") or ""
    correct, extracted = grader.grade(text, target)
    truncated = rec.get("finish_reason") == "length"
    if truncated:
        correct = False                      # pre-registered: length == wrong
    rec["extracted"] = extracted
    rec["normalized"] = grader.normalize(extracted)
    rec["correct"] = bool(correct)
    rec["truncated"] = truncated
    rec["grader_version"] = grader.GRADER_VERSION
    return rec


# ----------------------------------------------------------------------------- aggregate
def aggregate(cfg: RunConfig, recs: List[Dict[str, Any]], items_by_id: Dict[str, Any],
              elapsed_s: Optional[float], peak_rss: Optional[int], smoke: bool) -> Dict[str, Any]:
    by_item: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        if r.get("error"):
            continue
        by_item.setdefault(r["item_id"], []).append(r)
    per_item: List[Tuple[List[Optional[str]], List[bool]]] = []
    item_ids = []
    for iid, rs in by_item.items():
        rs.sort(key=lambda r: r["sample_idx"])
        per_item.append(([r["normalized"] for r in rs], [r["correct"] for r in rs]))
        item_ids.append(iid)
    K = cfg.k
    ks = sorted({k for k in (1, 2, 4, 8, 16, 32) if k <= K} | {K})
    curve = cons_at_k_curve(per_item, ks, seed=cfg.seed) if per_item else {}
    cons_flags = []
    for norm, corr in per_item:
        sel, _ = majority(norm)
        cons_flags.append(float(sel is not None and any(c for a, c in zip(norm, corr) if a == sel)))
    n_items = len(per_item)
    acc = sum(cons_flags) / max(n_items, 1)
    comp = [r.get("completion_tokens") or 0 for r in recs if not r.get("error")]
    comp_item = [sum(r.get("completion_tokens") or 0 for r in rs) for rs in by_item.values()]
    walls = [r.get("wall_s") or 0.0 for r in recs if not r.get("error")]
    total_comp = sum(comp)
    n_correct_items = int(sum(cons_flags))
    out = {
        "experiment": cfg.result_stem(),
        "smoke": smoke,
        "run_id": cfg.run_id,
        "task": cfg.task,
        "task_args": cfg.task_args,
        "model": cfg.model,
        "model_short": cfg.model_short(),
        "quant": cfg.quant,
        "strategy": cfg.strategy,
        "k": K,
        "sampler": cfg.sampler.__dict__,
        "config_sha": cfg.sha(),
        "git_hash": git_hash(),
        "date": time.strftime("%Y-%m-%d"),
        "n_items": n_items,
        "n_samples": len(recs),
        "n_errors": sum(1 for r in recs if r.get("error")),
        "accuracy": acc,
        "accuracy_metric": f"cons@{K}" if K > 1 else "acc@1",
        "accuracy_ci": list(bootstrap_mean_ci(cons_flags, seed=cfg.seed)) if cons_flags else [0, 0],
        "accuracy_wilson": list(wilson(acc, n_items)),
        "avg_at_1": avg_at_1(per_item) if per_item else 0.0,
        "cons_at_k": {str(k): v for k, v in curve.items()},
        "pass_at_k_diagnostic": {str(k): pass_at_k(per_item, k) for k in ks} if per_item else {},
        "truncation_rate": (sum(1 for r in recs if r.get("truncated")) / max(len(recs), 1)),
        "completion_tokens_total": total_comp,
        "completion_tokens_per_sample_mean": statistics.fmean(comp) if comp else 0.0,
        "completion_tokens_per_sample_p95": (sorted(comp)[int(0.95 * (len(comp) - 1))] if comp else 0),
        "completion_tokens_per_item_mean": statistics.fmean(comp_item) if comp_item else 0.0,
        "completion_tokens_per_item_p95": (sorted(comp_item)[int(0.95 * (len(comp_item) - 1))] if comp_item else 0),
        "tokens_per_correct": (total_comp / n_correct_items) if n_correct_items else None,
        "wall_s_per_sample_mean": statistics.fmean(walls) if walls else 0.0,
        "elapsed_s": elapsed_s,
        "wall_s_per_item": (elapsed_s / n_items) if (elapsed_s and n_items) else None,
        "aggregate_tok_s": (total_comp / elapsed_s) if elapsed_s else None,
        "peak_rss_gb": (peak_rss / 1e9) if peak_rss else None,
        "per_item_cons_correct": {iid: bool(f) for iid, f in zip(item_ids, cons_flags)},
        "per_item_avg1": {iid: (sum(c) / len(c)) for iid, (_, c) in zip(item_ids, per_item)},
    }
    return out


# ----------------------------------------------------------------------------- run
async def run(cfg: RunConfig, out_dir: str, limit: Optional[int], smoke: bool,
              max_tokens_override: Optional[int]) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    items = load_items(cfg.task, cfg.task_args)
    if limit:
        items = items[:limit]
    items_by_id = {it.id: it for it in items}
    grader = grader_for(cfg.task)
    traces_path = os.path.join(out_dir, f"{cfg.run_id}.jsonl")
    done = set()
    if os.path.exists(traces_path):
        for r in read_traces(traces_path):
            if not r.get("error"):
                done.add((r["item_id"], r["sample_idx"]))
    todo = [(it, s) for it in items for s in range(cfg.k) if (it.id, s) not in done]
    print(f"[runner] {cfg.run_id}: {len(items)} items x k={cfg.k} = {len(items) * cfg.k} samples, "
          f"{len(done)} done, {len(todo)} to go", flush=True)

    client = ChatClient(cfg.server_url, cfg.concurrency)
    health = await client.health()
    served = [m.get("id") for m in health.get("data", [])]
    print(f"[runner] server models: {served}", flush=True)
    mon = RssMonitor(server_pid())
    mon.start()
    t_start = time.time()
    progress_path = os.path.join(out_dir, "progress.json")
    n_done = 0
    lock = asyncio.Lock()
    f = open(traces_path, "a")

    async def one(item, s):
        nonlocal n_done
        c = await client.complete(item.messages, cfg.sampler, max_tokens=max_tokens_override)
        rec = {"run_id": cfg.run_id, "task": cfg.task, "item_id": item.id, "sample_idx": s,
               "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
               "reasoning_chars": len(c.reasoning or ""), "finish_reason": c.finish_reason,
               "ttft_s": c.ttft_s, "wall_s": c.wall_s, "error": c.error,
               "text": c.text, "reasoning": c.reasoning, "ts": time.time()}
        if not c.error:
            grade_record(grader, rec, item.target)
        async with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done += 1
            with open(progress_path, "w") as pf:
                json.dump({"run_id": cfg.run_id, "done": n_done + len(done),
                           "total": len(items) * cfg.k, "elapsed_s": time.time() - t_start,
                           "ts": time.time()}, pf)
            if n_done % 10 == 0 or n_done == len(todo):
                el = time.time() - t_start
                print(f"[runner] {n_done}/{len(todo)} samples, {el / 60:.1f} min, "
                      f"last: item={item.id} s={s} tokens={c.completion_tokens} "
                      f"finish={c.finish_reason} correct={rec.get('correct')}", flush=True)

    try:
        await asyncio.gather(*(one(it, s) for it, s in todo))
    finally:
        f.close()
        mon.stop()
        await client.aclose()
    elapsed = time.time() - t_start
    recs = read_traces(traces_path)
    agg = aggregate(cfg, recs, items_by_id, elapsed if todo else None, mon.peak_bytes or None, smoke)
    agg["samples_this_session"] = len(todo)
    return agg


def write_results(agg: Dict[str, Any], cfg: RunConfig, traces_src: Optional[str],
                  force: bool, smoke: bool) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = cfg.result_stem() + ("-smoke" if smoke else "")
    path = os.path.join(RESULTS_DIR, stem + ".json")
    if os.path.exists(path) and not force and not smoke:
        raise SystemExit(f"refusing to overwrite {path} (pass --force)")
    with open(path, "w") as fh:
        json.dump(agg, fh, indent=2)
    if traces_src and not smoke:
        os.makedirs(TRACES_DIR, exist_ok=True)
        dst = os.path.join(TRACES_DIR, f"{cfg.run_id}.jsonl.gz")
        with open(traces_src, "rb") as fi, gzip.open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default=None, help="default experiments/15-harness/runs/<run_id>")
    p.add_argument("--limit", type=int, default=None, help="first N items only (smoke)")
    p.add_argument("--smoke", action="store_true", help="mark the output as a smoke test")
    p.add_argument("--max-tokens", type=int, default=None, help="override sampler.max_tokens (smoke only)")
    p.add_argument("--regrade", action="store_true", help="re-aggregate from stored traces, no generation")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out_dir or os.path.join(ROOT, "experiments", "15-harness", "runs", cfg.run_id)
    if args.limit and not args.smoke:
        raise SystemExit("--limit is only allowed with --smoke (full sets always; no --limit headline)")
    if args.max_tokens and not args.smoke:
        raise SystemExit("--max-tokens override is only allowed with --smoke")
    if args.regrade:
        traces = os.path.join(out_dir, f"{cfg.run_id}.jsonl")
        if not os.path.exists(traces):
            traces = os.path.join(TRACES_DIR, f"{cfg.run_id}.jsonl.gz")
        recs = read_traces(traces)
        grader = grader_for(cfg.task)
        items = {it.id: it for it in load_items(cfg.task, cfg.task_args)}
        for r in recs:
            if not r.get("error") and r["item_id"] in items:
                grade_record(grader, r, items[r["item_id"]].target)
        agg = aggregate(cfg, recs, items, None, None, args.smoke)
        agg["regraded"] = True
        path = write_results(agg, cfg, None, force=True, smoke=args.smoke)
    else:
        agg = asyncio.run(run(cfg, out_dir, args.limit, args.smoke, args.max_tokens))
        path = write_results(agg, cfg, os.path.join(out_dir, f"{cfg.run_id}.jsonl"),
                             args.force, args.smoke)
    print(json.dumps({k: agg[k] for k in ("experiment", "n_items", "n_samples", "accuracy",
                                           "accuracy_metric", "avg_at_1", "cons_at_k",
                                           "truncation_rate", "completion_tokens_per_item_mean",
                                           "aggregate_tok_s", "peak_rss_gb")}, indent=1))
    print(f"[runner] results -> {path}")


if __name__ == "__main__":
    main()

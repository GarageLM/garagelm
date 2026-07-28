import argparse
import json
import os
import sys
import time

import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")

MAIN_TASKS = ["hellaswag", "piqa", "arc_easy", "winogrande"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="experiments/02-baseline-small-lm")
    parser.add_argument("--tasks", default=None,
                         help="comma-separated task override (skips the MMLU stage); "
                              "default keeps the standard suite")
    parser.add_argument("--limit", type=int, default=300,
                         help="examples per task, for hellaswag/piqa/arc_easy/winogrande")
    parser.add_argument("--mmlu-limit", type=int, default=5,
                         help="examples per MMLU subject (57 subjects -> ~285 total by default; "
                              "'mmlu' is a 57-subject group task and lm-eval-harness applies --limit "
                              "per subtask, so a much smaller per-subject limit keeps total MMLU "
                              "compute comparable to the other four tasks)")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--gen-tokens", type=int, default=200,
                         help="tokens generated for the tokens/sec decode measurement")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    # Name results by the path under experiments/, not the bare basename --
    # nested variant dirs (e.g. 05-data-frontier/gqa) would otherwise
    # collide with other milestones' variants of the same name.
    parts = exp_dir.rstrip("/").split(os.sep)
    if "experiments" in parts:
        exp_name = "-".join(parts[parts.index("experiments") + 1:])
    else:
        exp_name = parts[-1]
    sys.path.insert(0, exp_dir)
    sys.path.insert(0, BENCH_DIR)

    from config import GPTConfig
    from model import GPT
    from lm_eval_adapter import GPTLMEvalAdapter
    import lm_eval

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(os.path.join(exp_dir, "out", "ckpt.pt"), map_location=device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = model.num_params()

    adapter = GPTLMEvalAdapter(model, model_cfg.block_size, device)
    print(f"evaluating {exp_name}: params={n_params:,} device={device} block_size={model_cfg.block_size}")

    main_tasks = args.tasks.split(",") if args.tasks else MAIN_TASKS
    print(f"running {main_tasks} with limit={args.limit}/task")
    main_results = lm_eval.simple_evaluate(
        model=adapter, tasks=main_tasks, num_fewshot=args.num_fewshot,
        limit=args.limit, log_samples=False,
    )

    if args.tasks is None:
        print(f"running mmlu (57 subjects) with limit={args.mmlu_limit}/subject")
        mmlu_results = lm_eval.simple_evaluate(
            model=adapter, tasks=["mmlu"], num_fewshot=args.num_fewshot,
            limit=args.mmlu_limit, log_samples=False,
        )
        quality = {**main_results["results"], **mmlu_results["results"]}
    else:
        quality = dict(main_results["results"])

    # lightweight tokens/sec (decode) measurement -- NOT the rigorous MLX
    # prefill/decode/TTFT/KV-cache benchmark benchmarks/README.md scopes
    # separately; just enough to avoid reporting a bare accuracy number.
    prompt_ids = adapter.tok_encode("Once upon a time")
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    t0 = time.time()
    with torch.no_grad():
        model.generate(inp, args.gen_tokens, temperature=0.8, top_k=50)
    tokens_per_sec = args.gen_tokens / (time.time() - t0)

    summary = {
        "experiment": exp_name,
        "params": n_params,
        "device": device,
        "block_size": model_cfg.block_size,
        "tokens_per_sec_decode": tokens_per_sec,
        "num_fewshot": args.num_fewshot,
        "limit_main_tasks": args.limit,
        "limit_mmlu_per_subject": args.mmlu_limit,
        "quality": quality,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "-ext" if args.tasks else ""
    out_path = os.path.join(RESULTS_DIR, f"{exp_name}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nresults written to {out_path}")
    print(f"tokens/sec (decode): {tokens_per_sec:.1f}")
    for name in [*main_tasks] + ([] if args.tasks else ["mmlu"]):
        res = quality.get(name, {})
        acc_keys = {k: v for k, v in res.items() if "acc" in k and "stderr" not in k}
        print(f"  {name}: {acc_keys}")


if __name__ == "__main__":
    main()

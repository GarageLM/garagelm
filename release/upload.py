"""Upload an exported model dir to the Hugging Face Hub.

Requires a logged-in token (`hf auth login`) with write access.

  uv run python release/upload.py --hf-dir release/hf/hybrid-gpt-232m --repo trevino293/hybrid-gpt-232m
"""

import argparse

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--repo", required=True, help="e.g. trevino293/hybrid-gpt-232m")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as: {who['name']}")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.hf_dir, repo_id=args.repo, repo_type="model",
                      ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
                      commit_message="upload from llm-arch-explore release tooling")
    print(f"uploaded -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

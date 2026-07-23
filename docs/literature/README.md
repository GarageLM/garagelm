# Literature

Research foundation for the attention-architecture work in `experiments/`. Read
in roughly this order; each topic should be written up with what problem it
solves, what it costs (compute, memory, quality), and where it breaks down at
scale or at the small model sizes this repo actually trains (see
`CLAUDE.md`).

Write-ups so far: topic 7 → `data-quality-frontier.md`.

1. **Attention foundations: MHA → MQA → GQA.** Why multi-query and
   grouped-query attention exist (KV-cache size at inference), and what
   quality they trade away vs. full multi-head attention. This is the
   baseline everything else in this repo is compared against.

2. **Multi-head Latent Attention (MLA).** Introduced in DeepSeek-V2: low-rank
   joint compression of keys/values into a shared latent space, reconstructed
   per-head at inference. Compares favorably to GQA on KV-cache size.
   - [DeepSeek-V2 paper](https://arxiv.org/pdf/2405.04434)
   - [FlashMLA (DeepSeek's MLA kernels)](https://github.com/deepseek-ai/FlashMLA)

3. **Sparse attention: NSA / DeepSeek Sparse Attention (V3.2), sliding-window
   lineage.** Native Sparse Attention decomposes dense attention into
   sliding-window + compressed + selective components. DeepSeek V3.2 pairs
   MLA (compress what's cached) with sparse attention (reduce what's
   revisited) as complementary, not competing, techniques. Sliding-window
   attention (Longformer/BigBird → Gemma2/3, Mistral) is a simpler,
   longer-established version of the same idea.
   - [DeepSeek Sparse Attention — Sebastian Raschka](https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/)
   - [Efficient Attention Mechanisms for LLMs: a Survey](https://arxiv.org/html/2507.19595v3)

4. **Linear attention / state-space (Mamba-family) hybrids.** Read for
   context on a possible later milestone — not an initial implementation
   target in `experiments/03-attention-variants/`.

5. **Reference small models** — size/quality/latency reference points, and
   codebases worth reading before writing this repo's own training loop:
   - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — single-GPU
     friendly, ~300-line model/train loop. Closest fit for local work here.
   - [karpathy/nanochat](https://github.com/karpathy/nanochat) — full
     pretrain→chat pipeline, but targets 8xH100; read for architecture/eval
     ideas (e.g. its optimizer and evaluation-pipeline design), not directly
     runnable on this hardware.
   - [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) — 20+
     architectures with production pretrain/finetune recipes.
   - SmolLM2 ([paper](https://arxiv.org/pdf/2502.02737)), Qwen3-0.6B,
     Gemma3, Pythia, TinyLlama — small open-weight models to compare against
     on the quality/size/latency axes this repo cares about.

6. **Benchmark methodology.** What eval tooling and tasks are actually
   meaningful at this repo's model sizes — see `benchmarks/README.md` for the
   concrete plan.
   - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

7. **The data-quality frontier.** Refined corpora as the main capability
   lever below ~1B params: phi/"Textbooks Are All You Need", FineWeb-Edu,
   SmolLM2/Cosmopedia, DCLM. Write-up: `data-quality-frontier.md` — the
   foundation for `experiments/05-data-frontier/`'s corpus choice.
   - [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   - [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
   - [SmolLM2 paper](https://arxiv.org/pdf/2502.02737)
   - [DCLM paper](https://arxiv.org/abs/2406.11794)

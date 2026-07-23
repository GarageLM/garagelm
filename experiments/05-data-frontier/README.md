# 05-data-frontier

The corpus upgrade and the deferred long-range attention question, run as
one experiment. Milestones 02–04 established that on TinyStories, attention
beyond ~64 local tokens buys nothing — but TinyStories has no long-range
structure to exploit. This milestone swaps in a refined corpus with real
long-form documents (grounding: `docs/literature/data-quality-frontier.md`)
and re-runs the attention showdown where full attention finally has
something a 64-token window can't see.

## Data

`data.py` builds a **1.777B-token pool** (uint16 gpt2-token bins, same
memmap format as 02/04), downloaded as plain parquet over HTTPS:

| Source | Share | Tokens |
|---|---|---|
| FineWeb-Edu `sample-10BT` (2 shards) | 84.5% | 1.503B |
| Cosmopedia-v2 (1 shard, synthetic textbooks) | 15.5% | 0.276B |

Val = 2M tokens held out from the tail of each source (stratified, same mix
as train). Perplexities on this corpus are **not comparable** to 02–04
(different data). The pool is ~3.5x the 07 flagship budget, so each token
is seen well under once per run — which is why dropout drops to 0.0 here.

## Models — fair-comparison rule, only the attention pattern differs

The proven 114M recipe (n_layer=12, n_head=12, n_kv_head=4, n_embd=768,
ffn_hidden=2048) at **block_size 1024** (the 64-token window now covers
1/16th of context). All three: 114,114,048 params exactly (masks carry no
parameters). 16,384 tokens/optimizer step (micro-batch 4 × accum 4), 6,100
steps = **100M tokens each**, seed 1337, identical TrainConfigs.

- `gqa/` — full-attention control (verbatim `02` model)
- `sliding-window/` — the incumbent winner (verbatim `03` model, w=64)
- `hybrid/` — **new**: sliding-window everywhere except layers 3/7/11,
  which run full attention (Gemma-2/3 lineage). At T=1024 its KV cache is
  (9·64 + 3·1024)/(12·1024) ≈ **30% of full attention's**, while keeping a
  long-range path through the model.

## Evaluation

- Val loss/PPL (primary), lm-eval suite at n=300 via
  `benchmarks/run_quality_eval.py`
- **`benchmarks/long_range_probe.py`** — per-position val loss curves, the
  diagnostic this milestone adds: if gqa/hybrid keep improving past
  position ~64–128 where sliding-window flattens, long-range attention is
  finally earning something. (On TinyStories, even full attention was flat
  past ~128 — measured on the 04 checkpoint.)

**Decision rule for 07**: quality-per-KV-byte. Hybrid matching GQA at ~30%
cache ⇒ hybrid advances. Sliding-window still tying everything ⇒ it stays
the winner. GQA clearly ahead ⇒ locality was a TinyStories artifact, and
the honest answer is full attention.

## Results

All three trained cleanly, ~7.0h each (one earlier gqa attempt hit the
in-process MPS allocator pathology at ~iter 2500 — 7x step slowdown with a
healthy GPU — and was restarted; that incident is why train.py now saves
resumable state every 1000 steps).

| Model | Val loss | PPL | HellaSwag | PIQA | ARC-E | WinoGrande | MMLU | tok/s |
|---|---|---|---|---|---|---|---|---|
| **hybrid** | **3.8091** | **45.1** | .323/.337 | .583/.563 | .363/.347 | .517 | .267 | 54.3 |
| gqa (full) | 3.8680 | 47.9 | .320/.323 | .580/.540 | .383/.343 | .513 | .256 | 52.7 |
| sliding-window | 3.9278 | 50.8 | .320/.340 | .583/.563 | .390/.340 | .533 | .260 | 52.2 |

(Benchmark cells are acc/acc_norm at n=300 — differences between the three
are within stderr; val loss is the discriminating metric, as in `03`.)

**Per-position val loss** (the long-range diagnostic, 256 fixed windows;
full curves in `benchmarks/results/05-data-frontier-*-per-position.json`):

| Position bucket | gqa | sliding-window | hybrid |
|---|---|---|---|
| 64–127 | 3.904 | 3.875 | 3.848 |
| 256–319 | 3.811 | 3.895 | **3.753** |
| 704–767 | 3.794 | 3.874 | **3.736** |
| 960–1023 | 3.887 | 3.987 | **3.839** |

Sliding-window flattens at ~position 64 and never improves — on this
corpus it *degrades* toward far positions. gqa and hybrid keep improving
deep into the context; the sliding-window↔hybrid gap grows to ~0.15 nats
by position 1000. **Long-range attention finally earns something here** —
the exact signal TinyStories could never produce.

**The data-quality effect** (same 114M architecture, `02` vs this
milestone): ARC-Easy .260→.383 (chance→real signal), HellaSwag .287→.320,
PIQA .547→.583 — with just 100M tokens of refined data vs 12M of
TinyStories. For scale: Cerebras-GPT-111M (2.2B Pile tokens) reports
ARC-E .380, HellaSwag .268 — matched/beaten here at 1/22 the tokens.

## Verdict

**Hybrid wins by the pre-registered decision rule, and not on a
technicality.** It beats full attention outright (3.809 vs 3.868, ~0.06
nats) while using ~30% of its KV cache, and it beats it at essentially
every position bucket. Pure sliding-window — the TinyStories champion —
clearly loses on data with real long-range structure (0.12 nats behind
hybrid), confirming the `04` caveat that its earlier "win" was a corpus
artifact. That 3 global layers beat 12 suggests the local layers act as a
useful inductive bias while a few global layers suffice to carry
long-range information — consistent with the Gemma-2/3 design lineage.

**Hybrid advances to `07-flagship-slm`.**

## Run it

```
uv run python experiments/05-data-frontier/data.py          # ~5.5GB download + tokenize
uv run python experiments/05-data-frontier/gqa/train.py     # then sliding-window/, hybrid/
uv run python benchmarks/run_quality_eval.py --experiment-dir experiments/05-data-frontier/<v>
uv run python benchmarks/long_range_probe.py --experiment-dir experiments/05-data-frontier/<v>
```

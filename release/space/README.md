---
title: GarageLM Chat
emoji: 🛠️
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
models:
  - garagelm/hybrid-gpt-232m-chat
---

# GarageLM Chat

Chat with **hybrid-gpt-232m-chat** — a 232M-parameter model pretrained,
fine-tuned, and benchmarked **entirely on one Apple M4 Pro laptop**
(1B refined tokens, 127 hours, ~$3 of electricity).

At matched evaluation the base model ties Pythia-160M (trained on 300×
more tokens) and beats GPT-2 on ARC-Easy and PIQA. It is also an honest
~200M research artifact: fluent, format-reliable, and factually
unreliable — expect confident nonsense, delivered politely.

This Space runs on free CPU hardware, so replies stream slowly; the
laptop that trained the model serves it at ~530 tok/s at home.

Model cards: [base](https://huggingface.co/garagelm/hybrid-gpt-232m) ·
[chat](https://huggingface.co/garagelm/hybrid-gpt-232m-chat)

*Small hardware. Real research.*

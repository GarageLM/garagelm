"""GarageLM chat demo Space: hybrid-gpt-232m-chat via transformers on free
CPU hardware. Uses the exact SFT template (User:/Assistant: lines,
<|endoftext|> terminates assistant turns -- the gpt2 tokenizer maps the
literal string to token 50256, so the text-level template is faithful).

The model wrapper has no KV cache (research release), so generation cost
grows with context -- history is trimmed to keep the demo responsive on
2 vCPUs.
"""

import threading

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

REPO = "garagelm/hybrid-gpt-232m-chat"
EOT = "<|endoftext|>"
MAX_CONTEXT = 640  # prompt-token budget: keeps no-cache CPU generation snappy

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True,
                                             dtype=torch.float32)
model.eval()
torch.set_num_threads(2)


def build_prompt(message, history):
    parts = []
    for turn in history:
        if turn["role"] == "user":
            parts.append(f"User: {turn['content']}\n")
        elif turn["role"] == "assistant":
            parts.append(f"Assistant: {turn['content']}{EOT}\n")
    parts.append(f"User: {message}\nAssistant:")
    text = "".join(parts)
    ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, -MAX_CONTEXT:]


def reply(message, history, max_tokens, temperature):
    input_ids = build_prompt(message, history)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    kwargs = dict(input_ids=input_ids, streamer=streamer,
                  max_new_tokens=int(max_tokens), do_sample=True,
                  temperature=float(temperature), top_k=50,
                  eos_token_id=tokenizer.eos_token_id,
                  pad_token_id=tokenizer.eos_token_id)
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    acc = ""
    for piece in streamer:
        acc += piece
        yield acc.strip()
    thread.join()


demo = gr.ChatInterface(
    fn=reply,
    type="messages",
    title="GarageLM · hybrid-gpt-232m-chat",
    description=(
        "A 232M model trained end-to-end on **one laptop** (1B refined "
        "tokens, ~$3 of electricity). Ties Pythia-160M at matched eval "
        "with 300× fewer training tokens. Honest caveat: it is a small "
        "research model — fluent, polite, and frequently wrong. "
        "*(Free-CPU Space: replies stream slowly; the training laptop "
        "itself serves ~530 tok/s.)*"
    ),
    additional_inputs=[
        gr.Slider(16, 256, value=96, step=16, label="Max new tokens"),
        gr.Slider(0.1, 1.2, value=0.7, step=0.1, label="Temperature"),
    ],
    examples=[
        ["What is photosynthesis?", 96, 0.7],
        ["Give me three tips for learning guitar.", 96, 0.7],
        ["Why is the sky blue?", 96, 0.7],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()

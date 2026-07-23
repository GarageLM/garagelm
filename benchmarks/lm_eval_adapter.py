import torch
import torch.nn.functional as F
import tiktoken
from lm_eval.api.model import TemplateLM


class GPTLMEvalAdapter(TemplateLM):
    """Wraps a raw PyTorch GPT checkpoint (this repo's model.py, not a
    HuggingFace transformers model) + tiktoken gpt2 tokenizer so
    lm-evaluation-harness can score it via simple_evaluate(model=<instance>).

    Requests are processed one at a time (no padding/batching) — simpler to
    get correct than HFLM's batched implementation, and fast enough at the
    --limit sample sizes this repo uses for local runs. Reused across every
    experiment (01, 02, 03's future variants), not tied to one checkpoint.
    """

    def __init__(self, model, block_size, device):
        super().__init__()
        self.model = model.eval()
        self.block_size = block_size
        self._device = device  # LM base class exposes this via a read-only `device` property
        self.backend = "causal"
        self.batch_size = 1
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def eot_token_id(self):
        return self.enc.eot_token

    @property
    def max_length(self):
        return self.block_size

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.enc.encode_ordinary(string)

    def tok_decode(self, tokens, **kwargs):
        return self.enc.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        res = []
        for _, context_enc, continuation_enc in requests:
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0

            # Left-truncate to the model's context window: keep the last
            # (block_size + 1) tokens, then drop the final one so the model
            # never sees the last continuation token as input (matches
            # lm-eval-harness's own HFLM convention).
            full = (context_enc + continuation_enc)[-(self.block_size + 1):]
            inp_ids = full[:-1]
            cont_len = len(continuation_enc)

            inp = torch.tensor([inp_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(inp)

            cont_logits = logits[0, -cont_len:, :]
            log_probs = F.log_softmax(cont_logits, dim=-1)
            cont_ids = torch.tensor(continuation_enc, dtype=torch.long, device=self.device)

            token_logprobs = log_probs.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
            greedy_ids = log_probs.argmax(dim=-1)
            is_greedy = bool((greedy_ids == cont_ids).all().item())
            res.append((token_logprobs.sum().item(), is_greedy))
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        # Not used by any task this repo runs (all are loglikelihood-based
        # multiple choice, not rolling-perplexity tasks like wikitext) —
        # implemented simply (single window, no sliding-window stitching for
        # texts longer than block_size) only to satisfy the ABC.
        res = []
        for req in requests:
            (text,) = req.args
            tokens = [self.eot_token_id] + self.tok_encode(text)
            tokens = tokens[: self.block_size + 1]
            if len(tokens) < 2:
                res.append(0.0)
                continue
            inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(inp)
            log_probs = F.log_softmax(logits[0], dim=-1)
            token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            res.append(token_logprobs.sum().item())
        return res

    def generate_until(self, requests, disable_tqdm=False):
        # Not used by any task this repo runs — implemented via the model's
        # own .generate() only to satisfy the ABC.
        res = []
        for req in requests:
            context, gen_kwargs = req.args
            gen_kwargs = gen_kwargs or {}
            until = gen_kwargs.get("until", []) or []
            max_gen_toks = gen_kwargs.get("max_gen_toks", 128)

            ctx_ids = self.tok_encode(context)[-self.block_size:]
            inp = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
            out = self.model.generate(inp, max_gen_toks, temperature=1.0, top_k=None)
            gen_ids = out[0, len(ctx_ids):].tolist()
            text = self.tok_decode(gen_ids)
            for stop in until:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
            res.append(text)
        return res

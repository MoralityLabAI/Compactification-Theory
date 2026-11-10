"""Explicit block-loop recurrence for a pretrained Qwen3.5 text decoder.

Qwen3.5-0.8B has no native recurrence. ARG-02 adds one, untrained: decoder
layers [a, b) form a block B that is applied `depth` times to the residual
stream, with the pretrained weights unchanged:

    x   = layers[0:a](embed(tokens))              block input
    h_1 = B(x)                                    = the ordinary forward
    h_t = B(h_{t-1})                  (loop)      residual feedback
    h_t = h_{t-1} + (h_1 - x)         (open)      no feedback: the block only
                                                  ever sees x; its single update
                                                  is re-applied
    h_t = h_1 + D_t                   (shuffle)   D_t = h_t - h_1 of a donor
                                                  case's loop, right-aligned
    logits = lm_head(norm(layers[b:](h_t)))       readout after any step t

depth 1 is the unmodified model (checked against the HF forward in tests).
Every layer is called directly with the module's own masks and rotary
embeddings, so hidden states after every step are exact, inspectable tensors.
"""
from __future__ import annotations

import time

import torch

from .prompts import OPS

SITE = 5  # right-aligned answer site: "\n", "Program", ":" and the two candidate tokens


class Timer:
    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"

    def start(self):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def stop(self, mark):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            ev.synchronize()
            return mark.elapsed_time(ev)
        return (time.perf_counter() - mark) * 1000.0


def candidate_rows(prompt_ids, op_ids):
    """16 rows: prompt + (op1, op2) for every pair; row index 4*i + j."""
    rows = []
    for a in OPS:
        for b in OPS:
            rows.append(list(prompt_ids) + [op_ids[a], op_ids[b]])
    return torch.tensor(rows, dtype=torch.long)


def extract(logprobs, op_ids):
    """Answer record from (16, 3, V) log-softmax over the last three positions."""
    tok = [op_ids[o] for o in OPS]
    lp = logprobs.float()
    lp1 = [float(lp[0, 0, t]) for t in tok]
    g1 = int(lp[0, 0].argmax())
    lp2, g2, lp3, g3 = [], [], [], []
    for i in range(4):
        lp2.append([float(lp[4 * i, 1, t]) for t in tok])
        g2.append(int(lp[4 * i, 1].argmax()))
        row3, rowg = [], []
        for j in range(4):
            r = 4 * i + j
            row3.append([float(lp[r, 2, t]) for t in tok])
            rowg.append(int(lp[r, 2].argmax()))
        lp3.append(row3)
        g3.append(rowg)
    # Rows share a prefix, so these positions must agree across rows.
    spread1 = float((lp[:, 0] - lp[0:1, 0]).abs().max())
    spread2 = max(float((lp[4 * i:4 * i + 4, 1] - lp[4 * i:4 * i + 1, 1]).abs().max()) for i in range(4))
    return {"lp1": lp1, "lp2": lp2, "lp3": lp3, "g1": g1, "g2": g2, "g3": g3,
            "row_spread": max(spread1, spread2)}


def align_index(recipient_len, donor_len, device):
    """Recipient position p takes donor position donor_len-1-((recipient_len-1-p) mod donor_len)."""
    k = torch.arange(recipient_len - 1, -1, -1, device=device) % donor_len
    return donor_len - 1 - k


class BlockLoop:
    def __init__(self, model, block=None):
        self.model = model
        self.base = model.model
        self.layers = self.base.layers
        self.cfg = self.base.config
        if getattr(self.cfg, "model_type", None) != "qwen3_5_text":
            raise ValueError("BlockLoop supports Qwen3.5 text decoders only")
        self.n = len(self.layers)
        self.types = list(self.cfg.layer_types)
        if block is not None:
            a, b = block
            if not (0 <= a < b <= self.n):
                raise ValueError("block outside the decoder")
        self.block = tuple(block) if block is not None else (0, self.n)

    @property
    def device(self):
        return self.base.embed_tokens.weight.device

    @torch.no_grad()
    def prepare(self, ids):
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        ids = ids.to(self.device)
        emb = self.base.embed_tokens(ids)
        rows, length = ids.shape
        pos = torch.arange(length, device=self.device).view(1, 1, -1).expand(4, rows, -1)
        text_pos, rope_pos = pos[0], pos[1:]
        kw = {"config": self.cfg, "inputs_embeds": emb, "attention_mask": None,
              "past_key_values": None, "position_ids": text_pos}
        masks = {"full_attention": create_causal_mask(**kw),
                 "linear_attention": create_recurrent_attention_mask(**kw)}
        pe = self.base.rotary_emb(emb, rope_pos)
        return emb, {"pe": pe, "masks": masks, "pos": text_pos}

    @torch.no_grad()
    def run(self, h, lo, hi, ctx):
        for i in range(lo, hi):
            h = self.layers[i](h, position_embeddings=ctx["pe"], attention_mask=ctx["masks"][self.types[i]],
                               position_ids=ctx["pos"], past_key_values=None, use_cache=False)
        return h

    @torch.no_grad()
    def readout(self, h, ctx):
        h = self.run(h, self.block[1], self.n, ctx)
        h = self.base.norm(h[:, -3:, :])
        return torch.log_softmax(self.model.lm_head(h).float(), dim=-1)

    @torch.no_grad()
    def trajectory(self, ids, op_ids, depth, *, mode="loop", readout_steps=(), capture=(-3,),
                   donor_delta=None, patch=None, perturb=None, keep_delta=(), keep_site=(),
                   clean_states=None, keep_states=False):
        """Run one block-looped forward over the 16 candidate rows of one prompt.

        donor_delta: {t: (rows, Ld, H)} donor h_t - h_1, for mode "shuffle".
        patch: {"step": t, "site": (rows, SITE, H)} donor site states written at step t.
        perturb: {"step": t, "sigma": s, "seed": n} Gaussian kick of s * RMS(h_t).
        clean_states: [row-0 states x, h_1..h_T] of an unperturbed run, for distances.
        """
        if mode not in ("loop", "open", "shuffle"):
            raise ValueError("unknown recurrence mode")
        if mode == "shuffle" and donor_delta is None:
            raise ValueError("shuffle needs donor deltas")
        a, b = self.block
        timer = Timer(self.device)
        timing = {"prelude_ms": 0.0, "block_ms": [], "readout_ms": {}}
        mark = timer.start()
        emb, ctx = self.prepare(ids)
        x = self.run(emb, 0, a, ctx)
        timing["prelude_ms"] = timer.stop(mark)
        rows, length, _ = x.shape
        out = {"records": {}, "states": [x[0, list(capture)].float().cpu()], "residual": [],
               "residual_site": [], "deltas": {}, "sites": {}, "distance": [], "timing": timing,
               "rows": rows, "length": length, "full_states": [x[0].float()] if keep_states else None}
        index = None
        if mode == "shuffle":
            any_delta = next(iter(donor_delta.values()))
            index = align_index(length, any_delta.shape[1], x.device)
        h_prev, h1, upd = x, None, None
        for t in range(1, depth + 1):
            mark = timer.start()
            if t == 1 or mode == "loop":
                h = self.run(h_prev, a, b, ctx)
            elif mode == "open":
                h = h_prev + upd
            elif t in donor_delta:
                # The recipient never runs its own block after step 1, so only
                # steps that are read out need a donor displacement.
                h = h1 + donor_delta[t].to(h1.dtype)[:, index, :]
            else:
                h = h_prev
            if t == 1:
                h1, upd = h, h - x
            if perturb is not None and t == perturb["step"]:
                gen = torch.Generator(device="cpu").manual_seed(int(perturb["seed"]))
                noise = torch.randn((1, length, h.shape[-1]), generator=gen).to(h.device, torch.float32)
                rms = h[0].float().pow(2).mean().sqrt()
                h = (h.float() + perturb["sigma"] * rms * noise).to(h.dtype)
                out["kick_norm"] = float((perturb["sigma"] * rms * noise[0]).norm())
            if patch is not None and t == patch["step"]:
                h = h.clone()
                h[:, -SITE:, :] = patch["site"].to(h.dtype)
            timing["block_ms"].append(timer.stop(mark))
            diff = (h[0].float() - h_prev[0].float())
            out["residual"].append(float(diff.norm() / h_prev[0].float().norm().clamp_min(1e-12)))
            out["residual_site"].append(float(diff[-SITE:].norm() / h_prev[0, -SITE:].float().norm().clamp_min(1e-12)))
            out["states"].append(h[0, list(capture)].float().cpu())
            if keep_states:
                out["full_states"].append(h[0].float())
            if clean_states is not None and t < len(clean_states):
                out["distance"].append(float((h[0].float() - clean_states[t]).norm()))
            if t in keep_delta:
                out["deltas"][t] = (h - h1).detach()
            if t in keep_site:
                out["sites"][t] = h[:, -SITE:, :].detach().clone()
            if t in readout_steps:
                mark = timer.start()
                rec = extract(self.readout(h, ctx), op_ids)
                timing["readout_ms"][t] = timer.stop(mark)
                out["records"][t] = rec
            h_prev = h
        return out

    def latency_ms(self, timing, depth):
        """Standalone latency of a depth-`depth` forward from step timings."""
        return timing["prelude_ms"] + sum(timing["block_ms"][:depth]) + timing["readout_ms"][depth]

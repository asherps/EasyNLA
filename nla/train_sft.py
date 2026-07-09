"""Self-contained NLA SFT (AV + AR), no Miles dependency.

Single entry point with `--mode {av,ar}`:
  - AV: AutoModelForCausalLM + Karvonen layer-1 injection hook
        loss = cross-entropy on response tokens only
        target: actor learns to verbalise injected activations
  - AR: NLACriticModel (truncated K+1-layer backbone + Linear(d,d) value_head)
        loss = MSE on L2-normalised (pred, gold) at last-token position
        target: critic learns to reconstruct activation from explanation text

Replaces the old Miles-era pipeline (FSDP actor subclass, loss plug-ins,
rollout adapters, shell wrappers, and a separate critic-init script — all
removed in the repo consolidation). AR backbone truncation now happens
in-script.

Loads bf16 model + bitsandbytes AdamW8bit (~4 GB optim states on 8B model
instead of 64 GB for fp32 AdamW). Single GPU; activation memory bounded by
gradient_checkpointing on the AV path.

Saves HF format checkpoints directly — no DCP→HF conversion step.
"""

import argparse
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from nla.utils import critic_predict, register_karvonen_hook
from nla.utils.run_config import add_config_arg, apply_config_defaults, save_resolved_config
from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import (
    INJECT_PLACEHOLDER,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)


# ----------------------------------------------------------------------------
# Helpers shared with train_rl_self_contained.py (kept inline so this file
# stays self-contained — they're small, and importing creates an awkward
# coupling between SFT and RL trainers).
# ----------------------------------------------------------------------------




def load_sft_dataset(parquet_path, n_max=None, *, mode):
    """Stream-load AV (prompt: list[dict], response: str, activation_vector)
    or AR (prompt: str, activation_vector). Slice rowgroups so n_max=N takes
    only N rows, not the full first rowgroup."""
    cols = (
        ["prompt", "response", "activation_vector"] if mode == "av"
        else ["prompt", "activation_vector"]
    )
    pf = pq.ParquetFile(parquet_path)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n_in_rg = rg.num_rows
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        rg = rg.slice(0, take)
        # activation_vector via flatten→numpy (zero-copy) — ~100× faster than
        # to_pylist() on 4096-float lists, which builds ~1B PyFloats at 250k rows
        # (GPUs sit idle for 10-20 min otherwise). Same pattern as schema.py.
        acts_col = rg.column("activation_vector").combine_chunks()  # ChunkedArray→Array
        acts_np = (acts_col.flatten().to_numpy(zero_copy_only=False)
                   .astype(np.float32).reshape(len(acts_col), -1))
        prompts = rg.column("prompt").to_pylist()
        responses = rg.column("response").to_pylist() if mode == "av" else None
        for i in range(take):
            row = {"prompt": prompts[i], "activation_vector": acts_np[i]}
            if mode == "av":
                row["response"] = responses[i]
            rows.append(row)
    return rows


def load_heldout_explanation_pairs(parquet_path, n_rows):
    """(explanation, activation) pairs from an AV-split parquet (has `response`).

    The AV split is DOC-DISJOINT from the AR training data by stage-1
    construction, so FVE on these pairs is a genuine held-out number —
    training-batch FVE overstates quality once the data is multi-epoch.
    """
    pf = pq.ParquetFile(parquet_path)
    pairs = []
    for rg_idx in range(pf.num_row_groups):
        if len(pairs) >= n_rows:
            break
        rg = pf.read_row_group(rg_idx, columns=["response", "activation_vector"])
        responses = rg.column("response").to_pylist()
        acts_col = rg.column("activation_vector").combine_chunks()
        acts = (acts_col.flatten().to_numpy(zero_copy_only=False)
                .astype(np.float32).reshape(len(acts_col), -1))
        for resp, act in zip(responses, acts):
            expl = extract_explanation(resp)
            if expl is None:
                continue
            pairs.append((expl, act))
            if len(pairs) >= n_rows:
                break
    return pairs


@torch.no_grad()
def heldout_fve_mse(critic, tokenizer, pairs, template, mse_scale_f, device,
                    micro_batch=16, max_len=1024):
    """Mean per-sample MSE on normalized (pred, gold) over held-out pairs.

    Returns (mean_mse, n_scored). Caller divides by a predict-the-mean
    baseline for FVE. Skips pairs whose critic prompt exceeds max_len
    (would truncate the suffix anchor).
    """
    mses = []
    for cs in range(0, len(pairs), micro_batch):
        chunk = pairs[cs:cs + micro_batch]
        ids_list, golds = [], []
        for expl, act in chunk:
            ids = tokenizer.encode(template.format(explanation=expl),
                                   add_special_tokens=False)
            if not 0 < len(ids) <= max_len:
                continue
            ids_list.append(torch.tensor(ids, dtype=torch.long))
            golds.append(act)
        if not ids_list:
            continue
        bs = len(ids_list)
        T = max(t.numel() for t in ids_list)
        batch_ids = torch.full((bs, T), tokenizer.eos_token_id,
                               dtype=torch.long, device=device)
        attn = torch.zeros((bs, T), dtype=torch.long, device=device)
        for i, t in enumerate(ids_list):
            batch_ids[i, : t.numel()] = t.to(device)
            attn[i, : t.numel()] = 1
        pred = critic_predict(critic, batch_ids, attn, mse_scale_f)
        gold = torch.tensor(np.stack(golds), dtype=torch.float32, device=device)
        pred_n = normalize_activation(pred, mse_scale_f)
        gold_n = normalize_activation(gold, mse_scale_f)
        mses.extend(((pred_n - gold_n) ** 2).mean(dim=-1).tolist())
    return float(np.mean(mses)) if mses else float("nan"), len(mses)


def ar_debug_stats(pred, gold, mse_scale_f):
    """Cheap per-batch AR diagnostics: norms + direction match (cosine).

    pred/gold are raw [B, d]. Returns dict for wandb. Helps tell apart "wrong
    scale" (norm mismatch) from "wrong direction" (low cosine) failures that a
    single normalized-MSE number hides.
    """
    with torch.no_grad():
        cos = F.cosine_similarity(pred.float(), gold.float(), dim=-1).mean().item()
        return {
            "pred_norm": pred.float().norm(dim=-1).mean().item(),
            "gold_norm": gold.float().norm(dim=-1).mean().item(),
            "cos_pred_gold": cos,
        }


@torch.no_grad()
def av_generate_samples(model, tokenizer, rows, cfg, device, *,
                        max_new_tokens=256):
    """Generate explanations for a few fixed activations (AV debug table).

    Returns list of dicts: {idx, gen_len, explanation}.
    """
    model.eval()
    out = []
    vref = getattr(model, "_nla_vectors_ref", None)
    for i, row in enumerate(rows):
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, cfg.injection_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        ptxt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(ptxt, add_special_tokens=False)
        pt = torch.tensor([ids], dtype=torch.long, device=device)
        act = torch.tensor(row["activation_vector"], dtype=torch.float32).unsqueeze(0).to(device)
        if vref is not None:
            vref[0] = act
        try:
            gen = model.generate(
                input_ids=pt, attention_mask=torch.ones_like(pt),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
            )
        finally:
            if vref is not None:
                vref[0] = None
        resp = tokenizer.decode(gen.sequences[0, pt.shape[1]:], skip_special_tokens=True)
        expl = extract_explanation(resp)
        out.append({
            "idx": i,
            "gen_len": int(gen.sequences.shape[1] - pt.shape[1]),
            "explanation": (expl or "<no extraction>")[:600],
        })
    model.train()
    return out


# ----------------------------------------------------------------------------
# AR critic init: truncate base Qwen3 to K+1 layers + Linear(d, d) value_head,
# identity-init the head. (Previously a separate critic-init script; now in-process.)
# ----------------------------------------------------------------------------

def _resolve_device_map(device_map_mode, max_gpu_mem, quant_config):
    """Return (device_map, max_memory) for from_pretrained.

    'single' → whole 4-bit model on GPU0 (bf16: None, caller does .to(device)).
    'auto'   → accelerate splits weights across visible GPUs (naive MP). A
               positive max_gpu_mem (GiB/GPU) forces a split — used to validate
               the 397B sharding path on a small model that would otherwise fit
               on one GPU.
    """
    if quant_config is None:
        return None, None
    if device_map_mode == "auto":
        max_memory = None
        if max_gpu_mem and max_gpu_mem > 0:
            max_memory = {
                i: f"{max_gpu_mem}GiB" for i in range(torch.cuda.device_count())
            }
        return "auto", max_memory
    return {"": 0}, None


def init_critic_from_base(base_ckpt: str, num_layers: int, dtype, quant_config=None,
                          device_map=None, max_memory=None, strip_final_norm=True):
    """Truncate base to first `num_layers` transformer blocks, attach an
    identity-init Linear(d, d) value_head. NLACriticModel handles the wrapping.

    identity-init is critical: at step 0, pred = value_head(last_h) = last_h
    when value_head = I, so the initial reconstruction loss starts at the
    backbone's own representational ceiling instead of `kaiming_uniform`'s
    1/√3 scaling which would crush pred_norm. See TRAINING_NOTES.md.

    quant_config (BitsAndBytesConfig) loads the backbone in 4-bit (QLoRA); the
    value_head stays full-precision (tiny, fully trainable).
    """
    # First load the full base, truncate the layers list, then construct
    # NLACriticModel around it.
    from copy import deepcopy
    base = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
        quantization_config=quant_config,
        device_map=device_map, max_memory=max_memory,
    )
    # Route through arch_adapters: multimodal wrappers (Gemma-3) expose the text
    # model under .language_model with config under .text_config, and GPT-2 /
    # Falcon keep decoder blocks at .transformer.h — the old bare
    # `while hasattr(.model)` walk crashed or truncated the wrong module there.
    from nla.utils.arch_adapters import resolve_text_model
    base = resolve_text_model(base)   # CausalLM-shaped text model (pass-through for Qwen/Llama)
    cfg = deepcopy(base.config)
    cfg.num_hidden_layers = num_layers
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        cfg.layer_types = list(cfg.layer_types)[:num_layers]
    # Inner decoder container: .model (llama family) or .transformer (GPT-2/Falcon),
    # holding the block list at .layers / .h respectively.
    if hasattr(base, "model"):
        inner, _layers_key = base.model, "layers"
    elif hasattr(base, "transformer"):
        inner, _layers_key = base.transformer, "h"
    else:
        raise AssertionError(
            f"{type(base).__name__} has neither .model nor .transformer — "
            f"extend init_critic_from_base for this architecture"
        )
    # Keep only the first num_layers blocks
    setattr(inner, _layers_key, torch.nn.ModuleList(
        list(cast(torch.nn.ModuleList, getattr(inner, _layers_key)))[:num_layers]))
    # Truncate the BACKBONE's own config too — NLACriticModel.save_pretrained
    # delegates to backbone.save_pretrained, which writes backbone.config.
    # Leaving it at the full depth makes a full (non-LoRA) AR save claim 36
    # layers with weights for 25; a later from_pretrained would then randomly
    # initialize the missing blocks and silently predict garbage.
    base.config.num_hidden_layers = num_layers
    if getattr(base.config, "layer_types", None) is not None:
        base.config.layer_types = list(base.config.layer_types)[:num_layers]
    if strip_final_norm:
        # RAW residual stream → value head. The full model's final
        # RMSNorm was trained for the LAST layer's output; applying it to the
        # layer-K stream bakes a per-channel γ reweighting into every critic
        # prediction. NLACriticModel.from_pretrained already strips it — this
        # makes the fresh-truncation path consistent. ar_meta.json records the
        # choice so RL reloads match (pre-2026-06 ckpts trained with norm kept).
        for _attr in ("norm", "final_layernorm", "ln_f", "final_layer_norm"):
            if hasattr(inner, _attr):
                setattr(inner, _attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"could not find final layernorm on {type(inner).__name__}"
            )
    # lm_head is never used by the critic — drop it (frees ~1.2GB on 8B-class).
    if hasattr(base, "lm_head"):
        base.lm_head = torch.nn.Identity()
    d_model = cfg.hidden_size
    # NLACriticModel wraps backbone + value_head. Constructor takes both.
    critic = NLACriticModel(cfg, base)
    # Identity init the value head (Linear has bias=False per models.py:82)
    with torch.no_grad():
        critic.value_head.weight.copy_(torch.eye(d_model, dtype=dtype))
    # value_head stays FP32 regardless of backbone dtype: AdamW steps it directly
    # (no fp32 master), and in bf16 the identity diagonal (1.0, ULP≈0.0039)
    # can't absorb lr~1e-4 updates — they round to zero and the head never
    # moves. critic_predict casts in/out, so fp32 is compute-transparent.
    if quant_config is None:
        critic = critic.to(dtype)
        critic.value_head.to(dtype=torch.float32)
    else:
        # 4-bit backbone already placed (device_map); align value_head to the
        # LAST layer's device so forward's value_head(last_hidden) matches.
        last_dev = next(getattr(inner, _layers_key)[-1].parameters()).device
        critic.value_head.to(device=last_dev, dtype=torch.float32)
    print(f"[ar] truncated to {num_layers} layers, value_head identity-init "
          f"(weight norm = {critic.value_head.weight.float().norm().item():.3f})")
    return critic


# ----------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay to min_lr
# ----------------------------------------------------------------------------

def build_lr_lambda(warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    return fn


# ----------------------------------------------------------------------------
# AV forward: encode chat-template prompt + response, build response-only loss
# mask, forward through model with Karvonen hook firing on the marker token.
# ----------------------------------------------------------------------------

@torch.no_grad()
def heldout_av_ce(model, tokenizer, rows, cfg, vectors_ref, device, *,
                  max_len=1024, micro_batch=16):
    """Held-out AV val loss: mean token-CE on response tokens over doc-disjoint
    held-out AV rows — the SAME per-response-token CE the AV trains on, so it's
    directly comparable to the train `loss` (train loss is a memorization proxy;
    this is the generalization number). Returns (mean_ce, n_rows)."""
    tot_loss, tot_tok = 0.0, 0
    for cs in range(0, len(rows), micro_batch):
        chunk = rows[cs:cs + micro_batch]
        ids, attn, loss_mask, v_batch = _av_prepare_chunk(
            chunk, tokenizer, cfg.injection_char, device,
            max_len=max_len)
        vectors_ref[0] = v_batch
        try:
            logits = model(input_ids=ids, attention_mask=attn).logits.float()
        finally:
            vectors_ref[0] = None
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = ids[:, 1:].to(shift_logits.device).contiguous()
        shift_mask = loss_mask[:, 1:].to(shift_logits.device).contiguous()
        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1), reduction="none").view(shift_targets.shape)
        tot_loss += float((per_tok * shift_mask).sum().item())
        tot_tok += int(shift_mask.sum().item())
    return (tot_loss / max(tot_tok, 1)), len(rows)


def _av_prepare_chunk(rows, tokenizer, inject_char, device, max_len=1024):
    """Return (input_ids, attn, loss_mask, v_batch) — all [B, T] (or [B, d])."""
    full_ids_list = []
    prompt_lens = []
    for row in rows:
        # row["prompt"] is list[{"role","content"}] with INJECT_PLACEHOLDER inside.
        # Replace with the actual injection char so the tokenizer emits the
        # marker token id at the right position.
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        # Response gets a trailing EOS so the model learns to stop.
        resp = row["response"] + (tokenizer.eos_token or "")
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)
        full = prompt_ids + resp_ids
        if len(full) > max_len:
            # Truncate response from the right to fit. Prompt is fixed.
            full = full[:max_len]
        full_ids_list.append(torch.tensor(full, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))

    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    loss_mask = torch.zeros((bs, T), dtype=torch.float32, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
        # 1 on response positions, 0 on prompt + pad. The shift-by-one for CE
        # is applied later (in the loss computation), so this mask is in
        # "target token" space — positions whose CE we want to count.
        loss_mask[i, prompt_lens[i]:L] = 1
    v_batch = torch.tensor(
        np.stack([r["activation_vector"] for r in rows]),
        dtype=torch.float32, device=device,
    )
    return batch_ids, attn, loss_mask, v_batch


# ----------------------------------------------------------------------------
# AR forward: tokenize the already-built critic prompt, forward, take MSE on
# normalised (pred, gold).
# ----------------------------------------------------------------------------

def _ar_prepare_chunk(rows, tokenizer, device, max_len=1024):
    full_ids_list = []
    kept_rows = []
    n_skipped = 0
    for row in rows:
        # AR's prompt is the already-filled critic template string.
        # add_special_tokens=False matches RL-time critic scoring and stage-3's
        # build-time suffix verification (True is a no-op on Qwen but prepends
        # BOS on Llama/Gemma-family tokenizers → train/reward token mismatch).
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) > max_len:
            # Right-truncating would cut the "</text> <summary>" suffix and the
            # last-token extraction would land mid-explanation — silently wrong.
            # Skip the row instead (RL-side rejects over-length the same way).
            n_skipped += 1
            continue
        full_ids_list.append(torch.tensor(ids, dtype=torch.long))
        kept_rows.append(row)
    if n_skipped:
        print(f"[ar] skipped {n_skipped}/{len(rows)} rows with critic prompt "
              f"> {max_len} tokens (suffix anchor would be truncated)")
    assert full_ids_list, f"all {len(rows)} rows exceeded max_len={max_len}"
    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
    gold = torch.tensor(
        np.stack([r["activation_vector"] for r in kept_rows]),
        dtype=torch.float32, device=device,
    )
    return batch_ids, attn, gold


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--mode", required=True, choices=["av", "ar"])
    p.add_argument("--base-ckpt", required=True,
                   help="HF dir for AV (base model) or AR (base model to truncate, "
                        "OR an already-prepared NLACriticModel checkpoint).")
    p.add_argument("--parquet", required=True, help="SFT data parquet")
    p.add_argument("--sidecar", default=None,
                   help="Sidecar source (defaults to --parquet for the dataset sidecar)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Per-forward batch (= 'micro batch'). Effective batch = "
                        "batch_size × gradient_accumulation_steps.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--ar-num-layers", type=int, default=25,
                   help="K+1 for AR mode — truncate base to this many transformer blocks")
    p.add_argument("--freeze-backbone", action="store_true", default=False,
                   help="AR mode: freeze the backbone (+ value_head), training "
                        "nothing — used to sanity-check. This flag hard-freezes "
                        "the AR here.")
    # ---- Debug sampling: periodically dump example generations to a wandb Table ----
    p.add_argument("--sample-every", type=int, default=0,
                   help="Every N steps, log example generations to an accumulating "
                        "wandb Table (AV: generate explanations from injected "
                        "activations; AR: reconstruct). 0 = off.")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Number of fixed examples per --sample-every dump.")
    p.add_argument("--sample-max-new-tokens", type=int, default=256,
                   help="AV sampling: max new tokens when generating explanations.")
    p.add_argument("--heldout-parquet", default=None,
                   help="AR mode: AV-split parquet for held-out FVE (doc-disjoint "
                        "from AR training data by stage-1 construction). "
                        "AV mode: held-out AV parquet (prompt/response/activation) "
                        "for inline val token-CE + ppl (the generalization metric, "
                        "vs the memorization-proxy train loss). "
                        "Evaluated every --heldout-every steps.")
    p.add_argument("--heldout-rows", type=int, default=1000)
    p.add_argument("--heldout-every", type=int, default=100)
    p.add_argument("--strip-final-norm", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="AR mode: replace the backbone's final RMSNorm with "
                        "Identity so the value head sees the raw layer-K "
                        "residual (matches NLACriticModel."
                        "from_pretrained). --no-strip-final-norm reproduces "
                        "pre-2026-06 checkpoints. Recorded in ar_meta.json.")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=None,
                   help="If omitted: AV-mode default 1e-4 (best for a 1-epoch warm-start "
                        "in our sweeps), AR-mode default 2e-5.")
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Default: ON for AV (fits 8B + batch=64 + FA2 on 141 GB H200), "
                        "OFF for AR (smaller model + shorter seq fits comfortably).")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--quant", choices=["none", "4bit"], default="none",
                   help="4bit = bitsandbytes nf4 (QLoRA). Required for models too "
                        "big for bf16; validates the GLM-5 path on Qwen3-8B.")
    p.add_argument("--use-lora", action="store_true", default=False,
                   help="Train a LoRA adapter on a frozen base instead of full-FT. "
                        "Mandatory for 4bit. (AR value_head stays fully trainable.)")
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--device-map", choices=["single", "auto"], default="single",
                   help="single = whole 4-bit model on GPU0 (fits up to ~70B on "
                        "a B200). auto = accelerate splits weights across all "
                        "visible GPUs (naive MP) — required for 397B-class bases.")
    p.add_argument("--max-gpu-mem", type=int, default=0,
                   help="GiB/GPU cap for device_map=auto weight placement. >0 "
                        "forces a multi-GPU split (used to validate sharding on a "
                        "small model). 0 = use full GPU memory.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap training rows (smoke runs)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="warmstart",
                   help="wandb group for organizing the workspace (warmstart/rl/eval).")
    p.add_argument("--wandb-tags", default=None,
                   help="comma-separated wandb tags for explicit experiments (e.g. 'sweep,lr3e5').")
    p.add_argument("--no-wandb", action="store_true")
    apply_config_defaults(p)   # YAML (--config) -> argparse defaults; CLI still overrides
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    if args.lr is None:
        # Mode-aware default: non-comp AV warmstart is 1e-4 (2x-data 1-epoch best, held-out
        # val ppl 3.86; optimum dropped from the old 1x 2e-4 after the data doubled);
        # AR keeps 2e-5 (AR launchers pass --lr explicitly anyway).
        args.lr = 1e-4 if args.mode == "av" else 2e-5
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = (args.mode == "av")
    if args.sidecar is None:
        args.sidecar = args.parquet

    # ---- tokenizer + nla config ----
    # From --base-ckpt, NOT hardcoded — the sidecar asserts below catch a
    # wrong-family tokenizer, but only if we load the one the run targets.
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    print(f"[cfg] mode={args.mode} d_model={cfg.d_model} mse_scale={mse_scale_f}")

    # ---- model ----
    if args.mode == "av":
        print(f"[av] loading {args.base_ckpt} (quant={args.quant}, lora={args.use_lora})")
        quant_config = None
        if args.quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=dtype,  # FSDP-friendly storage (harmless single-GPU)
            )
        dmap, max_mem = _resolve_device_map(args.device_map, args.max_gpu_mem, quant_config)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_ckpt, torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            quantization_config=quant_config,
            device_map=dmap, max_memory=max_mem,
        )
        if dmap is None:
            model = model.to(device)
        elif args.device_map == "auto" and hasattr(model, "hf_device_map"):
            print(f"[av] device_map=auto → GPUs used: "
                  f"{sorted({d for d in model.hf_device_map.values() if isinstance(d, int)})}")
        if args.use_lora:
            if quant_config is not None:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=args.gradient_checkpointing,
                )
            from nla.utils.arch_adapters import resolve_lora_target_modules
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM", use_rslora=True,
                target_modules=resolve_lora_target_modules(model.config),
            ))
            model.print_trainable_parameters()
        vectors_ref = [None]
        register_karvonen_hook(
            model, vectors_ref,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
        model._nla_vectors_ref = vectors_ref  # av_generate_samples reaches it here
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print("[av] gradient_checkpointing ENABLED")
    else:  # ar
        quant_config = None
        if args.quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=dtype,
            )
        dmap, max_mem = _resolve_device_map(args.device_map, args.max_gpu_mem, quant_config)
        # Check if --base-ckpt is already a critic ckpt (has value_head.safetensors)
        is_prepared_critic = (Path(args.base_ckpt) / "value_head.safetensors").exists()
        if is_prepared_critic:
            print(f"[ar] loading pre-prepared critic from {args.base_ckpt}")
            model = NLACriticModel.from_pretrained(
                args.base_ckpt, torch_dtype=dtype,
                attn_implementation=args.attn_implementation,
                quantization_config=quant_config,
                device_map=dmap, max_memory=max_mem,
            )
            if dmap is None:
                model = model.to(device)
            # from_pretrained ALWAYS strips the final norm, regardless of the
            # CLI flag — record what actually happened, or RL would rebuild
            # the critic differently than it was trained.
            if not args.strip_final_norm:
                print("[ar] NOTE: --no-strip-final-norm ignored on the "
                      "prepared-critic path (from_pretrained always strips); "
                      "recording final_norm_stripped=true")
            args.strip_final_norm = True
        else:
            print(f"[ar] truncating base {args.base_ckpt} to {args.ar_num_layers} "
                  f"layers (quant={args.quant})")
            model = init_critic_from_base(
                args.base_ckpt, args.ar_num_layers, dtype, quant_config,
                device_map=dmap, max_memory=max_mem,
                strip_final_norm=args.strip_final_norm,
            )
            if dmap is None:
                model = model.to(device)
        if args.use_lora:
            # Inject LoRA IN-PLACE into the backbone's attn projections. Unlike
            # get_peft_model this does NOT wrap the backbone in a PeftModel, so
            # NLACriticModel.forward (which calls the inner transformer directly)
            # is unchanged and the value_head stays a plain trainable module.
            from peft import inject_adapter_in_model
            if quant_config is not None:
                model.backbone = prepare_model_for_kbit_training(
                    model.backbone, use_gradient_checkpointing=args.gradient_checkpointing,
                )
            from nla.utils.arch_adapters import resolve_lora_target_modules
            inject_adapter_in_model(LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM", use_rslora=True,
                target_modules=resolve_lora_target_modules(model.backbone.config),
            ), model.backbone)
            # Train ONLY the LoRA adapters + the value_head; freeze the rest.
            for n_, p_ in model.named_parameters():
                p_.requires_grad_(("lora_" in n_) or n_.startswith("value_head"))
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ar] LoRA-injected; trainable={n_tr/1e6:.1f}M (lora + value_head)")
        vectors_ref = None
        if args.gradient_checkpointing and not args.use_lora:
            # NLACriticModel wraps backbone; enable on inner module
            # (use_lora+4bit path already enabled it via prepare_model_for_kbit_training)
            if hasattr(model.backbone, "gradient_checkpointing_enable"):
                model.backbone.gradient_checkpointing_enable()
                print("[ar] gradient_checkpointing ENABLED (backbone)")
        if args.freeze_backbone:
            for p_ in model.parameters():
                p_.requires_grad_(False)
            print("[ar] backbone + value_head FROZEN (--freeze-backbone)")
    model.train()

    # ---- data ----
    print(f"[data] loading {args.parquet} (max_rows={args.max_rows})", flush=True)
    rows = load_sft_dataset(args.parquet, n_max=args.max_rows, mode=args.mode)
    print(f"[data] {len(rows)} rows", flush=True)
    if args.mode == "ar" and cfg.critic_suffix_ids:
        # One-time suffix-anchor sanity check (the sidecar field's stated
        # purpose): the tokenized critic prompt must end with the expected
        # "</text> <summary>" ids, or last-token extraction trains on the
        # wrong position. Row 0 suffices — template drift hits every row.
        from nla.config import verify_critic_suffix
        _row0_ids = tokenizer.encode(rows[0]["prompt"], add_special_tokens=False)
        verify_critic_suffix(_row0_ids, cfg.critic_suffix_ids, context="ar row 0")
        print(f"[ar] critic suffix anchor verified (row 0)")

    # ---- optimizer + LR schedule ----
    try:
        import bitsandbytes as bnb
        optim_cls = bnb.optim.AdamW8bit
        print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        optim_cls = torch.optim.AdamW
        print("[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, (
        "no trainable parameters — --freeze-backbone freezes the whole AR, so "
        "there is nothing to optimize in AR-SFT. Drop --freeze-backbone."
    )
    optim = optim_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        build_lr_lambda(args.lr_warmup_steps, args.num_steps,
                        args.min_lr / max(args.lr, 1e-12)),
    )
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[optim] trainable params: {n_trainable / 1e9:.2f} B")

    # ---- AR-only: predict-the-mean baseline for FVE logging ----
    # Paper definition: baseline = E[||v_norm - μ||²] (raw variance of the
    # normalized distribution, ≈0.72), NOT MSE against normalize(μ) (≈0.94)
    # which runs before 2026-06-09 used and which inflates FVE.
    fve_baseline = None
    if args.mode == "ar":
        from nla.schema import compute_predict_mean_baselines
        _act = torch.tensor(
            np.stack([r["activation_vector"] for r in rows[: min(len(rows), 4000)]]),
            dtype=torch.float32,
        )
        _bl_meannorm, fve_baseline = compute_predict_mean_baselines(_act, mse_scale_f)
        print(f"[ar] predict-the-mean MSE baseline = {fve_baseline:.4f} "
              f"(paper def; meannorm baseline = {_bl_meannorm:.4f})")

    # ---- AR-only: held-out FVE pairs (doc-disjoint AV split) ----
    heldout_pairs = None
    heldout_baseline = None
    heldout_av_rows = None
    if args.mode == "av" and args.heldout_parquet:
        heldout_av_rows = load_sft_dataset(
            args.heldout_parquet, args.heldout_rows, mode="av")
        print(f"[av] {len(heldout_av_rows)} held-out AV rows from {args.heldout_parquet} "
              f"-> inline val token-CE/ppl every {args.heldout_every} steps", flush=True)
    if args.mode == "ar" and args.heldout_parquet:
        assert cfg.critic_prompt_template is not None, (
            "--heldout-parquet needs critic_prompt_template in the sidecar"
        )
        heldout_pairs = load_heldout_explanation_pairs(
            args.heldout_parquet, args.heldout_rows,
        )
        _h_acts = torch.tensor(
            np.stack([a for _, a in heldout_pairs]), dtype=torch.float32,
        )
        _, heldout_baseline = compute_predict_mean_baselines(_h_acts, mse_scale_f)
        del _h_acts
        print(f"[ar] {len(heldout_pairs)} held-out pairs from "
              f"{args.heldout_parquet}; baseline (paper def) = {heldout_baseline:.4f}")

    # ---- wandb ----
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   group=args.wandb_group,
                   tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
                   config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(args, save_dir)   # snapshot merged config for reproducibility

    # ---- debug sampling: fixed example set + accumulating table ----
    sample_rows = rows[: args.n_samples] if args.sample_every > 0 else []
    sample_table_data = []

    # ---- training loop ----
    rng = np.random.default_rng(args.seed)
    perm = list(range(len(rows)))
    rng.shuffle(perm)
    cursor = 0

    grad_accum = args.gradient_accumulation_steps
    eff_batch = args.batch_size * grad_accum
    print(f"[loop] {args.num_steps} steps, batch={args.batch_size} × "
          f"grad_accum={grad_accum} = eff_batch={eff_batch}")

    for step in range(args.num_steps):
        t0 = time.time()
        optim.zero_grad()
        accum_loss = 0.0
        accum_resp_tokens = 0  # AV only: total response tokens for normalization
        accum_av_entropy = 0.0  # AV only: mean policy entropy over response tokens (nats)
        accum_n = 0
        ar_dbg = {}            # AR: last-chunk norms/cosine snapshot

        for accum_idx in range(grad_accum):
            # ---- pick batch ----
            if cursor + args.batch_size > len(perm):
                rng.shuffle(perm)
                cursor = 0
            chunk_rows = [rows[i] for i in perm[cursor:cursor + args.batch_size]]
            cursor += args.batch_size

            # ---- forward + loss ----
            if args.mode == "av":
                ids, attn, loss_mask, v_batch = _av_prepare_chunk(
                    chunk_rows, tokenizer, cfg.injection_char, device,
                    max_len=args.max_len,
                )
                # vectors_ref stays set through .backward() below: AV mode runs
                # gradient checkpointing BY DEFAULT, the backward-time recompute
                # re-fires the injection hook, and clearing before backward made
                # the recompute skip the injection's Jacobian — a silent gradient
                # error on the marker pathway (verified vs no-checkpoint grads).
                vectors_ref[0] = v_batch
                logits = model(input_ids=ids, attention_mask=attn).logits.float()
                # Shift-by-one CE on response tokens. Predict ids[:, t+1] from
                # logits[:, t]. Mask is in TARGET space (positions of tokens
                # to predict), so mask[:, 1:] aligned with logits[:, :-1].
                shift_logits = logits[:, :-1].contiguous()
                # device_map=auto can return logits on a non-zero GPU; align.
                shift_targets = ids[:, 1:].to(shift_logits.device).contiguous()
                shift_mask = loss_mask[:, 1:].to(shift_logits.device).contiguous()
                V = shift_logits.size(-1)
                per_tok = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_targets.view(-1),
                    reduction="none",
                ).view(shift_targets.shape)
                n_resp = shift_mask.sum().clamp(min=1)
                loss = (per_tok * shift_mask).sum() / n_resp
                accum_resp_tokens += int(n_resp.item())
                # mean token entropy over response positions (nats), logging only.
                # Gather response tokens first so the softmax is over n_resp rows, not B*T.
                with torch.no_grad():
                    resp_logits = shift_logits[shift_mask.bool()]   # [n_resp, V]
                    lsm = F.log_softmax(resp_logits, dim=-1)
                    accum_av_entropy += float((-(lsm.exp() * lsm).sum(-1)).mean())
            else:  # ar, single-vector
                ids, attn, gold = _ar_prepare_chunk(
                    chunk_rows, tokenizer, device, max_len=args.max_len,
                )
                pred = critic_predict(model, ids, attn, mse_scale_f)
                pred_n = normalize_activation(pred, mse_scale_f)
                gold_n = normalize_activation(gold, mse_scale_f)
                loss = F.mse_loss(pred_n, gold_n)
                ar_dbg = ar_debug_stats(pred, gold, mse_scale_f)

            # Scale loss for accumulation; gradients sum correctly.
            try:
                (loss / grad_accum).backward()
            finally:
                vectors_ref[0] = None   # clear only AFTER backward (checkpoint recompute done)
            accum_loss += loss.item()
            accum_n += 1

        # ---- step ----
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step()
        sched.step()

        mean_loss = accum_loss / max(accum_n, 1)
        cur_lr = sched.get_last_lr()[0]

        log = {
            "step": step,
            "loss": mean_loss,
            "lr": cur_lr,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "wall_s": time.time() - t0,
        }
        line = (f"step {step:04d} | loss {mean_loss:.4f} | lr {cur_lr:.2e} "
                f"| grad {log['grad_norm']:.3f} | t {log['wall_s']:.1f}s")
        if args.mode == "ar" and fve_baseline is not None:
            fve = (1.0 - mean_loss / fve_baseline) * 100.0
            log["fve_pct"] = fve
            line += f" | FVE {fve:.1f}%"
        if args.mode == "av":
            n_seen = max(1, args.batch_size * accum_n)
            log["resp_tokens"] = accum_resp_tokens
            log["mean_resp_len"] = accum_resp_tokens / n_seen
            log["ppl"] = math.exp(min(20.0, mean_loss))
            log["entropy"] = accum_av_entropy / max(accum_n, 1)  # mean response-token entropy (nats)
            line += (f" | resp_toks {accum_resp_tokens} | ppl {log['ppl']:.2f}"
                     f" | ent {log['entropy']:.3f}")
        # AR debug scalars (norms + direction match)
        if args.mode == "ar" and ar_dbg:
            log.update(ar_dbg)
            line += (f" | cos {ar_dbg['cos_pred_gold']:.3f} "
                     f"| |p|/|g| {ar_dbg['pred_norm']:.1f}/{ar_dbg['gold_norm']:.1f}")
        print(line, flush=True)

        # ---- periodic example-generation table (debug) ----
        if args.sample_every > 0 and (
            (step + 1) % args.sample_every == 0 or (step + 1) == args.num_steps
        ):
            if args.mode == "av":
                samps = av_generate_samples(
                    model, tokenizer, sample_rows, cfg, device,
                    max_new_tokens=args.sample_max_new_tokens,
                )
                for s in samps:
                    sample_table_data.append([step, s["idx"],
                                              s["gen_len"], s["explanation"]])
                print(f"  [sample@{step}] {len(samps)} gens; "
                      f"e.g. idx0: {samps[0]['explanation'][:160]!r}", flush=True)
                if not args.no_wandb:
                    log["samples"] = wandb.Table(
                        columns=["step", "idx", "gen_len", "explanation"],
                        data=list(sample_table_data),
                    )
            else:  # ar: reconstruction examples
                model.eval()
                for i, row in enumerate(sample_rows):
                    ids, attn, gold = _ar_prepare_chunk(
                        [row], tokenizer, device, max_len=args.max_len,
                    )
                    with torch.no_grad():
                        pred = critic_predict(model, ids, attn, mse_scale_f)
                    pn = normalize_activation(pred, mse_scale_f)
                    gn = normalize_activation(gold, mse_scale_f)
                    rmse = F.mse_loss(pn, gn).item()
                    sample_table_data.append([
                        step, i, round(rmse, 4), row["prompt"][:400],
                    ])
                model.train()
                print(f"  [sample@{step}] logged {len(sample_rows)} AR reconstructions",
                      flush=True)
                if not args.no_wandb:
                    log["samples"] = wandb.Table(
                        columns=["step", "idx", "recon_mse", "prompt"],
                        data=list(sample_table_data),
                    )

        # ---- held-out val token-CE/ppl (AV mode, doc-disjoint) ----
        if heldout_av_rows is not None and (
            (step + 1) % args.heldout_every == 0 or (step + 1) == args.num_steps
        ):
            model.eval()
            h_ce, h_n = heldout_av_ce(
                model, tokenizer, heldout_av_rows, cfg, vectors_ref, device,
                max_len=args.max_len)
            model.train()
            log["heldout_loss"] = h_ce
            log["heldout_ppl"] = math.exp(h_ce) if h_ce < 30 else float("inf")
            print(f"  [heldout@{step}] val_loss {h_ce:.4f} | val_ppl "
                  f"{log['heldout_ppl']:.3f} (n={h_n})", flush=True)
        # ---- held-out FVE (AR mode, doc-disjoint) ----
        if heldout_pairs is not None and (
            (step + 1) % args.heldout_every == 0 or (step + 1) == args.num_steps
        ):
            model.eval()
            h_mse, h_n = heldout_fve_mse(
                model, tokenizer, heldout_pairs, cfg.critic_prompt_template,
                mse_scale_f, device, max_len=args.max_len,
            )
            model.train()
            h_fve = (1.0 - h_mse / heldout_baseline) * 100.0
            log["heldout_fve_pct"] = h_fve
            log["heldout_mse"] = h_mse
            print(f"  [heldout@{step}] mse {h_mse:.4f} | FVE {h_fve:.1f}% "
                  f"(n={h_n})", flush=True)

        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save ----
        if (step + 1) % args.save_every == 0 or (step + 1) == args.num_steps:
            out_dir = save_dir / f"iter_{step + 1:07d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[save] → {out_dir}", flush=True)
            if args.mode == "av":
                model.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
            elif args.use_lora:
                # AR + LoRA: save just the adapter weights + value_head (NOT the
                # 4-bit backbone). RL reloads via init_critic_from_base + inject.
                from safetensors.torch import save_file
                sd = {n: p.detach().cpu().contiguous()
                      for n, p in model.named_parameters()
                      if ("lora_" in n) or n.startswith("value_head")}
                save_file(sd, str(out_dir / "ar_lora_value_head.safetensors"))
                from nla.utils.arch_adapters import resolve_lora_target_modules
                (out_dir / "ar_meta.json").write_text(json.dumps({
                    "ar_num_layers": args.ar_num_layers,
                    "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                    "quant": args.quant,
                    # Echo the ACTUAL resolved targets (list or regex str; both
                    # json-safe and both accepted by LoraConfig at merge time) —
                    # a hardcoded list here silently broke fused-QKV archs and
                    # would mis-merge all-linear adapters as attn-only.
                    "target_modules": resolve_lora_target_modules(model.backbone.config),
                    # Whether the backbone's final RMSNorm was stripped at init.
                    # RL must rebuild the critic the same way or predictions
                    # silently shift (pre-2026-06 ckpts: norm kept = False).
                    "final_norm_stripped": args.strip_final_norm,
                }, indent=2))
                tokenizer.save_pretrained(str(out_dir))
            else:
                model.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
            # Copy the sidecar so the RL trainer can find injection_token_id etc.
            import shutil
            sidecar_src = Path(args.sidecar)
            if sidecar_src.is_file() and sidecar_src.suffix == ".parquet":
                sidecar_yaml = sidecar_src.with_suffix(".parquet.nla_meta.yaml")
                if sidecar_yaml.exists():
                    shutil.copy2(sidecar_yaml, out_dir / "nla_meta.yaml")

    print("done.", flush=True)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

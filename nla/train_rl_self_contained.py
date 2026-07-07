"""Self-contained NLA GRPO training: Karvonen injection, LoRA actor.

Architecture:
  - Actor: Qwen3-8B + Karvonen layer-1 ADD injection (from AV-SFT ckpt).
           Wrapped with LoRA so backbone stays frozen (memory: 16GB base + ~100MB
           LoRA + small Adam states + activations all fit on one H200).
  - Reference policy: same model with LoRA adapter disabled (PEFT context manager).
           No second model copy.
  - Critic: NLACriticModel (truncated K+1-layer backbone + value head). Frozen,
            bf16, eval-only — produces predicted activation given the actor's
            explanation. Reward = -MSE(pred, gold).
  - Rollout: HF model.generate() with the same Karvonen hook used in training.
             Slower than vLLM but no weight-sync complexity. On-policy.

GRPO objective (DeepSeekMath / DeepSeek-R1):
  L = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)] + beta * KL(pi || pi_ref)
  where r = exp(log_p_new - log_p_old), token-level
        A = group-relative reward, per-prompt baseline
        KL ≈ exp(log_p_ref - log_p_new) - (log_p_ref - log_p_new) - 1 (k3 estimator)

Per step:
  1. Sample B prompts from rl_shuf.parquet (each carries a gold activation v).
  2. Generate G samples per prompt with sampling temperature.
     Collect old log_probs from generate's output_scores.
  3. Extract <explanation>; failed extractions get reward = -2.0 (paper default,
     equals MSE on fully-orthogonal unit vectors — i.e. maximally bad).
  4. Score with critic → r_ij = -mse_nrm.
  5. Group-relative advantage: A_ij = (r_ij - mean_j) / std_j (per prompt group).
  6. Training-mode forward of the actor: compute new log_probs (LoRA active).
  7. Reference forward (same batch, LoRA disabled): compute ref log_probs.
  8. GRPO loss, backward + Adam.
"""

import argparse
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import (LoraConfig, PeftModel, get_peft_model,
                  inject_adapter_in_model, prepare_model_for_kbit_training)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb

from nla.utils import rl_logging
from nla.utils import build_prompt_text, cjk_fraction, critic_predict, register_karvonen_hook
from nla.utils.run_config import add_config_arg, apply_config_defaults, save_resolved_config

# Evals selectable via the config `evals:` list. base_fve is the core FVE.
KNOWN_EVALS = ("base_fve", "text_judges")
from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual, marker_well_formed
from nla.models import NLACriticModel
from nla.schema import (
    compute_predict_mean_baselines,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)
from nla.train_sft import _resolve_device_map, init_critic_from_base




def load_rl_dataset(parquet_path, n_max=None, exclude_doc_pred=None):
    """Streaming + vectorized load — reads only the columns/rows we need, and keeps
    activations as numpy float32 (zero-copy from arrow), NEVER python floats.

    The old `.to_pylist()` on the activation column materialized n_rows x 4096
    python float objects — at the auto-split's ~450k rows that's ~1.8B objects,
    ~60GB RAM and ~10 min of pure-python. numpy rows keep the same row-dict
    interface (torch.tensor(np_array) downstream) at ~7GB / ~seconds.

    exclude_doc_pred: optional doc_id -> bool; rows whose doc matches are DROPPED
    (the auto-split's held-out val docs — see nla/val_split.py). n_max counts
    kept rows.
    """
    import pyarrow.parquet as pq_inner
    pf = pq_inner.ParquetFile(parquet_path)
    cols = ["prompt", "activation_vector"] + (["doc_id"] if exclude_doc_pred else [])
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n_in_rg = rg.num_rows
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        if exclude_doc_pred is None:
            rg = rg.slice(0, take)   # slice-first only valid without a row filter
        prompts = rg.column("prompt").to_pylist()
        # list<float> column -> flat values array -> [n, dim] float32 view.
        # .flatten() respects the slice offsets; np.asarray is zero-copy.
        col = rg.column("activation_vector").combine_chunks()
        acts = np.asarray(col.flatten(), dtype=np.float32).reshape(len(prompts), -1)
        if exclude_doc_pred is not None:
            dids = rg.column("doc_id").to_pylist()
            for i, p in enumerate(prompts):
                if n_max is not None and len(rows) >= n_max:
                    break
                if not exclude_doc_pred(dids[i]):
                    rows.append({"prompt": p, "activation": acts[i]})
        else:
            for i, p in enumerate(prompts):
                rows.append({"prompt": p, "activation": acts[i]})
    return rows



@torch.no_grad()
def rollout_one_prompt(
    actor, tokenizer, prompt_text, activation, vectors_ref,
    inj_id, group_size, max_new_tokens, temperature, device,
    eos_ids=None,
):
    """Generate `group_size` samples for one prompt (pure on-policy: no log-probs
    are captured at rollout time — the single per-rollout update recomputes them)."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    batched = prompt_t.expand(group_size, -1).contiguous()
    v_batch = activation.unsqueeze(0).expand(group_size, -1).contiguous().to(device).float()
    vectors_ref[0] = v_batch
    try:
        gen_out = actor.generate(
            input_ids=batched,
            attention_mask=torch.ones_like(batched),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            # Explicitly override sampler — Qwen3's generation_config.json may
            # set top_p<1.0 / top_k>0 / repetition_penalty by default. Keep the
            # sampler unclamped so the sampled tokens match the policy the
            # single on-policy update scores them under.
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    finally:
        vectors_ref[0] = None
    full_ids = gen_out.sequences  # [G, prompt_len + new_len]
    prompt_len = prompt_t.shape[1]
    # Trim at ANY stop id, not just tokenizer.eos_token_id: Qwen3's
    # generation_config lists two ([151645 <|im_end|>, 151643 <|endoftext|>]);
    # a sample terminating via the second would otherwise keep one forced-pad
    # token in the loss.
    if eos_ids is None:
        eos_ids = {tokenizer.eos_token_id}
    responses = []
    for g in range(group_size):
        resp_ids = full_ids[g, prompt_len:].tolist()
        # Trim at the first stop token (inclusive). Batched generate() pads
        # samples that finish early to the group max with pad(=eos) tokens;
        # those positions were never sampled from the policy. Without
        # trimming, the recomputed logp + KL get gradient on garbage positions.
        n_real = next(
            (i + 1 for i, t in enumerate(resp_ids) if t in eos_ids),
            len(resp_ids),
        )
        resp_ids = resp_ids[:n_real]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        responses.append({
            "text": text,
            "full_ids": full_ids[g, : prompt_len + n_real],
            "prompt_len": prompt_len,
            "n_resp": n_real,
        })
    return responses



def score_with_critic(
    critic, tokenizer, explanations, activations, template, mse_scale_f, device,
    batch_size=32,
):
    """Returns list of rewards (None for failed extractions), reward = -recon_MSE.

    BATCHED critic forward (port of the vLLM twin's): right-padded + attention
    mask so critic_predict extracts each row's last REAL token. Identical
    rewards to the old per-rollout batch-1 loop, but ~batch_size fewer forwards
    (at B*G=512 the batch-1 loop was 512 forwards of the 5.4B critic)."""
    n = len(explanations)
    rewards = [None] * n
    pad_id = tokenizer.eos_token_id
    ids_list = [None] * n
    for i, expl in enumerate(explanations):
        if expl is None:
            continue
        ids = tokenizer.encode(template.format(explanation=expl), add_special_tokens=False)
        if 0 < len(ids) <= 1024:
            ids_list[i] = ids
    valid = [i for i in range(n) if ids_list[i] is not None]
    for cs in range(0, len(valid), batch_size):
        chunk = valid[cs:cs + batch_size]
        maxlen = max(len(ids_list[i]) for i in chunk)
        bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            L = len(ids_list[i])
            bx[r, :L] = torch.tensor(ids_list[i], dtype=torch.long, device=device)
            attn[r, :L] = 1
        with torch.no_grad():
            preds = critic_predict(critic, bx, attn, mse_scale_f)            # [B, d]
        gold = torch.stack([activations[i].to(device).float() for i in chunk], dim=0)
        pred_n = normalize_activation(preds, mse_scale_f)
        gold_n = normalize_activation(gold, mse_scale_f)
        mse = ((pred_n - gold_n) ** 2).mean(dim=1)                           # [B] per-row MSE
        for r, i in enumerate(chunk):
            m = mse[r].item()
            rewards[i] = (-m) if math.isfinite(m) else None
    return rewards


def grpo_token_loss(new_lp, ref_lp, advantage, kl_beta=0.04, kl_tok=None):
    """Pure ON-POLICY GRPO per-sample token loss + KL to reference.

    This trainer rolls out with the current policy and does exactly ONE
    gradient update per rollout, so training IS on-policy: the classic
    importance ratio between the sampling and training policies is identically
    1 and PPO clipping never engages. We therefore use the plain
    policy-gradient surrogate `advantage * new_lp` (whose gradient equals the
    ratio surrogate's at ratio == 1) and drop all ratio / clip machinery and
    the stale-logp diagnostics.

    new_lp: [n_resp] per-token policy log-probs (with grad).
    ref_lp: [n_resp] per-token reference log-probs (detached).
    advantage: scalar advantage for this sample (broadcast over tokens).
    kl_tok: optional per-token KL tensor (full analytic KL(policy||ref) — see
    --kl-estimator dist). When given it replaces the k3 term (bounded gradient,
    no exp of a single-sample Δ).

    Returns (loss_mean, kl_mean) as tensors (loss carries grad; kl detached).
    """
    ref_lp = ref_lp.detach()
    surrogate = advantage * new_lp
    if kl_tok is not None:
        kl = kl_tok                                    # full analytic KL (see --kl-estimator dist)
    else:
        # k3 KL estimator (unbiased, low-variance): kl ~= exp(d) - d - 1 with
        # d = ref - new; always >= 0.
        # k3 with a delta clamp — bounds the exp(delta) spike gradient while
        # keeping a correctly-signed pull toward the ref (see the vLLM twin).
        delta = (ref_lp - new_lp).clamp(max=12.0)
        kl = torch.exp(delta) - delta - 1.0
    per_tok = -(surrogate - kl_beta * kl)
    return per_tok.mean(), kl.detach().mean()


def grpo_update_microbatched(
    actor, optim, tokenizer, full_ids_list, prompt_lens, activations,
    advantages, vectors_ref, device,
    micro_batch=2, kl_beta=0.04, max_grad_norm=1.0,
    kl_estimator="k3", n_total=None,
):
    """Fused micro-batched forward+loss+backward for GRPO.

    Each micro-batch: forward (LoRA on, grad) → ref forward (LoRA off, no grad)
    → per-chunk GRPO loss → backward → release graph → next chunk. Single
    optim.step() at the end. Peak memory = one micro-batch graph instead of
    N retained graphs (which is what OOMs at B*G=256).

    Returns (mean_loss, grad_norm, metrics_dict).
    """
    optim.zero_grad()
    n = len(full_ids_list)
    sample_losses_log = []
    sample_kls_log = []
    sample_entropy_log = []   # mean per-token policy entropy over response tokens (nats)
    advantages = advantages.detach()  # no grad through advantage
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        bs = len(idxs)
        max_len = max(full_ids_list[i].numel() for i in idxs)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            batch_ids[row, :L] = full_ids_list[i].to(device)
            attn[row, :L] = 1
        v_batch = torch.stack(
            [activations[i].to(device).float() for i in idxs], dim=0,
        )
        # --- new logits (with grad) ---
        # vectors_ref stays set from here through this chunk's .backward(): under
        # gradient checkpointing the backward-time recompute re-fires the injection
        # hook, and clearing early makes the recompute SKIP the injection's
        # Jacobian (I + v_hat h_hat^T from the norm-match) — a silent gradient
        # error on exactly the marker pathway (verified vs no-checkpoint grads).
        vectors_ref[0] = v_batch
        new_logits = actor(input_ids=batch_ids, attention_mask=attn).logits   # [B,L,V] bf16
        # --- ref logits: switch to the frozen "reference" adapter (= AV-SFT init).
        #     Not disable_adapter() — the policy is a LoRA, so that would anchor
        #     KL to the bare base instead of the SFT init. ---
        try:
            with torch.no_grad():
                actor.set_adapter("reference")
                ref_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            actor.set_adapter("default")
        # SELECTIVE log-prob (ported from the vLLM twin): materialize fp32
        # log-probs ONLY at the response positions per row, never a full
        # [B,L,V] fp32 log_softmax (that double materialization was the memory
        # wall that kept this trainer at --logp-micro-batch 2). Identities:
        #   logp(tok) = logit[tok] - logsumexp(logits);  H = lse - E_p[logit].
        # Bit-for-bit the same as F.log_softmax(...).gather(...).
        # --- per-sample GRPO loss for this chunk ---
        chunk_losses = []
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                continue
            target_ids = batch_ids[row, p_len:L]
            pred_idx = torch.arange(p_len - 1, L - 1, device=device)
            resp_logits = new_logits[row].index_select(0, pred_idx).float()  # [n_resp, V]
            lse = torch.logsumexp(resp_logits, dim=-1)
            new_lp = resp_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - lse
            with torch.no_grad():   # policy entropy over response tokens (nats), logging only
                p_resp = (resp_logits - lse.unsqueeze(-1)).exp()
                sample_entropy_log.append(
                    float((lse - (p_resp * resp_logits).sum(-1)).mean()))
                del p_resp
            ref_resp_logits = ref_logits[row].index_select(0, pred_idx).float()
            ref_lse = torch.logsumexp(ref_resp_logits, dim=-1)
            ref_lp = (ref_resp_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                      - ref_lse).detach()
            if new_lp.numel() == 0:
                continue
            kl_tok = None
            if kl_estimator == "dist":
                # Full analytic KL(policy||ref) per token — bounded gradient, no
                # exp() of a single-sample log-ratio (the k3 heavy tail behind the
                # occasional grad/KL spikes). Computed on the response rows only;
                # unlike the vLLM twin's top-k truncation this keeps the EXACT sum.
                resp_logp = resp_logits - lse.unsqueeze(-1)
                ref_row_logp = ref_resp_logits - ref_lse.unsqueeze(-1)
                kl_tok = (resp_logp.exp() * (resp_logp - ref_row_logp)).sum(-1)
            # Pure on-policy surrogate (advantage * new_lp) + KL to ref.
            sample_loss, sample_kl = grpo_token_loss(
                new_lp, ref_lp, advantages[i], kl_beta=kl_beta, kl_tok=kl_tok,
            )
            chunk_losses.append(sample_loss)
            sample_kls_log.append(sample_kl.item())
        # Free logits before backward to bound peak (new_logits retained by graph).
        del ref_logits
        if not chunk_losses:
            vectors_ref[0] = None
            del new_logits
            continue
        # Scale so summed chunk losses give batch-mean.
        # Fixed-budget normalizer (see vLLM twin): dropped samples act as zeros.
        denom = n_total if n_total is not None else n
        chunk_loss = torch.stack(chunk_losses).sum() / denom
        chunk_loss.backward()
        vectors_ref[0] = None   # clear only AFTER backward (checkpoint recompute done)
        sample_losses_log.append(chunk_loss.item() * denom / len(chunk_losses))
        del new_logits
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in actor.parameters() if p.requires_grad], max_grad_norm,
    )
    gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    # Guard BEFORE stepping: clip_grad_norm_ does not sanitize nan/inf (it
    # scales by max_norm/total_norm, and nan propagates). Stepping Adam on
    # non-finite grads corrupts the moment estimates AND the weights.
    if math.isfinite(gn):
        optim.step()
    else:
        optim.zero_grad(set_to_none=True)
        print(f"[grpo] non-finite grad norm ({gn}) — skipping optimizer step",
              flush=True)
    metrics = {
        "kl_mean": float(np.mean(sample_kls_log)) if sample_kls_log else 0.0,
        "entropy": float(np.mean(sample_entropy_log)) if sample_entropy_log else 0.0,
    }
    mean_loss = float(np.mean(sample_losses_log)) if sample_losses_log else 0.0
    return mean_loss, gn, metrics


def main():
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--av-ckpt", required=True,
                   help="AV-SFT LoRA adapter dir (sits on --base-ckpt).")
    p.add_argument("--ar-ckpt", required=True,
                   help="AR-LoRA dir (ar_lora_value_head.safetensors + ar_meta.json).")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B",
                   help="Base the AV/AR LoRA adapters sit on (4-bit if --quant 4bit).")
    p.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--max-gpu-mem", type=int, default=0)
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-prompts", type=int, default=64,
                   help="prompts per step")
    p.add_argument("--group-size", type=int, default=8,
                   help="samples per prompt (for group baseline)")
    p.add_argument("--max-new-tokens", type=int, default=150)  # paper's rollout cap
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--text-judges-every", type=int, default=50,
                   help="Run the Opus text-attribute judges every N steps, reusing that "
                        "step's held-out eval generations (see the vLLM twin / "
                        "nla/utils/text_judges.py). Multiple of --eval-every; needs "
                        "ANTHROPIC_API_KEY; only active with `--evals ... text_judges`.")
    p.add_argument("--judge-concurrency", type=int, default=64,
                   help="Concurrent judge API calls for text_judges.")
    p.add_argument("--eval-temperature", type=float, default=None,
                   help="Sampling temperature for the held-out FVE eval only "
                        "(default: --temperature). 0 = greedy/deterministic — "
                        "matches the vLLM twin's knob so eval noise floors are "
                        "comparable across trainers.")
    p.add_argument("--lr", type=float, default=1e-4)   # matches train_rl_vllm
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True,
                   help="Use rsLoRA scaling (alpha/sqrt(r) instead of alpha/r). "
                        "Default ON because we use r=128 where vanilla LoRA's "
                        "alpha/r=0.125 collapses the effective learning rate.")
    p.add_argument("--train-ar", "--train-critic", dest="train_critic",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Co-train the AR (paper-faithful): a separate optimizer "
                        "for the AR + supervised reconstruction MSE on "
                        "(explanation, gold_activation) pairs each step. "
                        "(--train-critic kept as a deprecated alias.)")
    p.add_argument("--ar-lr", "--critic-lr", dest="critic_lr", type=float, default=8e-5)  # matches train_rl_vllm
    p.add_argument("--length-penalty", type=float, default=0.01,
                   help="HINGED length penalty: subtract length_penalty * "
                        "max(0, n_response_tokens - length_threshold) from the GRPO "
                        "signal (see the vLLM twin). 0 disables.")
    p.add_argument("--length-threshold", type=int, default=0,
                   help="Hinge point. 0 (default) => max_new_tokens - 64.")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False,
                   help="Recompute activations during backward (saves ~50% "
                        "activation memory at ~30%% compute cost). Off by "
                        "default — 8-bit Adam on critic gives bigger savings.")
    p.add_argument("--ar-micro-batch", "--critic-micro-batch", dest="critic_micro_batch",
                   type=int, default=4,
                   help="Micro-batch size for the AR's training-time forward. "
                        "Single full-batch forward OOMs at B*G=256.")
    p.add_argument("--logp-micro-batch", type=int, default=2)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume-from-lora", type=str, default=None,
                   help="Directory containing a saved LoRA adapter (iter_NNNNNN); "
                        "loaded as the policy adapter so training continues from "
                        "those weights (KL reference stays the AV-SFT init). If "
                        "the dir has a critic/ subdir, the co-trained critic is "
                        "resumed too.")
    p.add_argument("--start-step", type=int, default=0,
                   help="Initial step counter — useful when resuming so wandb "
                        "x-axis lines up with the previous run.")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Run the held-out FVE eval every N steps (default 5 for a "
                        "dense FVE curve). Logs eval/fve_pct + explanation Table and "
                        "per-eval wall-time under time/eval_*_s; 0 disables.")
    p.add_argument("--eval-n-prompts", type=int, default=64,
                   help="Number of fixed held-out prompts for per-step eval. "
                        "128+ for a trustworthy held-out FVE (20 is too noisy).")
    p.add_argument("--eval-skip-rows", type=int, default=0,
                   help="Take eval prompts from rl_shuf rows starting here. "
                        "0 (default) => AUTO = corpus - --val-rows.")
    p.add_argument("--val-rows", type=int, default=50000,
                   help="Rows reserved at the END of the corpus for held-out eval "
                        "(doc-disjoint); used when --max-rows/--eval-skip-rows are auto (0).")
    p.add_argument("--evals", nargs="*", default=["base_fve"],
                   help="Which evals to run each eval step (set this in the run YAML). "
                        f"Choices: {', '.join(KNOWN_EVALS)}. base_fve = held-out FVE.")
    p.add_argument("--eval-gen-batch", type=int, default=32,
                   help="Batch size for the held-out FVE eval's HF generation. "
                        "Eval was sequential (1 prompt at a time, ~8min for "
                        "n=128); batching cuts it to ~1min. Lower if it OOMs.")
    p.add_argument("--max-rows", type=int, default=0,
                   help="cap training rows. 0 (default) => AUTO = ENTIRE corpus minus the "
                        "last --val-rows (held-out). Set a small positive value for smoke runs.")
    p.add_argument("--kl-beta", type=float, default=None,
                   help="KL-penalty coefficient. Unset => estimator-conditional "
                        "default (k3: 0.01, dist: 0.2 — the analytic KL needs a "
                        "higher beta than k3 for equal regularization).")
    p.add_argument("--kl-estimator", choices=["k3", "dist"], default="k3",
                   help="Per-token KL-penalty form. k3 (DEFAULT) = exp(Δ)-Δ-1 on the "
                        "sampled token, Δ clamped to max=12 (bounds the spike "
                        "gradient). dist = full analytic KL(policy||ref) — "
                        "needs a higher beta (see the resolution below). Same "
                        "flag as the vLLM trainer (which truncates to top-k).")
    p.add_argument("--log-reward", action="store_true",
                   help="GRPO reward = -log(mse) instead of -mse (paper). Its gradient "
                        "(-1/mse) stays strong as mse shrinks, avoiding the -mse "
                        "advantage-collapse plateau. FVE/logging always use raw -mse, "
                        "so curves stay comparable across runs.")
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="rl",
                   help="wandb group for organizing the workspace (warmstart/rl/eval).")
    p.add_argument("--wandb-tags", default=None,
                   help="comma-separated wandb tags for explicit experiments (e.g. 'sweep,reward-ab').")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    apply_config_defaults(p)   # YAML (--config) -> argparse defaults; CLI still overrides
    args = p.parse_args()

    # ---- fail-fast checks (BEFORE any model/engine loading) ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite an existing run: iter_* checkpoints present
    # but no --resume-from-lora means a re-launch would clobber them.
    _existing_iters = sorted(save_dir.glob("iter_*"))
    if _existing_iters and args.resume_from_lora is None:
        raise SystemExit(
            f"[save] {save_dir} already contains {len(_existing_iters)} iter_* "
            f"checkpoints (latest: {_existing_iters[-1].name}) — refusing to "
            f"overwrite. Resume with --resume-from-lora {_existing_iters[-1]} "
            f"(+ --start-step, auto-defaulted from optim_latest) or use a fresh "
            f"--save-dir."
        )
    # RL checkpoints become self-describing: snapshot the resolved sidecar next
    # to them, and on resume ASSERT the tokens/extraction contract still agrees
    # (a wrong --sidecar silently retargets injection/extraction).
    from nla.schema import sidecar_path_for as _spf
    _side_src = _spf(args.sidecar)
    _side_dst = save_dir / "nla_meta.yaml"
    if _side_dst.exists():
        import yaml as _yaml
        _prev = _yaml.safe_load(_side_dst.read_text())
        _cur = _yaml.safe_load(_side_src.read_text())
        for _k in ("tokens", "extraction"):
            assert _prev.get(_k) == _cur.get(_k), (
                f"save-dir sidecar snapshot disagrees with --sidecar on {_k!r}: "
                f"this run would score/inject differently than the checkpoints "
                f"it resumes. Fix --sidecar or use a fresh --save-dir."
            )
    elif True:
        import shutil as _sh
        _sh.copy2(_side_src, _side_dst)


    _bad_evals = [e for e in args.evals if e not in KNOWN_EVALS]
    assert not _bad_evals, f"--evals: unknown {_bad_evals}; choices are {list(KNOWN_EVALS)}"
    if "text_judges" in args.evals:
        from nla.utils.text_judges import require_judge_key
        require_judge_key()
        assert args.text_judges_every > 0, "--text-judges-every must be > 0"
        assert args.eval_every > 0 and args.text_judges_every % args.eval_every == 0, (
            f"--text-judges-every {args.text_judges_every} must be a multiple of "
            f"--eval-every {args.eval_every}."
        )

    if args.kl_beta is None:
        # dist (full analytic KL) needs a higher beta than k3 for equal
        # regularization — k3's heavy-tail exp(Δ) gradient is itself an anchor.
        args.kl_beta = {"k3": 0.01, "dist": 0.2}[args.kl_estimator]  # matches train_rl_vllm
    print(f"[kl] estimator={args.kl_estimator} beta={args.kl_beta}", flush=True)
    if args.length_threshold <= 0:
        args.length_threshold = max(1, args.max_new_tokens - 64)
    print(f"[len] hinged penalty {args.length_penalty}/token past "
          f"{args.length_threshold} tokens (cap {args.max_new_tokens})", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"

    # On-policy consistency: rollouts are sampled at `temperature`, but the
    # update's forward scores tokens under the untempered policy (T=1). At T=1
    # the sampling and scoring distributions match, so the on-policy surrogate
    # is exact; at any other T the sampled tokens no longer come from the
    # distribution the update trains, silently biasing the gradient.
    assert args.temperature == 1.0, (
        f"--temperature {args.temperature} != 1.0: rollouts would be sampled "
        f"from a distribution the on-policy update does not score under. "
        f"Remove this assert only if you also temperature-scale the update's logp."
    )
    # Auto data split (mirror of train_rl_vllm): default (max_rows<=0 / eval_skip_rows<=0)
    # trains on the ENTIRE corpus minus a held-out ~--val-rows worth of DOCS, split by
    # DOC-HASH (nla/val_split.py) — a row boundary can't be doc-disjoint on the
    # row-shuffled corpus at ~90% train coverage. Positive values keep legacy behavior.
    val_permille = 0   # >0 => doc-hash split active
    if (args.max_rows is None or args.max_rows <= 0) or args.eval_skip_rows <= 0:
        import pyarrow.parquet as _pq
        from nla.val_split import val_doc_permille
        _total = _pq.ParquetFile(args.rl_parquet).metadata.num_rows
        val_permille = val_doc_permille(args.val_rows, _total)
        if args.eval_skip_rows is None or args.eval_skip_rows <= 0:
            args.eval_skip_rows = max(1, _total - args.val_rows)
        if args.max_rows is None or args.max_rows <= 0:
            args.max_rows = _total   # loader drops val-doc rows itself
        print(f"[data] auto-split (doc-hash): corpus={_total} rows, "
              f"~{val_permille / 10:.1f}% of docs held out for eval "
              f"(~{args.val_rows} rows); train = all rows of the other docs",
              flush=True)
    # Eval rows are taken past --eval-skip-rows; training samples rows[:max_rows].
    # Without this cap the training pool contains the literal eval rows.
    if args.eval_every > 0 and args.eval_n_prompts > 0 and val_permille == 0:
        assert args.max_rows is not None and args.max_rows <= args.eval_skip_rows, (
            f"evals enabled but --max-rows ({args.max_rows}) is unset or exceeds "
            f"--eval-skip-rows ({args.eval_skip_rows}) — training would include "
            f"the eval rows themselves."
        )

    # ---- tokenizer + nla config ----
    # From --base-ckpt, NOT hardcoded — the sidecar asserts below catch a
    # wrong-family tokenizer, but only if we load the one the run targets.
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    inject_char = cfg.injection_char
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template
    assert template is not None, "critic_prompt_template missing"
    print(f"[cfg] inj_id={inj_id} mse_scale_f={mse_scale_f} d_model={cfg.d_model}")

    # ---- actor: (4-bit) base + AV-SFT LoRA ("default", trainable) + frozen
    #      "reference" adapter (= AV-SFT init) for the KL anchor.
    #      The policy is now a LoRA *on* the frozen base, so disable_adapter()
    #      would anchor KL to the bare base, not the SFT init — hence a second
    #      frozen adapter. (See feedback_rl_ref_policy.)
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    dmap, max_mem = _resolve_device_map(args.device_map, args.max_gpu_mem, quant_config)
    print(f"[actor] base={args.base_ckpt} + AV-LoRA={args.av_ckpt} "
          f"(quant={args.quant}, device_map={args.device_map})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=quant_config, device_map=dmap, max_memory=max_mem,
    )
    if dmap is None:
        base = base.to(device)
    if quant_config is not None:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=args.gradient_checkpointing,
        )
    # Policy adapter: AV-SFT init, or a saved RL LoRA when resuming. The KL
    # "reference" adapter ALWAYS comes from the AV-SFT ckpt — resuming must
    # not move the KL anchor.
    policy_ckpt = args.resume_from_lora or args.av_ckpt
    if args.resume_from_lora:
        print(f"[actor] RESUMING policy LoRA from {args.resume_from_lora} "
              f"(KL reference stays {args.av_ckpt})")
    actor = PeftModel.from_pretrained(
        base, policy_ckpt, adapter_name="default", is_trainable=True,
    )
    actor.load_adapter(args.av_ckpt, adapter_name="reference")  # frozen KL ref
    actor.set_adapter("default")
    if args.resume_from_lora:
        # Sanity check: a resumed adapter should differ from the reference.
        # diff == 0 is either the historical resume-ignored bug OR a genuinely
        # zero-update leg (e.g. all extractions failed → adv ≡ 0); a hard
        # assert here would brick-loop the self-chaining sbatch on the latter,
        # so warn loudly instead. NOTE: this does NOT catch a partial adapter
        # load (PEFT load_adapter never raises on missing keys) — fresh-init
        # tensors also differ from the reference.
        _diff = 0.0
        _sd = actor.state_dict()
        for n in list(_sd):
            if ".default." in n and "lora_" in n:
                _ref = _sd.get(n.replace(".default.", ".reference."))
                if _ref is not None:
                    _diff += (_sd[n].float() - _ref.float()).pow(2).sum().item()
        print(f"[actor] resumed; sum((lora_default - lora_reference)²) = {_diff:.3e}")
        if _diff == 0.0:
            print(f"[actor] WARNING: resumed adapter is IDENTICAL to the AV-SFT "
                  f"reference — either {args.resume_from_lora} is untrained or "
                  f"the resume load silently failed.", flush=True)
    # The reference adapter is the frozen KL anchor — pin requires_grad False
    # explicitly. (PEFT's set_adapter toggles trainability when switching
    # adapters during the update loop; the optimizer snapshot taken below
    # protects against drift today, but be explicit so a future optimizer
    # rebuild can't silently start training the anchor.)
    for _n, _p in actor.named_parameters():
        if ".reference." in _n:
            _p.requires_grad_(False)
    actor.print_trainable_parameters()
    actor.train()
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        actor.enable_input_require_grads()
        print(f"[actor] gradient_checkpointing ENABLED")

    # ---- critic: AR-LoRA (4-bit base + injected LoRA + value_head). Rebuild
    #      the exact structure train_sft saved, then load the adapter + head. ----
    import json as _json
    from safetensors.torch import load_file as _load_file
    # When resuming and the resume dir has a saved co-trained critic, load
    # that instead of the AR-SFT init — otherwise the reward model snaps back
    # to its SFT state and the reward scale is discontinuous across the resume.
    ar_src = Path(args.ar_ckpt)
    if args.resume_from_lora is not None:
        _resumed_critic = Path(args.resume_from_lora) / "critic"
        if (_resumed_critic / "ar_lora_value_head.safetensors").exists():
            ar_src = _resumed_critic
            print(f"[ar] RESUMING co-trained critic from {ar_src}")
    ar_meta = _json.loads((ar_src / "ar_meta.json").read_text())
    print(f"[ar] AR-LoRA from {ar_src}: {ar_meta}")
    assert ar_meta.get("quant") != "4bit" or quant_config is not None, (
        f"AR-LoRA was trained on a 4-bit backbone (ar_meta quant=4bit) but this "
        f"run uses --quant {args.quant}: the LoRA's baked-in quantization-error "
        f"compensation would silently mismatch a bf16 backbone."
    )
    ar_quant = quant_config if ar_meta.get("quant") == "4bit" else None
    ar_dmap, ar_maxmem = _resolve_device_map(args.device_map, args.max_gpu_mem, ar_quant)
    critic = init_critic_from_base(
        args.base_ckpt, ar_meta["ar_num_layers"], torch.bfloat16,
        ar_quant, device_map=ar_dmap, max_memory=ar_maxmem,
        # Checkpoints record whether their backbone ran with the final RMSNorm
        # stripped (design §4) or kept (pre-2026-06 ckpts). Must match training.
        strip_final_norm=ar_meta.get("final_norm_stripped", False),
    )
    if ar_dmap is None:
        critic = critic.to(device)
    inject_adapter_in_model(LoraConfig(
        r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=ar_meta["target_modules"],
    ), critic.backbone)
    _ar_sd = _load_file(str(ar_src / "ar_lora_value_head.safetensors"))
    _miss, _unexp = critic.load_state_dict(_ar_sd, strict=False)
    # `missing` is always the whole frozen backbone (uninformative);
    # `unexpected` non-empty means a key-schema drift left the reward model
    # at init — fail loudly, not via a print nobody reads.
    _n_lora = sum(1 for k in _ar_sd if "lora_" in k)
    assert _n_lora > 0 and not _unexp, (
        f"AR weights load mismatch: {_n_lora} lora tensors in file, "
        f"unexpected={_unexp[:3]} — PEFT key-schema drift?"
    )
    print(f"[ar] loaded {len(_ar_sd)} AR tensors "
          f"(missing={len(_miss)} backbone keys, unexpected=0)")
    # Freeze everything; conditionally unfreeze LoRA + value_head for co-training.
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    critic_optim = None
    if args.train_critic:
        # Per paper §RL training: AR is co-trained simultaneously with AV on
        # the SAME explanations the actor produces this step. Loss = MSE against
        # the gold activation, normalised. AR's gradient does NOT flow back into
        # the actor (the explanation tokens are discrete — gradient stops there
        # automatically). Both backbone AND value_head train; the bf16+Adam
        # blow-up that NaN'd AR SFT is now neutralised by critic_predict's
        # normalize-before-value_head trick (bounds value_head input norm).
        # Co-train only the AR LoRA adapters + value_head (4-bit base frozen).
        for n_, p_ in critic.named_parameters():
            if ("lora_" in n_) or n_.startswith("value_head"):
                p_.requires_grad_(True)
        critic_trainable = [p for p in critic.parameters() if p.requires_grad]
        try:
            import bitsandbytes as _bnb
            critic_optim = _bnb.optim.AdamW8bit(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        except ImportError:
            critic_optim = torch.optim.AdamW(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        n_trainable = sum(p.numel() for p in critic_trainable)
        print(f"[ar] CO-TRAINED, lr={args.critic_lr}, "
              f"trainable={n_trainable/1e9:.2f}B (backbone + value_head)")
    else:
        print(f"[ar] FROZEN (eval-only scorer)")
    critic.eval()  # Qwen3 has no dropout — eval mode is fine for both grad/no-grad
    print(f"[ar] value_head shape={tuple(critic.value_head.weight.shape)}")

    # ---- karvonen hook on actor ----
    vectors_ref = [None]
    register_karvonen_hook(actor, vectors_ref, inj_id, left_id, right_id, layer_idx=1)

    # All stop ids for rollout EOS-trimming (Qwen3 lists two in its
    # generation_config; trimming on tokenizer.eos_token_id alone would leave
    # one forced-pad token in the loss for sequences stopping on the other).
    eos_ids = {tokenizer.eos_token_id}
    _gc_eos = getattr(getattr(actor, "generation_config", None), "eos_token_id", None)
    if _gc_eos is not None:
        eos_ids.update(_gc_eos if isinstance(_gc_eos, (list, tuple)) else [_gc_eos])
    eos_ids.discard(None)
    print(f"[rollout] EOS-trim ids: {sorted(eos_ids)}")

    # ---- dataset ----
    print(f"[data] loading {args.rl_parquet} (max_rows={args.max_rows}, "
          f"val_doc_permille={val_permille})", flush=True)
    from nla.val_split import is_val_doc
    _val_pred = (lambda d: is_val_doc(d, val_permille)) if val_permille else None
    rows = load_rl_dataset(args.rl_parquet, n_max=args.max_rows,
                           exclude_doc_pred=_val_pred)
    print(f"[data] {len(rows)} rows", flush=True)

    # ---- FVE baseline: predict-the-mean MSE on this dataset ----
    # FVE = 1 - mse_actual / baseline_mse, with the PAPER's baseline:
    # E[||v_norm - μ||²], the raw variance of the normalized distribution
    # (≈0.72 on Qwen 7B-class). NOTE: runs before 2026-06-09 used the looser
    # "meannorm" baseline MSE(v_norm, normalize(μ)) (≈0.94), which inflates
    # FVE vs the paper's definition — old wandb curves are not comparable.
    # Both are logged; `fve` uses the paper definition.
    _act_stack = torch.tensor(
        [r["activation"] for r in rows[: min(len(rows), 4000)]],
        dtype=torch.float32,
    )
    fve_baseline_meannorm, fve_baseline = compute_predict_mean_baselines(
        _act_stack, mse_scale_f,
    )
    del _act_stack
    print(f"[fve] predict-the-mean baseline mse_nrm = {fve_baseline:.4f} "
          f"(paper def; meannorm baseline = {fve_baseline_meannorm:.4f})",
          flush=True)

    # ---- optimizer ----
    # 8-bit Adam (bitsandbytes) for both actor LoRA and critic — block-wise
    # int8 quantization of (m, v) state cuts optimizer memory ~4×. "Paged"
    # variant CPU-offloads pages under memory pressure. Standard choice for
    # memory-constrained LLM fine-tuning; numerically equivalent to fp32 Adam
    # within bf16 noise for our use case.
    try:
        import bitsandbytes as bnb
        _adam_cls = bnb.optim.AdamW8bit
        print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        _adam_cls = torch.optim.AdamW
        print(f"[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = _adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Resume: restore Adam moments (saved latest-only alongside checkpoints) —
    # same layout as the vLLM twin.
    if args.resume_from_lora is not None:
        from nla.utils.resume import find_optim_ckpt, warn_cold_adam
        # Search save_dir AND the resumed LoRA's parent dir: branch-style
        # resumes (old run's LoRA, new save-dir) used to silently miss the old
        # run's optim_latest -> cold Adam (no second-moment history) ->
        # entropy-death spiral on late-stage policies (2/2 in the full repo).
        _opt_ckpt = find_optim_ckpt(args.save_dir, args.resume_from_lora)
        if _opt_ckpt is not None:
            print(f"[resume] optimizer state: {_opt_ckpt}", flush=True)
            _opt_st = torch.load(str(_opt_ckpt), map_location="cpu", weights_only=True)
            # Default --start-step from the saved step: forgetting the flag used
            # to silently replay data from index 0, restart the wandb x-axis,
            # and overwrite iter_ checkpoints.
            _saved_step = int(_opt_st.get("step", 0))
            if args.start_step == 0 and _saved_step > 0:
                args.start_step = _saved_step
                print(f"[resume] --start-step not given — defaulting to "
                      f"optim_latest's saved step {_saved_step}.", flush=True)
            elif args.start_step != _saved_step:
                print(f"[resume] WARN: --start-step {args.start_step} != saved "
                      f"step {_saved_step} — trusting the explicit flag.", flush=True)
            # Graceful on mismatch (config drift between save and resume):
            # restart the affected moments instead of killing the resume.
            try:
                optim.load_state_dict(_opt_st["actor_optim"])
                if critic_optim is not None and "critic_optim" in _opt_st:
                    critic_optim.load_state_dict(_opt_st["critic_optim"])
                print(f"[resume] optimizer state restored (saved at step "
                      f"{_opt_st.get('step', '?')})", flush=True)
            except (ValueError, KeyError, RuntimeError) as _e:
                print(f"[resume] WARN: optimizer state incompatible ({_e}) — "
                      f"Adam moments restart.", flush=True)
        else:
            warn_cold_adam(args.start_step)

    # Snapshot the fully-resolved run config (defaults+YAML+CLI) next to the ckpt.
    save_resolved_config(args, args.save_dir)
    print(f"[cfg] evals: {args.evals}", flush=True)

    # ---- wandb (shared init — see nla/rl_logging.py) ----
    if not args.no_wandb:
        rl_logging.init_wandb(
            args, rollout_tag="single-gpu",
            fve_baseline=fve_baseline, fve_baseline_meannorm=fve_baseline_meannorm,
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pending_idxs = list(range(len(rows)))
    rng.shuffle(pending_idxs)
    cursor = 0

    # ---- Fixed held-out eval prompts, DOC-DISJOINT from training rows.
    # Stage-1 only guarantees disjointness BETWEEN av_sft/ar_sft/rl FILES;
    # within rl_shuf.parquet, rows past --eval-skip-rows can share doc_id
    # with rows before it (the file is row-shuffled, not doc-partitioned
    # internally). Without explicit filtering we measured ~50% doc-overlap.
    # Fix: scan training-window (rows 0..eval_skip_rows) to collect doc_ids,
    # then take eval rows past the cursor whose doc_id is NOT in that set.
    eval_rows = []
    if args.eval_every > 0 and args.eval_n_prompts > 0 and val_permille > 0:
        # Doc-hash split: eval rows = first eval_n_prompts rows of HELD-OUT docs
        # (~val_permille/1000 of rows, scattered) — doc-disjoint from training by
        # construction (see nla/val_split.py).
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.rl_parquet)
        for _rg_idx in range(_pf.num_row_groups):
            if len(eval_rows) >= args.eval_n_prompts:
                break
            _tj_cols = (["detokenized_text_truncated"]
                        if "detokenized_text_truncated" in _pf.schema_arrow.names else [])
            _rg = _pf.read_row_group(
                _rg_idx, columns=["prompt", "activation_vector", "doc_id"] + _tj_cols)
            _prompts = _rg.column("prompt").to_pylist()
            _acts = np.asarray(
                _rg.column("activation_vector").combine_chunks().flatten(),
                dtype=np.float32).reshape(len(_prompts), -1)
            _dids = _rg.column("doc_id").to_pylist()
            _srcs = (_rg.column("detokenized_text_truncated").to_pylist()
                     if _tj_cols else [""] * len(_prompts))
            for _i, _d in enumerate(_dids):
                if is_val_doc(_d, val_permille):
                    eval_rows.append({"prompt": _prompts[_i], "activation": _acts[_i],
                                      "source": _srcs[_i] or ""})
                    if len(eval_rows) >= args.eval_n_prompts:
                        break
        print(f"[eval] {len(eval_rows)} held-out-doc prompts loaded "
              f"(doc-hash split, {val_permille / 10:.1f}% of docs)", flush=True)
    elif args.eval_every > 0 and args.eval_n_prompts > 0:
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.rl_parquet)
        # Pass 1: training-window doc_ids
        _train_doc_ids: set = set()
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if _seen >= args.eval_skip_rows:
                break
            _rg = _pf.read_row_group(_rg_idx, columns=["doc_id"])
            _ids = _rg.column("doc_id").to_pylist()
            _nrg = len(_ids)
            _take = min(_nrg, args.eval_skip_rows - _seen)
            _train_doc_ids.update(_ids[:_take])
            _seen += _nrg
        # Pass 2: doc-disjoint rows past the cursor
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if len(eval_rows) >= args.eval_n_prompts:
                break
            _rg = _pf.read_row_group(
                _rg_idx, columns=["prompt", "activation_vector", "doc_id"]
                + (["detokenized_text_truncated"]
                   if "detokenized_text_truncated" in _pf.schema_arrow.names else []),
            )
            _n = _rg.num_rows
            if _seen + _n <= args.eval_skip_rows:
                _seen += _n
                continue
            _start = max(0, args.eval_skip_rows - _seen)
            _prompts = _rg.column("prompt").to_pylist()
            _acts = _rg.column("activation_vector").to_pylist()
            _dids = _rg.column("doc_id").to_pylist()
            _srcs = (_rg.column("detokenized_text_truncated").to_pylist()
                     if "detokenized_text_truncated" in _rg.schema.names else [""] * _n)
            for _i in range(_start, _n):
                if _dids[_i] in _train_doc_ids:
                    continue
                eval_rows.append({"prompt": _prompts[_i], "activation": _acts[_i],
                                  "source": _srcs[_i] or ""})
                if len(eval_rows) >= args.eval_n_prompts:
                    break
            _seen += _n
        print(f"[eval] {len(eval_rows)} doc-disjoint prompts loaded "
              f"(rows past {args.eval_skip_rows}, excluding "
              f"{len(_train_doc_ids)} training doc_ids)", flush=True)
    # ---- eval-set FVE baseline (see the vLLM twin): held-out FVE must divide by
    # the variance of the population actually being scored, not the train set's.
    eval_fve_baseline = fve_baseline
    if eval_rows:
        _e_acts = torch.stack([
            torch.as_tensor(r["activation"], dtype=torch.float32) for r in eval_rows
        ])
        _, eval_fve_baseline = compute_predict_mean_baselines(_e_acts, mse_scale_f)
        del _e_acts
        print(f"[fve] eval-set baseline = {eval_fve_baseline:.4f} "
              f"(train-set baseline = {fve_baseline:.4f})", flush=True)

    eval_table_data = []  # accumulates [step, idx, reward, fve, extracted, explanation]

    # Resume (--start-step > 0): fast-forward the data cursor + RNG through the
    # steps already taken, replaying the exact shuffle-on-wrap sequence (seeded
    # rng => deterministic), so resumed training continues on the data the
    # original run would have seen next instead of replaying from index 0.
    if args.start_step > 0:
        for _ in range(args.start_step):
            if cursor + args.batch_prompts > len(pending_idxs):
                rng.shuffle(pending_idxs)
                cursor = 0
            cursor += args.batch_prompts
        print(f"[data] fast-forwarded cursor through {args.start_step} steps "
              f"(cursor={cursor})", flush=True)

    for step in range(args.start_step, args.num_steps):
        t0 = time.time()
        # ---- batch select ----
        if cursor + args.batch_prompts > len(pending_idxs):
            rng.shuffle(pending_idxs)
            cursor = 0
        batch_idxs = pending_idxs[cursor : cursor + args.batch_prompts]
        cursor += args.batch_prompts

        # ---- rollouts ----
        actor.eval()
        all_full_ids = []
        all_prompt_lens = []
        all_activations = []
        all_explanations = []
        all_response_text = []
        all_prompt_group = []
        all_resp_lens = []  # response token count per sample (for length/throughput logging)
        for gi, row_idx in enumerate(batch_idxs):
            row = rows[row_idx]
            prompt_text = build_prompt_text(row["prompt"], inject_char, tokenizer)
            activation = torch.tensor(row["activation"], dtype=torch.float32)
            responses = rollout_one_prompt(
                actor, tokenizer, prompt_text, activation, vectors_ref,
                inj_id, args.group_size, args.max_new_tokens, args.temperature, device,
                eos_ids=eos_ids,
            )
            for r in responses:
                expl = extract_explanation(r["text"])
                all_full_ids.append(r["full_ids"])
                all_prompt_lens.append(r["prompt_len"])
                all_activations.append(activation)
                all_explanations.append(expl)
                all_response_text.append(r["text"])
                all_prompt_group.append(gi)
                all_resp_lens.append(int(r["n_resp"]))
        t_gen_end = time.time()  # pure HF generation time (the rollout loop above)

        # ---- per-rollout injection-success checks -> training mask ----
        # Don't train on rollouts whose injection failed (no usable signal; corrupts
        # the AR's targets). marker_ok = mechanism check (prompt still has exactly one
        # well-formed marker, distribution-invariant); cjk_fail = the output-symptom
        # backstop (RL erodes it). A rollout failing EITHER is dropped from the AV
        # update, AR co-training, and the GRPO group baseline. (No vLLM steering-apply
        # check here — this path injects in-process via the HF Karvonen hook.)
        cjk_fail = [cjk_fraction(t) > 0.05 for t in all_response_text]
        marker_ok = [
            marker_well_formed(
                all_full_ids[i][: all_prompt_lens[i]].tolist(), inj_id, left_id, right_id
            )
            for i in range(len(all_full_ids))
        ]
        inject_ok = [
            (not cjk_fail[i]) and marker_ok[i] for i in range(len(all_full_ids))
        ]
        n_inject_fail = int(sum(cjk_fail))
        n_marker_bad = int(sum(1 for m in marker_ok if not m))
        n_inject_masked = int(sum(1 for ok in inject_ok if not ok))
        inject_ok_t = torch.tensor(inject_ok, dtype=torch.bool, device=device)
        # Truncated rollouts (hit the max_new_tokens cap) are scored as FAILED
        # (-2 floor) and TRAINED ON — the failure reward is the anti-runaway
        # gradient (masking them out collapsed the policy; see the vLLM twin).
        # Match the vLLM twin's finish_reason semantics: a sample whose EOS
        # lands exactly on the cap's last token STOPPED (not truncated) — only
        # cap-length rollouts that never emitted a stop id count as truncated.
        truncated = [
            (all_full_ids[i].numel() - all_prompt_lens[i] >= args.max_new_tokens)
            and (int(all_full_ids[i][-1]) not in eos_ids)
            for i in range(len(all_full_ids))
        ]
        n_truncated = int(sum(truncated))

        # ---- scoring ----
        # `rewards` holds the reconstruction reward (-MSE) and feeds FVE logging.
        # Length shaping is applied only to `rewards_t` (the GRPO signal), so FVE
        # stays a pure reconstruction metric comparable across runs.
        rewards = score_with_critic(
            critic, tokenizer, all_explanations, all_activations,
            template, mse_scale_f, device,
        )
        # TRUNCATED -> FAILED: a cap-truncated rollout must not be scored as if
        # its explanation were complete — it gets the -2 failure reward, which IS
        # trained on (the anti-runaway gradient). Keeps FVE/extraction honest.
        rewards = [None if t else r for r, t in zip(rewards, truncated)]
        # GRPO reward fill + optional -log transform. `rewards` holds raw -mse
        # (or None for failed extraction); FVE below uses these raw values, so
        # the FVE curve is identical regardless of --log-reward.
        # Failed-extraction floor = the orthogonal-vector outcome (mse=2.0):
        #   default -mse -> -2.0 ;  --log-reward -log(mse) -> -log(2.0) ≈ -0.69.
        # -log(mse) keeps the reward gradient (-1/mse) strong as mse shrinks,
        # avoiding the -mse advantage-collapse plateau (FVE flatlines ~0.47).
        if args.log_reward:
            _floor = -math.log(2.0)
            rewards_filled = [
                _floor if r is None else -math.log(min(max(-r, 1e-3), 2.0))
                for r in rewards
            ]
        else:
            rewards_filled = [-2.0 if r is None else r for r in rewards]
        rewards_t = torch.tensor(rewards_filled, dtype=torch.float32, device=device)

        # ---- reward shaping (length penalty) ----
        # Subtracted from the GRPO signal only. Default (0) is a no-op.
        shape_terms = {}
        if args.length_penalty > 0:
            n_tok = torch.tensor(
                all_resp_lens, dtype=torch.float32, device=device,
            )
            overage = (n_tok - float(args.length_threshold)).clamp_min(0.0)
            rewards_t = rewards_t - args.length_penalty * overage
            shape_terms["av/len_pen_mean"] = (args.length_penalty * overage).mean().item()
            shape_terms["av/len_overage_frac"] = float((overage > 0).float().mean())

        # ---- GRPO group-relative advantage (per-prompt mean & std) ----
        group_t = torch.tensor(all_prompt_group, dtype=torch.long, device=device)
        adv = torch.zeros_like(rewards_t)
        shape_terms["av/truncated_count"] = n_truncated
        for gi in range(args.batch_prompts):
            # exclude injection-failed rollouts from the group baseline; truncated
            # participate with the -2 failure reward.
            mask = (group_t == gi) & inject_ok_t
            if mask.sum() == 0:
                continue
            group_r = rewards_t[mask]
            mu = group_r.mean()
            sd = group_r.std() if group_r.numel() > 1 else torch.tensor(1.0, device=device)
            adv[mask] = (group_r - mu) / (sd + 1e-6)

        # ---- GRPO update: fused forward+loss+backward per micro-batch ----
        # Previous code did all forwards then all backwards, which retained
        # every micro-batch's compute graph and OOM'd at B*G=256. The fused
        # version releases each chunk's graph before starting the next.
        # Drop ONLY injection-failed rollouts; truncated stay in with -2.
        keep = [i for i, ok in enumerate(inject_ok) if ok]
        if not keep:
            print(f"step {step}: all {len(inject_ok)} rollouts failed the injection "
                  f"checks (cjk/marker) — skipping AV+AR update this step.", flush=True)
            continue
        upd_full_ids = [all_full_ids[i] for i in keep]
        upd_prompt_lens = [all_prompt_lens[i] for i in keep]
        upd_activations = [all_activations[i] for i in keep]
        upd_adv = adv.index_select(0, torch.tensor(keep, device=device))
        actor.train()
        mean_loss_val, grad_norm_val, grpo_metrics = grpo_update_microbatched(
            actor, optim, tokenizer,
            upd_full_ids, upd_prompt_lens, upd_activations,
            upd_adv, vectors_ref, device,
            micro_batch=args.logp_micro_batch,
            kl_beta=args.kl_beta,
            max_grad_norm=args.max_grad_norm,
            kl_estimator=args.kl_estimator,
            n_total=len(inject_ok),   # fixed budget: dropped rollouts act as zeros
        )
        # Build a scalar-tensor stand-in for the existing logging path that
        # expects a `loss` tensor with .item().
        loss = torch.tensor(mean_loss_val, device=device)
        grad_norm = torch.tensor(grad_norm_val, device=device)
        if not math.isfinite(mean_loss_val):
            print(
                f"step {step}: loss={mean_loss_val} non-finite "
                f"(kl={grpo_metrics.get('kl_mean')}). Skipping critic update.",
                flush=True,
            )
            # The helper already refused to optim.step() on a non-finite grad
            # norm, so weights are intact; skip the critic update + logging.
            continue

        # ---- AR critic co-training (paper-faithful, optional) ----
        # Per paper §RL: "Update the AR by one step of gradient descent on the
        # regression loss ||h_l − AR_θ(z)||²_2". Inputs z = the explanations the
        # actor just produced this step; targets h_l = the gold activations.
        # Gradient from this update does NOT flow into the actor (z is discrete).
        critic_loss_val = float("nan")
        critic_grad_norm_val = float("nan")
        if args.train_critic and critic_optim is not None:
            crit_inputs = []
            crit_golds = []
            # `keep` excludes injection-failed rollouts (cjk/marker) from AR targets.
            for i in keep:
                expl = all_explanations[i]
                act = all_activations[i]
                if expl is None:
                    continue
                text = template.format(explanation=expl)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 1024 or len(ids) == 0:
                    continue
                crit_inputs.append(torch.tensor(ids, dtype=torch.long))
                crit_golds.append(act)
            if crit_inputs:
                # Micro-batch the critic update — single forward on 256 sequences
                # × 200 tokens × 5.5B-param critic with grad blows past 130GB.
                # Accumulate gradient across micro-batches, single step at the
                # end (loss is divided by total bs so it averages correctly).
                bs_total = len(crit_inputs)
                pad_id = tokenizer.eos_token_id
                critic_optim.zero_grad()
                accumulated = 0.0
                finite = True
                cmb = max(1, args.critic_micro_batch)
                for cs in range(0, bs_total, cmb):
                    chunk = list(range(cs, min(cs + cmb, bs_total)))
                    max_len = max(crit_inputs[i].numel() for i in chunk)
                    bs = len(chunk)
                    batch_ids = torch.full(
                        (bs, max_len), pad_id, dtype=torch.long, device=device,
                    )
                    attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
                    for row, i in enumerate(chunk):
                        L = crit_inputs[i].numel()
                        batch_ids[row, :L] = crit_inputs[i].to(device)
                        attn[row, :L] = 1
                    pred = critic_predict(critic, batch_ids, attn, mse_scale_f)
                    gold = torch.stack([crit_golds[i] for i in chunk]).to(device).float()
                    pred_n = normalize_activation(pred, mse_scale_f)
                    gold_n = normalize_activation(gold, mse_scale_f)
                    # Scale so the sum across micro-batches = MSE over full batch.
                    chunk_loss = F.mse_loss(pred_n, gold_n) * (bs / bs_total)
                    if not torch.isfinite(chunk_loss):
                        print(f"step {step}: critic loss non-finite (chunk {cs}), skipping", flush=True)
                        finite = False
                        break
                    chunk_loss.backward()
                    accumulated += chunk_loss.item()
                if finite:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        critic_trainable, args.max_grad_norm,
                    )
                    critic_optim.step()
                    critic_loss_val = accumulated  # already the full-batch mean
                    critic_grad_norm_val = (
                        critic_grad_norm.item()
                        if hasattr(critic_grad_norm, "item")
                        else float(critic_grad_norm)
                    )

        # ---- logging ----
        valid_rewards = [r for r in rewards if r is not None]
        n_valid = len(valid_rewards)
        n_total = len(rewards)
        extraction_rate = n_valid / n_total if n_total else 0
        # n_inject_fail (CJK), n_marker_bad, n_inject_masked were computed at rollout
        # time above (over the full rollout set, before the inject_ok filter).
        # Response lengths come from the rollout (one entry per sample).
        n_resps_t = torch.tensor(
            all_resp_lens, dtype=torch.float32, device=device,
        )
        # Truncation canary (truncated rollouts are scored -2 and trained on).
        frac_cut_off = float(np.mean(truncated)) if truncated else 0.0
        if frac_cut_off > 0.02:
            print(f"[WARN step {step}] {frac_cut_off:.0%} of rollouts hit the "
                  f"max_new_tokens={args.max_new_tokens} cap (truncated -> trained with "
                  f"the failure reward). Raise --max-new-tokens if persistent.", flush=True)
        # FVE on valid (non-extraction-failed) samples — gives an
        # interpretable curve in wandb that maps to paper's reported numbers.
        # Use valid rewards only so extraction failures don't bias FVE down.
        fve = (
            1.0 - (-float(np.mean(valid_rewards))) / fve_baseline
            if valid_rewards else float("nan")
        )
        # wandb x-axis is `step` (passed to wandb.log below) — do NOT also log
        # "step" as a metric, or it shows up as a useless step-vs-step chart.
        # Two model-named groups (av_, ar_) + a reward_ group; headline numbers
        # (fve_pct, wall_s) are top-level. wandb x-axis is `step` (passed below) —
        # don't log "step" as a metric.
        log = rl_logging.build_step_log(
            fve=fve,
            grad_norm=grad_norm,
            grpo_metrics=grpo_metrics,
            adv=adv,
            valid_rewards=valid_rewards,
            extraction_rate=extraction_rate,
            inject_fail_count=n_inject_fail,
            resp_len_mean=n_resps_t.mean().item(),
            ar_recon_mse=critic_loss_val,
            ar_grad_norm=critic_grad_norm_val,
            wall_s=time.time() - t0,
            shape_terms=shape_terms,
            marker_bad_count=n_marker_bad,
            inject_masked_count=n_inject_masked,
            frac_cut_off=frac_cut_off,
        )
        # rollout generation throughput (tok/s) — same metric as the vLLM path for comparison
        _gen_s = max(1e-6, t_gen_end - t0)
        _gen_tokens = sum(int(n) for n in all_resp_lens)
        log["rollout/gen_tok_per_s"] = _gen_tokens / _gen_s
        log["rollout/gen_tokens"] = float(_gen_tokens)
        log["rollout/n_rollouts"] = float(len(all_resp_lens))
        log["rollout/gen_s"] = _gen_s
        print(rl_logging.format_console_line(step, log, train_ar=args.train_critic), flush=True)

        # ---- per-step eval: every N steps, run actor (current weights) on a
        # FIXED set of held-out prompts and log explanations as a wandb Table.
        # Lets you scrub through the run and watch explanations evolve.
        if args.eval_every > 0 and step % args.eval_every == 0:
            actor.eval()
            _t_eval_fve = time.time()   # [timing] base_fve = HF gen + critic scoring
            eval_rewards_s = []
            eval_records = []
            # --- Phase 1: BATCHED HF generation. Was sequential (one prompt at
            # a time, ~8 min for n=128); batching cuts it to ~1 min. The marker
            # hook scans for the token per-row, so a [B,d] vectors_ref injects
            # each row's activation at its own marker — batch-consistency is
            # locked down by tests/test_eval_batching.py.
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            _eval_prompts = [
                build_prompt_text(r["prompt"], inject_char, tokenizer)
                for r in eval_rows
            ]
            _eval_acts = [torch.tensor(r["activation"], dtype=torch.float32) for r in eval_rows]
            _all_resp = []
            _orig_pad = tokenizer.padding_side
            tokenizer.padding_side = "left"  # generation: left-pad so completions align
            try:
                for _c0 in range(0, len(_eval_prompts), args.eval_gen_batch):
                    _cp = _eval_prompts[_c0:_c0 + args.eval_gen_batch]
                    _ca = _eval_acts[_c0:_c0 + args.eval_gen_batch]
                    _enc = tokenizer(
                        _cp, return_tensors="pt", padding=True, add_special_tokens=False,
                    ).to(device)
                    vectors_ref[0] = torch.stack(_ca).to(device).float()
                    try:
                        with torch.no_grad():
                            _et = (args.eval_temperature
                                   if args.eval_temperature is not None
                                   else args.temperature)
                            _gen = actor.generate(
                                input_ids=_enc.input_ids, attention_mask=_enc.attention_mask,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=(_et > 0),
                                **({"temperature": _et} if _et > 0 else {}),
                                top_p=1.0, top_k=0, repetition_penalty=1.0,
                                pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=True,
                            )
                    finally:
                        vectors_ref[0] = None
                    _new = _gen.sequences[:, _enc.input_ids.shape[1]:]
                    for _i in range(len(_cp)):
                        _all_resp.append(tokenizer.decode(_new[_i], skip_special_tokens=True))
            finally:
                tokenizer.padding_side = _orig_pad
            # --- Phase 2: scoring (per-row, reads the pre-generated responses) ---
            for ei, row in enumerate(eval_rows):
                activation = _eval_acts[ei]
                resp = _all_resp[ei]
                expl = extract_explanation(resp)
                e_reward = -2.0
                if expl is not None:
                    ctext = template.format(explanation=expl)
                    cids = tokenizer.encode(ctext, add_special_tokens=False)
                    if 0 < len(cids) <= 1024:
                        x = torch.tensor([cids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            pred = critic_predict(critic, x, None, mse_scale_f)[0]
                        gold = activation.to(device).float()
                        pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
                        gn = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
                        mse = F.mse_loss(pn, gn).item()
                        if math.isfinite(mse):
                            e_reward = -mse
                eval_rewards_s.append(e_reward)
                eval_records.append({
                    "step": step, "idx": ei, "reward": e_reward,
                    "fve": (1.0 - (-e_reward) / eval_fve_baseline) if e_reward > -2.0 else float("nan"),
                    "extracted": expl is not None,
                    "explanation": expl if expl is not None else "<extraction failed>",
                })
            # Aggregate eval scalars
            valid_e = [r for r in eval_rewards_s if r > -2.0]
            log["eval/reward_mean"] = (
                float(np.mean(eval_rewards_s)) if eval_rewards_s else float("nan")
            )
            log["eval/reward_mean_valid"] = (
                float(np.mean(valid_e)) if valid_e else float("nan")
            )
            log["eval/fve_pct"] = (
                (1.0 - (-float(np.mean(valid_e))) / eval_fve_baseline) * 100.0
                if valid_e else float("nan")
            )
            log["eval/extraction_rate"] = (
                sum(1 for r in eval_records if r["extracted"]) / len(eval_records)
                if eval_records else 0.0
            )
            log["time/eval_base_fve_s"] = time.time() - _t_eval_fve  # FVE eval cost (gen+scoring)
            # Persistent table — accumulates example generations across the whole
            # run (scrub by step to watch explanations evolve).
            for r in eval_records:
                eval_table_data.append([
                    r["step"], r["idx"], r["reward"], r["fve"],
                    r["extracted"], r["explanation"][:600],
                ])
            if not args.no_wandb:
                log["eval/samples"] = wandb.Table(
                    columns=["step", "idx", "reward", "fve", "extracted", "explanation"],
                    data=list(eval_table_data),
                )
            print(
                f"  [eval@{step}] reward {log['eval/reward_mean']:.3f} "
                f"| FVE {log['eval/fve_pct']:.1f}% "
                f"| ext {log['eval/extraction_rate']:.0%}",
                flush=True,
            )
            # ---- Opus text-attribute judges (opt-in; reuses THIS round's
            # generations — see nla/utils/text_judges.py) ----
            if "text_judges" in args.evals and step % args.text_judges_every == 0:
                from nla.utils.text_judges import RUBRIC_PROMPTS, judge_explanations
                _t_tj = time.time()
                _tj_expl = [r["explanation"] if r["extracted"] else None
                            for r in eval_records]
                _tj_src = [row.get("source", "") for row in eval_rows]
                tj_metrics, _ = judge_explanations(
                    _tj_expl, _tj_src, seed=args.seed,
                    concurrency=args.judge_concurrency)
                log.update({f"eval_judge/{k}": v for k, v in tj_metrics.items()})
                log["time/eval_text_judges_s"] = time.time() - _t_tj
                print(
                    f"  [text_judges@{step}] "
                    + " ".join(f"{d} {tj_metrics[d + '_mean']:.2f}"
                               for d in RUBRIC_PROMPTS)
                    + f" | match {tj_metrics['source_match_acc']:.0%}"
                    f" | judge_fail {tj_metrics['judge_fail_rate']:.0%}"
                    f" | {log['time/eval_text_judges_s']:.0f}s",
                    flush=True,
                )
            # Print 3 sample explanations so the log itself shows how outputs
            # evolve. Pick indices 0, 7, 14 — spread across the eval set.
            for _ei in (0, 7, 14):
                if _ei < len(eval_records):
                    _r = eval_records[_ei]
                    _expl = _r["explanation"][:200].replace("\n", " ")
                    print(
                        f"    [eval@{step} idx={_ei} r={_r['reward']:.3f}] {_expl}",
                        flush=True,
                    )

        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save LoRA periodically ----
        if (step + 1) % args.save_every == 0:
            out_dir = save_dir / f"iter_{step + 1:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Critic FIRST: resume keys off the actor's adapter_config.json,
            # so a crash between the two saves must leave critic-without-actor
            # (resume fails loudly) rather than actor-without-critic (resume
            # silently falls back to the SFT critic → reward discontinuity).
            if args.train_critic:
                from safetensors.torch import save_file as _save_file
                crit_dir = out_dir / "critic"
                crit_dir.mkdir(exist_ok=True)
                _crit_sd = {n: p_.detach().cpu().contiguous()
                            for n, p_ in critic.named_parameters()
                            if ("lora_" in n) or n.startswith("value_head")}
                _save_file(_crit_sd, str(crit_dir / "ar_lora_value_head.safetensors"))
                (crit_dir / "ar_meta.json").write_text(_json.dumps(ar_meta, indent=2))
            actor.save_pretrained(str(out_dir))
            # Optimizer state (latest-only, atomic write): without it, resume
            # restarts Adam moments from zero — a reproducible loss bump at the
            # resume boundary. Same layout as the vLLM twin.
            _opt_tmp = save_dir / "optim_latest.pt.tmp"
            _opt_dst = save_dir / "optim_latest.pt"
            _opt_state = {"step": step + 1, "actor_optim": optim.state_dict()}
            if args.train_critic and critic_optim is not None:
                _opt_state["critic_optim"] = critic_optim.state_dict()
            torch.save(_opt_state, str(_opt_tmp))
            os.replace(str(_opt_tmp), str(_opt_dst))
            print(f"[save] LoRA → {out_dir} (+ optim_latest)"
                  + (" (+ co-trained critic)" if args.train_critic else ""))

    print("done.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

"""Shared wandb logging for the RL trainers (single-GPU + vLLM).

Both `train_rl_self_contained.py` and `train_rl_vllm.py` must log identically —
same run-init (project/group/tags/config + FVE baselines in config, not per-step)
and the same canonical metric names. Centralizing here is the single source of
truth so the two paths can't drift (the vLLM path had silently kept the old flat
`critic_loss`/per-step-baseline naming).

Metric convention:
  - headline: `fve_pct`, `wall_s` (top-level)
  - `av/…`   verbalizer / GRPO policy (kl_to_ref, advantage,
             extraction_rate, inject_fail_count, resp_len, …)
  - `ar/…`   reconstructor co-training (recon_mse, grad_norm)
  - `reward/…` reward stats over valid rollouts
  - av/ar, never actor/critic. `step` is the x-axis — never a metric.
"""

import numpy as np
import wandb


def init_wandb(args, *, rollout_tag, fve_baseline, fve_baseline_meannorm):
    """wandb.init with the shared project/group/tags/config + baselines in config.

    rollout_tag: "single-gpu" or "vllm" — auto-appended to --wandb-tags so the
    rollout path is always filterable. FVE baselines are run constants → config,
    not per-step metrics (which would draw flat lines and clutter charts).
    """
    tags = (args.wandb_tags.split(",") if getattr(args, "wandb_tags", None) else []) + [rollout_tag]
    wandb.init(
        project=args.wandb_project, name=args.wandb_name,
        group=getattr(args, "wandb_group", None), tags=tags, config=vars(args),
    )
    wandb.config.update({
        # _paper           = MSE(v_norm, mean)            — variance-around-mean,
        #                    the paper's FVE denominator (what fve_pct uses).
        # _meannorm_legacy = MSE(v_norm, normalize(mean)) — looser pre-2026-06-09
        #                    denominator, kept only to map old wandb curves.
        "fve_baseline_mse_paper": fve_baseline,
        "fve_baseline_mse_meannorm_legacy": fve_baseline_meannorm,
    })


def build_step_log(
    *,
    fve,
    grad_norm,
    grpo_metrics,
    adv,
    valid_rewards,
    extraction_rate,
    inject_fail_count,
    resp_len_mean,
    ar_recon_mse,
    ar_grad_norm,
    wall_s,
    shape_terms=None,
    marker_bad_count=None,
    inject_masked_count=None,
    steer_apply_count=None,
    steer_apply_rate=None,
    frac_cut_off=None,
):
    """Assemble the canonical per-step wandb log dict (identical for both paths).

    grad_norm may be a tensor or float. shape_terms = reward-shaping penalty means
    (e.g. length penalty) or None.
    """
    gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    vr = valid_rewards
    log = {
        # --- headline ---
        "fve_pct": fve * 100.0,  # % activation variance the AR recovers from the AV's words (paper baseline)
        "wall_s": wall_s,
        # --- av/: the verbalizer (GRPO policy) + its generations ---
        # NB: grpo_loss intentionally NOT logged — the surrogate part averages to
        # ~0 (zero-mean advantages, on-policy), so its value ≈ kl_beta·kl_to_ref,
        # i.e. redundant with av/kl_to_ref. Watch reward/mean, fve_pct, kl_to_ref, grad_norm.
        "av/grad_norm": gn,
        "av/kl_to_ref": grpo_metrics.get("kl_mean", 0.0),
        "av/entropy": grpo_metrics.get("entropy", 0.0),   # mean policy entropy over response tokens (nats)
        "av/advantage_mean": adv.mean().item(),
        "av/advantage_std": adv.std().item(),
        "av/extraction_rate": extraction_rate,   # frac of rollouts with a parseable <explanation>
        "av/inject_fail_count": inject_fail_count,  # # CJK-garbage rollouts = injection failed, output-symptom check (want 0)
        "av/resp_len": resp_len_mean,
        # frac of rollouts that hit the max_new_tokens cap (truncated mid-explanation).
        # Truncated rollouts are scored -2 and trained on; a growing fraction means
        # response length is outgrowing the cap — raise --max-new-tokens. Want ~0.
        "av/frac_cut_off": frac_cut_off if frac_cut_off is not None else 0.0,
        # --- reward = -(normalized AR reconstruction MSE), over valid rollouts ---
        "reward/mean": float(np.mean(vr)) if vr else float("nan"),
        "reward/std": float(np.std(vr)) if vr else float("nan"),
        "reward/min": float(np.min(vr)) if vr else float("nan"),
        "reward/max": float(np.max(vr)) if vr else float("nan"),
        # --- ar/: the reconstructor co-training (supervised recon MSE vs gold) ---
        "ar/recon_mse": ar_recon_mse,
        "ar/grad_norm": ar_grad_norm,
    }
    # --- explicit injection-success checks (mechanism-level, distribution-invariant) ---
    # marker_bad: rollouts whose AV prompt lost its single well-formed marker (HF-side
    #   precondition for a correct Karvonen inject). inject_masked: rollouts excluded
    #   from the AV+AR updates this step (cjk_fail OR marker_bad). steer_apply_count /
    #   _rate: how many rollouts vLLM actually wrote a steering vector for (the explicit
    #   vLLM-side check; -1/None when the patch counter isn't available). Unlike CJK,
    #   these can't be eroded by RL shifting the output distribution.
    if marker_bad_count is not None:
        log["av/marker_bad_count"] = marker_bad_count
    if inject_masked_count is not None:
        log["av/inject_masked_count"] = inject_masked_count
    if steer_apply_count is not None:
        log["av/steer_apply_count"] = float(steer_apply_count)
    if steer_apply_rate is not None:
        log["av/steer_apply_rate"] = steer_apply_rate
    if shape_terms:
        log.update(shape_terms)  # length penalty means
    return log


def format_console_line(step, log, *, train_ar):
    """One-line stdout summary from the canonical log dict (shared by both paths)."""
    crit_str = (f"| ar_mse {log['ar/recon_mse']:.4f} " if train_ar else "")
    return (
        f"step {step:04d} "
        f"| r {log['reward/mean']:.3f} | FVE {log['fve_pct']:.1f}% {crit_str}"
        f"| kl {log['av/kl_to_ref']:.4f} | ent {log.get('av/entropy', 0.0):.3f} "
        f"| ext {log['av/extraction_rate']:.0%} | t {log['wall_s']:.0f}s"
    )

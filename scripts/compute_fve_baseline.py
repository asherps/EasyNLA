"""Compute predict-the-mean MSE baselines on a parquet, then convert reward → FVE.

Two baselines (see nla/schema.py compute_predict_mean_baselines):
  - paper-def (rawvar): E[||v_norm − μ||²] — the paper's FVE denominator.
    This is what the trainers log as `fve_baseline` since 2026-06-09.
  - meannorm: MSE(v_norm, normalize(μ)) — the looser pre-2026-06-09 baseline;
    printed only for comparing against historical wandb curves.

FVE = 1 − mse_actual / baseline (paper-def unless you opt into meannorm).
"""

import argparse

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from nla.config import load_nla_config
from nla.schema import compute_predict_mean_baselines, resolve_target_scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                   help="tokenizer for sidecar verification (use the run's base model)")
    p.add_argument("--n", type=int, default=2000, help="rows to use for the baseline")
    p.add_argument("--reward-pre", type=float, default=None,
                   help="if set, also print FVE for this pre-RL reward")
    p.add_argument("--reward-post", type=float, default=None,
                   help="if set, also print FVE for this post-RL reward")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    print(f"mse_scale_f = {mse_scale_f}, d_model = {cfg.d_model}")

    pf = pq.ParquetFile(args.parquet)
    rg = pf.read_row_group(0, columns=["activation_vector"]).slice(0, args.n)
    acts = rg.column("activation_vector").to_pylist()
    acts_t = torch.tensor(acts, dtype=torch.float32)  # [N, d]
    print(f"loaded {acts_t.shape[0]} activations")

    bl_meannorm, baseline = compute_predict_mean_baselines(acts_t, mse_scale_f)
    print("\n=== Predict-the-mean baselines (normalized MSE) ===")
    print(f"  paper-def (rawvar) = {baseline:.4f}   <- FVE denominator (matches trainers)")
    print(f"  meannorm           = {bl_meannorm:.4f}   (pre-2026-06-09 metric, inflates FVE)")
    print(f"  fully-random ceiling = 2.0 (== FAILED extraction floor)")

    fve_pre = fve_post = None
    if args.reward_pre is not None:
        mse_pre = -args.reward_pre
        fve_pre = 1.0 - mse_pre / baseline
        print(f"\n=== Pre-RL ===")
        print(f"  reward = {args.reward_pre:.4f}  mse_nrm = {mse_pre:.4f}")
        print(f"  FVE = 1 - {mse_pre:.4f}/{baseline:.4f} = {fve_pre*100:.2f}%")

    if args.reward_post is not None:
        mse_post = -args.reward_post
        fve_post = 1.0 - mse_post / baseline
        print(f"\n=== Post-RL ===")
        print(f"  reward = {args.reward_post:.4f}  mse_nrm = {mse_post:.4f}")
        print(f"  FVE = 1 - {mse_post:.4f}/{baseline:.4f} = {fve_post*100:.2f}%")

    if fve_pre is not None and fve_post is not None:
        print(f"\n=== Delta ===")
        print(f"  FVE Δ = {(fve_post - fve_pre)*100:+.2f} pp")


if __name__ == "__main__":
    main()

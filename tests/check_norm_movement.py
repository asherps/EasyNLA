"""Post-smoke check: did norm/scale params MOVE during full-FT SFT?

The fp32 fix exists because in pure bf16, an Adam update of ~lr rounds to zero
on any weight with |w| > lr*512 — which freezes ~1.0-scale norm weights at ANY
practical lr. If the fp32 path works, the trained checkpoint's layernorm
weights must differ from the base model's. Runs on CPU (weights only).

Usage: python tests/check_norm_movement.py <ckpt_dir> [base_id]
"""

import sys

import torch
from safetensors import safe_open
from pathlib import Path
from huggingface_hub import snapshot_download


def norm_tensors(model_dir, patterns=("input_layernorm.weight", "post_attention_layernorm.weight",
                                      "q_norm.weight", "k_norm.weight")):
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        if f.name == "value_head.safetensors":
            continue
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if any(k.endswith(p) for p in patterns):
                    out[k] = sf.get_tensor(k).float()
    return out


def main():
    ckpt = sys.argv[1]
    base_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
    base_dir = snapshot_download(base_id, allow_patterns=["*.safetensors", "*.json"])
    trained = norm_tensors(ckpt)
    base = norm_tensors(base_dir)
    assert trained, f"no norm tensors found in {ckpt}"

    moved = 0
    total = 0
    worst = (0.0, None)
    for k, t in trained.items():
        # AR critic checkpoints prefix keys with backbone./model.; match by suffix
        bk = next((b for b in base if b.endswith(k) or k.endswith(b)), None)
        if bk is None:
            continue
        total += 1
        d = (t - base[bk]).abs().max().item()
        if d > 0:
            moved += 1
        if d > worst[0]:
            worst = (d, k)
    print(f"norm params compared: {total} | MOVED: {moved} ({moved/max(total,1):.0%}) "
          f"| max abs change {worst[0]:.2e} at {worst[1]}")
    if moved == 0:
        print("FAIL: no norm parameter moved — bf16-style freezing still present")
        sys.exit(1)
    print("PASS: norm params are training (fp32 path effective)")


if __name__ == "__main__":
    main()

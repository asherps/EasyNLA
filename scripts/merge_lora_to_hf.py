"""Merge AV / AR LoRA adapters (trained on a 4-bit base) into full bf16 HF
checkpoints, so train_rl_vllm.py can load them (it expects merged models +
tokenizer, not base+LoRA).

NOTE (lossy): the adapters were trained against a 4-bit (dequantized) base;
merging into bf16 reintroduces quantization-error mismatch. We VALIDATE the
merged AR against its held-out FVE downstream before trusting a long run.

AV merge: standard PeftModel.merge_and_unload.
AR merge: rebuild the critic exactly as train_sft saved it
  (init_critic_from_base -> inject_adapter_in_model -> load ar_lora_value_head),
  merge each LoRA layer, then rewrite the state dict (.base_layer.weight ->
  .weight, drop lora_*) and load into a fresh un-injected NLACriticModel.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.models import NLACriticModel
from nla.train_sft import init_critic_from_base


def _copy_sidecar(src: Path, dst: Path):
    for name in ("nla_meta.yaml", "ar_meta.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
            print(f"  copied {name}")


def merge_av(base_ckpt: str, av_dir: Path, out: Path):
    print(f"[av] base={base_ckpt} + LoRA={av_dir} -> {out}")
    base = AutoModelForCausalLM.from_pretrained(base_ckpt, torch_dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(base, str(av_dir))
    merged = peft.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    # AV LoRA dir carries the tokenizer files; preserve them so vLLM can load.
    AutoTokenizer.from_pretrained(str(av_dir)).save_pretrained(out)
    _copy_sidecar(av_dir, out)
    del base, peft, merged
    torch.cuda.empty_cache()
    print(f"[av] done -> {out}")


def merge_ar(base_ckpt: str, ar_dir: Path, out: Path):
    meta = json.loads((ar_dir / "ar_meta.json").read_text())
    print(f"[ar] meta={meta}")
    n_layers = meta["ar_num_layers"]
    strip = meta.get("final_norm_stripped", False)
    # Rebuild on a bf16 base (NOT 4-bit) so the LoRA can be merged into real weights.
    critic = init_critic_from_base(
        base_ckpt, n_layers, torch.bfloat16, None,
        device_map=None, max_memory=None, strip_final_norm=strip,
    ).to("cuda")
    inject_adapter_in_model(LoraConfig(
        r=meta["lora_r"], lora_alpha=meta["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=meta["target_modules"],
    ), critic.backbone)
    sd = load_file(str(ar_dir / "ar_lora_value_head.safetensors"))
    miss, unexp = critic.load_state_dict(sd, strict=False)
    n_lora = sum(1 for k in sd if "lora_" in k)
    assert n_lora > 0 and not unexp, f"AR load mismatch: n_lora={n_lora} unexpected={unexp[:3]}"
    print(f"[ar] loaded {len(sd)} tensors ({n_lora} lora), missing={len(miss)} unexpected=0")

    # Merge each LoRA layer into its base_layer.weight.
    n_merged = 0
    for m in critic.backbone.modules():
        if isinstance(m, BaseTunerLayer):
            m.merge()
            n_merged += 1
    print(f"[ar] merged {n_merged} LoRA layers")

    # Rewrite to clean (un-injected) keys: drop lora_*, base_layer.weight -> weight.
    merged_sd = {}
    for k, v in critic.state_dict().items():
        if "lora_" in k:
            continue
        nk = k.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
        merged_sd[nk] = v

    fresh = init_critic_from_base(
        base_ckpt, n_layers, torch.bfloat16, None,
        device_map=None, max_memory=None, strip_final_norm=strip,
    )
    fmiss, funexp = fresh.load_state_dict(merged_sd, strict=False)
    # fresh has no lora; merged_sd should cover everything fresh needs.
    assert not funexp, f"fresh load unexpected keys: {funexp[:5]}"
    assert not fmiss, f"fresh load missing keys: {fmiss[:5]}"
    out.mkdir(parents=True, exist_ok=True)
    fresh.save_pretrained(out)
    AutoTokenizer.from_pretrained(str(ar_dir)).save_pretrained(out)
    _copy_sidecar(ar_dir, out)
    print(f"[ar] done -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--av-dir", required=True)
    p.add_argument("--ar-dir", required=True)
    p.add_argument("--av-out", required=True)
    p.add_argument("--ar-out", required=True)
    p.add_argument("--mode", choices=["av", "ar", "both"], default="both")
    args = p.parse_args()
    if args.mode in ("av", "both"):
        merge_av(args.base_ckpt, Path(args.av_dir), Path(args.av_out))
    if args.mode in ("ar", "both"):
        merge_ar(args.base_ckpt, Path(args.ar_dir), Path(args.ar_out))
    print("MERGE COMPLETE")


if __name__ == "__main__":
    main()

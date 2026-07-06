"""Print full NLA generations from a trained verbalizer (AV).

Loads the base model + the AV policy LoRA, injects held-out activations at the
marker token, greedily generates the explanation, and prints it. If an AR critic
checkpoint is given (--ar-ckpt), it also reconstructs the activation from the
explanation and reports the normalized reconstruction MSE / reward.

Example:
    python scripts/show_nla_generations.py \
        --av-lora  <av_adapter_dir> \
        --sidecar  <dataset>/rl_shuf.parquet \
        --parquet  <dataset>/rl_shuf.parquet \
        --ar-ckpt  <critic_dir>   # optional; omit to just see the words
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from nla.config import load_nla_config
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.train_sft import _resolve_device_map, init_critic_from_base


def load_ar_critic(ar_ckpt, base_ckpt, device):
    """Rebuild the AR critic (AR-LoRA + value head) from a saved checkpoint dir.

    Expects `ar_meta.json` + `ar_lora_value_head.safetensors` — the format saved
    by the AR SFT / RL co-training. Mirrors how the trainers reconstruct it.
    """
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import load_file

    ar_src = Path(ar_ckpt)
    ar_meta = json.loads((ar_src / "ar_meta.json").read_text())
    print(f"[critic] AR from {ar_src}: {ar_meta}")
    ar_quant = None
    if ar_meta.get("quant") == "4bit":
        ar_quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    ar_dmap, ar_maxmem = _resolve_device_map("single", 0, ar_quant)
    critic = init_critic_from_base(
        base_ckpt, ar_meta["ar_num_layers"], torch.bfloat16, ar_quant,
        device_map=ar_dmap, max_memory=ar_maxmem,
        strip_final_norm=ar_meta.get("final_norm_stripped", False),
    )
    if ar_dmap is None:
        critic = critic.to(device)
    inject_adapter_in_model(LoraConfig(
        r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=ar_meta["target_modules"],
    ), critic.backbone)
    sd = load_file(str(ar_src / "ar_lora_value_head.safetensors"))
    _miss, unexp = critic.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys loading AR: {unexp[:3]}"
    critic.eval()
    return critic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-lora", required=True, help="AV policy LoRA adapter dir (SFT or RL)")
    p.add_argument("--sidecar", required=True, help="dataset parquet / sidecar (the NLA contract)")
    p.add_argument("--parquet", required=True, help="parquet with prompt + activation_vector rows")
    p.add_argument("--ar-ckpt", default=None, help="optional AR critic dir → also print reconstruction FVE")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--skip-rows", type=int, default=36000, help="skip the training rows (held-out only)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--4bit", dest="four_bit", action="store_true",
                   help="load the base in 4-bit (matches the single-GPU training path)")
    args = p.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template

    # --- held-out rows (detokenized_text_truncated = the source the activation came from) ---
    pf = pq.ParquetFile(args.parquet)
    has_src = "detokenized_text_truncated" in pf.schema_arrow.names
    cols = ["prompt", "activation_vector"] + (["detokenized_text_truncated"] if has_src else [])
    rows, seen = [], 0
    for rg_idx in range(pf.num_row_groups):
        if len(rows) >= args.n:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n = rg.num_rows
        if seen + n <= args.skip_rows:
            seen += n
            continue
        start = max(0, args.skip_rows - seen)
        seen += start
        pr = rg.column("prompt").to_pylist()
        ac = rg.column("activation_vector").to_pylist()
        src = rg.column("detokenized_text_truncated").to_pylist() if has_src else [None] * n
        for i in range(start, n):
            rows.append({"prompt": pr[i], "activation": ac[i], "source": src[i]})
            if len(rows) >= args.n:
                break

    quant = None
    if args.four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=quant).to(device)
    actor = PeftModel.from_pretrained(base, args.av_lora).eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
                           layer_idx=1)
    critic = load_ar_critic(args.ar_ckpt, args.base_ckpt, device) if args.ar_ckpt else None

    for i, row in enumerate(rows):
        ptxt = build_prompt_text(row["prompt"], cfg.injection_char, tok)
        ids = tok.encode(ptxt, add_special_tokens=False)
        pt = torch.tensor([ids], dtype=torch.long, device=device)
        act = torch.tensor(row["activation"], dtype=torch.float32).unsqueeze(0).to(device)
        vref[0] = act
        try:
            with torch.no_grad():
                out = actor.generate(
                    input_ids=pt, attention_mask=torch.ones_like(pt),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id, return_dict_in_generate=True)
        finally:
            vref[0] = None
        resp = tok.decode(out.sequences[0, pt.shape[1]:], skip_special_tokens=True)
        m = EXPLANATION_RE.search(resp)
        expl = m.group(1).strip() if m else None

        print(f"\n{'=' * 88}\n### GENERATION {i}\n{'=' * 88}")
        if row.get("source"):
            s = row["source"]
            tail = s if len(s) <= 700 else "…" + s[-700:]
            print(f"SOURCE TEXT (activation = layer-{cfg.critic_num_layers or 24} state at its END):\n{tail}\n")
        print("FULL RESPONSE:\n" + resp)
        if not expl:
            print("\n(no <explanation> extracted)")
            continue
        if critic is None:
            continue
        # Reconstruct the activation from the explanation → normalized MSE / reward.
        cids = tok.encode(template.format(explanation=expl), add_special_tokens=False)
        x = torch.tensor([cids], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = critic_predict(critic, x, None, mse_scale_f)[0]
        pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
        gn = normalize_activation(act[0].float().unsqueeze(0), mse_scale_f)[0]
        mse = F.mse_loss(pn, gn).item()
        print(f"\n  recon MSE (normalized) = {mse:.4f}  → reward {-mse:.3f}")


if __name__ == "__main__":
    main()

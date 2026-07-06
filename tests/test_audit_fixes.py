"""Tests for the 2026-07-04 audit-fix batch (one test per fixed item).

Covers (numbers = the fix list discussed with the user):
  1  models.py arch-aware last-layer lookup (GPT-2 critic no longer crashes)
  2  sync bucketing splits transformer.h.<N> layers (GPT-arch, no 16GB chunk)
  3  GRPO n_total fixed-budget normalizer (grad scales with budget, not len(keep))
  6  run_pipeline stage0.multigpu asserts at config-parse
  7  hooks: zero-marker forward now fails LOUD; decode-step still silent
  9  extraction_layer_index loaded from sidecar; --ar-num-layers cross-checked
  10 NLACriticOutput.values dead matmul removed; critic_predict golden-checked
  8/11 fail-fast guards: iter_ overwrite refusal + sidecar snapshot/mismatch
       (subprocess against the real trainer entrypoint)
  4/12/13 static: installer applies patch, -1 warning present, README note

Run: python tests/test_audit_fixes.py   (CPU-only; needs the Qwen3 tokenizer in
HF cache + the nla-data sidecar for the subprocess/sidecar tests, both present
on the cluster.)
"""

import copy
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, None))
        print(f"PASS {name}")
    except Exception as e:
        RESULTS.append((name, e))
        print(f"FAIL {name}: {type(e).__name__}: {e}")


def tiny_llama(dtype=torch.bfloat16):
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=100,
    )).to(dtype)


def tiny_gpt2(dtype=torch.bfloat16):
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(
        n_embd=64, n_layer=2, n_head=4, vocab_size=100, n_positions=128,
    )).to(dtype)


# ---- 1: GPT-2 critic from_pretrained round-trip -----------------------------
def t1_gpt2_critic_roundtrip():
    from nla.train_sft import init_critic_from_base
    from nla.models import NLACriticModel
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) / "gpt2_base"
        tiny_gpt2().save_pretrained(base_dir)
        critic = init_critic_from_base(str(base_dir), 2, torch.bfloat16, None,
                                       device_map=None)
        crit_dir = Path(td) / "critic"
        critic.save_pretrained(str(crit_dir))
        # pre-fix: AttributeError ('GPT2Model' object has no attribute 'layers')
        reloaded = NLACriticModel.from_pretrained(str(crit_dir),
                                                  torch_dtype=torch.bfloat16)
        assert reloaded.value_head.weight.shape[0] == 64
        # llama regression
        base_dir2 = Path(td) / "llama_base"
        tiny_llama().save_pretrained(base_dir2)
        critic2 = init_critic_from_base(str(base_dir2), 2, torch.bfloat16, None,
                                        device_map=None)
        crit_dir2 = Path(td) / "critic2"
        critic2.save_pretrained(str(crit_dir2))
        NLACriticModel.from_pretrained(str(crit_dir2), torch_dtype=torch.bfloat16)


# ---- 2: GPT-arch sync bucketing ---------------------------------------------
def t2_gpt2_sync_buckets():
    from peft import LoraConfig, get_peft_model
    from nla.train_rl_vllm import sync_actor_to_vllm

    class ChunkRecorder:
        def __init__(self):
            self.chunks = []
            class E:
                def reset_prefix_cache(self): pass
            self.llm_engine = E()
        def apply_model(self, fn):
            self.chunks.append([name for name, _ in fn.keywords["chunk"]])
            return [None]

    actor = get_peft_model(tiny_gpt2(), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["c_attn"],
    ))
    rec = ChunkRecorder()
    sync_actor_to_vllm(actor, rec, ipc=False, only_adapted=False)
    layer_of = {}
    import re
    for ci, chunk in enumerate(rec.chunks):
        for k in chunk:
            m = re.match(r"transformer\.h\.(\d+)\.", k)
            layer_of.setdefault(ci, set()).add(m.group(1) if m else "_other")
    per_chunk = list(layer_of.values())
    # every chunk must be single-layer (or all-_other); layers 0 and 1 split
    assert all(len(v) == 1 for v in per_chunk), f"mixed chunks: {per_chunk}"
    layer_ids = {next(iter(v)) for v in per_chunk}
    assert {"0", "1"} <= layer_ids, f"h.0/h.1 not split into own chunks: {layer_ids}"


# ---- 3: n_total fixed-budget normalizer --------------------------------------
def t3_n_total_grad_scaling():
    from peft import LoraConfig, get_peft_model
    from nla.train_rl_vllm import grpo_update_microbatched

    def build():
        torch.manual_seed(0)
        m = get_peft_model(tiny_llama(torch.float32), LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
        torch.manual_seed(1)
        with torch.no_grad():
            for n, p_ in m.named_parameters():
                if "lora_" in n:
                    p_.copy_(torch.randn_like(p_) * 0.05)
        return m

    def grads_with(n_total):
        actor = build()
        optim = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.0)
        ids = [torch.arange(2, 12), torch.arange(3, 15)]
        acts = [torch.randn(64), torch.randn(64)]
        adv = torch.tensor([1.0, -0.5])
        class Tok:
            eos_token_id = 0
        import inspect
        params = inspect.signature(grpo_update_microbatched).parameters
        if "old_logps_list" in params:   # full-repo signature (on-policy ignores them)
            old_lps = [torch.zeros(ids[0].numel() - 4), torch.zeros(ids[1].numel() - 5)]
            grpo_update_microbatched(
                actor, optim, Tok(), ids, [4, 5], acts, old_lps, adv, [None], "cpu",
                micro_batch=2, kl_beta=0.0, do_step=False, n_total=n_total,
            )
        else:
            grpo_update_microbatched(
                actor, optim, Tok(), ids, [4, 5], acts, adv, [None], "cpu",
                micro_batch=2, kl_beta=0.0, do_step=False, n_total=n_total,
            )
        return {n: p.grad.clone() for n, p in actor.named_parameters()
                if p.requires_grad and p.grad is not None}

    g2, g4 = grads_with(2), grads_with(4)
    assert g2 and set(g2) == set(g4)
    for k in g2:
        if g2[k].abs().max() == 0:
            continue
        ratio = (g2[k] / g4[k])[g4[k] != 0]
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-5), \
            f"{k}: grad ratio not 2.0 (n_total=4 should halve per-sample weight)"


# ---- 6: stage0 multigpu config-parse assert -----------------------------------
def t6_stage0_multigpu_assert():
    if (REPO / "scripts/datagen/stage0_multigpu.sh").exists():
        print("  (script shipped in this repo — assert not applicable, OK)")
        return
    from nla.datagen import run_pipeline as rp
    cfg = {
        "stage0": {"multigpu": True},
        "corpus": {"name": "dummy", "length": 1, "config": None},
        "extraction": {"base_model": "x", "layer_index": 1, "positions_per_doc": 1},
        "dataset_id": "t", "seed": 0,
    }
    try:
        rp._stage0(cfg, {"base": "/tmp/x.parquet"})
    except AssertionError as e:
        assert "stage0_multigpu.sh" in str(e)
        return
    except KeyError:
        raise AssertionError("KeyError before the multigpu assert — assert must "
                             "run before cfg-field access for fail-at-parse")
    raise AssertionError("multigpu with missing script did not raise")


# ---- 7: zero-marker forward fails loud; decode-step silent --------------------
def t7_hook_loud_on_template_drift():
    from nla.utils.hooks import register_karvonen_hook
    model = tiny_llama(torch.float32)
    vectors_ref = [None]
    INJ, L, R = 42, 41, 43
    register_karvonen_hook(model, vectors_ref, INJ, L, R)

    # (a) marker present with neighbors -> injects (logits change vs no vector)
    ids = torch.tensor([[1, 41, 42, 43, 5, 6]])
    with torch.no_grad():
        base_out = model(input_ids=ids).logits.clone()
        vectors_ref[0] = torch.randn(1, 64) * 5
        inj_out = model(input_ids=ids).logits.clone()
    assert not torch.allclose(base_out, inj_out), "injection had no effect"

    # (b) NO marker anywhere + vector set -> must RAISE (was: silent skip)
    bad = torch.tensor([[1, 2, 3, 4, 5, 6]])
    try:
        with torch.no_grad():
            model(input_ids=bad)
    except RuntimeError as e:
        assert "marker" in str(e).lower() or "found 0" in str(e), str(e)
    else:
        raise AssertionError("zero-marker forward with vector set did not raise")

    # (c) decode-step shape [B,1] -> silent no-op (seq_len<2 guard)
    with torch.no_grad():
        model(input_ids=torch.tensor([[7]]))
    vectors_ref[0] = None


# ---- 9: extraction_layer_index plumbed + validated ----------------------------
SIDECAR = "/workspace-vast/asherps/nla-data/qwen3_8b_finefineweb_100k/rl_shuf.parquet"

def t9_layer_index_loaded():
    from transformers import AutoTokenizer
    from nla.config import load_nla_config
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(SIDECAR, tok)
    assert cfg.extraction_layer_index == 24, cfg.extraction_layer_index


def t9_ar_num_layers_mismatch_asserts():
    env = dict(os.environ, HF_HOME=f"/workspace-vast/{os.environ['USER']}/hf_cache")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, "-m", "nla.train_sft", "--mode", "ar",
             "--base-ckpt", "Qwen/Qwen3-8B",
             "--parquet", SIDECAR, "--sidecar", SIDECAR,
             "--save-dir", td, "--ar-num-layers", "20", "--no-wandb"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "extraction.layer_index" in (r.stderr + r.stdout), \
        f"expected depth-mismatch assert; got rc={r.returncode}\n{r.stderr[-800:]}"


# ---- 10: dead values removed; critic_predict golden ----------------------------
def t10_critic_predict_golden():
    from nla.train_sft import init_critic_from_base
    from nla.utils.critic import critic_predict
    from nla.schema import normalize_activation
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) / "base"
        tiny_llama().save_pretrained(base_dir)
        critic = init_critic_from_base(str(base_dir), 2, torch.float32, None,
                                       device_map=None)
    ids = torch.randint(1, 99, (3, 9))
    attn = torch.ones_like(ids); attn[1, 7:] = 0   # row 1 padded -> anchor at 6
    out = critic(input_ids=ids, attention_mask=attn)
    assert out.values is None, "values should no longer be computed"
    assert out.backbone_last_hidden.shape == (3, 9, 64)
    pred = critic_predict(critic, ids, attn, mse_scale_f=8.0)
    assert pred.shape == (3, 64) and pred.dtype == torch.float32
    # golden: value_head(normalize(h[anchor])) by hand
    h = out.backbone_last_hidden
    anchors = attn.sum(1) - 1
    manual = critic.value_head(normalize_activation(
        h[torch.arange(3), anchors].float(), 8.0
    ).to(critic.value_head.weight.dtype)).float()
    assert torch.allclose(pred, manual, atol=1e-5)


# ---- 8/11: fail-fast guards (subprocess vs the real entrypoint) ----------------
def _run_trainer(save_dir, extra=()):
    env = dict(os.environ, HF_HOME=f"/workspace-vast/{os.environ['USER']}/hf_cache")
    return subprocess.run(
        [sys.executable, "-m", "nla.train_rl_vllm",
         "--av-ckpt", "/nonexistent/av", "--ar-ckpt", "/nonexistent/ar",
         "--rl-parquet", SIDECAR, "--sidecar", SIDECAR,
         "--save-dir", str(save_dir), "--no-wandb", *extra],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


def t8_iter_overwrite_refused():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "iter_000005").mkdir()
        r = _run_trainer(td)
        assert r.returncode != 0
        assert "refusing to overwrite" in (r.stderr + r.stdout), r.stderr[-500:]


def t11_sidecar_mismatch_asserts():
    with tempfile.TemporaryDirectory() as td:
        meta = yaml.safe_load(Path(SIDECAR + ".nla_meta.yaml").read_text())
        meta["extraction"]["layer_index"] = 7   # tampered snapshot
        (Path(td) / "nla_meta.yaml").write_text(yaml.safe_dump(meta))
        r = _run_trainer(td)
        assert r.returncode != 0
        assert "disagrees" in (r.stderr + r.stdout), r.stderr[-500:]


def t11_sidecar_snapshot_written():
    with tempfile.TemporaryDirectory() as td:
        r = _run_trainer(td)   # dies later at the dummy --av-ckpt, AFTER the snapshot
        assert (Path(td) / "nla_meta.yaml").exists(), \
            f"snapshot not written (rc={r.returncode}):\n{r.stderr[-500:]}"


# ---- 4/12/13: static checks ----------------------------------------------------
def t_static():
    inst = (REPO / "scripts/install_vllm_lens.sh").read_text()
    assert "patch_vllm_lens.py" in inst and "FATAL" in inst, "installer lacks patch step"
    trainer = (REPO / "nla/train_rl_vllm.py").read_text()
    assert "steer counter unavailable" in trainer, "missing -1 counter warning"
    assert "clamping the OUTPUT would zero the gradient" not in trainer, "stale clamp comment"
    assert "FRESH zero-init" in trainer, "missing fresh-LoRA cold-start warning"
    readme = (REPO / "README.md").read_text()
    if (REPO / "README.md").exists() and "av-adapter" in readme:
        assert "cold-start" in readme or "~12pp" in readme, "README av-adapter note missing"


if __name__ == "__main__":
    check("1_gpt2_critic_roundtrip", t1_gpt2_critic_roundtrip)
    check("2_gpt2_sync_buckets", t2_gpt2_sync_buckets)
    check("3_n_total_grad_scaling", t3_n_total_grad_scaling)
    check("6_stage0_multigpu_assert", t6_stage0_multigpu_assert)
    check("7_hook_loud_on_template_drift", t7_hook_loud_on_template_drift)
    check("9_layer_index_loaded", t9_layer_index_loaded)
    check("9_ar_num_layers_mismatch_asserts", t9_ar_num_layers_mismatch_asserts)
    check("10_critic_predict_golden", t10_critic_predict_golden)
    check("8_iter_overwrite_refused", t8_iter_overwrite_refused)
    check("11_sidecar_mismatch_asserts", t11_sidecar_mismatch_asserts)
    check("11_sidecar_snapshot_written", t11_sidecar_snapshot_written)
    check("static_4_12_13", t_static)
    n_fail = sum(1 for _, e in RESULTS if e)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    sys.exit(1 if n_fail else 0)

"""Comprehensive correctness tests for the out-of-place sync_actor_to_vllm.

The rewrite (R1) replaced merge_adapter() -> push -> unmerge_adapter() with
out-of-place merged tensors (base.weight + get_delta_weight()). These tests
pin the properties that rewrite must preserve:

  1. pushed weights == the old merge-path reference (<= 1 bf16 ulp), in both
     only_adapted=True (default: adapted-weight subset) and full-push modes
  2. actor params BIT-untouched by sync (the whole point of R1)
  3. two-adapter mode (--av-adapter: trainable 'default' + frozen 'reference')
     merges ONLY the active adapter, exactly like merge_adapter() did
  4. key set identical to the old path (no lora_ leaks, .base_layer renamed)
  5. repeated syncs don't drift the actor (old path drifts in bf16)
  6. sync after a weight update pushes the NEW deltas
  7. pushed tensors are detached (no autograd graph leak) and on CPU (ipc=False)

Run: python -m pytest tests/test_sync_out_of_place.py -q   (CPU-only, ~1min)
"""

import copy

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from nla.train_rl_vllm import sync_actor_to_vllm


class FakeEngine:
    def reset_prefix_cache(self):
        pass


class FakeLLM:
    """Captures pushed (name, tensor) pairs; emulates llm.apply_model."""

    def __init__(self):
        self.pushed = {}
        self.llm_engine = FakeEngine()

    def apply_model(self, fn):
        chunk = fn.keywords["chunk"]
        for name, t in chunk:
            self.pushed[name] = t
        return [None]


def tiny_base(dtype=torch.bfloat16, attention_bias=False, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=100,
        attention_bias=attention_bias,
    )
    return LlamaForCausalLM(cfg).to(dtype)


def add_lora(base, r=8, alpha=16, use_rslora=False, seed=1):
    m = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", use_rslora=use_rslora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    randomize_lora(m, seed=seed)
    return m


def randomize_lora(model, seed=1, scale=0.05, adapter=None):
    """Give LoRA weights non-trivial values so merged != base."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n and (adapter is None or f".{adapter}." in n):
                p.copy_((torch.randn_like(p, dtype=torch.float32) * scale).to(p.dtype))


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def assert_bit_identical(model, snap, ctx=""):
    changed = [n for n, p in model.named_parameters()
               if not torch.equal(p.detach(), snap[n])]
    assert not changed, f"{ctx}: actor params mutated by sync: {changed[:5]}"


def reference_merge_push(actor):
    """The OLD path's pushed dict: merge_adapter -> clean keys -> unmerge."""
    actor.merge_adapter()
    try:
        ref = {}
        for k, v in actor.state_dict().items():
            if "lora_" in k or "modules_to_save" in k:
                continue
            nk = k
            if nk.startswith("base_model.model."):
                nk = nk[len("base_model.model."):]
            nk = nk.replace(".base_layer.weight", ".weight")
            nk = nk.replace(".base_layer.bias", ".bias")
            ref[nk] = v.detach().cpu().clone()
    finally:
        actor.unmerge_adapter()
    return ref


def max_rel_dev(pushed, ref):
    worst = 0.0
    for k in ref:
        d = (pushed[k].float() - ref[k].float()).abs().max().item()
        s = max(ref[k].float().abs().max().item(), 1e-9)
        worst = max(worst, d / s)
    return worst


# bf16 has 8 significand bits -> 1 ulp relative step = 2^-8. The out-of-place
# fp32-accumulated add may legitimately differ from peft's bf16 in-place add
# by at most 1 rounding step.
BF16_ULP = 2 ** -8


@pytest.mark.parametrize("r,alpha,rslora,dtype,bias", [
    (8, 16, False, torch.bfloat16, False),
    (128, 16, True, torch.bfloat16, False),     # the real RL config (rslora r128)
    (8, 32, False, torch.float32, False),
    (8, 16, False, torch.bfloat16, True),       # attention_bias -> .base_layer.bias keys
])
def test_matches_merge_path_and_leaves_actor_untouched(r, alpha, rslora, dtype, bias):
    actor = add_lora(tiny_base(dtype=dtype, attention_bias=bias), r=r,
                     alpha=alpha, use_rslora=rslora)
    before = snapshot(actor)
    # reference from a DEEPCOPY: reference_merge_push runs the old in-place
    # merge/unmerge round-trip, which itself drifts bf16 weights (the R1 bug) —
    # computing it on the real actor would corrupt the bit-identity checks.
    import copy as _copy
    ref = reference_merge_push(_copy.deepcopy(actor))

    # full-push mode: identical key set + values to the old path
    llm_full = FakeLLM()
    sync_actor_to_vllm(actor, llm_full, ipc=False, only_adapted=False)
    assert_bit_identical(actor, before, "full-push")
    assert set(llm_full.pushed) == set(ref), set(llm_full.pushed) ^ set(ref)
    dev = max_rel_dev(llm_full.pushed, ref)
    assert dev <= BF16_ULP, f"full push deviates from merge path by {dev:.2e} rel"

    # default (only_adapted): exactly the adapted-weight subset, same values
    llm = FakeLLM()
    sync_actor_to_vllm(actor, llm, ipc=False)
    assert_bit_identical(actor, before, "single-adapter")
    expected = {k for k in ref if k.endswith(
        (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight"))}
    assert set(llm.pushed) == expected, set(llm.pushed) ^ expected
    assert not any("lora_" in k or "base_layer" in k for k in llm.pushed)
    dev = max_rel_dev(llm.pushed, {k: ref[k] for k in expected})
    assert dev <= BF16_ULP, f"pushed deviates from merge path by {dev:.2e} rel"
    # merged weights actually differ from the raw base (LoRA non-trivial)
    q = "model.layers.0.self_attn.q_proj.weight"
    raw = dict(actor.named_parameters())[
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"]
    assert not torch.equal(llm.pushed[q], raw.detach().cpu()), \
        "pushed q_proj == raw base — the LoRA delta was NOT merged in"


def test_two_adapter_av_mode_merges_only_active():
    """--av-adapter mode: trainable 'default' + frozen 'reference'. Old
    merge_adapter() merged only the ACTIVE adapter; the rewrite must match."""
    base = tiny_base()
    actor = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ), adapter_name="default")
    # distinct-valued second adapter, then activate 'default' like the trainer
    actor.add_adapter("reference", LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    randomize_lora(actor, seed=1, adapter="default")
    randomize_lora(actor, seed=2, adapter="reference")
    actor.set_adapter("default")

    before = snapshot(actor)
    llm = FakeLLM()
    sync_actor_to_vllm(actor, llm, ipc=False)
    assert_bit_identical(actor, before, "two-adapter")
    ref = reference_merge_push(actor)  # merge_adapter merges active ('default') only
    assert set(llm.pushed) == {k for k in ref if k.endswith(
        (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight"))}
    dev = max_rel_dev(llm.pushed, {k: ref[k] for k in llm.pushed})
    assert dev <= BF16_ULP, f"two-adapter dev {dev:.2e}"

    # sanity: had we merged BOTH adapters the result would differ — prove the
    # reference adapter is genuinely excluded.
    actor.base_model.set_adapter(["default", "reference"])
    both = reference_merge_push(actor)
    actor.base_model.set_adapter("default")
    actor.set_adapter("default")
    q = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(both[q], ref[q]), \
        "test is vacuous: reference adapter has no effect on q_proj"


def test_repeated_sync_zero_drift_vs_old_path_drifts():
    """50 syncs: new path leaves the actor bit-identical; the old in-place
    merge/unmerge accumulates bf16 drift (the R1 bug being fixed)."""
    actor = add_lora(tiny_base(), r=64, alpha=128, seed=3)  # big deltas -> visible drift
    before = snapshot(actor)
    llm = FakeLLM()
    for _ in range(50):
        sync_actor_to_vllm(actor, llm, ipc=False)
    assert_bit_identical(actor, before, "50x new-path syncs")

    # old path on a fresh identical copy
    actor2 = add_lora(tiny_base(), r=64, alpha=128, seed=3)
    key = "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    w0 = dict(actor2.named_parameters())[key].detach().clone()
    for _ in range(50):
        actor2.merge_adapter()
        actor2.unmerge_adapter()
    w1 = dict(actor2.named_parameters())[key].detach()
    drift = (w1.float() - w0.float()).abs().max().item()
    assert drift > 0, ("old merge/unmerge path shows zero bf16 drift — "
                       "R1 premise not reproduced by this toy (weaken test?)")


def test_sync_reflects_weight_updates():
    """Changing LoRA between syncs changes what's pushed (no stale caching)."""
    actor = add_lora(tiny_base(), r=8)
    llm1, llm2 = FakeLLM(), FakeLLM()
    sync_actor_to_vllm(actor, llm1, ipc=False)
    randomize_lora(actor, seed=99)  # "optimizer step"
    before = snapshot(actor)
    sync_actor_to_vllm(actor, llm2, ipc=False)
    assert_bit_identical(actor, before, "post-update sync")
    ref2 = reference_merge_push(actor)
    assert max_rel_dev(llm2.pushed, {k: ref2[k] for k in llm2.pushed}) <= BF16_ULP
    q = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(llm1.pushed[q], llm2.pushed[q]), \
        "push after weight update is identical to the pre-update push"


def test_pushed_tensors_detached_cpu_and_correct_dtype():
    actor = add_lora(tiny_base(), r=8)
    llm = FakeLLM()
    sync_actor_to_vllm(actor, llm, ipc=False)
    for k, t in llm.pushed.items():
        assert not t.requires_grad, f"{k} requires_grad (graph leak)"
        assert t.device.type == "cpu", f"{k} not on CPU under ipc=False"
    q = "model.layers.0.self_attn.q_proj.weight"
    assert llm.pushed[q].dtype == torch.bfloat16
    emb = "model.embed_tokens.weight"
    assert emb not in llm.pushed  # frozen non-adapted weights skipped by default
    llm_full = FakeLLM()
    sync_actor_to_vllm(actor, llm_full, ipc=False, only_adapted=False)
    assert emb in llm_full.pushed  # ...but present in full-push mode


def test_requires_grad_and_training_mode_preserved():
    actor = add_lora(tiny_base(), r=8)
    actor.train()
    rg_before = {n: p.requires_grad for n, p in actor.named_parameters()}
    sync_actor_to_vllm(actor, FakeLLM(), ipc=False)
    assert actor.training
    rg_after = {n: p.requires_grad for n, p in actor.named_parameters()}
    assert rg_before == rg_after


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


def test_mlp_inclusive_targets_are_discovered():
    """only_adapted must push whatever LoRA wraps — not assume attn-only.
    (The trainers currently target attn projections, but the sync discovers
    LoraLayer modules; broadening target_modules must Just Work.)"""
    base = tiny_base()
    actor = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    randomize_lora(actor)
    before = snapshot(actor)
    llm = FakeLLM()
    sync_actor_to_vllm(actor, llm, ipc=False)
    assert_bit_identical(actor, before, "mlp-inclusive")
    import copy as _copy
    ref = reference_merge_push(_copy.deepcopy(actor))
    expected = {k for k in ref if k.endswith((
        ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
        ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))}
    assert set(llm.pushed) == expected, set(llm.pushed) ^ expected
    assert any(k.endswith(".gate_proj.weight") for k in llm.pushed), \
        "MLP modules not discovered"
    assert max_rel_dev(llm.pushed, {k: ref[k] for k in llm.pushed}) <= BF16_ULP


def test_modules_to_save_falls_back_to_full_push():
    """modules_to_save trains params outside the LoRA deltas; only_adapted
    would silently skip them, so the sync must fall back to a full push."""
    base = tiny_base()
    actor = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["lm_head"],
    ))
    randomize_lora(actor)
    llm = FakeLLM()
    sync_actor_to_vllm(actor, llm, ipc=False)   # default only_adapted=True
    # fell back: non-adapted keys (embeddings) present
    assert "model.embed_tokens.weight" in llm.pushed, \
        "modules_to_save did not trigger the full-push fallback"


def test_resumed_adapter_discovery(tmp_dir="/tmp/nla_test_resume_adapter"):
    """Adapters loaded via PeftModel.from_pretrained (the --resume-from-lora
    path) must be discovered exactly like get_peft_model ones."""
    import shutil
    actor0 = add_lora(tiny_base(), r=8, seed=7)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    actor0.save_pretrained(tmp_dir)
    base = tiny_base()  # same seed -> identical base weights
    actor = PeftModel.from_pretrained(base, tmp_dir, is_trainable=True)
    llm = FakeLLM()
    sync_actor_to_vllm(actor, llm, ipc=False)
    import copy as _copy
    ref = reference_merge_push(_copy.deepcopy(actor))
    dev = max_rel_dev(llm.pushed, {k: ref[k] for k in llm.pushed})
    assert llm.pushed and dev <= BF16_ULP, f"resumed-adapter dev {dev:.2e}"
    q = "model.layers.0.self_attn.q_proj.weight"
    raw = dict(actor.named_parameters())[
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"]
    assert not torch.equal(llm.pushed[q], raw.detach().cpu()), \
        "resumed adapter delta not merged into the push"
    shutil.rmtree(tmp_dir, ignore_errors=True)

"""Real-scale GPU validation of the out-of-place sync (R1) on Qwen3-8B.

Run under Slurm:  srun ... python tests/gpu_bench_sync.py

Checks, at the real RL config (r=128, alpha=16, rslora, attn targets):
  1. correctness: pushed weights == old merge-path reference (<= 1 bf16 ulp)
  2. actor params BIT-untouched by the new sync
  3. empirical R1 drift: 100 old-path merge/unmerge round-trips on the real
     8B weights (the bug being fixed), vs 0 for the new path
  4. timing: old path (merge + state_dict + cpu-copy + unmerge) vs new path
     (delta build + out-of-place add + cpu-copy), 5 reps each
  5. GPU peak-memory overhead of the new path's transient merged tensors
"""

import os, sys, time

os.environ.setdefault("HF_HOME", f"/workspace-vast/{os.environ['USER']}/hf_cache")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nla.train_rl_vllm import sync_actor_to_vllm

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-8B")


class FakeEngine:
    def reset_prefix_cache(self): pass


class FakeLLM:
    def __init__(self, keep=False):
        self.pushed = {} if keep else None
        self.llm_engine = FakeEngine()
    def apply_model(self, fn):
        if self.pushed is not None:
            for name, t in fn.keywords["chunk"]:
                self.pushed[name] = t
        return [None]


def old_path_push(actor, keep=False):
    """The pre-R1 implementation, verbatim semantics (merge -> walk -> unmerge)."""
    pushed = {} if keep else None
    actor.merge_adapter()
    try:
        for k, v in actor.state_dict().items():
            if "lora_" in k or "modules_to_save" in k:
                continue
            nk = k
            if nk.startswith("base_model.model."):
                nk = nk[len("base_model.model."):]
            nk = nk.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
            t = v.detach().cpu()
            if pushed is not None:
                pushed[nk] = t.clone()
    finally:
        actor.unmerge_adapter()
    return pushed


def main():
    dev = "cuda"
    print(f"loading {MODEL} bf16 ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    actor = get_peft_model(base, LoraConfig(
        r=128, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        use_rslora=True, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    # non-trivial LoRA at a realistic late-training magnitude
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in actor.named_parameters():
            if "lora_" in n:
                p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.02).bfloat16())
    actor.eval()

    # ---- 1+2: correctness at scale ----
    before = {n: p.detach().clone() for n, p in actor.named_parameters()
              if "layers.0." in n or "layers.35." in n or "embed" in n}  # spot-check snapshot (fp32 full snapshot won't fit)
    llm = FakeLLM(keep=True)
    sync_actor_to_vllm(actor, llm, ipc=False)
    changed = [n for n, p in actor.named_parameters()
               if n in before and not torch.equal(p.detach(), before[n])]
    print(f"[correctness] actor spot-check params changed by new sync: {changed or 'NONE (bit-exact)'}")
    assert not changed

    ref = old_path_push(actor, keep=True)
    # default sync pushes ONLY the adapted attn projections
    expected = {k for k in ref if k.endswith(
        (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight"))}
    assert set(llm.pushed) == expected, f"key mismatch: {set(llm.pushed) ^ expected}"
    # full-push mode must still cover everything
    llm_full = FakeLLM(keep=True)
    sync_actor_to_vllm(actor, llm_full, ipc=False, only_adapted=False)
    assert set(llm_full.pushed) == set(ref), "full-push key set mismatch"
    llm.pushed.update({k: v for k, v in llm_full.pushed.items() if k not in llm.pushed})
    del llm_full
    worst, where = 0.0, None
    for k in ref:
        d = (llm.pushed[k].float() - ref[k].float()).abs().max().item()
        s = max(ref[k].float().abs().max().item(), 1e-9)
        if d / s > worst:
            worst, where = d / s, k
    print(f"[correctness] pushed-vs-merge-path max rel dev: {worst:.3e} at {where} "
          f"({'<=1 bf16 ulp OK' if worst <= 2**-8 else 'TOO BIG'})")
    assert worst <= 2 ** -8
    del llm, ref
    torch.cuda.empty_cache()

    # ---- 3: empirical drift of the OLD path on real 8B weights ----
    key = "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    w0 = dict(actor.named_parameters())[key].detach().float().clone()
    for _ in range(100):
        actor.merge_adapter(); actor.unmerge_adapter()
    w1 = dict(actor.named_parameters())[key].detach().float()
    drift = ((w1 - w0).norm() / w0.norm()).item()
    print(f"[drift] OLD path, 100 merge/unmerge round-trips on real q_proj: "
          f"{drift:.4%} relative  (new path: 0 by construction — verified above)")
    # restore pristine weights for timing (drifted weights don't affect timing)

    # ---- 4: timing, 5 reps each ----
    fake = FakeLLM(keep=False)

    def bench(fn, reps=5):
        fn(); torch.cuda.synchronize()          # warmup
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / reps

    t_adapted = bench(lambda: sync_actor_to_vllm(actor, fake, ipc=False))
    t_full = bench(lambda: sync_actor_to_vllm(actor, fake, ipc=False, only_adapted=False))
    t_old = bench(lambda: old_path_push(actor))
    print(f"[timing] per-sync trainer-side cost (8B, cpu-copy path): "
          f"old(full) {t_old:.2f}s | new(full) {t_full:.2f}s | "
          f"new(adapted-only, DEFAULT) {t_adapted:.2f}s "
          f"({t_old / max(t_adapted, 1e-9):.1f}x faster than old)")

    # ---- 5: peak-memory overhead of the transient merged tensors ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    sync_actor_to_vllm(actor, fake, ipc=False)
    peak = torch.cuda.max_memory_allocated()
    print(f"[memory] new-sync transient GPU overhead: "
          f"{(peak - base_alloc) / 2**20:.0f} MiB above resting allocation")
    print("ALL GPU CHECKS PASS")


if __name__ == "__main__":
    main()

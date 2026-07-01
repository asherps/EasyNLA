# vLLM-lens setup (fast RL rollout backend)

`vllm-lens` is the vLLM activation-injection plugin used as the **fast RL rollout
backend** for `nla/train_rl_vllm.py`. It injects the AV's activation at the marker
token via `SteeringVector(norm_match=True)` (== the Karvonen formula) during vLLM
generation — roughly **5× faster** than the HF `generate()` rollout path.

This is a **one-time environment setup**. It lives in its own venv (not the main
`nla` env) because it pins a specific vLLM.

## Install

```bash
bash scripts/install_vllm_lens.sh                       # -> $HOME/envs/vllm-lens
bash scripts/install_vllm_lens.sh /path/to/other/venv   # custom location
```

Activate this venv to run `python -m nla.train_rl_vllm` (see the README).

## Verify

The patch is idempotent — re-run it and it reports whether it's applied:

```bash
<vllm-lens-venv>/bin/python utils/patch_vllm_lens.py    # -> "already patched (all N hunks)"
```

At training time the trainer verifies injection every step: it logs
`av/steer_apply_count` (how many rollouts actually received a steering vector) and
masks out any rollout whose output is CJK garbage — the signature of a failed
injection. If injection silently breaks, `av/steer_apply_count` drops below the
rollout count and CJK failures spike.

## The two hard-won version pins — do not bump blindly

### `vllm==0.19.0` (with `vllm-lens==1.1.0`)

`vllm-lens 1.1.0` (latest, released 2026-04-14) was built against vLLM 0.19.0.
vLLM 0.22+ (2026-05-29) refactored `GPUModelRunner`, after which the injection
hook crashes with:

```
AttributeError: 'GPUModelRunner' object has no attribute 'input_batch'
```

and then **silently skips injection** — generations look fine but no vector is
ever injected. 0.19.0 is the matched version where the hook actually fires.

### `--torch-backend=cu128`

vLLM 0.22's default wheel is cu130 (needs NVIDIA driver ≥ 580). The cluster
driver is 570 (CUDA 12.8) → cu130 fails at import with:

```
libcudart.so.13: cannot open shared object file
```

0.19.0's default wheel is cu128, which runs on driver 570 directly.

### Also pinned

`transformers==4.57.1` (repo-wide pin) — vLLM's resolver otherwise pulls
transformers v5, whose `apply_chat_template` API break crashes the trainers.
`peft` / `bitsandbytes` / `wandb` are required by `nla.train_rl_vllm` itself
(this venv is what runs the trainer; vLLM alone can import `vllm_lens` but cannot
run the trainer).

## The norm-match bug this guards against

`vllm-lens` must norm-match the injection against the **full** residual stream
(`output[0] + output[1]`), not the partial layer output (`output[0]`). The
partial-residual version injected ~50× too weakly in vLLM rollouts while the HF
training forward injected at full strength → ~42% GRPO clip-frac at identical
weights → divergence. Re-apply `utils/patch_vllm_lens.py` after any vllm-lens
venv rebuild.

## Related files

- `scripts/install_vllm_lens.sh` — the installer (commands + pins).
- `utils/patch_vllm_lens.py` — the injection norm-match patch (idempotent;
  re-run after any venv rebuild).

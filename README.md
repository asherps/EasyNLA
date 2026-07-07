# EasyNLA — fast, distributed training of Natural Language Autoencoders

**EasyNLA** trains **Natural Language Autoencoders (NLAs)** — models that *read* a
residual-stream activation and *write* a natural-language explanation of it, then
*reconstruct* the activation back from that text.

The focus here is **fast, distributed training**. It scales with **data
parallelism**: launch under `torchrun` with one rank per GPU, each running its own
single-GPU vLLM rollout engine + trainer on a slice of the batch, gradients
all-reduced every step. Weights resync to the sampler each step over fast GPU→GPU
IPC, and the whole NLA trains end-to-end in hours. On Qwen3-8B (layer 24)
the default config reaches **~70% held-out FVE in ~3–4 hours on 4×H200**.

An NLA has two learned parts:

- **AV (verbalizer)** — a LoRA on the base model. An activation is injected at a
  marker token (norm-matched, à la Karvonen et al.), and the AV writes an
  `<explanation>…</explanation>` of it.
- **AR (reconstructor)** — a truncated copy of the base model + a linear head that
  maps the explanation text back to the activation vector.

Training is on-policy GRPO: after a short SFT warm-start, the AV is rewarded by how
well the AR reconstructs the activation from its words (reward = −reconstruction
MSE). The headline metric is **FVE** (fraction of activation variance explained) on
a held-out split.

> Built on **Celeste's nanoNLA** (https://github.com/ceselder/nanoNLA), a minimal
> implementation of Natural Language Autoencoders (Anthropic, 2026).

> ⚠️ **Primarily built and tested on Qwen3-8B.** The code is written to be
> architecture-generic (tokenization is BOS-safe, layer/module resolution goes
> through `nla/utils/arch_adapters.py`), but other model families (Llama, Gemma,
> GPT-2, …) have not been tested nearly as thoroughly — expect rough edges and
> validate the injection path (e.g. `av/steer_apply_rate`, CJK-free generations)
> before trusting a run on a new architecture.

## Setup

```bash
git clone https://github.com/asherps/EasyNLA.git && cd EasyNLA
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # core deps (torch, transformers, peft, …)
pip install bitsandbytes               # optional: 4-bit (QLoRA) single-GPU training
```

Export credentials (`HF_TOKEN` to download the base model + corpus,
`WANDB_API_KEY` for logging, `ANTHROPIC_API_KEY` for the gold explanations used in
data generation) and point `HF_HOME` at a big disk.

**For the fast distributed (vLLM) path**, build the patched `vllm-lens` rollout venv
once — it lives in its own environment (it pins `vllm==0.19.0`):

```bash
bash scripts/install_vllm_lens.sh                 # creates the venv + applies the patch
# after any rebuild of that venv, re-apply the injection patch:
<vllm-lens-venv>/bin/python utils/patch_vllm_lens.py
```

See [`docs/vllm-lens-setup.md`](docs/vllm-lens-setup.md) for details.

## Fast distributed training (the default)

The tuned defaults live in **`configs/rl_vllm.yaml`** (AV lr 1e-4 / AR 8e-5, batch
256 × group 8, on-policy). Launch under **`torchrun` with one rank per GPU** — each
rank runs its own single-GPU vLLM rollout engine + trainer on a slice of the global
batch, and gradients are all-reduced every step (a full-batch step, N× faster). It
needs **bf16-merged** AV/AR checkpoints (`scripts/merge_lora_to_hf.py`) and the
`vllm-lens` venv:

```bash
torchrun --standalone --nproc_per_node=4 -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
    --base-ckpt Qwen/Qwen3-8B \
    --av-ckpt <merged_av>/hf --ar-ckpt <merged_ar>/hf \
    --rl-parquet <data>/rl_shuf.parquet --sidecar <data>/rl_shuf.parquet \
    --save-dir <ckpts>/rl_vllm \
    --wandb-project easynla --wandb-name rl_vllm
```

> 💡 **Warm-start the RL LoRA from the SFT adapter** when you trained the AV with
> LoRA: `--base-ckpt <raw base> --av-adapter <av_sft adapter dir>` continues tuning
> the SAME adapter SFT trained (and keeps a frozen copy as the KL reference).
> Without it, RL starts from a fresh zero-init LoRA on the merged AV — a measured
> ~12pp FVE cold-start (B=0 puts step 0 in a random rank-r subspace).

**~70% held-out FVE in ~3–4 hours on 4×H200** (Qwen3-8B, layer 24). Set
`--nproc_per_node` to your GPU count (`batch_prompts` is the global batch, split
across ranks). For a model too big for one GPU, raise `--vllm-tp` (tensor-parallel
per rank) so `nproc_per_node × vllm-tp = #GPUs`. Any CLI flag overrides the config;
the merged run config is snapshotted to `<save-dir>/run_config.yaml`.

> ⚠️ The IPC weight sync needs the legacy CUDA allocator — launch with
> `PYTORCH_CUDA_ALLOC_CONF` unset (not `expandable_segments:True`).

**Optional LLM-judge eval** — add `text_judges` to `evals:` (or `--evals base_fve
text_judges`) to score every held-out explanation on **unique_info, coherence,
writing_quality, specificity, repetitiveness** (Opus rubric 1-10 each;
repetitiveness is lower-better) plus
**source_match** (pick the true source among 4 candidates; chance 25%) every
`--text-judges-every` steps, reusing the FVE eval's generations. Needs
`ANTHROPIC_API_KEY`; logs to `eval_judge/*`. Tracks the classic RL text-degradation
modes (writing-quality drift, repetitive filler) that reconstruction FVE alone
can't see — NB: only extraction-successful outputs are judged, so read the
rubric means together with `eval/extraction_rate` (a full format collapse
shows up there, not in these means).

## Full recipe (any decoder LM)

You need three things before RL: **data** (activations + gold explanations) and an
**AV** + **AR** warm-start. The end-to-end recipe is in
**[`docs/train_new_model.md`](docs/train_new_model.md)**:

```bash
# 1. data — activations + gold explanations (edit the datagen config head first)
python -m nla.datagen.run_pipeline --config configs/datagen/qwen3_8b_finefineweb_100k.yaml
#    → av_sft_shuf.parquet · ar_sft_shuf.parquet · rl_shuf.parquet (+ .nla_meta.yaml sidecars)

# 2. warm-start the verbalizer (AV) and the reconstructor (AR) — one epoch each
python -m nla.train_sft --mode av --base-ckpt <model> --parquet <data>/av_sft_shuf.parquet \
    --sidecar <data>/av_sft_shuf.parquet --save-dir <ckpts>/av --use-lora --quant 4bit --lr 1e-4 ...
python -m nla.train_sft --mode ar --base-ckpt <model> --parquet <data>/ar_sft_shuf.parquet \
    --sidecar <data>/ar_sft_shuf.parquet --save-dir <ckpts>/ar --use-lora --quant 4bit --lr 2e-5 \
    --ar-num-layers <layer_index + 1> ...

# 3. RL — the fast distributed command above (or the single-GPU fallback below)
```

### Single-GPU fallback

No spare GPUs or don't want the vLLM setup? `configs/rl_sgpu.yaml` runs the whole
loop on one GPU in 4-bit with HF `generate()` — no checkpoint merge, no vllm-lens.
Slower, but the easiest way to get an NLA training:

```bash
python -m nla.train_rl_self_contained --config configs/rl_sgpu.yaml \
    --base-ckpt <model> --quant 4bit \
    --av-ckpt <ckpts>/av/iter_XXXX --ar-ckpt <ckpts>/ar/iter_XXXX \
    --rl-parquet <data>/rl_shuf.parquet --sidecar <data>/rl_shuf.parquet \
    --save-dir <ckpts>/rl_sgpu --wandb-project easynla --wandb-name rl_sgpu
```

### Inspect a trained NLA

```bash
python scripts/show_nla_generations.py --av-lora <av_dir> --ar-ckpt <ar_dir> \
    --sidecar <data>/rl_shuf.parquet --parquet <data>/rl_shuf.parquet
```

## Layout

```
nla/
  config.py schema.py storage.py     # the sidecar "contract" (marker token, scales, templates)
  injection.py                       # Karvonen norm-matched activation injection
  models.py                          # AR reconstructor (truncated backbone + value head)
  train_sft.py                       # AV / AR warm-start SFT
  train_rl_vllm.py                   # data-parallel vLLM GRPO RL (the fast path)
  train_rl_self_contained.py         # single-GPU GRPO RL
  datagen/                           # activation extraction + gold-explanation pipeline
  utils/                             # hooks, prompts, critic, logging, steering, config layer
configs/                             # tuned run configs (rl_vllm, rl_sgpu, datagen/*)
docs/                                # train_new_model.md, vllm-lens-setup.md
scripts/                             # merge_lora_to_hf, compute_fve_baseline, show_nla_generations, install_vllm_lens
utils/patch_vllm_lens.py            # required patch for the vLLM rollout path
```

## License

MIT. Built on [Celeste's nanoNLA](https://github.com/ceselder/nanoNLA).

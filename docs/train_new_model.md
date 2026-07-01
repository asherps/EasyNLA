# Training an NLA on a new model

End-to-end recipe for training a Natural Language Autoencoder (AV verbalizer +
AR reconstructor) on **any** decoder LM:
**activation extraction → AV/AR warm-start SFT → GRPO RL**, all logged to Weights
& Biases. Everything runs from this repo's self-contained trainers
(`nla/train_sft.py`, `nla/train_rl_self_contained.py`, `nla/train_rl_vllm.py`) —
no external RL framework needed.

> ⚠️ Only **Qwen3-8B** has been trained end-to-end with this exact code. The
> pipeline is written to be arch-generic (BOS-safe tokenization, layer/module
> resolution via `nla/utils/arch_adapters.py`), but other families are far less
> tested. On a new model, smoke-test each stage (a `--stages 0,1` extraction, a
> short SFT, a few RL steps) and watch the injection-health metrics
> (`av/steer_apply_rate`, `av/inject_fail_count`) before committing to a run.

Prereqs on the box:

```bash
export HF_TOKEN=...           # model + corpus download
export WANDB_API_KEY=...      # logging
export ANTHROPIC_API_KEY=...  # stage-2 gold explanations only
export HF_HOME=/somewhere/big # keep model/dataset caches off the boot disk
```

## 0 · Choose the layer

NLAs read a residual-stream layer about **two-thirds deep** (semantic features
have formed; the unembedding hasn't taken over). **No code change is needed for a
new model** — `run_pipeline` reads `base_model` + `layer_index` from the datagen
YAML:

```
layer_index = (2 * num_layers) // 3      # Qwen3-0.6B 28→18 · Qwen3-8B 36→24 · Llama-3.3-70B 80→53
```

(If you'll reuse a model, optionally add a `ModelPreset` to
`nla/datagen/model_presets.py` and reference it with `model: <key>`.)

## 1 · Generate data

Copy a `configs/datagen/*.yaml`, edit the head, run the orchestrator:

```yaml
base_model: <your/model>
layer_index: <2/3 depth>
output_dir: /data/nla/<run>
corpus:
  name: <hf-dataset | local.parquet>    # any text corpus; a .parquet path uses Dataset.from_parquet
  split: train
  text_column: text
  start: 0
  length: 100000                        # docs × positions_per_doc = #activations
stage0: {positions_per_doc: 10, chunk_size: 256, seed: 42,
         extractor_kwargs: {batch_size: 12, max_length: 4096}}
stage1: {av_sft_frac: 0.25, ar_sft_frac: 0.25, rl_frac: 0.50, seed: 42}
stage2:
  provider_cls: nla.datagen.providers.BatchAnthropicProvider
  provider_kwargs: {model: claude-sonnet-4-6, max_tokens: 300, max_batch_size: 10000}
  chunk_size: 50000
stage3: {keep_debug_metadata: true}
shuffle: {enabled: true, seed: 42}
storage_cls: nla.datagen.storage.LocalStorage
```

```bash
python -m nla.datagen.run_pipeline --config configs/datagen/<your>.yaml
#  → av_sft_shuf.parquet · ar_sft_shuf.parquet · rl_shuf.parquet  (+ .nla_meta.yaml sidecars)
```

Stage 0 (extraction) is the **only model-specific step** — it forward-hooks
`layer_index` and writes raw activations. Stages 1–3 are model-agnostic.
Smoke-test extraction cheaply with `--stages 0,1` (skips the paid API
explanations); a few dozen docs is enough to confirm the dims are right.

The sidecar (`*.nla_meta.yaml`) is the **contract**: it records the marker token,
prompt templates, `injection_scale`, `mse_scale`, and `d_model`, all asserted
against the live tokenizer at every trainer startup. Never hardcode these.

## 2 · AV SFT — verbalizer (activation → text)

`nla/train_sft.py --mode av` loads the base model (4-bit + LoRA by default for
single-GPU budgets), hooks the Karvonen norm-matched injection at the layer-1
output, and trains cross-entropy on response tokens only:

```bash
python -m nla.train_sft --mode av --base-ckpt <your/model> \
  --parquet /data/nla/<run>/av_sft_shuf.parquet \
  --sidecar /data/nla/<run>/av_sft_shuf.parquet \
  --save-dir /ckpts/nla/<run>_av \
  --num-steps 1000 --batch-size 64 --gradient-accumulation-steps 1 \
  --use-lora --quant 4bit --lora-r 128 --lora-alpha 16 \
  --lr 1e-4 --min-lr 1e-5 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --save-every 500 --wandb-project nla-<run> --wandb-name av_sft --seed 0
```

Train **one epoch** — a second epoch overfits (held-out/downstream FVE regresses
while train loss keeps dropping). For a bf16 (non-quantized) full-fine-tune warm
start — which the vLLM RL path needs, since vLLM loads a full HF model — drop
`--quant 4bit --use-lora` and merge is unnecessary (train_sft saves HF format
directly).

## 3 · AR SFT — reconstructor (text → activation)

Same entry point with `--mode ar`. The trainer truncates the base model
in-process to `--ar-num-layers` blocks + a `Linear(d, d)` value head — **set
`--ar-num-layers` to `layer_index + 1`** (the critic needs the *output of* block
K, so block K must exist). The final RMSNorm is stripped by default
(recorded in `ar_meta.json` inside the checkpoint):

```bash
python -m nla.train_sft --mode ar --base-ckpt <your/model> \
  --parquet /data/nla/<run>/ar_sft_shuf.parquet \
  --sidecar /data/nla/<run>/ar_sft_shuf.parquet \
  --save-dir /ckpts/nla/<run>_ar \
  --num-steps 1000 --batch-size 64 --gradient-accumulation-steps 1 \
  --use-lora --quant 4bit --lora-r 128 --lora-alpha 16 \
  --ar-num-layers <layer_index + 1> \
  --lr 2e-5 --min-lr 2e-6 --lr-warmup-steps 50 --max-grad-norm 1.0 \
  --save-every 500 --wandb-project nla-<run> --wandb-name ar_sft --seed 0
```

For an honest held-out FVE during training, pass `--heldout-parquet` pointing at
a doc-disjoint parquet (e.g. the AV split — disjoint from AR data by stage-1
construction). Checkpoints save in HF format directly — no conversion step.

## 4 · RL — GRPO (reward = −reconstruction MSE)

The trainer rolls out the AV, scores each explanation with the AR critic, and
does GRPO with a group-relative baseline. **`--train-critic` is on by default and
matters**: with a frozen critic the reward is static, advantages collapse to ≈0,
and RL does nothing.

**Simple path — single GPU, 4-bit, no merge/patch** (`nla.train_rl_self_contained`):

```bash
python -m nla.train_rl_self_contained --config configs/rl_sgpu.yaml \
  --av-ckpt /ckpts/nla/<run>_av/iter_0001000 \
  --ar-ckpt /ckpts/nla/<run>_ar/iter_0001000 \
  --base-ckpt <your/model> --quant 4bit \
  --rl-parquet /data/nla/<run>/rl_shuf.parquet \
  --sidecar    /data/nla/<run>/rl_shuf.parquet \
  --save-dir   /ckpts/nla/<run>_rl \
  --wandb-project nla-<run> --wandb-name rl_grpo --seed 0
```

`--config configs/rl_sgpu.yaml` supplies the tuned single-GPU hyperparameters;
any CLI flag overrides it. `--av-ckpt` is the AV LoRA dir; `--ar-ckpt` is the AR
LoRA dir (must contain `ar_meta.json`). Resume with `--resume-from-lora
<iter_dir> --start-step <N>`.

**Fast/best path — multi-GPU vLLM rollouts** (`nla.train_rl_vllm`, TP over N
GPUs). Needs bf16-merged AV/AR checkpoints (`scripts/merge_lora_to_hf.py`) and
the patched vllm-lens venv (`scripts/install_vllm_lens.sh` +
`docs/vllm-lens-setup.md`). This is what the tuned defaults in
`configs/rl_vllm.yaml` (AV lr 1e-4 / AR 8e-5) target — ~67% held-out FVE on
Qwen3-8B vs a ~50% single-vector baseline:

```bash
python -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
  --av-ckpt <merged_av>/hf --ar-ckpt <merged_ar>/hf --base-ckpt <your/model> \
  --rl-parquet /data/nla/<run>/rl_shuf.parquet --sidecar /data/nla/<run>/rl_shuf.parquet \
  --save-dir /ckpts/nla/<run>_rl_vllm --vllm-tp <N> \
  --wandb-project nla-<run> --wandb-name rl_vllm --seed 0
```

Every stage streams to wandb; checkpoints land in each `--save-dir`. Inspect a
trained NLA with `python scripts/show_nla_generations.py --av-lora <av_dir>
--ar-ckpt <ar_dir> --sidecar <rl_shuf.parquet> --parquet <rl_shuf.parquet>`.
That's the whole loop — extraction → warm-start → RL.

---

### Note

The datagen/extraction path is model-agnostic and has been run on Qwen3-0.6B
(layer 18, no preset, no code change) and Qwen3-8B (layer 24). Only stage 0
touches the model; everything downstream is generic.

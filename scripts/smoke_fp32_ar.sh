#!/bin/bash
#SBATCH --job-name=easynla_fp32_ar_smoke
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

DATA=/workspace-vast/asherps/nla-data/qwen3_8b_finefineweb_100k_v2
cd /workspace-vast/asherps/nla-train

# AR full-FT smoke on the NEW fp32+autocast path. lr 2e-5 is the case where
# pure bf16 froze ~61% of weights + all norm params — the sharpest test.
# Success: FVE trending up, loss decreasing, norm params MOVED post-hoc.
srun /workspace-vast/asherps/envs/nla/bin/python -m nla.train_sft \
    --mode ar --base-ckpt Qwen/Qwen3-8B \
    --parquet  $DATA/ar_sft_shuf.parquet \
    --sidecar  $DATA/ar_sft_shuf.parquet \
    --save-dir /workspace-vast/asherps/exp/training/easynla_fp32_smoke_ar \
    --num-steps 150 --save-every 150 --heldout-every 75 \
    --wandb-project easynla --wandb-group warmstart --wandb-name fp32_smoke_ar \
    --wandb-tags fp32-smoke

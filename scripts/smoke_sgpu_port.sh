#!/bin/bash
#SBATCH --job-name=easynla_sgpu_port_smoke
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

DATA=/workspace-vast/asherps/nla-data/qwen3_8b_finefineweb_100k
CKPT=/workspace-vast/asherps/nla-ckpts
cd /workspace-vast/asherps/nla-train

# End-to-end smoke of the sgpu twin AFTER the perf ports (selective logp,
# batched critic scoring) + twin-drift fixes (truncation boundary, eval temp).
# Success: steps run, FVE in the warmstart band and trending, ext ~100%,
# no OOM at the same micro-batch (memory should be LOWER than before).
srun /workspace-vast/asherps/envs/nla/bin/python -m nla.train_rl_self_contained \
    --config configs/rl_sgpu.yaml \
    --base-ckpt Qwen/Qwen3-8B --quant 4bit \
    --av-ckpt $CKPT/qwen3_8b_L24_av_sft_noncomp_2ep_4bit_2e4/iter_0003863 \
    --ar-ckpt $CKPT/qwen3_8b_L24_ar/iter_0001000 \
    --rl-parquet $DATA/rl_shuf.parquet --sidecar $DATA/rl_shuf.parquet \
    --save-dir /workspace-vast/asherps/exp/training/easynla_sgpu_port_smoke \
    --num-steps 12 --batch-prompts 8 --group-size 8 \
    --eval-every 6 --eval-n-prompts 32 --save-every 12 \
    --wandb-project easynla --wandb-name sgpu_port_smoke --wandb-tags sgpu-port-smoke

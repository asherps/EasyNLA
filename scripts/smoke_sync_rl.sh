#!/bin/bash
#SBATCH --job-name=easynla_sync_rl_smoke
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-03:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
# NO PYTORCH_CUDA_ALLOC_CONF: ipc_weight_sync needs the legacy allocator.
# vLLM v1 spawns EngineCore in a subprocess; apply_model ships a functools.partial
# which msgspec refuses without this (same as the main repo's rl_vllm launchers).
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

DATA=/workspace-vast/asherps/nla-data/qwen3_8b_finefineweb_100k
CKPT=/workspace-vast/asherps/nla-ckpts
cd /workspace-vast/asherps/nla-train

# END-TO-END validation of the out-of-place weight sync against a REAL vLLM
# engine (the unit tests use a fake). Every step pushes adapted-only merged
# weights via CUDA-IPC. Success criteria:
#   - step-0 eval FVE ~ warmstart level (~50%), does NOT crater after the
#     first syncs (a botched sync = vLLM diverges from HF -> FVE tanks fast)
#   - no CJK in generations, steer_apply_rate ~100%, KL/entropy sane
srun /workspace-vast/asherps/envs/vllm-lens/bin/python -m nla.train_rl_vllm \
    --config configs/rl_vllm.yaml \
    --base-ckpt Qwen/Qwen3-8B \
    --av-ckpt $CKPT/qwen3_8b_L24_av_sft_noncomp_2x_lr1e4/iter_0005801 \
    --ar-ckpt $CKPT/qwen3_8b_L24_ar_bf16/iter_0001000/hf \
    --rl-parquet $DATA/rl_shuf.parquet --sidecar $DATA/rl_shuf.parquet \
    --save-dir /workspace-vast/asherps/exp/training/easynla_sync_smoke_rl2 \
    --num-steps 25 --batch-prompts 32 --group-size 8 \
    --eval-every 5 --eval-n-prompts 64 --save-every 25 \
    --wandb-project easynla --wandb-name sync_smoke_rl --wandb-tags sync-smoke

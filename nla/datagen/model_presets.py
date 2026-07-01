"""Model presets — one source of truth for per-model datagen constants.

layer/d_model/batch_size were scattered across eight yaml configs and a half-
dozen scratch scripts, each re-deriving the 2/3-depth rule and getting
the Gemma device_map gotcha right (or not). Yaml sets `model: gemma27b`,
run_pipeline calls resolve(), and everyone reads the same numbers.

Zero torch/transformers imports — this module loads in <10ms so the wildchat
extractors can pull constants without the full datagen import tree.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelPreset:
    hf_name: str
    num_layers: int
    d_model: int
    extractor_kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    # Chat-template structure — only matters for conversation-corpus extractors
    # (wildchat etc.) that need to find where the interesting tokens start.
    # UltraFineWeb goes through stage0's raw-text path, doesn't touch these.
    turn_marker: str = ""
    accepts_system_role: bool = True

    @property
    def default_layer(self) -> int:
        """2/3 depth — semantic features have formed, prediction-head prep
        hasn't yet dominated the residual stream. qwen7b 28→18, gemma12b 48→32,
        gemma27b 62→41. qwen configs explicitly set layer_index=20 (historical
        choice existing checkpoints depend on — not derived from this rule)."""
        return (2 * self.num_layers) // 3


# device_map="cuda:0" on Gemma: multigpu.sh sets CUDA_VISIBLE_DEVICES=$i so
# only one GPU is visible per process. "auto" triggers accelerate's memory
# estimator which, on residual 1-2GB from prior contexts, decides to CPU-
# offload → meta-tensor crash at forward ("Some parameters are on the meta
# device"). Explicit "cuda:0" bypasses the estimator — fail loud with honest
# CUDA OOM instead of silent offload. Observed 2026-03-13 on 27b.
MODELS: dict[str, ModelPreset] = {
    "qwen7b": ModelPreset(
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        num_layers=28,
        d_model=3584,
        extractor_kwargs={"batch_size": 16, "max_length": 4096},
        turn_marker="<|im_start|>",
        accepts_system_role=True,
    ),
    "qwen3_8b": ModelPreset(
        hf_name="Qwen/Qwen3-8B",
        num_layers=36,
        d_model=4096,
        extractor_kwargs={"batch_size": 12, "max_length": 4096},
        turn_marker="<|im_start|>",
        accepts_system_role=True,
    ),
    "gemma12b": ModelPreset(
        hf_name="google/gemma-3-12b-it",
        num_layers=48,
        d_model=3840,
        extractor_kwargs={"batch_size": 8, "max_length": 4096, "device_map": "cuda:0"},
        turn_marker="<start_of_turn>",
        accepts_system_role=False,
    ),
    "gemma27b": ModelPreset(
        hf_name="google/gemma-3-27b-it",
        num_layers=62,
        d_model=5376,
        extractor_kwargs={"batch_size": 4, "max_length": 4096, "device_map": "cuda:0"},
        turn_marker="<start_of_turn>",
        accepts_system_role=False,
    ),
    "llama70b": ModelPreset(
        hf_name="meta-llama/Llama-3.3-70B-Instruct",
        num_layers=80,
        d_model=8192,
        # 70B on a single GPU needs offload; device_map="auto" is fine here since
        # there's no batch-size-driven meta-device flap (batch_size=1, max_length=4096
        # → stable memory footprint). Multi-GPU shard works too.
        extractor_kwargs={"batch_size": 1, "max_length": 4096, "device_map": "auto"},
        turn_marker="<|start_header_id|>",
        accepts_system_role=True,
    ),
}


def resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand `model: <key>` into concrete fields. Explicit yaml keys win.

    base_model can still be a local path even with a preset: the 27b HF download
    rate-limited after 8/12 shards; we staged to NFS and pointed base_model
    there while keeping `model: gemma27b` for everything else.
    """
    key = cfg.get("model")
    if key is None:
        return cfg
    assert key in MODELS, f"unknown model preset {key!r}, have: {sorted(MODELS)}"
    m = MODELS[key]
    cfg.setdefault("base_model", m.hf_name)
    cfg.setdefault("layer_index", m.default_layer)
    cfg.setdefault("stage0", {})
    cfg["stage0"].setdefault("extractor_kwargs", dict(m.extractor_kwargs))
    return cfg

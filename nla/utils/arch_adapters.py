"""Architecture adapters — one place for multimodal-wrapper unwrapping.

Some HF checkpoints (Gemma-3, LLaVA-style) load as multimodal wrappers even
via `AutoModelForCausalLM`. NLA only cares about the text side. Rather than
scattering `getattr(x, "text_config", x)` / `getattr(x, "language_model", x)`
across train_actor/models/extractors, these functions centralize the unwrap
with explicit arch detection.

Pass-through for plain text models (Qwen/Llama/Mistral): they have none of
the wrapper attributes, so all functions return the input unchanged.

Add new wrapped architectures by extending _WRAPPER_MODEL_ATTRS /
_WRAPPER_CONFIG_ATTRS — don't duck-type new getattr fallbacks at callsites.
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM

# Multimodal wrapper → text model attribute name.
# Gemma3ForConditionalGeneration.language_model → Gemma3ForCausalLM
# (LLaVA-style wrappers would go here too if we ever support them)
_WRAPPER_MODEL_ATTRS = ("language_model",)

# Multimodal wrapper config → text config attribute name.
# Gemma3Config.text_config → Gemma3TextConfig (has hidden_size, num_hidden_layers)
_WRAPPER_CONFIG_ATTRS = ("text_config",)


def resolve_text_config(config: Any) -> Any:
    """Return the text-side config for multimodal wrappers; pass-through otherwise.

    Gemma3Config nests hidden_size/num_hidden_layers under .text_config.
    Qwen2Config/LlamaConfig/MistralConfig have those at top level and no
    .text_config attr → return as-is.
    """
    for attr in _WRAPPER_CONFIG_ATTRS:
        nested = getattr(config, attr, None)
        if nested is not None:
            return nested
    return config


def resolve_text_model(model: Any) -> Any:
    """Return the text-side CausalLM for multimodal wrappers; pass-through otherwise.

    Invariant: always returns a CausalLM-shaped model (has .model + .lm_head),
    so `save_pretrained()` / `AutoModelForCausalLM.from_pretrained()` roundtrip.

    Gemma3ForConditionalGeneration.language_model is a Gemma3TextModel (bare
    transformer, NO .lm_head, NO .model wrapper). Returning it directly means
    `save_pretrained` writes keys like `layers.0.*` but `from_pretrained` via
    AutoModelForCausalLM loads Gemma3ForCausalLM expecting `model.layers.0.*`
    → zero keys match → everything random-inits. Observed Mar 13 2026:
    pred_norm=507 vs gold_norm=75616 on same input, step-0 loss=2.0 (orthogonal).

    Qwen/Llama/Mistral have no .language_model → pass through unchanged
    (already CausalLM-shaped).
    """
    for attr in _WRAPPER_MODEL_ATTRS:
        nested = getattr(model, attr, None)
        if nested is None:
            continue
        if hasattr(nested, "lm_head"):
            return nested  # already CausalLM-shaped
        # Bare TextModel — wrap in CausalLM so keys roundtrip. meta device
        # avoids materializing a throwaway 12B random model.
        with torch.device("meta"):
            wrapper = AutoModelForCausalLM.from_config(nested.config)
        wrapper.model = nested  # transplant pretrained weights
        # lm_head is on meta. Critic caller strips it to Identity (harmless).
        # Actor caller (train_actor.NLATextOnlyCausalLM) needs a real one to
        # generate — Gemma ties it to embed_tokens, so tie_weights() points
        # lm_head.weight at the real embedding tensor (no extra alloc).
        # Non-tied archs would need the caller to load lm_head separately;
        # cross that bridge when we hit one.
        if getattr(nested.config, "tie_word_embeddings", False):
            wrapper.tie_weights()
        return wrapper
    return model


def resolve_decoder_layers(model: Any) -> torch.nn.ModuleList:
    """Find the decoder layers ModuleList, unwrapping multimodal wrappers first.

    After resolve_text_model (always returns CausalLM-shaped):
      Llama/Qwen/Mistral/Gemma: model.model.layers
      GPT-2/Falcon: model.transformer.h
    """
    model = resolve_text_model(model)
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise AssertionError(
            f"{type(model).__name__} has neither .model nor .transformer — "
            f"extend arch_adapters.resolve_decoder_layers for this architecture"
        )
    assert isinstance(layers, torch.nn.ModuleList), (
        f"resolved {type(layers).__name__}, expected nn.ModuleList. "
        f"Module path is wrong for {type(model).__name__}."
    )
    return layers


def is_multimodal_wrapper(config_or_model: Any) -> bool:
    """True if this is a known multimodal wrapper (has nested text config/model)."""
    for attr in (*_WRAPPER_CONFIG_ATTRS, *_WRAPPER_MODEL_ATTRS):
        if getattr(config_or_model, attr, None) is not None:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Embedding forward scale — for RL rollout injection.
#
# Some archs' embedding modules multiply by a scale in their forward():
#   - Gemma: Gemma3TextScaledWordEmbedding × √d_model
#   - T5-style: similar √d_model scaling
# Plain nn.Embedding (Qwen/Llama/Mistral): no scale.
#
# Training-side injection is unaffected (the forward hook captures POST-scale
# output). RL rollout builds embeds manually from raw weights → must apply
# the scale or injection magnitude is off by ~62× (Gemma).
#
# This is an EXPLICIT registry — don't try to auto-detect. If you add a
# scaled-embed arch, add it here AND the callsite will visibly multiply.
# ─────────────────────────────────────────────────────────────────────────

# model_type → scale expression. √d_model is the common case; if an arch
# uses a different formula, add a lambda here.
_SCALED_EMBED_MODEL_TYPES: dict[str, str] = {
    "gemma3": "sqrt_d_model",
    "gemma3_text": "sqrt_d_model",
    "gemma2": "sqrt_d_model",
    "gemma": "sqrt_d_model",
    "t5": "sqrt_d_model",
}


def resolve_embed_scale(config: Any) -> float:
    """Return the scalar that the model's embedding forward() multiplies by.

    For plain nn.Embedding archs (Qwen/Llama/Mistral): 1.0.
    For Gemma/T5-style: √d_model.

    RL rollout uses this to scale raw-weight-lookup embeds to match what
    the model's forward would produce. Training doesn't need it (forward
    hook captures post-scale output). See `_SCALED_EMBED_MODEL_TYPES`
    and extend there for new scaled-embed archs.
    """
    text_config = resolve_text_config(config)
    model_type = getattr(text_config, "model_type", "")
    rule = _SCALED_EMBED_MODEL_TYPES.get(model_type)
    if rule is None:
        return 1.0
    if rule == "sqrt_d_model":
        return float(text_config.hidden_size) ** 0.5
    raise AssertionError(
        f"unknown embed-scale rule {rule!r} for model_type={model_type!r} — "
        f"extend resolve_embed_scale in arch_adapters.py"
    )


# LoRA target modules for the attention block, by architecture. The llama
# lineage (Qwen/Mistral/Gemma/...) all use q/k/v/o_proj; GPT-2 fuses QKV into
# c_attn; Falcon/GPT-NeoX fuse into query_key_value. Hardcoding q_proj at a
# callsite makes PEFT raise "target modules not found" (best case) or silently
# adapt nothing on fused-QKV archs.
_LLAMA_FAMILY_MODEL_TYPES = {
    "llama", "llama4", "mistral", "mixtral",
    "qwen2", "qwen2_5", "qwen2_5_vl", "qwen3", "qwen3_moe",
    "gemma", "gemma2", "gemma3", "gemma3_text",
    "phi", "internlm2", "olmo", "olmo2",
}


def resolve_attn_target_modules(config: Any) -> list[str]:
    """Attention-projection module names for LoRA, resolved by model_type.

    Unwraps multimodal configs first (Gemma-3 nests model_type under
    text_config). Raises loudly for unknown archs instead of letting PEFT
    silently adapt nothing / crash mid-init.
    """
    model_type = getattr(resolve_text_config(config), "model_type", "")
    if model_type in _LLAMA_FAMILY_MODEL_TYPES:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if model_type == "gpt2":
        return ["c_attn", "c_proj"]
    if model_type in ("falcon", "gpt_neox"):
        return ["query_key_value", "dense"]
    if model_type == "phi3":
        return ["qkv_proj", "o_proj"]
    raise AssertionError(
        f"model_type={model_type!r}: unknown attention module naming — extend "
        f"arch_adapters.resolve_attn_target_modules for this architecture."
    )
